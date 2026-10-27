using Test
using SmoothPeriodicStatsModels
using Distributions, Random
using Optimization, OptimizationOptimJL
using SimpleNonlinearSolve, ForwardDiff # with Optimization, these trigger the fitting extension
using Ipopt, OptimizationMOI

const SPSM = SmoothPeriodicStatsModels

# * Fitting extension * #

# Fitting lives in an extension, so a missing `using` or `import` in it shows up only when a
# given entry point is first called. This calls every one of them on a tiny problem, so that an
# import regression fails here in seconds instead of somewhere deep in the suite.
@testset "every fitting entry point is reachable" begin
    @test !isnothing(Base.get_extension(SPSM, :SmoothPeriodicStatsModelsOptimizationExt))

    K, D, T, order, deg, N = 2, 3, 5, 1, 1, 400
    rng = MersenneTwister(1)
    hmm = Trig2ARPeriodicHMM(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1),
        0.5randn(rng, K, D, 2^order, 2deg + 1), T)
    n2t = n_to_t(N, T)
    y_past = rand(rng, Bool, D, order)
    z, y = rand(MersenneTwister(2), hmm, n2t; y_ini=y_past, z_ini=1, seq=true)

    @test length(fit_θ(hmm, deg)) == 2
    @test length(fit_θ!(copy(hmm), deg)) == 2
    @test length(fit_mle(hmm, 0.5randn(rng, K, K - 1, 2deg + 1),
        0.5randn(rng, K, D, 2^order, 2deg + 1), y, y_past; n2t=n2t, maxiter=2, tol=0.0)) == 4
    @test length(fit_mle_RO(y, y_past, z, n2t, deg, K, T)) == 2

    y_ar = rand(AR1([0.1t for t = 1:T], fill(0.3, T), fill(0.5, T)), 500; y₁=0.0)
    @test length(fit_mle_AR1(y_ar, deg, T)) == 3
    @test length(fit_mle_AR1(y_ar, rand(rng, 1:2, 500), deg, T)) == 3

    mix(t, θ) = MixtureModel([Exponential(σₜ(t, θ[:, 1])), Exponential(σₜ(t, θ[:, 2]))],
        [αₜ(t, θ[:, 3]), 1 - αₜ(t, θ[:, 3])])
    θtrue = hcat([1.0, 1, -0.2], [3.0, -0.5, 0.6], [0.1, -0.5, 1.0])
    θ0 = hcat([0.0, 0, 0], [5.0, 0, 0], [0.0, 0.2, 0])
    Tm, Nm = 20, 400
    n2tm = repeat(1:Tm, Nm ÷ Tm)
    ym = [rand(rng, mix(t / Tm, θtrue)) for t in n2tm]
    @test length(fit_mle([mix(t / Tm, θ0) for t = 1:Tm], permutedims(θ0[:, 3]),
        [θ0[:, 1], θ0[:, 2]], ym, n2tm; maxiter=2, tol=0.0)) == 4

    ℓ(θ, x) = sum(abs2, θ .- 1.0)
    @test fit_mle(OptimMLE(ℓ, [0.0, 0.0]), [1.0]).u ≈ [1.0, 1.0] atol = 1e-6
    @test fit_loss_optim(ℓ, [1.0], [0.0, 0.0]).u ≈ [1.0, 1.0] atol = 1e-6
end


# A user-supplied emission family, added from outside the package by a single `update_dist`
# method -- the extension point documented in the README. Guards that the seasonal EM is generic
# in the component type: nothing in the E-step or the weight M-step may assume `Exponential`.
@testset "Extensibility: a user-defined seasonal family" begin
    rng = MersenneTwister(1234)

    # Seasonal Rayleigh, scale S(t) = exp(P(t)): logpdf = log y - 2log S - y^2/(2S^2)
    rayleigh_nll(theta, (s_pi, s_piy2, basis)) =
        sum(2 * s_pi[t] * P + s_piy2[t] * exp(-2P) / 2 for (t, P) in enumerate(basis * theta))

    function SPSM.update_dist(dist::AbstractVector{<:Rayleigh}, theta_Y, gamma, obs, n_in_t,
        basis; solver=default_solver(), adtype=default_adtype(), solve_kwargs=(;))
        T = length(dist)
        theta_Y = copy(theta_Y)
        s_pi = [sum(gamma[n] for n in n_in_t[t]; init=0.0) for t = 1:T]
        s_piy2 = [sum(gamma[n] * obs[n]^2 for n in n_in_t[t]; init=0.0) for t = 1:T]
        theta_Y[:] = solve_unconstrained(rayleigh_nll, collect(theta_Y[:]),
            (s_pi, s_piy2, basis); solver, adtype, solve_kwargs)
        return Rayleigh.([exp(polynomial_trigo(t / T, theta_Y[:])) for t = 1:T]), theta_Y
    end

    T, N = 20, 4000
    theta_true = hcat([0.3, 0.4, -0.2], [1.6, -0.3, 0.5], [0.1, -0.5, 1.0])
    mixR(t, th) = MixtureModel(
        [Rayleigh(exp(polynomial_trigo(t, th[:, 1]))), Rayleigh(exp(polynomial_trigo(t, th[:, 2])))],
        [αₜ(t, th[:, 3]), 1 - αₜ(t, th[:, 3])])

    n2t = repeat(1:T, N ÷ T)
    y = [rand(rng, mixR(t / T, theta_true)) for t in n2t]

    theta0 = hcat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.2, 0.0])
    mix_fit, _, theta_Y, history = fit_mle([mixR(t / T, theta0) for t = 1:T],
        permutedims(theta0[:, 3]), [theta0[:, 1], theta0[:, 2]], y, n2t; maxiter=200, tol=1e-5)

    @test history["converged"]
    @test all(diff(history["logtots"]) .>= -1e-8)             # EM still monotone
    @test components(mix_fit[1])[1] isa Rayleigh              # the family survived the loop
    @test theta_Y[1] ≈ theta_true[:, 1] atol = 0.15      # and the truth is recovered
    @test theta_Y[2] ≈ theta_true[:, 2] atol = 0.15
end

# * Utilities * #

@testset "Utilities: trigonometric parametrization" begin
    β = [0.3, -0.2, 0.5, 0.1, -0.4] # degree 2
    T = 12

    # Degree 0 is the constant polynomial
    @test SPSM.polynomial_trigo(0.3, [1.5]) == 1.5
    @test SPSM.polynomial_trigo([0.1, 0.7], [1.5]) == [1.5, 1.5]

    # Vectorized versions agree with the scalar ones
    @test SPSM.polynomial_trigo(collect(1:T) / T, β) ≈ [SPSM.polynomial_trigo(t / T, β) for t = 1:T]

    # Periodicity: `t` is in fractions of a period
    @test SPSM.polynomial_trigo((3 + T) / T, β) ≈ SPSM.polynomial_trigo(3 / T, β)
    @test SPSM.polynomial_trigo(0.25 + 1, β) ≈ SPSM.polynomial_trigo(0.25, β)

    # Link functions: μ is the raw polynomial, the others map it to the parameter domain
    for t in range(0, 1, length=21)
        P = SPSM.polynomial_trigo(t, β)
        @test μₜ(t, β) ≈ P
        @test σₜ(t, β) ≈ exp(P)
        @test αₜ(t, β) ≈ 1 / (1 + exp(-P))
        @test ρₜ(t, β) ≈ 2αₜ(t, β) - 1
        @test σₜ(t, β) > 0
        @test 0 < αₜ(t, β) < 1
        @test -1 < ρₜ(t, β) < 1
    end
end

@testset "Utilities: periodic indexing and memory encoding" begin
    @test n_to_t(7, 3) == [1, 2, 3, 1, 2, 3, 1]
    @test n_to_t(6, 3) == [1, 2, 3, 1, 2, 3]
    @test length(n_to_t(100, 7)) == 100
    @test all(1 .≤ n_to_t(100, 7) .≤ 7)

    # `bin2digit` indexes the 2^order possible memories, first entry = most significant bit
    @test SPSM.bin2digit([false]) == 1
    @test SPSM.bin2digit([true]) == 2
    @test SPSM.bin2digit((true, false)) == 3
    @test sort(vec([SPSM.bin2digit([b₁, b₂]) for b₁ in (false, true), b₂ in (false, true)])) == 1:4

    Y = Bool[0 1 0; 1 1 0]           # D = 2 sites, N = 3 observations
    # Without memory every observation falls in the single category 1
    @test SPSM.conditional_to(Y, zeros(Bool, 2, 0)) == ones(Int, 2, 3)
    # With memory 1 the category is `1 + previous value`, the first column coming from `Y_past`
    Y_past = Bool[1; 0;;]            # D × order
    @test SPSM.conditional_to(Y, Y_past) == [Y_past Y[:, 1:end-1]] .+ 1
end

@testset "Utilities: number of parameters and BIC" begin
    K, D, d = 3, 6, 1
    # (2d+1) coefficients for each of the K·2^memory·D emissions and the K(K-1) free transitions
    @test SPSM.nb_param(K, 1, d, D) == 126
    @test SPSM.nb_param(K, 0, d, D) == (2d + 1) * (K * D + K * (K - 1))
    @test SPSM.nb_param(K, 1, 0, D) < SPSM.nb_param(K, 1, 1, D)
    @test SPSM.nb_param(K, 0, d, D) < SPSM.nb_param(K, 1, d, D)

    LL, N = -1234.5, 1000
    @test SPSM.bic_sel(LL, K, 1, d, D, 0.0) == LL
    @test SPSM.bic_sel(LL, K, 1, d, D, log(N)) == LL - log(N) * SPSM.nb_param(K, 1, d, D) / 2
    # At equal loglikelihood the smaller model wins
    @test SPSM.bic_sel(LL, K, 0, d, D, log(N)) > SPSM.bic_sel(LL, K, 1, d, D, log(N))
end

# * Mixture * #

# EM for Mixture
@testset "Mixture: fit_mle_trig_exp2_EM" begin

    Random.seed!(1234)
    # K = 2
    # 𝔻𝕖𝕘 = 1
    f(θ) = MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
    f(t, θ) = f([σₜ(t, θ[:, 1]), σₜ(t, θ[:, 2]), αₜ(t, θ[:, 3])])

    θσ1 = [1, 1, -0.2]
    θσ2 = [3, -0.5, 0.6]
    θα = [0.1, -0.5, 1]
    θtrue = hcat(θσ1, θσ2, θα)

    # Data
    T = 100
    N = 20000
    n2t = repeat(1:T, N ÷ T)
    y = [rand(f(t / T, θtrue)) for t in n2t]

    #* guess
    θσ10 = [0, 0, 0]
    θσ20 = [5, 0, 0.0]
    θα0 = [0.0, 0.2, 0]
    θ0 = hcat(θσ10, θσ20, θα0)
    mix0 = [f(t / T, θ0) for t in 1:T]
    θσ0 = [θσ10, θσ20]

    mixt, θ_α, θ_Y, history = fit_mle(mix0, permutedims(θα0), θσ0, y, n2t;
        display=:none, maxiter=1000, tol=1e-5, robust=false)

    @test all(diff(history["logtots"]) .> 0) # increasing loglikelihood
    @test history["converged"]
    @test history["iterations"] == length(history["logtots"])

    # Parameter recovery
    @test θ_Y[1] ≈ θσ1 rtol = 10e-2
    @test θ_Y[2] ≈ θσ2 rtol = 10e-2
    @test vec(θ_α) ≈ θα rtol = 10e-2

    # The returned sequence of mixtures is the one encoded by the returned parameters
    @test length(mixt) == T
    @test [scale(components(mixt[t])[1]) for t = 1:T] ≈ [σₜ(t / T, θ_Y[1]) for t = 1:T]
    @test [scale(components(mixt[t])[2]) for t = 1:T] ≈ [σₜ(t / T, θ_Y[2]) for t = 1:T]
    @test [probs(mixt[t])[1] for t = 1:T] ≈ [αₜ(t / T, θ_α[1, :]) for t = 1:T]
    @test all(sum(probs(m)) ≈ 1 for m in mixt)

    # The fit improves on the guess it started from
    ll(mix) = sum(logpdf(mix[t], y[n]) for (n, t) in enumerate(n2t))
    @test ll(mixt) > ll(mix0)
    @test ll(mixt) ≈ history["logtots"][end]
end


# The seasonal EM loop is `ExpectationMaximization`'s; this package supplies only the seasonal
# parametrization. These cover what that buys and what it must not break.
@testset "Mixture: the ExpectationMaximization driver" begin
    Random.seed!(1234)
    g(th) = MixtureModel([Exponential(th[1]), Exponential(th[2])], [th[3], 1 - th[3]])
    g(t, th) = g([σₜ(t, th[:, 1]), σₜ(t, th[:, 2]), αₜ(t, th[:, 3])])
    thtrue = hcat([1, 1, -0.2], [3, -0.5, 0.6], [0.1, -0.5, 1])
    T, N = 20, 2000
    n2t = repeat(1:T, N / T |> Int)
    y = [rand(g(t / T, thtrue)) for t in n2t]
    th0 = hcat([0, 0, 0], [5, 0, 0.0], [0.0, 0.2, 0])
    mix0 = [g(t / T, th0) for t in 1:T]
    base() = (mix0, permutedims(th0[:, 3]), [th0[:, 1], th0[:, 2]], y, n2t)

    # `display=:iter` used to throw `UndefVarError: method` from this package's own loop
    @test_nowarn fit_mle(base()...; display=:iter, maxiter=2, tol=0.0)

    # `rtol`, which the hand-rolled loop had no equivalent of
    _, _, _, h_rtol = fit_mle(base()...; maxiter=1000, tol=0.0, rtol=1e-6)
    @test h_rtol["converged"]
    @test h_rtol["iterations"] < 1000

    # Stochastic EM: a different driver over the same seasonal M-step
    mS, _, thS, hS = fit_mle(base()...; method=StochasticEM(MersenneTwister(1)), maxiter=200, tol=1e-5)
    @test length(mS) == T
    @test components(mS[1])[1] isa Exponential
    @test hS["logtots"][end] > hS["logtots"][1]
    @test thS[2] ≈ thtrue[:, 2] rtol = 2e-1

    # Per-observation weights agree with replicating the observations they weight
    w = [isodd(n) ? 2.0 : 1.0 for n = 1:N]
    rep = findall(isodd, 1:N)
    _, a_w, Y_w, _ = fit_mle(mix0, permutedims(th0[:, 3]), [th0[:, 1], th0[:, 2]], y, n2t, w;
        maxiter=500, tol=1e-8)
    _, a_r, Y_r, _ = fit_mle(mix0, permutedims(th0[:, 3]), [th0[:, 1], th0[:, 2]],
        vcat(y, y[rep]), vcat(n2t, n2t[rep]); maxiter=500, tol=1e-8)
    @test Y_w[1] ≈ Y_r[1] rtol = 1e-4
    @test Y_w[2] ≈ Y_r[2] rtol = 1e-4
    @test vec(a_w) ≈ vec(a_r) rtol = 1e-4

    # The caller's starting coefficients must survive the fit. `theta_Y` is a vector of vectors,
    # so a shallow `copy` would leave the inner vectors shared and the refit would overwrite them.
    # All-Float64 inputs are the case that bites: a mixed Int/Float64 pair is promoted on
    # construction, which silently copies and hides the aliasing.
    thY_keep = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    tha_keep = permutedims([0.0, 0.2, 0.0])
    thY_snapshot = deepcopy(thY_keep); tha_snapshot = copy(tha_keep)
    _, a_out, Y_out, _ = fit_mle(mix0, tha_keep, thY_keep, y, n2t; maxiter=20, tol=1e-6)
    @test thY_keep == thY_snapshot
    @test tha_keep == tha_snapshot
    @test Y_out[1] != thY_snapshot[1]   # and the fit did move, so the check is not vacuous
end


@testset "OptimMLE is concrete and bounds θ₀" begin
    loss(th, x) = sum(abs2, th .- 1.0)
    o = OptimMLE(loss, [0.0, 0.0])

    # Every field parametric, so the struct is concrete. `ℓ::Function` would not be: `Function`
    # is abstract, and the compiler then cannot see through `OptimizationFunction(opt.ℓ, adtype)`.
    @test isconcretetype(typeof(o))
    @test all(isconcretetype, fieldtypes(typeof(o)))
    @test fieldtype(typeof(o), :ℓ) === typeof(loss)

    # `θ₀` must be a vector. The bound used to be written as a trailing `where` on the supertype,
    # where it silently did nothing and a scalar was accepted, failing much later inside `solve`.
    @test_throws MethodError OptimMLE(loss, default_solver(), 5)
    @test_throws MethodError OptimMLE(loss, 5)

    @test fit_mle(o, [1.0]).u ≈ [1.0, 1.0] atol = 1e-6
end

# Optim for Mixture
@testset "Mixture: fit_mle_trig_exp2_Optim" begin
    Random.seed!(1234)

    f(θ) = MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
    f(t, θ) = f([σₜ(t, θ[1:3]), σₜ(t, θ[4:6]), αₜ(t, θ[7:9])])

    θσ1 = [1, 1, -0.2]
    θσ2 = [3, -0.5, 0.6]
    θα = [0.1, -0.5, 1]
    θtrue = hcat(θσ1, θσ2, θα)

    # Data
    T = 100
    N = 20000
    n2t = repeat(1:T, N ÷ T)
    y = [rand(f(t / T, vec(θtrue))) for t in n2t]

    ℓ(θ, x) = -sum(logpdf(f(t / T, θ), x[n]) for (n, t) in enumerate(n2t)) # = -loglikelihood

    #* guess
    θσ10 = [0, 0, 0]
    θσ20 = [5, 0, 0.0]
    θα0 = [0.0, 0.2, 0]
    θ0 = hcat(θσ10, θσ20, θα0)

    sol_Ipopt = fit_mle(OptimMLE(ℓ, Ipopt.Optimizer(), vec(θ0)), y)
    sol_NewtonTR = fit_mle(OptimMLE(ℓ, NewtonTrustRegion(), vec(θ0)), y) #TODO figure out SecondOrder to supress warning (and have better perf?)

    @test sol_Ipopt.u ≈ vec(θtrue) rtol = 5e-2
    @test sol_NewtonTR.u ≈ vec(θtrue) rtol = 5e-2

    # Both solvers land on the same optimum, which beats both the guess and the true parameters
    @test ℓ(sol_Ipopt.u, y) ≈ ℓ(sol_NewtonTR.u, y) rtol = 1e-4
    @test ℓ(sol_Ipopt.u, y) < ℓ(vec(θ0), y)
    @test ℓ(sol_Ipopt.u, y) ≤ ℓ(vec(θtrue), y) + 1e-6
end

# * AR(1) * #

@testset "AR1: rand" begin
    T = 10
    μ = [0.5cos(2π * t / T) for t = 1:T]
    ρ = fill(0.4, T)
    σ = [0.3 + 0.05cos(2π * t / T) for t = 1:T]
    AR = AR1(μ, ρ, σ)

    Random.seed!(1234)
    y = rand(AR, 1000; y₁=0.0)
    @test y isa Vector{Float64}
    @test length(y) == 1000
    @test y[1] == 0.0

    # Same seed, same chain
    Random.seed!(1234)
    y₁ = rand(AR, 500; y₁=0.2)
    Random.seed!(1234)
    y₂ = rand(AR, 500; y₁=0.2)
    @test y₁ == y₂

    # Giving `N` or the season index map `n2t` is equivalent
    Random.seed!(7)
    yₐ = rand(AR, 300; y₁=0.0)
    Random.seed!(7)
    y_b = rand(AR, n_to_t(300, T); y₁=0.0)
    @test yₐ == y_b
end

@testset "AR1: fit_mle recovers the trigonometric parameters" begin
    T, d = 20, 1
    θμ, θρ, θσ = [0.5, 0.3, -0.2], [0.4, 0.2, 0.1], [-0.5, 0.1, 0.2]
    μ = [μₜ(t / T, θμ) for t = 1:T]
    ρ = [ρₜ(t / T, θρ) for t = 1:T]
    σ = [σₜ(t / T, θσ) for t = 1:T]
    AR = AR1(μ, ρ, σ)

    Random.seed!(2024)
    y = rand(AR, 20_000; y₁=0.0)

    θμ̂, θρ̂, θσ̂ = fit_mle_AR1(y, d, T)

    # The seasonal curves are what the model is about, check them rather than the raw coefficients
    @test maximum(abs, [μₜ(t / T, θμ̂) for t = 1:T] - μ) < 5e-2
    @test maximum(abs, [ρₜ(t / T, θρ̂) for t = 1:T] - ρ) < 5e-2
    @test maximum(abs, [σₜ(t / T, θσ̂) for t = 1:T] - σ) < 5e-2
    @test maximum(abs, θμ̂ - θμ) < 5e-2
    @test maximum(abs, θσ̂ - θσ) < 5e-2
end

# * HMM * #

@testset "Sample from HMM" begin
    K = 4
    T = 20
    D = 10
    N = 100
    ξ = [1; zeros(K - 1)]
    ref_station = 1

    # Test that the HMM is well definied with different order of chain (= "local memory" in my jargon)
    for order in 0:3
        hmm_random = randARPeriodicHMM(K, T, D, order; ξ=ξ, ref_station=ref_station)

        z, y = rand(hmm_random, N, seq=true, z_ini=1, y_ini=zeros(Int, D, order))

        y = rand(hmm_random, N)
    end

    #TODO add test comparing order = 0 to PeriodicHMM (it should be exactly the same)
end

@testset "ARPeriodicHMM: constructor invariants" begin
    K, D, T, order, deg = 3, 4, 7, 1, 1
    rng = MersenneTwister(1234)
    θᴬ = randn(rng, K, K - 1, 2deg + 1)
    θᴮ = randn(rng, K, D, 2^order, 2deg + 1)
    a = [0.2, 0.3, 0.5]
    hmm = Trig2ARPeriodicHMM(a, θᴬ, θᴮ, T)

    @test size(hmm) == (K, D, T, 2^order)
    @test (size(hmm, 1), size(hmm, 2), size(hmm, 3), size(hmm, 4)) == (K, D, T, 2^order)
    @test hmm.a == a
    @test all(sum(hmm.A, dims=2) .≈ 1)
    @test all(0 .≤ hmm.A .≤ 1)
    @test SPSM.assert_hmm(hmm)

    # The default initial distribution is uniform
    @test Trig2ARPeriodicHMM(θᴬ, θᴮ, T).a == fill(1 / K, K)

    # A single state leaves no freedom in the chain
    hmm₁ = Trig2ARPeriodicHMM([1.0], zeros(1, 0, 2deg + 1), θᴮ[1:1, :, :, :], T)
    @test size(hmm₁) == (1, D, T, 2^order)
    @test all(hmm₁.A .== 1)

    # `copy` really copies
    hmm_c = copy(hmm)
    hmm_c.A[1, 1, 1] /= 2
    @test hmm_c.A[1, 1, 1] != hmm.A[1, 1, 1]

    # Malformed arguments are rejected
    @test_throws ArgumentError ARPeriodicHMM([0.5, 0.6, 0.5], hmm.A, hmm.B)
    bad_A = copy(hmm.A)
    bad_A[1, :, 1] .= 0.0
    @test_throws ArgumentError ARPeriodicHMM(a, bad_A, hmm.B)
end

@testset "ARPeriodicHMM: randARPeriodicHMM" begin
    K, D, T, order = 4, 5, 6, 2
    @test randARPeriodicHMM(MersenneTwister(42), K, T, D, order).A ==
          randARPeriodicHMM(MersenneTwister(42), K, T, D, order).A

    hmm = randARPeriodicHMM(MersenneTwister(42), K, T, D, order)
    @test size(hmm) == (K, D, T, 2^order)
    @test SPSM.assert_hmm(hmm)
    @test succprob.(hmm.B) == succprob.(randARPeriodicHMM(MersenneTwister(42), K, T, D, order).B)

    # States are sorted by decreasing success probability at the reference station
    ref_station = 3
    hmm_s = randARPeriodicHMM(MersenneTwister(7), K, T, D, order; ref_station=ref_station)
    @test all(issorted([succprob(hmm_s.B[k, t, ref_station, 1]) for k = 1:K], rev=true) for t = 1:T)

    ξ = [0.1, 0.2, 0.3, 0.4]
    @test randARPeriodicHMM(MersenneTwister(7), K, T, D, order; ξ=ξ).a == ξ
end

@testset "ARPeriodicHMM: trigonometric round trip" begin
    K, D, T, order, deg = 2, 3, 8, 1, 1
    rng = MersenneTwister(2023)
    θᴬ = 0.5randn(rng, K, K - 1, 2deg + 1)
    θᴮ = 0.5randn(rng, K, D, 2^order, 2deg + 1)
    hmm = Trig2ARPeriodicHMM(fill(1 / K, K), θᴬ, θᴮ, T)

    # `fit_θ` on a model that is already trigonometric must give the coefficients back
    θᴬ_fit, θᴮ_fit = fit_θ(hmm, deg)
    @test size(θᴬ_fit) == size(θᴬ)
    @test size(θᴮ_fit) == size(θᴮ)
    @test maximum(abs, θᴬ_fit - θᴬ) < 1e-5
    @test maximum(abs, θᴮ_fit - θᴮ) < 1e-5

    hmm_rt = Trig2ARPeriodicHMM(hmm.a, θᴬ_fit, θᴮ_fit, T)
    @test maximum(abs, hmm_rt.A - hmm.A) < 1e-6
    @test maximum(abs, succprob.(hmm_rt.B) - succprob.(hmm.B)) < 1e-6

    # `fit_θ!` does the same in place, and leaves this model essentially unchanged
    hmm_ip = copy(hmm)
    θᴬ_ip, θᴮ_ip = fit_θ!(hmm_ip, deg)
    @test maximum(abs, hmm_ip.A - hmm.A) < 1e-6
    @test maximum(abs, succprob.(hmm_ip.B) - succprob.(hmm.B)) < 1e-6
    @test maximum(abs, θᴬ_ip - θᴬ_fit) < 1e-5
    @test maximum(abs, θᴮ_ip - θᴮ_fit) < 1e-5
end

@testset "ARPeriodicHMM: rand" begin
    K, D, T, order, deg = 2, 3, 5, 1, 1
    rng = MersenneTwister(3)
    hmm = Trig2ARPeriodicHMM([0.4, 0.6], 0.5randn(rng, K, K - 1, 2deg + 1),
        0.5randn(rng, K, D, 2^order, 2deg + 1), T)
    y_ini = rand(MersenneTwister(4), Bool, D, order)
    N = 50
    n2t = n_to_t(N, T)

    z, y = rand(MersenneTwister(5), hmm, n2t; y_ini=y_ini, z_ini=1, seq=true)
    @test y isa Matrix{Bool}
    @test size(y) == (D, N)
    @test size(z) == (N,)
    @test all(1 .≤ z .≤ K)
    @test z[1] == 1
    @test y[:, 1:order] == y_ini

    # `seq=false` returns the observations of the very same draw
    @test rand(MersenneTwister(5), hmm, n2t; y_ini=y_ini, z_ini=1) == y
    # Giving `N` or the season index map `n2t` is equivalent
    @test rand(MersenneTwister(5), hmm, N; y_ini=y_ini, z_ini=1) == y

    @test_throws ArgumentError rand(MersenneTwister(5), hmm, n2t; y_ini=zeros(Bool, D, order + 1), z_ini=1)
end

@testset "ARPeriodicHMM: rand follows the emission probabilities" begin
    # A single state: the chain plays no role, so y conditioned on its memory must follow B
    D, T, order = 2, 3, 1
    p = [0.1 + 0.25(h - 1) + 0.15t / T + 0.05j for k = 1:1, t = 1:T, j = 1:D, h = 1:2^order]
    hmm = ARPeriodicHMM([1.0], ones(1, 1, T), Bernoulli.(p))

    N = 300_000
    n2t = n_to_t(N, T)
    y_past = zeros(Bool, D, order)
    y = rand(MersenneTwister(42), hmm, n2t; y_ini=y_past, z_ini=1)
    lag_cat = SPSM.conditional_to(y, y_past)

    for t = 1:T, j = 1:D, h = 1:2^order
        n = findall((n2t .== t) .& (lag_cat[j, :] .== h))
        @test mean(y[j, n]) ≈ p[1, t, j, h] atol = 1e-2
    end
end

@testset "ARPeriodicHMM: rand follows the transition matrix" begin
    K, T = 3, 2
    A = zeros(K, K, T)
    A[:, :, 1] = [0.7 0.2 0.1; 0.1 0.8 0.1; 0.3 0.3 0.4]
    A[:, :, 2] = [0.5 0.25 0.25; 0.2 0.5 0.3; 0.1 0.1 0.8]
    hmm = ARPeriodicHMM(fill(1 / K, K), A, Bernoulli.(fill(0.5, K, T, 1, 1)))

    N = 200_000
    n2t = n_to_t(N, T)
    z, _ = rand(MersenneTwister(7), hmm, n2t; y_ini=zeros(Bool, 1, 0), z_ini=1, seq=true)

    Â = zeros(K, K, T)
    for n = 1:N-1
        Â[z[n], z[n+1], n2t[n]] += 1
    end
    Â ./= sum(Â, dims=2)
    @test maximum(abs, Â - A) < 2e-2
end

@testset "ARPeriodicHMM: likelihoods, viterbi and complete loglikelihood" begin
    K, D, T, order, deg = 2, 3, 5, 1, 1
    rng = MersenneTwister(11)
    hmm = Trig2ARPeriodicHMM([0.4, 0.6], 0.5randn(rng, K, K - 1, 2deg + 1),
        0.5randn(rng, K, D, 2^order, 2deg + 1), T)
    N = 200
    n2t = n_to_t(N, T)
    y_past = rand(rng, Bool, D, order)
    z, y = rand(MersenneTwister(12), hmm, n2t; y_ini=y_past, z_ini=1, seq=true)
    lag_cat = SPSM.conditional_to(y, y_past)

    LL = loglikelihoods(hmm, y, y_past; n2t=n2t)
    @test size(LL) == (N, K)

    L = zeros(N, K)
    SPSM.likelihoods!(L, hmm, y, lag_cat; n2t=n2t)
    @test log.(L) ≈ LL

    # Column k is the emission loglikelihood of the whole site vector in state k
    manual = [sum(logpdf(hmm.B[k, n2t[n], j, lag_cat[j, n]], y[j, n]) for j = 1:D) for n = 1:N, k = 1:K]
    @test LL ≈ manual

    # Complete loglikelihood = chain term + emission term along the given path
    chain = sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n = 1:N-1)
    emission = sum(manual[n, z[n]] for n = 1:N)
    @test complete_loglikelihood(hmm, y, y_past, z; n2t=n2t) ≈ chain + emission

    # On a well separated model, viterbi finds the hidden path back
    Ts = 4
    A = zeros(2, 2, Ts)
    for t = 1:Ts
        A[:, :, t] = [0.9 0.1; 0.1 0.9]
    end
    p = zeros(2, Ts, 4, 1)
    p[1, :, :, 1] .= 0.05
    p[2, :, :, 1] .= 0.95
    hmm_sep = ARPeriodicHMM([0.5, 0.5], A, Bernoulli.(p))
    y_past_sep = zeros(Bool, 4, 0)
    n2t_sep = n_to_t(2000, Ts)
    z_sep, y_sep = rand(MersenneTwister(13), hmm_sep, n2t_sep; y_ini=y_past_sep, z_ini=1, seq=true)

    ẑ = viterbi(hmm_sep, y_sep, y_past_sep; n2t=n2t_sep)
    @test length(ẑ) == length(z_sep)
    @test mean(ẑ .== z_sep) > 0.95
    # `a` is uniform here, so viterbi maximizes exactly the complete loglikelihood
    @test complete_loglikelihood(hmm_sep, y_sep, y_past_sep, ẑ; n2t=n2t_sep) ≥
          complete_loglikelihood(hmm_sep, y_sep, y_past_sep, z_sep; n2t=n2t_sep)
end

@testset "ARPeriodicHMM: Baum-Welch on a small toy model" begin
    K, D, T, order, deg = 2, 3, 5, 1, 1
    N = 5_000
    rng = MersenneTwister(2021)
    θᴬ = 2(rand(rng, K, K - 1, 2deg + 1) .- 1 / 2)
    θᴮ = 2(rand(rng, K, D, 2^order, 2deg + 1) .- 1 / 2)
    hmm = Trig2ARPeriodicHMM(fill(1 / K, K), θᴬ, θᴮ, T)

    n2t = n_to_t(N, T)
    y_past = rand(rng, Bool, D, order)
    z, y = rand(MersenneTwister(2022), hmm, n2t; y_ini=y_past, z_ini=1, seq=true)

    # Guess: right mean level, no seasonality at all
    θᴬ0 = zeros(K, K - 1, 2deg + 1)
    θᴬ0[:, :, 1] = θᴬ[:, :, 1]
    θᴮ0 = zeros(K, D, 2^order, 2deg + 1)
    θᴮ0[:, :, :, 1] = θᴮ[:, :, :, 1]
    hmm0 = Trig2ARPeriodicHMM(fill(1 / K, K), θᴬ0, θᴮ0, T)

    hmm_fit, θᴬ_fit, θᴮ_fit, hist = fit_mle(hmm0, θᴬ0, θᴮ0, y, y_past;
        n2t=n2t, maxiter=100, tol=1e-4, robust=true, display=:none)

    logtots = hist["logtots"]
    @test length(logtots) == hist["iterations"]
    @test all(diff(logtots) .> -1e-8 * abs(logtots[1])) # Baum-Welch increases the loglikelihood
    @test logtots[end] > logtots[1]
    @test SPSM.assert_hmm(hmm_fit)
    @test size(θᴬ_fit) == size(θᴬ) && size(θᴮ_fit) == size(θᴮ)

    # The guess is not modified in place by the non-mutating `fit_mle`
    @test θᴬ0[:, :, 2:end] == zeros(K, K - 1, 2deg)
    @test hmm0.A == Trig2ARPeriodicHMM(fill(1 / K, K), θᴬ0, θᴮ0, T).A

    # The fitted model explains the realized path better than the guess did
    @test complete_loglikelihood(hmm_fit, y, y_past, z; n2t=n2t) >
          complete_loglikelihood(hmm0, y, y_past, z; n2t=n2t)

    # Decoding agrees with the true states, up to the label switching EM is free to do
    ẑ = viterbi(hmm_fit, y, y_past; n2t=n2t)
    @test max(mean(ẑ .== z), mean(ẑ .!= z)) > 0.6
end

@testset "sARPeriodicHMM: site-dependent memory" begin
    K, T, deg = 2, 5, 1
    orders = [0, 1, 2] # each site carries its own memory order
    D = length(orders)
    rng = MersenneTwister(33)
    θᴬ = 0.5randn(rng, K, K - 1, 2deg + 1)
    θᴮ = [[0.5randn(rng, 2deg + 1) for _ = 1:2^orders[j]] for k = 1:K, j = 1:D]
    hmm = Trig2ARPeriodicHMM(fill(1 / K, K), θᴬ, θᴮ, T)
    @test size(hmm) == (K, D, T, 2 .^ orders)

    N = 2_000
    n2t = n_to_t(N, T)
    y_past = [rand(rng, Bool, orders[j]) for j = 1:D]
    z, y = rand(MersenneTwister(34), hmm, n2t; y_ini=y_past, z_ini=1, seq=true)
    @test size(y) == (D, N)
    @test all(1 .≤ z .≤ K)

    # Guess: right mean level, no seasonality
    θᴬ0 = zeros(K, K - 1, 2deg + 1)
    θᴬ0[:, :, 1] = θᴬ[:, :, 1]
    θᴮ0 = [[vcat(θᴮ[k, j][h][1], zeros(2deg)) for h = 1:2^orders[j]] for k = 1:K, j = 1:D]
    hmm0 = Trig2ARPeriodicHMM(fill(1 / K, K), θᴬ0, θᴮ0, T)

    hmm_fit, θᴬ_fit, θᴮ_fit, hist = fit_mle(hmm0, θᴬ0, θᴮ0, y, y_past;
        n2t=n2t, maxiter=30, tol=1e-4, robust=true)
    logtots = hist["logtots"]
    @test all(diff(logtots) .> -1e-8 * abs(logtots[1])) # Baum-Welch increases the loglikelihood
    @test logtots[end] > logtots[1]
    @test size(θᴬ_fit) == size(θᴬ)

    # The M-step problems are independent, so the parallel backend must not change the answer
    _, θᴬ_thr, θᴮ_thr, _ = fit_mle(hmm0, θᴬ0, θᴮ0, y, y_past;
        n2t=n2t, maxiter=30, tol=1e-4, robust=true, ensemble=EnsembleThreads())
    @test θᴬ_thr ≈ θᴬ_fit
    @test all(θᴮ_thr[k, j][h] ≈ θᴮ_fit[k, j][h] for k = 1:K, j = 1:D for h = 1:2^orders[j])
end

@testset "PeriodicHMM" begin
    Random.seed!(2020)
    K = 3 # Number of Hidden states
    T = 11 # Period
    N = 49_586 # Length of observation
    D = 6 # dimension of observed variables
    autoregressive_order = 1
    size_order = 2^autoregressive_order
    degree_of_P = 1
    size_degree_of_P = 2 * degree_of_P + 1
    trans_θ = 4 * (rand(K, K - 1, size_degree_of_P) .- 1 / 2)
    Bernoulli_θ = 2 * (rand(K, D, size_order, size_degree_of_P) .- 1 / 2)
    hmm = Trig2ARPeriodicHMM([1 / 3, 1 / 6, 1 / 2], trans_θ, Bernoulli_θ, T)
    z_ini = 1
    y_past = rand(Bool, D, autoregressive_order)
    n2t = n_to_t(N, T)
    z, y = rand(hmm, n2t; y_ini=y_past, z_ini=z_ini, seq=true)

    trans_θ_guess = rand(K, K - 1, size_degree_of_P)
    trans_θ_guess[:, :, 1] .= trans_θ[:, :, 1]
    Bernoulli_θ_guess = zeros(K, D, size_order, size_degree_of_P)
    Bernoulli_θ_guess[:, :, :, 1] = Bernoulli_θ[:, :, :, 1]
    hmm_guess = Trig2ARPeriodicHMM([1 / 4, 1 / 4, 1 / 2], trans_θ_guess, Bernoulli_θ_guess, T)

    @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_guess, trans_θ_guess, Bernoulli_θ_guess, y, y_past, maxiter=10000, robust=true; display=:iter, tol=1e-3, θ_iters=true, n2t=n2t)
    # The transition coefficients are only weakly identified: a state visited 15% of the time
    # leaves ~1300 observations per (state, period) cell. Measured over seeds 2020 and 1-4 the
    # relative error spans 0.10-0.28 and tracks the rarest state occupancy, so `rtol = 0.2`
    # held for one draw only. The emission coefficients below stay well inside 0.2.
    # Better fixes than a looser bound, if ever worth the CI time: raise `N`, or assert on
    # `hmm_fit.A` rather than on the coefficients behind it.
    @test θq_fit ≈ trans_θ rtol = 30e-2
    @test θy_fit ≈ Bernoulli_θ rtol = 20e-2

    @test hist["converged"]
    @test all(diff(hist["logtots"]) .> -1e-8 * abs(hist["logtots"][1]))
    @test length(histo_A) == length(histo_B) == hist["iterations"] + 1
    @test histo_A[end] == θq_fit && histo_B[end] == θy_fit

    # With the fitted model, decoding recovers the simulated states far better than chance (1/K)
    ẑ = viterbi(hmm_fit, y, y_past; n2t=n2t)
    @test mean(ẑ .== z) > 0.6
end

@testset "Hierachical PeriodicHMM for many hyperparams" begin
    Random.seed!(2020)
    T = 5 # Period
    N = 100 # Length of observation
    D = 6 # dimension of observed variables
    for K in 1:2, autoregressive_order in 0:2, degree_of_P in 0:2
        size_order = 2^autoregressive_order
        size_degree_of_P = 2 * degree_of_P + 1
        trans_θ = 4 * (rand(K, K - 1, size_degree_of_P) .- 1 / 2)
        Bernoulli_θ = 2 * (rand(K, D, size_order, size_degree_of_P) .- 1 / 2)
        ξ = (1:K) / sum(1:K)
        hmm = Trig2ARPeriodicHMM(ξ, trans_θ, Bernoulli_θ, T)
        z_ini = 1
        y_past = rand(Bool, D, autoregressive_order)
        n2t = n_to_t(N, T)
        z, y = rand(hmm, n2t; y_ini=y_past, z_ini=z_ini, seq=true)

        trans_θ_guess = rand(K, K - 1, size_degree_of_P)
        trans_θ_guess[:, :, 1] .= trans_θ[:, :, 1]
        Bernoulli_θ_guess = zeros(K, D, size_order, size_degree_of_P)
        Bernoulli_θ_guess[:, :, :, 1] = Bernoulli_θ[:, :, :, 1]
        hmm_guess = Trig2ARPeriodicHMM(ξ, trans_θ_guess, Bernoulli_θ_guess, T)

        @time "FitMLE SHMM (Baum Welch) K = $K, autoregressive_order = $autoregressive_order, degree_of_P = $degree_of_P" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_guess, trans_θ_guess, Bernoulli_θ_guess, y, y_past, maxiter=10000, robust=true; tol=1e-3, θ_iters=true, n2t=n2t)
        z_hat = viterbi(hmm_fit, y, y_past; n2t=n2t)
        ll_fit = complete_loglikelihood(hmm_fit, y, y_past, z_hat; n2t=n2t)
        ll_true = complete_loglikelihood(hmm, y, y_past, z; n2t=n2t)

        # Whatever the hyperparameters, the shapes are preserved and EM does not go downhill
        @test size(hmm_fit) == (K, D, T, size_order)
        @test SPSM.assert_hmm(hmm_fit)
        @test size(θq_fit) == size(trans_θ) && size(θy_fit) == size(Bernoulli_θ)
        @test all(diff(hist["logtots"]) .> -1e-8 * abs(hist["logtots"][1]))
        @test all(1 .≤ z_hat .≤ K)
        # The decoded path is the most likely one for the fitted model
        @test ll_fit ≥ complete_loglikelihood(hmm_fit, y, y_past, z; n2t=n2t)
        @test isfinite(ll_true)

        # @test ll_fit ≈ ll_true rtol = 20e-2
        # @test z_hat ≈ z rtol = 20e-2
        # @test θq_fit ≈ trans_θ rtol = 20e-2
        # @test θy_fit ≈ Bernoulli_θ rtol = 20e-2
    end
end

using Distributed
addprocs(2)
@everywhere using SmoothPeriodicStatsModels
@testset "Distributed PeriodicHMM" begin
    Random.seed!(2020)
    K = 4 # Number of Hidden states
    T = 11 # Period
    N = 49_586 # Length of observation
    D = 10 # dimension of observed variables
    autoregressive_order = 1
    size_order = 2^autoregressive_order
    degree_of_P = 1
    size_degree_of_P = 2 * degree_of_P + 1
    trans_θ = 4 * (rand(K, K - 1, size_degree_of_P) .- 1 / 2)
    Bernoulli_θ = 2 * (rand(K, D, size_order, size_degree_of_P) .- 1 / 2)
    ξ = (1:K) / sum(1:K)
    hmm = Trig2ARPeriodicHMM(ξ, trans_θ, Bernoulli_θ, T)
    z_ini = 1
    y_past = rand(Bool, D, autoregressive_order)
    n2t = n_to_t(N, T)
    z, y = rand(hmm, n2t; y_ini=y_past, z_ini=z_ini, seq=true)

    trans_θ_guess = rand(K, K - 1, size_degree_of_P)
    trans_θ_guess[:, :, 1] .= trans_θ[:, :, 1]
    Bernoulli_θ_guess = zeros(K, D, size_order, size_degree_of_P)
    Bernoulli_θ_guess[:, :, :, 1] = Bernoulli_θ[:, :, :, 1]
    hmm_guess = Trig2ARPeriodicHMM(ξ, trans_θ_guess, Bernoulli_θ_guess, T)

    @time "pmap worker" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_guess, trans_θ_guess, Bernoulli_θ_guess, y, y_past, maxiter=10000, robust=true; tol=1e-3, θ_iters=true, n2t=n2t)
    # pmap 0 worker: 31.229542 seconds (161.79 M allocations: 17.413 GiB, 2.89% gc time, 0.01% compilation time)
    # map alone: 32.788183 seconds (161.73 M allocations: 17.411 GiB, 3.53% gc time)
    # pmap 10 worker: 19.509379 seconds (165.01 M allocations: 18.707 GiB, 7.18% gc time)

    @test SPSM.assert_hmm(hmm_fit)
    @test all(diff(hist["logtots"]) .> -1e-8 * abs(hist["logtots"][1]))
end

include("test_hmmspa.jl")
