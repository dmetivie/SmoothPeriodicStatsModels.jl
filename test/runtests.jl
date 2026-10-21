using Test
using SmoothPeriodicStatsModels
using Distributions, Random
using Optimization, OptimizationOptimJL
using Ipopt, OptimizationMOI

const SPSM = SmoothPeriodicStatsModels

# * Utilities * #

@testset "Utilities: trigonometric parametrization" begin
    β = [0.3, -0.2, 0.5, 0.1, -0.4] # degree 2
    T = 12

    # Degree 0 is the constant polynomial
    @test SPSM.polynomial_trigo(0.3, [1.5]) == 1.5
    @test SPSM.polynomial_trigo(3, [1.5], T) == 1.5
    @test SPSM.polynomial_trigo([0.1, 0.7], [1.5]) == [1.5, 1.5]
    @test SPSM.polynomial_trigo([1, 2], [1.5], T) == [1.5, 1.5]

    # The `T` flavour and the normalized flavour are the same parametrization
    @test all(SPSM.polynomial_trigo(t, β, T) ≈ SPSM.polynomial_trigo(t / T, β) for t = 1:T)

    # Vectorized versions agree with the scalar ones
    @test SPSM.polynomial_trigo(collect(1:T), β, T) ≈ [SPSM.polynomial_trigo(t, β, T) for t = 1:T]
    @test SPSM.polynomial_trigo(collect(1:T) / T, β) ≈ [SPSM.polynomial_trigo(t / T, β) for t = 1:T]

    # Periodicity: period `T` in the `T` flavour, period 1 in the normalized one
    @test SPSM.polynomial_trigo(3 + T, β, T) ≈ SPSM.polynomial_trigo(3, β, T)
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

    @test all(μₜ(t, β, T) ≈ μₜ(t / T, β) for t = 1:T)
    @test all(σₜ(t, β, T) ≈ σₜ(t / T, β) for t = 1:T)
    @test all(ρₜ(t, β, T) ≈ ρₜ(t / T, β) for t = 1:T)
end

@testset "Utilities: periodic indexing and memory encoding" begin
    @test n_to_t(7, 3) == [1, 2, 3, 1, 2, 3, 1]
    @test n_to_t(6, 3) == [1, 2, 3, 1, 2, 3]
    @test length(n_to_t(100, 7)) == 100
    @test all(1 .≤ n_to_t(100, 7) .≤ 7)

    # `bin2digit` indexes the 2^order possible memories, last entry = most recent lag
    @test SPSM.bin2digit([false]) == 1
    @test SPSM.bin2digit([true]) == 2
    @test SPSM.bin2digit((true, false)) == 3
    @test sort(vec([SPSM.bin2digit([b₁, b₂]) for b₁ in (false, true), b₂ in (false, true)])) == 1:4

    Y = Bool[0 1; 1 1; 0 0]
    # Without memory every observation falls in the single category 1
    @test SPSM.conditional_to(Y, zeros(Bool, 0, 2)) == ones(Int, 3, 2)
    # With memory 1 the category is `1 + previous value`, the first row coming from `Y_past`
    Y_past = Bool[1 0]
    @test SPSM.conditional_to(Y, Y_past) == [Y_past; Y[1:end-1, :]] .+ 1
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
        display=:none, maxiter=1000, tol=1e-5, robust=false, silence=true, warm_start=true)

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
    μ = [μₜ(t, θμ, T) for t = 1:T]
    ρ = [ρₜ(t, θρ, T) for t = 1:T]
    σ = [σₜ(t, θσ, T) for t = 1:T]
    AR = AR1(μ, ρ, σ)

    Random.seed!(2024)
    y = rand(AR, 20_000; y₁=0.0)

    θμ̂, θρ̂, θσ̂ = model_for_loglikelihood_AR1_full(y, d, T)

    # The seasonal curves are what the model is about, check them rather than the raw coefficients
    @test maximum(abs, [μₜ(t, θμ̂, T) for t = 1:T] - μ) < 5e-2
    @test maximum(abs, [ρₜ(t, θρ̂, T) for t = 1:T] - ρ) < 5e-2
    @test maximum(abs, [σₜ(t, θσ̂, T) for t = 1:T] - σ) < 5e-2
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

        z, y = rand(hmm_random, N, seq=true, z_ini=1, y_ini=zeros(Int, order, D))

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
    y_ini = rand(MersenneTwister(4), Bool, order, D)
    N = 50
    n2t = n_to_t(N, T)

    z, y = rand(MersenneTwister(5), hmm, n2t; y_ini=y_ini, z_ini=1, seq=true)
    @test y isa Matrix{Bool}
    @test size(y) == (N, D)
    @test size(z) == (N,)
    @test all(1 .≤ z .≤ K)
    @test z[1] == 1
    @test y[1:order, :] == y_ini

    # `seq=false` returns the observations of the very same draw
    @test rand(MersenneTwister(5), hmm, n2t; y_ini=y_ini, z_ini=1) == y
    # Giving `N` or the season index map `n2t` is equivalent
    @test rand(MersenneTwister(5), hmm, N; y_ini=y_ini, z_ini=1) == y

    @test_throws ArgumentError rand(MersenneTwister(5), hmm, n2t; y_ini=zeros(Bool, order + 1, D), z_ini=1)
end

@testset "ARPeriodicHMM: rand follows the emission probabilities" begin
    # A single state: the chain plays no role, so y conditioned on its memory must follow B
    D, T, order = 2, 3, 1
    p = [0.1 + 0.25(h - 1) + 0.15t / T + 0.05j for k = 1:1, t = 1:T, j = 1:D, h = 1:2^order]
    hmm = ARPeriodicHMM([1.0], ones(1, 1, T), Bernoulli.(p))

    N = 300_000
    n2t = n_to_t(N, T)
    y_past = zeros(Bool, order, D)
    y = rand(MersenneTwister(42), hmm, n2t; y_ini=y_past, z_ini=1)
    lag_cat = SPSM.conditional_to(y, y_past)

    for t = 1:T, j = 1:D, h = 1:2^order
        n = findall((n2t .== t) .& (lag_cat[:, j] .== h))
        @test mean(y[n, j]) ≈ p[1, t, j, h] atol = 1e-2
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
    z, _ = rand(MersenneTwister(7), hmm, n2t; y_ini=zeros(Bool, 0, 1), z_ini=1, seq=true)

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
    y_past = rand(rng, Bool, order, D)
    z, y = rand(MersenneTwister(12), hmm, n2t; y_ini=y_past, z_ini=1, seq=true)
    lag_cat = SPSM.conditional_to(y, y_past)

    LL = loglikelihoods(hmm, y, y_past; n2t=n2t)
    @test size(LL) == (N, K)

    L = zeros(N, K)
    SPSM.likelihoods!(L, hmm, y, lag_cat; n2t=n2t)
    @test log.(L) ≈ LL

    # Column k is the emission loglikelihood of the whole site vector in state k
    manual = [sum(logpdf(hmm.B[k, n2t[n], j, lag_cat[n, j]], y[n, j]) for j = 1:D) for n = 1:N, k = 1:K]
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
    y_past_sep = zeros(Bool, 0, 4)
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
    y_past = rand(rng, Bool, order, D)
    z, y = rand(MersenneTwister(2022), hmm, n2t; y_ini=y_past, z_ini=1, seq=true)

    # Guess: right mean level, no seasonality at all
    θᴬ0 = zeros(K, K - 1, 2deg + 1)
    θᴬ0[:, :, 1] = θᴬ[:, :, 1]
    θᴮ0 = zeros(K, D, 2^order, 2deg + 1)
    θᴮ0[:, :, :, 1] = θᴮ[:, :, :, 1]
    hmm0 = Trig2ARPeriodicHMM(fill(1 / K, K), θᴬ0, θᴮ0, T)

    hmm_fit, θᴬ_fit, θᴮ_fit, hist = fit_mle(hmm0, θᴬ0, θᴮ0, y, y_past;
        n2t=n2t, maxiter=100, tol=1e-4, robust=true, silence=true, display=:none)

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
    y_past = rand(Bool, autoregressive_order, D)
    n2t = n_to_t(N, T)
    z, y = rand(hmm, n2t; y_ini=y_past, z_ini=z_ini, seq=true)

    trans_θ_guess = rand(K, K - 1, size_degree_of_P)
    trans_θ_guess[:, :, 1] .= trans_θ[:, :, 1]
    Bernoulli_θ_guess = zeros(K, D, size_order, size_degree_of_P)
    Bernoulli_θ_guess[:, :, :, 1] = Bernoulli_θ[:, :, :, 1]
    hmm_guess = Trig2ARPeriodicHMM([1 / 4, 1 / 4, 1 / 2], trans_θ_guess, Bernoulli_θ_guess, T)

    @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_guess, trans_θ_guess, Bernoulli_θ_guess, y, y_past, maxiter=10000, robust=true; display=:iter, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)
    @test θq_fit ≈ trans_θ rtol = 20e-2
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
        y_past = rand(Bool, autoregressive_order, D)
        n2t = n_to_t(N, T)
        z, y = rand(hmm, n2t; y_ini=y_past, z_ini=z_ini, seq=true)

        trans_θ_guess = rand(K, K - 1, size_degree_of_P)
        trans_θ_guess[:, :, 1] .= trans_θ[:, :, 1]
        Bernoulli_θ_guess = zeros(K, D, size_order, size_degree_of_P)
        Bernoulli_θ_guess[:, :, :, 1] = Bernoulli_θ[:, :, :, 1]
        hmm_guess = Trig2ARPeriodicHMM(ξ, trans_θ_guess, Bernoulli_θ_guess, T)

        @time "FitMLE SHMM (Baum Welch) K = $K, autoregressive_order = $autoregressive_order, degree_of_P = $degree_of_P" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_guess, trans_θ_guess, Bernoulli_θ_guess, y, y_past, maxiter=10000, robust=true; silence=true, tol=1e-3, θ_iters=true, n2t=n2t)
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
    y_past = rand(Bool, autoregressive_order, D)
    n2t = n_to_t(N, T)
    z, y = rand(hmm, n2t; y_ini=y_past, z_ini=z_ini, seq=true)

    trans_θ_guess = rand(K, K - 1, size_degree_of_P)
    trans_θ_guess[:, :, 1] .= trans_θ[:, :, 1]
    Bernoulli_θ_guess = zeros(K, D, size_order, size_degree_of_P)
    Bernoulli_θ_guess[:, :, :, 1] = Bernoulli_θ[:, :, :, 1]
    hmm_guess = Trig2ARPeriodicHMM(ξ, trans_θ_guess, Bernoulli_θ_guess, T)

    @time "pmap worker" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_guess, trans_θ_guess, Bernoulli_θ_guess, y, y_past, maxiter=10000, robust=true; silence=true, tol=1e-3, θ_iters=true, n2t=n2t)
    # pmap 0 worker: 31.229542 seconds (161.79 M allocations: 17.413 GiB, 2.89% gc time, 0.01% compilation time)
    # map alone: 32.788183 seconds (161.73 M allocations: 17.411 GiB, 3.53% gc time)
    # pmap 10 worker: 19.509379 seconds (165.01 M allocations: 18.707 GiB, 7.18% gc time)

    @test SPSM.assert_hmm(hmm_fit)
    @test all(diff(hist["logtots"]) .> -1e-8 * abs(hist["logtots"][1]))
end

include("test_hmmspa.jl")
