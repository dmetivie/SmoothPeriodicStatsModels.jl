# The `additional_loss` keyword of the fits: penalized likelihood with NeuralPDE.jl's
# `additional_loss(phi, θ, p)` convention, on every model that has an optimizer step. These
# tests look for what a penalty can break: the unpenalized fit must not move, a weight of zero
# must change nothing, the loss must reach the optimizer on the loglikelihood scale, the
# penalized loglikelihood of an EM must be the one that increases, and a loss that does not
# return a finite real must say so.

import SciMLBase
using ComponentArrays: getaxes

# What a loss was called with, recorded without the dual numbers of the gradient. The solvers copy
# the objective of an ensemble, closure included, so the records live in a global that the loss
# reaches by name; the lock is for the fits that solve their subproblems in threads.
const RECORDED_CALLS = Dict{Int,Vector{NamedTuple}}()
const RECORD_LOCK = ReentrantLock()

function recording_loss(f=(phi, θ, p) -> 0.0)
    id = length(RECORDED_CALLS) + 1
    calls = RECORDED_CALLS[id] = NamedTuple[]
    strip(x::Number) = ForwardDiff.value(x)
    strip(x::AbstractArray) = strip.(x)
    strip(x::NamedTuple) = map(strip, x)
    loss = function (phi, θ, p)
        rec = (; part=p.part, index=p.index, T=p.T, θ=collect(strip(θ)), names=θ isa ComponentArray ? propertynames(θ) : nothing,
            exact=eltype(θ) === Float64, curve=strip(phi(1:p.T, θ)), one=strip(phi(1, θ)))
        @lock RECORD_LOCK push!(RECORDED_CALLS[id], rec)
        return f(phi, θ, p)
    end
    return loss, calls
end

# The same fit, penalized or not, from fixtures that each model keeps small.
function penalty_fixtures()
    fixtures = NamedTuple[]
    T = 12

    # the periodic AR(1), alone, given states, and weighted
    N = 900
    y_ar = rand(StableRNG(1), AR1([0.1t for t = 1:T], fill(0.3, T), fill(0.5, T)), N; y₁=0.0)
    z_ar = rand(StableRNG(2), 1:2, N)
    m_ar = SmoothAR1(1, T; init=(μ=0.0, ρ=0.1, σ=0.5))
    push!(fixtures, (name="SmoothAR1", parts=Set([(:AR, 1)]), model=m_ar, data=y_ar,
        run=(; kw...) -> fit_mle(m_ar, y_ar; solver=SOLVER, kw...),
        select=(; kw...) -> fit(SeasonalFit, m_ar, y_ar; solver=SOLVER, kw...)))
    push!(fixtures, (name="SmoothAR1 given states", parts=Set([(:AR, 1), (:AR, 2)]), model=m_ar, data=y_ar,
        run=(; kw...) -> fit_mle(m_ar, y_ar, z_ar; solver=SOLVER, kw...),
        select=(; kw...) -> fit(SeasonalFit, m_ar, y_ar, z_ar; solver=SOLVER, kw...)))
    w_ar = 0.5 .+ rand(StableRNG(3), N)
    push!(fixtures, (name="SmoothAR1 weighted", parts=Set([(:AR, 1)]), model=m_ar, data=y_ar,
        run=(; kw...) -> fit_mle(m_ar, y_ar, w_ar; solver=SOLVER, kw...)))

    # a distribution per hidden state
    Td, Nd = 20, 1500
    z_d = rand(StableRNG(4), 1:2, Nd)
    truth = SmoothDistribution(Gamma, Td; K=2, season=1, init=[(α=3.0, θ=1.0), (α=1.5, θ=3.0)])
    n2t_d = n_to_t(Nd, Td)
    y_d = rand(StableRNG(5), truth, n2t_d; z=z_d)
    m_d = SmoothDistribution(Gamma, Td; K=2, season=1)
    push!(fixtures, (name="SmoothDistribution", parts=Set([(:distribution, 1), (:distribution, 2)]),
        model=m_d, data=y_d, run=(; kw...) -> fit_mle(m_d, y_d, OneHot(z_d, 2); n2t=n2t_d, solver=SOLVER, kw...),
        select=(; kw...) -> fit(SeasonalFit, m_d, y_d, OneHot(z_d, 2); n2t=n2t_d, solver=SOLVER, kw...)))

    # a seasonal mixture, with and without weights
    Tm, Nm = 20, 1200
    mix(t, θ) = MixtureModel([Exponential(σₜ(t, θ[:, 1])), Exponential(σₜ(t, θ[:, 2]))],
        [αₜ(t, θ[:, 3]), 1 - αₜ(t, θ[:, 3])])
    θtrue = hcat([1.0, 1, -0.2], [3.0, -0.5, 0.6], [0.1, -0.5, 1.0])
    n2t_m = repeat(1:Tm, Nm ÷ Tm)
    y_m = [rand(StableRNG(100 + n), mix(t / Tm, θtrue)) for (n, t) in enumerate(n2t_m)]
    m_m = SmoothSeasonalMixture(Exponential, 2, 1, Tm; init=[1.0, 5.0])
    w_m = 0.5 .+ rand(StableRNG(7), Nm)
    mixture_parts = Set([(:mixture, nothing), (:component, 1), (:component, 2)])
    push!(fixtures, (name="SmoothSeasonalMixture", parts=mixture_parts, model=m_m, data=y_m,
        run=(; kw...) -> fit_mle(m_m, y_m; solver=SOLVER, n2t=n2t_m, maxiter=8, tol=0.0, kw...),
        select=(; kw...) -> fit(SeasonalFit, m_m, y_m; solver=SOLVER, n2t=n2t_m, maxiter=8, tol=0.0, kw...)))
    push!(fixtures, (name="SmoothSeasonalMixture weighted", parts=mixture_parts, model=m_m, data=y_m,
        run=(; kw...) -> fit_mle(m_m, y_m, w_m; solver=SOLVER, n2t=n2t_m, maxiter=8, tol=0.0, kw...)))

    # the Bernoulli HMMs
    K, D, order, deg, Nb = 2, 3, 1, 1, 600
    rng = MersenneTwister(1)
    truth_b = Trig2ARPeriodicHMM(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1),
        0.5randn(rng, K, D, 2^order, 2deg + 1), T)
    n2t_b = n_to_t(Nb, T)
    y_past = rand(rng, Bool, D, order)
    z_b, y_b = rand(MersenneTwister(2), truth_b, n2t_b; Y_past=y_past, z_ini=1, seq=true)
    m_b = SmoothARPeriodicHMM(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1),
        0.5randn(rng, K, D, 2^order, 2deg + 1), T)
    bern_parts = Set(vcat([(:B, (k, s, h)) for k = 1:K, s = 1:D, h = 1:2^order][:], [(:A, k) for k = 1:K]))
    push!(fixtures, (name="SmoothARPeriodicHMM", parts=bern_parts, model=m_b, data=(y_b, y_past, n2t_b),
        run=(; kw...) -> fit_mle(m_b, y_b, y_past; solver=SOLVER, n2t=n2t_b, maxiter=4, tol=0.0, kw...),
        select=(; kw...) -> fit(SeasonalFit, m_b, y_b, y_past; solver=SOLVER, n2t=n2t_b, maxiter=4, tol=0.0, kw...)))
    # the fit given the states, with the transitions
    push!(fixtures, (name="SmoothARPeriodicHMM given states", parts=bern_parts, model=m_b, data=(y_b, y_past, n2t_b),
        run=(; kw...) -> (fit_mle(m_b, y_b, OneHot(z_b, K); n2t=n2t_b, past=(; Y=y_past), fixed=(),
                solver=SOLVER, kw...), nothing)))

    # sites with their own memory
    orders = [0, 1, 2]
    θᴮs = [[0.5randn(rng, 2deg + 1) for _ = 1:2^orders[j]] for k = 1:K, j = 1:length(orders)]
    truth_s = Trig2ARPeriodicHMM(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1), θᴮs, T)
    y_past_s = [rand(rng, Bool, orders[j]) for j in eachindex(orders)]
    _, y_s = rand(MersenneTwister(3), truth_s, n2t_b; Y_past=y_past_s, z_ini=1, seq=true)
    m_s = SmoothsARPeriodicHMM(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1), θᴮs, T)
    sites_parts = Set(vcat([(:B, (k, s, h)) for k = 1:K for s in eachindex(orders) for h = 1:2^orders[s]],
        [(:A, k) for k = 1:K]))
    push!(fixtures, (name="SmoothsARPeriodicHMM", parts=sites_parts, model=m_s, data=(y_s, y_past_s, n2t_b),
        run=(; kw...) -> fit_mle(m_s, y_s, y_past_s; solver=SOLVER, n2t=n2t_b, maxiter=4, tol=0.0, kw...),
        select=(; kw...) -> fit(SeasonalFit, m_s, y_s, y_past_s; solver=SOLVER, n2t=n2t_b, maxiter=4, tol=0.0, kw...)))

    # any emission law
    mh = SmoothPeriodicHMM(Normal, 2, 1, T; init=[(μ=0.0, σ=1.0), (μ=2.0, σ=1.0)])
    yh = vec(rand(MersenneTwister(4), mh, n2t_b))
    push!(fixtures, (name="SmoothPeriodicHMM", parts=Set([(:B, (1, 1)), (:B, (2, 1)), (:A, 1), (:A, 2)]),
        model=mh, data=(yh, n2t_b),
        run=(; kw...) -> fit_mle(mh, yh; solver=SOLVER, n2t=n2t_b, maxiter=4, tol=0.0, kw...),
        select=(; kw...) -> fit(SeasonalFit, mh, yh; solver=SOLVER, n2t=n2t_b, maxiter=4, tol=0.0, kw...)))
    return fixtures
end

# The coefficients of a model, flat, whatever the layout.
flat_coefficients(m) = flat_coefficients(parameters(m))
flat_coefficients(x::Number) = [x]
flat_coefficients(x::AbstractArray{<:Number}) = vec(collect(x))
flat_coefficients(x::AbstractArray) = reduce(vcat, flat_coefficients.(x))
flat_coefficients(x::Union{NamedTuple,ComponentArray}) = reduce(vcat, [flat_coefficients(getproperty(x, k)) for k in propertynames(x)])
fitted_model(r::Tuple) = r[1]
fitted_model(r) = r
fitted_history(r::Tuple) = r[2]
fitted_history(r) = nothing

@testset "additional_loss: the helpers" begin
    r = RidgePenalty(2)
    @test r(nothing, [1, 2, 3], nothing) == 28
    @test r(nothing, ComponentArray(a=[1.0, 2.0], b=[3.0]), nothing) == 28
    @test RidgePenalty(0.0)(nothing, [1.0, 2.0], nothing) == 0
    @test_throws ArgumentError RidgePenalty(-1.0)
    @test_throws ArgumentError RidgePenalty(NaN)
    @test_throws ArgumentError RidgePenalty(Inf)
    @test sprint(show, RidgePenalty(3)) == "RidgePenalty(3)"

    # differences, by hand: β = (1, 2, 4)
    β = [1.0, 2.0, 4.0]
    @test DifferencePenalty(1.0; order=1, cyclic=false)(nothing, β, nothing) == 1 + 4
    @test DifferencePenalty(1.0; order=1)(nothing, β, nothing) == 1 + 4 + 9           # wraps: 4 - 1
    @test DifferencePenalty(1.0; order=2, cyclic=false)(nothing, β, nothing) == 1
    @test DifferencePenalty(1.0; order=2)(nothing, β, nothing) == 1 + 25 + 16
    @test DifferencePenalty(3.0)(nothing, β, nothing) == 3 * 42                         # the default is order 2, cyclic
    # a constant has no differences, cyclic or not, whatever the order
    for order in 1:3, cyclic in (true, false)
        @test DifferencePenalty(1.0; order, cyclic)(nothing, fill(2.5, 7), nothing) == 0
    end
    # a block no longer than the order has no differences when not cyclic
    @test DifferencePenalty(1.0; order=2, cyclic=false)(nothing, [1.0, 5.0], nothing) == 0
    @test DifferencePenalty(1.0; order=2, cyclic=false)(nothing, [1.0], nothing) == 0
    @test DifferencePenalty(1.0)(nothing, [1.0], nothing) == 0
    # blocks are differenced one by one, not across their border
    cv = ComponentArray(a=[1.0, 2.0, 4.0], b=[10.0, 10.0, 10.0])
    @test DifferencePenalty(1.0)(nothing, cv, nothing) == 42
    @test DifferencePenalty(1.0)(nothing, vcat(cv.a, cv.b), nothing) != 42
    # dual numbers go through
    @test ForwardDiff.gradient(b -> DifferencePenalty(1.0; order=1, cyclic=false)(nothing, b, nothing), β) ≈ [-2.0, -2.0, 4.0]
    @test_throws ArgumentError DifferencePenalty(-1.0)
    @test_throws ArgumentError DifferencePenalty(1.0; order=0)
    @test sprint(show, DifferencePenalty(2.0; order=1, cyclic=false)) == "DifferencePenalty(2.0; order = 1, cyclic = false)"

    # CurvePenalty: the differences of `phi(1:T, θ)`, over the natural scale
    phi_vec(t, θ) = θ[t]
    @test CurvePenalty(1.0)(phi_vec, [1.0, 2.0, 4.0], (; T=3)) == 42
    phi_nt(t, θ) = (a=θ[t], b=2θ[t])
    @test CurvePenalty(1.0; order=1)(phi_nt, [1.0, 2.0, 4.0], (; T=3)) == 14 * 5
    phi_mat(t, θ) = [θ[t]'; 2θ[t]']
    @test CurvePenalty(1.0; order=1)(phi_mat, [1.0, 2.0, 4.0], (; T=3)) == 14 * 5
    @test_throws ArgumentError CurvePenalty(-1.0)
    @test_throws ArgumentError CurvePenalty(1.0; order=0)
    @test sprint(show, CurvePenalty(2.0)) == "CurvePenalty(2.0; order = 2)"

    # what a loss may be
    @test SPSM._resolve_loss(nothing) === nothing
    f3(phi, θ, p) = 0.0
    @test SPSM._resolve_loss(f3) === f3
    @test_throws ArgumentError SPSM._resolve_loss(3)
    @test_throws ArgumentError SPSM._resolve_loss(θ -> 0.0)            # the one-argument form is gone
    @test_throws ArgumentError SPSM._resolve_loss((a, b) -> 0.0)
    @test SPSM._resolve_loss(RidgePenalty(1.0)) isa RidgePenalty
end

@testset "additional_loss: nothing changes nothing, and a weight of zero is no loss" begin
    for fx in penalty_fixtures()
        @testset "$(fx.name)" begin
            base = fx.run()
            # `nothing` is the default: the same arithmetic, to the last bit
            same = fx.run(; additional_loss=nothing)
            @test flat_coefficients(fitted_model(same)) == flat_coefficients(fitted_model(base))
            h0, h1 = fitted_history(base), fitted_history(same)
            if h0 !== nothing
                @test h1.logtots == h0.logtots && loglikelihood(h1) == loglikelihood(h0)
                @test !h0.penalized && !h1.penalized
            end
            # a loss of zero: the same optimum
            # (the third records its calls: every subproblem, with the documented part, index and period)
            recorder, calls = recording_loss(CurvePenalty(0.0))
            for zero_loss in (RidgePenalty(0.0), DifferencePenalty(0.0), recorder)
                zero_fit = fx.run(; additional_loss=zero_loss)
                @test flat_coefficients(fitted_model(zero_fit)) ≈ flat_coefficients(fitted_model(base)) rtol = 1e-8
                hz = fitted_history(zero_fit)
                if h0 !== nothing
                    @test hz.penalized
                    @test hz.logtots ≈ h0.logtots rtol = 1e-8
                end
            end
            @test Set((c.part, c.index) for c in calls) == fx.parts
            @test all(c -> c.T == fx.model.nT, calls)
        end
    end
end

# The last call that saw the coefficients of subproblem `(part, index)` as plain floats, which an
# EM fit makes at the coefficients it returns.
last_exact(calls, part, index) = calls[something(findlast(c -> c.exact && c.part == part && c.index == index, calls))]

@testset "additional_loss: phi is the curve of the fitted model" begin
    fxs = Dict(fx.name => fx for fx in penalty_fixtures())

    @testset "Bernoulli HMM" begin
        fx = fxs["SmoothARPeriodicHMM"]
        loss, calls = recording_loss()
        m, _ = fx.run(; additional_loss=loss)
        ps = parameters(m)
        K, D, so = size(ps.B)[1:3]
        for k = 1:K, s = 1:D, h = 1:so
            c = last_exact(calls, :B, (k, s, h))
            @test c.θ == ps.B[k, s, h, :]
            @test c.names === nothing
            @test c.curve ≈ curve(m, :p)[k, s, h, :] rtol = 1e-12
            @test c.one == c.curve[1]
        end
        for k = 1:K
            c = last_exact(calls, :A, k)
            @test c.θ == ps.A[k, 1, :] && c.names == (:logit1,)
            @test c.curve ≈ curve(m, :A)[k, :, :] rtol = 1e-12
            @test all(≈(1), sum(c.curve; dims=1))
            @test c.one ≈ curve(m, :A)[k, :, 1] rtol = 1e-12
        end
    end

    @testset "three states: one block per free category" begin
        K, D, T, N = 3, 2, 6, 400
        rng = StableRNG(11)
        truth = SmoothARPeriodicHMM(fill(1 / K, K), randn(rng, K, K - 1, 3), randn(rng, K, D, 1, 3), T)
        n2t = n_to_t(N, T)
        (; z, Y) = rand(StableRNG(12), truth, n2t; past=(; Y=falses(D, 0)))
        m0 = SmoothARPeriodicHMM(K, D, T; memory=0, season=1)
        loss, calls = recording_loss()
        m = fit_mle(m0, Y, OneHot(z, K); n2t, past=(; Y=falses(D, 0)), fixed=(), solver=SOLVER, additional_loss=loss)
        ps = parameters(m)
        for k = 1:K
            target = vcat(ps.A[k, 1, :], ps.A[k, 2, :])
            c = calls[findlast(c -> c.part == :A && c.index == k && isapprox(c.θ, target; atol=1e-12), calls)]
            @test c.names == (:logit1, :logit2)
            @test c.curve ≈ curve(m, :A)[k, :, :] rtol = 1e-12
            @test size(c.curve) == (K, T)
        end
        @test Set(c.part for c in calls) == Set([:A, :B])
    end

    @testset "site-dependent memory" begin
        fx = fxs["SmoothsARPeriodicHMM"]
        loss, calls = recording_loss()
        m, _ = fx.run(; additional_loss=loss)
        ps = parameters(m)
        for k = 1:2, s = 1:3, h in eachindex(ps.B[k, s])
            c = last_exact(calls, :B, (k, s, h))
            @test c.θ == ps.B[k, s][h]
            @test c.curve ≈ curve(m, :p)[k, s][h, :] rtol = 1e-12
        end
    end

    @testset "any emission" begin
        fx = fxs["SmoothPeriodicHMM"]
        loss, calls = recording_loss()
        m, _ = fx.run(; additional_loss=loss)
        for k = 1:2
            c = last_exact(calls, :B, (k, 1))
            @test c.θ == SPSM.emission_coefs(m, k, 1)
            @test c.names == (:μ, :σ)
            @test collect(c.curve.μ) ≈ curve(m, :μ; block=1)[k, :] rtol = 1e-12
            @test collect(c.curve.σ) ≈ curve(m, :σ; block=1)[k, :] rtol = 1e-12
            @test c.one == (μ=c.curve.μ[1], σ=c.curve.σ[1])
        end
    end

    @testset "a distribution per state" begin
        fx = fxs["SmoothDistribution"]
        loss, calls = recording_loss()
        m = fx.run(; additional_loss=loss)
        for k = 1:2
            c = last_exact(calls, :distribution, k)
            @test c.θ == column(m.family.spec, parameters(m), k)
            @test c.names == (:α, :θ)
            @test c.curve.α ≈ curve(m, :α; state=k) rtol = 1e-12
            @test c.curve.θ ≈ curve(m, :θ; state=k) rtol = 1e-12
        end
    end

    @testset "the mixture" begin
        fx = fxs["SmoothSeasonalMixture"]
        loss, calls = recording_loss()
        m, _ = fx.run(; additional_loss=loss)
        c = last_exact(calls, :mixture, nothing)
        @test c.θ == vec(parameters(m).α) && c.names == (:logit1,)
        @test c.curve ≈ mixture_weights(m) rtol = 1e-12
        for k = 1:2
            c = last_exact(calls, :component, k)
            @test c.θ == component_coefs(m, k)
            @test collect(c.curve.θ) ≈ curve(m, :θ)[k, :] rtol = 1e-12
        end
    end

    @testset "the AR(1)" begin
        fx = fxs["SmoothAR1"]
        loss, calls = recording_loss()
        m, _ = fx.run(; additional_loss=loss)
        c = calls[findlast(c -> isapprox(c.θ, collect(parameters(m)); atol=1e-12), calls)]
        @test c.names == (:μ, :ρ, :σ) && c.part == :AR && c.index == 1
        for name in (:μ, :ρ, :σ)
            @test collect(c.curve[name]) ≈ curve(m, name) rtol = 1e-12
        end
        # one state per column
        fz = fxs["SmoothAR1 given states"]
        loss, calls = recording_loss()
        mz, _ = fz.run(; additional_loss=loss)
        for k = 1:2
            θk = column(mz.spec, parameters(mz), k)
            c = calls[findlast(c -> c.index == k && isapprox(c.θ, θk; atol=1e-12), calls)]
            @test collect(c.curve.σ) ≈ curve(mz, :σ)[k, :] rtol = 1e-12
        end
    end

    @testset "phi at one period, or at several" begin
        T = 8
        θ = [0.3, -0.5, 0.2, 0.1, -0.2]
        phi = SPSM._BernoulliPhi(design_matrix(TrigBasis(T, 2), 1:T))
        @test phi(3, θ) == phi(1:T, θ)[3]
        @test phi([2, 5], θ) ≈ phi(1:T, θ)[[2, 5]] rtol = 1e-14
        @test phi(3, θ) isa Float64 && phi(3:3, θ) isa Vector
        @test_throws BoundsError phi(T + 1, θ)
        sm = SPSM._SoftmaxPhi(design_matrix(TrigBasis(T, 1), 1:T), 3)
        θs = SPSM._coefficients(sm, collect(1.0:6.0) ./ 10)
        @test propertynames(θs) == (:logit1, :logit2)
        @test sum(sm(4, θs)) ≈ 1 && length(sm(4, θs)) == 3
        @test sm([1, 4], θs) == hcat(sm(1, θs), sm(4, θs))
        # the flat vector the optimizer holds gives the same probabilities as its named blocks
        @test sm(4, collect(1.0:6.0) ./ 10) ≈ sm(4, θs)
        # a huge linear predictor does not overflow
        big = SPSM._coefficients(sm, [800.0, -900.0, 0, 0, 0, 0])
        @test all(isfinite, sm(2, big)) && sm(2, big)[1] ≈ 1
    end
end

@testset "additional_loss: the loss counts on the scale of the summed loglikelihood" begin
    # a distribution, whose objective is an average inside: against the summed objective
    Td, Nd = 20, 800
    n2t = n_to_t(Nd, Td)
    m1 = SmoothDistribution(Gamma, Td; season=1)
    truth = SmoothDistribution(Gamma, Td; season=1, init=(α=3.0, θ=1.0))
    y1 = rand(StableRNG(13), truth, n2t)
    λ = 50.0
    mp = fit_mle(m1, y1; n2t, solver=SOLVER, additional_loss=RidgePenalty(λ))
    nll = PeriodicNLL(m1.family, PeriodicDesign(m1.family.spec, 1:Td), y1, group_by_time(n2t, Td))
    ref = fit_loss_optim((θ, _) -> (nll(θ) + λ * sum(abs2, θ)) / Nd, y1, zeros(6); solver=SOLVER).u
    @test column(mp.family.spec, parameters(mp), 1) ≈ ref rtol = 1e-5
    mu = fit_mle(m1, y1; n2t, solver=SOLVER)
    @test sum(abs2, parameters(mp)) < sum(abs2, parameters(mu))
    f = fit(SeasonalFit, m1, y1; n2t, solver=SOLVER, additional_loss=RidgePenalty(λ))
    @test loglikelihood(f) ≈ loglikelihood(mp, y1; n2t) - λ * sum(abs2, parameters(mp)) rtol = 1e-10
    # weights of two count twice: the same fit as the observations twice over
    W2 = fill(2.0, Nd, 1)
    m2 = fit_mle(m1, y1, W2; n2t, solver=SOLVER, additional_loss=RidgePenalty(λ))
    d2 = fit_mle(m1, [y1; y1]; n2t=[n2t; n2t], solver=SOLVER, additional_loss=RidgePenalty(λ))
    @test collect(parameters(m2)) ≈ collect(parameters(d2)) rtol = 1e-5

    # the AR(1) objective is a sum already
    y = rand(StableRNG(14), AR1([0.1t for t = 1:12], fill(0.3, 12), fill(0.5, 12)), 600; y₁=0.0)
    ma = SmoothAR1(1, 12; init=(μ=0.0, ρ=0.1, σ=0.5))
    fa, _ = fit_mle(ma, y; solver=SOLVER, additional_loss=RidgePenalty(7.0))
    stats = ar1_sufficient_statistics(y, n_to_t(600, 12), 12)
    design = PeriodicDesign(ma.spec, 1:12)
    ref_a = SPSM.solve_unconstrained((θ, p) -> ar1_nll(θ, p) + 7.0 * sum(abs2, θ), collect(float.(parameters(ma))),
        (stats, design); solver=SOLVER)
    @test collect(parameters(fa)) ≈ ref_a rtol = 1e-6

    # a Bernoulli cell, against the root of its score
    N = 400
    Y = reshape(rand(StableRNG(15), N) .< 0.1, 1, N)
    mb = SmoothARPeriodicHMM(1, 1, 5; memory=0, season=0)
    n0, n1 = count(!, Y), count(Y)
    for λ in (0.5, 5.0, 50.0)
        fb = fit_mle(mb, Y, OneHot(ones(Int, N), 1); n2t=n_to_t(N, 5), past=falses(1, 0), solver=SOLVER,
            additional_loss=RidgePenalty(λ))
        β = only(parameters(fb).B)
        score(b) = (n0 + n1) / (1 + exp(-b)) - n1 + 2λ * b
        lo, hi = -30.0, 30.0
        for _ = 1:200
            mid = (lo + hi) / 2
            score(mid) > 0 ? (hi = mid) : (lo = mid)
        end
        @test β ≈ lo rtol = 1e-5
    end
end

@testset "additional_loss: an EM fit tracks and increases the penalized loglikelihood" begin
    fxs = Dict(fx.name => fx for fx in penalty_fixtures())
    λ = 5.0
    ridge = RidgePenalty(λ)
    bern, sites, dists = fxs["SmoothARPeriodicHMM"], fxs["SmoothsARPeriodicHMM"], fxs["SmoothPeriodicHMM"]
    loglik = Dict(
        "SmoothARPeriodicHMM" => m -> loglikelihood(m, bern.data[1], bern.data[2]; n2t=bern.data[3]),
        "SmoothsARPeriodicHMM" => m -> loglikelihood(m, sites.data[1], sites.data[2]; n2t=sites.data[3]),
        "SmoothPeriodicHMM" => m -> loglikelihood(m, dists.data[1]; n2t=dists.data[2]),
    )
    for name in ("SmoothARPeriodicHMM", "SmoothsARPeriodicHMM", "SmoothPeriodicHMM")
        @testset "$name" begin
            fx = fxs[name]
            m, h = fx.run(; additional_loss=ridge, maxiter=12)
            @test h.penalized
            @test all(diff(h.logtots) .>= -1e-6 * abs(h.logtots[1]))
            # what the history holds is the loglikelihood of the returned model, less the loss
            @test h.logtots[end] ≈ loglik[name](m) - λ * sum(abs2, flat_coefficients(m)) rtol = 1e-8
            @test loglikelihood(h) == h.logtots[end]
            # the start counts too: no iteration, the loglikelihood of the start less its loss
            m0, h0 = fx.run(; additional_loss=ridge, maxiter=0)
            @test isempty(h0.logtots)
            @test loglikelihood(h0) ≈ loglik[name](fx.model) - λ * sum(abs2, flat_coefficients(fx.model)) rtol = 1e-8
            # the loss pulls the coefficients in
            mu, hu = fx.run(; maxiter=12)
            @test sum(abs2, flat_coefficients(m)) < sum(abs2, flat_coefficients(mu))
            # the unpenalized fit has the higher loglikelihood, the penalized one the higher objective
            @test loglik[name](mu) > loglik[name](m)
            @test h.logtots[end] > loglik[name](mu) - λ * sum(abs2, flat_coefficients(mu)) - 1e-6
        end
    end

    @testset "the mixture, with and without weights" begin
        fx = fxs["SmoothSeasonalMixture"]
        y, n2t = fx.data, repeat(1:20, 60)
        for (run, ll) in (
            ((; kw...) -> fit_mle(fx.model, y; solver=SOLVER, n2t, maxiter=12, tol=0.0, kw...),
                m -> loglikelihood(m, y; n2t)),
            ((; kw...) -> fit_mle(fx.model, y, fill(2.5, length(y)); solver=SOLVER, n2t, maxiter=12, tol=0.0, kw...),
                m -> 2.5 * loglikelihood(m, y; n2t)))
            m, h = run(; additional_loss=ridge)
            @test h.penalized && all(diff(h.logtots) .>= -1e-6 * abs(h.logtots[1]))
            @test h.logtots[end] ≈ ll(m) - λ * sum(abs2, flat_coefficients(m)) rtol = 1e-8
            mu, _ = run()
            @test sum(abs2, flat_coefficients(m)) < sum(abs2, flat_coefficients(mu))
        end
        # a stochastic EM works too, and its history is penalized
        ms, hs = fit_mle(fx.model, y; solver=SOLVER, n2t, maxiter=3, tol=0.0, method=StochasticEM(StableRNG(1)),
            additional_loss=ridge)
        @test hs.penalized && length(hs.logtots) == 3
    end

    @testset "stationarity: the M-step and the tracked loss are the same loss" begin
        # at the end of a long EM the gradient of the penalized loglikelihood vanishes; had the
        # M-step weighed the loss differently, a gradient of the order of λθ would remain
        fx = fxs["SmoothSeasonalMixture"]
        y, n2t = fx.data, repeat(1:20, 60)
        m, h = fit_mle(fx.model, y; solver=SOLVER, n2t, maxiter=400, tol=1e-9, additional_loss=RidgePenalty(20.0))
        ps = parameters(m)
        ax = getaxes(ps)
        penalized(x) = loglikelihood(SmoothSeasonalMixture(m.components, m.spec_α, m.nT, ComponentArray(x, ax)), y; n2t) -
                       20.0 * sum(abs2, x)
        x0, ε = collect(ps), 1e-5
        g = map(eachindex(x0)) do i
            e = zeros(length(x0))
            e[i] = ε
            (penalized(x0 .+ e) - penalized(x0 .- e)) / 2ε
        end
        @test maximum(abs, g) < 0.5
        @test maximum(abs, 40.0 .* collect(ps)) > 5            # the loss really does bind
    end
end

@testset "additional_loss: what it is for" begin
    # a wet day is always followed by a dry one: that cell of the Bernoulli model is separated
    T, N = 5, 600
    Y = reshape([n % 4 == 0 for n = 1:N], 1, N)
    W = OneHot(ones(Int, N), 1)
    m0 = SmoothARPeriodicHMM(1, 1, T; memory=1, season=0)
    kw = (; n2t=n_to_t(N, T), past=falses(1, 1), solver=SOLVER)
    pmin(m) = minimum(curve(m, :p))

    mu = fit_mle(m0, Y, W; kw...)
    @test minimum(parameters(mu).B) < -10               # the unpenalized coefficient runs away
    @test pmin(mu) < 1e-4
    for λ in (0.1, 1.0, 10.0)
        mr = fit_mle(m0, Y, W; kw..., additional_loss=RidgePenalty(λ))
        @test all(isfinite, parameters(mr).B)
        @test -8 < minimum(parameters(mr).B) < 0
        @test pmin(mu) < pmin(mr) < 0.25
        # the cell that is not separated moves little
        @test maximum(abs, parameters(mr).B) < maximum(abs, parameters(mu).B)
    end
    # more ridge, a less extreme coefficient
    β(λ) = minimum(parameters(fit_mle(m0, Y, W; kw..., additional_loss=RidgePenalty(λ))).B)
    @test β(0.1) < β(1.0) < β(10.0)

    # phi keeps a probability off 0 and 1: a hinge on p(t)
    ε = 0.1
    hinge(phi, θ, p) = 1e6 * (sum(abs2, max.(0, ε .- phi(1:p.T, θ))) + sum(abs2, max.(0, phi(1:p.T, θ) .- (1 - ε))))
    mh = fit_mle(m0, Y, W; kw..., additional_loss=hinge)
    @test pmin(mh) > ε - 1e-3
    @test maximum(curve(mh, :p)) < 1 - ε + 1e-3
    @test pmin(mh) < ε + 1e-3                              # and it binds only where it must

    # a loss that is not finite away from the start is the solver's to handle, as the likelihood's own is
    wall(phi, θ, p) = maximum(abs, θ) > 30 ? Inf : 0.0
    mw = fit_mle(m0, Y, W; kw..., additional_loss=wall, solve_kwargs=(; maxiters=200))
    @test all(isfinite, parameters(mw).B) && maximum(abs, parameters(mw).B) <= 30
end

@testset "additional_loss: smooth curves from a spline or a trigonometric basis" begin
    T, N = 12, 3000
    n2t = n_to_t(N, T)
    p_true = [0.2 + 0.6 * (1 + sin(2π * t / T)) / 2 for t = 1:T]
    Y = reshape(rand(StableRNG(16), N) .< p_true[n2t], 1, N)
    W = OneHot(ones(Int, N), 1)
    kw = (; n2t, past=falses(1, 0), solver=SOLVER)
    spread(m) = (p = vec(curve(m, :p)); maximum(p) - minimum(p))

    # P-splines: a huge cyclic difference penalty makes the coefficients, and so the curve, constant
    mspline = SmoothARPeriodicHMM(1, 1, T; memory=0, season=(A=0, B=CyclicBSplineBasis(T, 8)))
    mu = fit_mle(mspline, Y, W; kw...)
    @test spread(mu) > 0.4
    flat = fit_mle(mspline, Y, W; kw..., additional_loss=DifferencePenalty(1e6))
    @test spread(flat) < 0.01
    @test maximum(abs, diff(vec(parameters(flat).B))) < 0.05
    # in between, in between
    spreads = [spread(fit_mle(mspline, Y, W; kw..., additional_loss=DifferencePenalty(λ))) for λ in (1.0, 100.0, 1e4)]
    @test spread(mu) > spreads[1] > spreads[2] > spreads[3] > spread(flat)
    # the constant it flattens to is the maximum likelihood constant: the overall frequency
    @test only(unique(round.(vec(curve(flat, :p)); digits=2))) ≈ count(Y) / N atol = 0.02
    # order 1 flattens too, and a penalty that does not wrap does not flatten a periodic curve to a line
    @test spread(fit_mle(mspline, Y, W; kw..., additional_loss=DifferencePenalty(1e6; order=1))) < 0.01

    # a trigonometric basis has no neighbouring coefficients to difference: the curve is penalized
    mtrig = SmoothARPeriodicHMM(1, 1, T; memory=0, season=3)
    mut = fit_mle(mtrig, Y, W; kw...)
    @test spread(mut) > 0.4
    flat_t = fit_mle(mtrig, Y, W; kw..., additional_loss=CurvePenalty(1e6))
    @test spread(flat_t) < 0.02
    spreads_t = [spread(fit_mle(mtrig, Y, W; kw..., additional_loss=CurvePenalty(λ))) for λ in (1.0, 100.0, 1e4)]
    @test spread(mut) > spreads_t[1] > spreads_t[2] > spreads_t[3] > spread(flat_t)
    # the same penalty on a spline curve
    @test spread(fit_mle(mspline, Y, W; kw..., additional_loss=CurvePenalty(1e6))) < 0.02
    # a ridge on the harmonics: not a flat curve, but a lower one in the higher harmonics
    harmonics(phi, θ, p) = 1e3 * sum(abs2, θ[4:end])
    mh = fit_mle(mtrig, Y, W; kw..., additional_loss=harmonics)
    @test maximum(abs, parameters(mh).B[4:end]) < 0.25 * maximum(abs, parameters(mut).B[4:end])

    # every parameter of an AR(1) curve at once: μ, ρ and σ
    y = rand(StableRNG(17), AR1([0.2 * sin(2π * t / T) for t = 1:T], fill(0.3, T), [0.5 + 0.3 * cos(2π * t / T) for t = 1:T]), 3000; y₁=0.0)
    ma = SmoothAR1(2, T; init=(μ=0.0, ρ=0.1, σ=0.5))
    fa, _ = fit_mle(ma, y; solver=SOLVER)
    pa, _ = fit_mle(ma, y; solver=SOLVER, additional_loss=CurvePenalty(1e7))
    for name in (:μ, :ρ, :σ)
        @test maximum(curve(pa, name)) - minimum(curve(pa, name)) < 0.1 * (maximum(curve(fa, name)) - minimum(curve(fa, name))) + 1e-3
    end
end

@testset "additional_loss: what a loss may not do" begin
    fxs = penalty_fixtures()
    for fx in fxs
        fx.name in ("SmoothAR1", "SmoothAR1 given states", "SmoothDistribution", "SmoothSeasonalMixture",
            "SmoothARPeriodicHMM", "SmoothARPeriodicHMM given states", "SmoothPeriodicHMM") || continue
        @testset "$(fx.name)" begin
            parts = Set(first.(fx.parts))
            for bad in (NaN, Inf, -Inf)
                err = try
                    fx.run(; additional_loss=(phi, θ, p) -> bad)
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                msg = sprint(showerror, err)
                @test occursin("finite real number", msg) && occursin(repr(bad), msg)
                @test any(part -> occursin("part = $(repr(part))", msg), parts)
                @test occursin("index = ", msg)
            end
            # not a number
            for bad in ("x", missing, nothing, [1.0], (1.0, 2.0))
                @test_throws ArgumentError fx.run(; additional_loss=(phi, θ, p) -> bad)
            end
            # not a function of (phi, θ, p)
            err = try
                fx.run(; additional_loss=θ -> 0.0)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError && occursin("additional_loss(phi, θ, p)", sprint(showerror, err))
            @test_throws ArgumentError fx.run(; additional_loss=3)
            @test_throws ArgumentError fx.run(; additional_loss=(a, b) -> 0.0)
            # an error of its own is not swallowed
            @test_throws ErrorException fx.run(; additional_loss=(phi, θ, p) -> error("mine"))
        end
    end
end

@testset "additional_loss: penalized fits have no information criteria" begin
    for fx in penalty_fixtures()
        hasproperty(fx, :select) || continue
        @testset "$(fx.name)" begin
            f = fx.select(; additional_loss=RidgePenalty(1.0))
            @test f isa SeasonalFit && f.penalized && f.history.penalized && :penalized in propertynames(f)
            for criterion in (aic, aicc, bic)
                err = try
                    criterion(f)
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                msg = sprint(showerror, err)
                @test occursin("penalized", msg) && occursin("held-out", msg) && occursin("dof", msg)
            end
            @test loglikelihood(f) == loglikelihood(f.history)
            @test dof(f) == SPSM.nparams(f.model) + SPSM._ninitial(f.model)
            shown = sprint(show, MIME"text/plain"(), f)
            @test occursin("penalized loglikelihood", shown) && !occursin("bic", shown)
            @test occursin("penalized loglikelihood", sprint(show, MIME"text/plain"(), f.history))
            # without it, they are there
            if fx.name in ("SmoothAR1", "SmoothARPeriodicHMM")
                u = fx.select()
                @test !u.penalized && !u.history.penalized
                @test isfinite(aic(u)) && isfinite(aicc(u)) && isfinite(bic(u))
                @test dof(u) == dof(f)
                @test occursin("bic", sprint(show, MIME"text/plain"(), u))
                @test !occursin("penalized", sprint(show, MIME"text/plain"(), u))
            end
        end
    end
end

@testset "additional_loss: the spatial HMM, its range included" begin
    Random.seed!(99)
    K, T, D, G = 2, 8, 4, 4
    h = [60.0 * abs(i - j) for i = 1:D, j = 1:D]
    θᴬ = 0.5 * (rand(K, K - 1, 3) .- 0.5)
    θᴮ = rand(K, D, 1, 3) .- 0.5
    θᴿ = rand(K, 3) .- 0.5
    θᴿ[:, 1] = log.(100.0 .* (1:K))
    m0 = SmoothARPeriodicHMMSpatial(fill(1 / K, K), θᴬ, θᴮ, θᴿ, T, h; QMC_m=10)
    n2ts = [n_to_t(2T, T) for _ = 1:G]
    Y_pasts = [zeros(Bool, D, 0) for _ = 1:G]
    Ys = [last(rand(ARPeriodicHMMSpatial(m0), n2ts[g]; Y_past=Y_pasts[g], seq=true)) for g = 1:G]
    run(; kw...) = fit_mle(m0, Ys, Y_pasts; n2t=n2ts, solver=SOLVER, maxiter=2, maxiters_R=3, rng=Xoshiro(1), kw...)

    loss, calls = recording_loss()
    m, hist = run(; additional_loss=loss)
    @test hist.penalized
    @test Set((c.part, c.index) for c in calls) ==
          Set(vcat([(:B, (k, s, 1)) for k = 1:K, s = 1:D][:], [(:A, k) for k = 1:K], [(:R, k) for k = 1:K]))
    @test all(c -> c.T == T, calls)
    for k = 1:K
        i = findlast(c -> c.part == :R && c.index == k && c.θ == collect(parameters(m).R[k, :]), calls)
        @test i !== nothing
        @test collect(calls[i].curve) ≈ curve(m, :R)[k, :] rtol = 1e-12
        @test calls[i].names === nothing
    end

    # nothing, and a loss of zero, change nothing
    m_plain, h_plain = run()
    m_none, h_none = run(; additional_loss=nothing)
    @test parameters(m_none) == parameters(m_plain) && h_none.logtots == h_plain.logtots && !h_none.penalized
    m_zero, h_zero = run(; additional_loss=RidgePenalty(0.0))
    @test collect(parameters(m_zero)) ≈ collect(parameters(m_plain)) rtol = 1e-8

    # a loss on the range alone moves the range and nothing else's loss
    pin(phi, θ, p) = p.part === :R ? 1e8 * sum(abs2, θ .- θᴿ[p.index, :]) : 0.0
    m_pin, _ = run(; additional_loss=pin)
    @test parameters(m_pin).R ≈ θᴿ atol = 1e-3
    @test maximum(abs, parameters(m_plain).R .- θᴿ) > 1e-2
end

@testset "additional_loss: a state without data goes to the minimizer of the loss" begin
    y = rand(StableRNG(18), AR1([0.1t for t = 1:12], fill(0.3, 12), fill(0.5, 12)), 400; y₁=0.0)
    z = rand(StableRNG(19), 1:2, 400)
    m = SmoothAR1(1, 12; K=3, init=(μ=0.5, ρ=0.2, σ=0.7))
    plain, _ = fit_mle(m, y, z; solver=SOLVER)
    @test column(plain.spec, parameters(plain), 3) ≈ column(m.spec, parameters(m), 3)
    ridge, h = fit_mle(m, y, z; solver=SOLVER, additional_loss=RidgePenalty(1.0))
    @test maximum(abs, column(ridge.spec, parameters(ridge), 3)) < 1e-4
    @test h.penalized && isfinite(loglikelihood(h))
    # the states with data are fitted as with a loss of their own
    @test column(ridge.spec, parameters(ridge), 1) != column(plain.spec, parameters(plain), 1)
end
