# # Fitting a periodic AR(1)
#
# `fit_mle(model, data...) -> (model, history)`, the convention every sequence-model package in
# the ecosystem uses: `HiddenMarkovModels.baum_welch`, `HMMBase.fit_mle` and
# `ExpectationMaximization.fit_mle` all take the model
# as the starting point and hand the same kind of model back. The coefficients travel inside it,
# so nothing has to be returned beside it.
#
# One honest difference from the EM fits: the AR(1) likelihood is maximized in one solve from
# per-period sufficient statistics, so `parameters(m)` is a starting point for the optimizer but
# the fit does not iterate on the model itself.
#
# The objective is the summed negative loglikelihood, as in version 4, so that a fit given
# `OneHot(z, K)` is the fit given `z` to the last bit.

"""
    fit_mle(m::SmoothAR1, y; n2t, past, solver, adtype, solve_kwargs) -> (m, history)

Maximum likelihood fit of a periodic AR(1)

    y[n] = μ(t) + ρ(t) * y[n-1] + σ(t) * ε[n],   with t = n2t[n]

starting from `parameters(m)`, whose layout `m.spec` describes.

Returns a `SmoothAR1` of the same shape and a [`FitHistory`](@ref). `AR1(m)` is the evaluated
model to simulate from, `curve(m, :σ)` the seasonal curve, `parameters(m).σ` the coefficients.
Warns when the fitted `ρ(t)` reaches ±1, the sign of a start too far from the data: build `m`
with `init` (see [`SmoothAR1`](@ref)).

`past`, the value before `y[1]` (a number, or `(; Y = values)` whose last is used), adds the
transition into `y[1]`; without it the fit is conditional on `y[1]`.
"""
function fit_mle(m::SmoothAR1, y::AbstractVector;
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer}, past=nothing,
    solver=_required_solver(), adtype=default_adtype(), solve_kwargs=(;))
    @argcheck ncolumns(m.spec, parameters(m)) == 1 "`m` holds one column of coefficients per hidden state; pass `z` to fit them"
    design = PeriodicDesign(m.spec, 1:m.nT)
    y, n2t = _ar1_with_past(y, n2t, past)
    stats = ar1_sufficient_statistics(y, n2t, m.nT)

    sol = solve_unconstrained(ar1_nll, collect(float.(parameters(m))), (stats, design);
        solver, adtype, solve_kwargs, return_sol=true)
    return _ar1_fitted(m, sol, sum(stats.N))
end

# A function barrier. `solve`'s return type is not inferred, so in `fit_mle` itself every use of
# `sol` would be a dynamic dispatch, and `FitHistory`'s type -- which holds the whole solution's --
# would be built at run time on every fit.
function _ar1_fitted(m::SmoothAR1, sol, N::Real)
    fitted = SmoothAR1(m.spec, m.nT, _params_like(m.spec, parameters(m), sol.u))
    _warn_if_unit_root(fitted)
    history = FitHistory(;
        converged=successful_retcode(sol.retcode),
        iterations=sol.stats.iterations,
        logtots=[_ar1_loglikelihood(sol.objective, N)],
        solution=sol)
    return fitted, history
end

"""
    fit_mle(m::SmoothAR1, y, z; n2t, K, past, solver, adtype, ensemble, solve_kwargs) -> (m, history)
    fit_mle(m::SmoothAR1, y, W; n2t, past, solver, adtype, ensemble, solve_kwargs) -> (m, history)

Maximum likelihood fit of one periodic AR(1) per hidden state, given the state sequence `z`, or
`W = OneHot(z, K)`, or given state weights: an `N × K` matrix `W` whose row `n` weights the
transition into `y[n]` in each state.

`parameters(m).ρ` is `(2d+1) × K`, column `k` holding the coefficients of state `k`, so
`curve(m, :ρ)` is `K × T` and `AR1(m)` carries `K × T` curves. A model built with
`SmoothAR1(d, T)` rather than `SmoothAR1(d, T; K)` starts every state from the same column.

`history.retcode` is the `K` return codes. Pass `ensemble = EnsembleThreads()` to run the `K`
fits in parallel. A state that `z` never visits, or whose weights are all zero, has no data and
keeps its starting value. Warns, naming the states, when a fitted `ρ(t)` reaches ±1.

A model of one state also takes a vector of float weights, one per transition, and returns a
model of one state. Weights must be finite and nonnegative, and not all zero. `past` is as for
the fit of one AR(1); the first row of `W` counts only with it.
"""
function fit_mle(m::SmoothAR1, y::AbstractVector, z::AbstractVector{<:Integer};
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer},
    K=_nstates(m, z), past=nothing,
    solver=_required_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;))
    @argcheck length(z) == length(y) "`z` has $(length(z)) states and `y` $(length(y)) observations"
    @argcheck all(in(1:K), z) "the states in `z` must lie in 1:$K"
    T = m.nT
    yp, n2tp = _ar1_with_past(y, n2t, past)
    shift = length(yp) - length(y)   # 1 when the value before `y[1]` leads the series
    stats = [ar1_sufficient_statistics(yp, n2tp, T, [n + shift for n = 2-shift:length(y) if z[n] == k]) for k = 1:K]
    return _fit_ar1_states(m, stats; solver, adtype, ensemble, solve_kwargs)
end

fit_mle(m::SmoothAR1, y::AbstractVector, W::OneHot; kwargs...) = fit_mle(m, y, W.z; K=W.K, kwargs...)

function fit_mle(m::SmoothAR1, y::AbstractVector, W::AbstractMatrix{<:Real};
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer}, past=nothing,
    solver=_required_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;))
    cols = ncolumns(m.spec, parameters(m))
    K = cols > 1 ? cols : size(W, 2)
    N = length(y)
    _check_n2t(n2t, N, m.nT)
    W = _state_weights(W, N, K)
    yp, n2tp = _ar1_with_past(y, n2t, past)
    stats = [_ar1_weighted_statistics(yp, n2tp, m.nT, _transition_weights(view(W, :, k), length(yp) - N)) for k = 1:K]
    return _fit_ar1_states(m, stats; solver, adtype, ensemble, solve_kwargs)
end

function fit_mle(m::SmoothAR1, y::AbstractVector, w::AbstractVector{<:AbstractFloat};
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer}, past=nothing,
    solver=_required_solver(), adtype=default_adtype(), solve_kwargs=(;))
    cols = ncolumns(m.spec, parameters(m))
    cols == 1 || throw(ArgumentError(
        "`m` holds $cols columns of coefficients, one per hidden state: pass an N × $cols matrix of weights, or OneHot(z, $cols)"))
    N = length(y)
    _check_n2t(n2t, N, m.nT)
    _state_weights(w, N, 1)
    yp, n2tp = _ar1_with_past(y, n2t, past)
    stats = _ar1_weighted_statistics(yp, n2tp, m.nT, _transition_weights(w, length(yp) - N))
    design = PeriodicDesign(m.spec, 1:m.nT)
    sol = solve_unconstrained(ar1_nll, collect(float.(parameters(m))), (stats, design);
        solver, adtype, solve_kwargs, return_sol=true)
    return _ar1_fitted(m, sol, sum(stats.N))
end

fit_mle(::SmoothAR1, ::AbstractVector, w::AbstractVector; kwargs...) = throw(ArgumentError(
    "the third argument of an AR(1) fit is the state sequence, OneHot(z, K) or a vector of integers, an N × K matrix of weights, or for a model of one state a vector of float weights; got a $(typeof(w))"))

# The weights of the series the statistics read, in which the value before `y[1]`, when there is
# one, comes first and has no transition of its own.
_transition_weights(w::AbstractVector, shift::Integer) = shift == 0 ? w : vcat(false, w)

# The weighted per-period statistics of `ar1_sufficient_statistics`: transition `n` counts with
# weight `w[n]`, and a transition of weight zero is left out.
function _ar1_weighted_statistics(y::AbstractVector, n2t::AbstractVector{<:Integer}, T::Integer, w::AbstractVector)
    N = zeros(T)
    ∑y², ∑y₋², ∑yy₋, ∑y, ∑y₋ = (zeros(T) for _ = 1:5)
    for n = 2:length(y)
        wₙ = w[n]
        iszero(wₙ) && continue
        t = n2t[n]
        yₙ, yₙ₋₁ = y[n], y[n-1]
        N[t] += wₙ
        ∑y²[t] += wₙ * yₙ^2
        ∑y₋²[t] += wₙ * yₙ₋₁^2
        ∑yy₋[t] += wₙ * (yₙ * yₙ₋₁)
        ∑y[t] += wₙ * yₙ
        ∑y₋[t] += wₙ * yₙ₋₁
    end
    return (; N, ∑y², ∑y₋², ∑yy₋, ∑y, ∑y₋)
end

# One solve per state, from the statistics of each, as an ensemble.
function _fit_ar1_states(m::SmoothAR1, stats::AbstractVector; solver, adtype, ensemble, solve_kwargs)
    spec, T, K = m.spec, m.nT, length(stats)
    design = PeriodicDesign(spec, 1:T)
    starts = _starting_points(spec, parameters(m), K)

    base = OptimizationProblem(OptimizationFunction(ar1_nll, _optimization_adtype(adtype)), starts[1], (stats[1], design))
    eprob = EnsembleProblem(base;
        prob_func=(prob, ctx) -> remake(prob; u0=starts[ctx.sim_id], p=(stats[ctx.sim_id], design)))
    sol = solve(eprob, solver, ensemble; trajectories=K, solve_kwargs...)

    θ = Matrix{Float64}(undef, ncoef(spec), K)
    for k = 1:K
        successful_retcode(sol.u[k].retcode) || @warn "AR(1) state $k: $(sol.u[k].retcode)"
        θ[:, k] = sol.u[k].u
    end

    # The states partition the transitions, so the loglikelihoods add up.
    N = sum(sum(s.N) for s in stats)
    fitted = SmoothAR1(spec, T, seasonal_params(spec, θ))
    _warn_if_unit_root(fitted)
    history = FitHistory(;
        converged=all(s -> successful_retcode(s.retcode), sol.u),
        iterations=maximum(s -> s.stats.iterations, sol.u),
        logtots=[_ar1_loglikelihood(sum(s.objective for s in sol.u), N)],
        solution=sol.u)
    return fitted, history
end

"""
    fit(SeasonalFit, m::SmoothAR1, y, args...; n2t, past, solver, kwargs...) -> SeasonalFit

Fit a periodic AR(1) with [`fit_mle`](@ref), same keywords, and return a [`SeasonalFit`](@ref).
`nobs` is the number of transitions, those of positive weight when weights are given.
"""
function StatsAPI.fit(::Type{SeasonalFit}, m::SmoothAR1, y::AbstractVector, args...;
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer}, past=nothing, kwargs...)
    fitted, history = fit_mle(m, y, args...; n2t, past, kwargs...)
    # `nobs` of a conditional likelihood is the number of terms in it, not the number of points.
    yp, n2tp = _ar1_with_past(y, n2t, past)
    N = _ar1_nobs(yp, n2tp, m.nT, length(yp) - length(y), args...)
    return SeasonalFit(fitted, history, N, loglikelihood(history))
end

_ar1_nobs(y, n2t, T, _, _...) = sum(ar1_sufficient_statistics(y, n2t, T).N)
_ar1_nobs(y, _, _, shift, w::AbstractVector{<:AbstractFloat}) = _positive_transitions(w, shift, length(y))
_ar1_nobs(y, _, _, shift, W::AbstractMatrix{<:Real}) =
    _positive_transitions(vec(sum(W; dims=2)), shift, length(y))

_positive_transitions(w, shift, n) = count(>(0), view(_transition_weights(w, shift), 2:n))

# `ρ = ±1` to machine precision. From a start far from the data the optimizer's first steps can
# push `ρ`'s link to where it rounds to ±1; its gradient is then zero, the fit cannot come back,
# and it reports success with a random walk. A warning, since the solver does not see it.
function _warn_if_unit_root(m::SmoothAR1)
    ρ = curve(m, :ρ)
    at_one(r) = abs(r) >= 1 - 1e-8
    if ρ isa AbstractMatrix
        states = [k for k in axes(ρ, 1) if any(at_one, view(ρ, k, :))]
        isempty(states) || @warn "AR(1) fit: ρ(t) reaches ±1 in state(s) $states, so the fitted process is a random walk there. The start was probably too far from the data: pass one nearer it, as `SmoothAR1(d, T; K, init = (μ = …, ρ = …, σ = …))`."
    elseif any(at_one, ρ)
        @warn "AR(1) fit: ρ(t) reaches ±1, so the fitted process is a random walk there. The start was probably too far from the data: pass one nearer it, as `SmoothAR1(d, T; init = (μ = …, ρ = …, σ = …))`."
    end
    return nothing
end

# The model supplies the number of states when it has them, as `hmm` does for the HMM fits.
function _nstates(m::SmoothAR1, z)
    cols = ncolumns(m.spec, parameters(m))
    return cols > 1 ? cols : maximum(z)
end

# One starting point per state: the same coefficients for all of them, or one column each.
function _starting_points(spec::ParamSpec, ps::AbstractVector, K::Integer)
    cols = ncolumns(spec, ps)
    cols == 1 && return [collect(float.(ps)) for _ = 1:K]
    @argcheck cols == K "`parameters(m)` holds $cols columns, the fit wants $K"
    return [column(spec, ps, k) for k = 1:K]
end
