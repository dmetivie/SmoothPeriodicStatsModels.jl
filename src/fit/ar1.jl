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

"""
    fit_mle(m::SmoothAR1, y; n2t, solver, adtype, solve_kwargs) -> (m, history)

Maximum likelihood fit of a periodic AR(1)

    y[n] = μ(t) + ρ(t) * y[n-1] + σ(t) * ε[n],   with t = n2t[n]

starting from `parameters(m)`, whose layout `m.spec` describes.

Returns a `SmoothAR1` of the same shape and a [`FitHistory`](@ref). `AR1(m)` is the evaluated
model to simulate from, `curve(m, :σ)` the seasonal curve, `parameters(m).σ` the coefficients.
"""
function fit_mle(m::SmoothAR1, y::AbstractVector;
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer},
    solver=default_solver(), adtype=default_adtype(), solve_kwargs=(;))
    @argcheck ncolumns(m.spec, parameters(m)) == 1 "`m` holds one column of coefficients per hidden state; pass `z` to fit them"
    design = PeriodicDesign(m.spec, 1:m.nT)
    stats = ar1_sufficient_statistics(y, n2t, m.nT)

    sol = solve_unconstrained(ar1_nll, collect(float.(parameters(m))), (stats, design);
        solver, adtype, solve_kwargs, return_sol=true)
    return _ar1_fitted(m, sol, sum(stats.N))
end

# A function barrier. `solve`'s return type is not inferred, so in `fit_mle` itself every use of
# `sol` would be a dynamic dispatch, and `FitHistory`'s type -- which holds the whole solution's --
# would be built at run time on every fit.
function _ar1_fitted(m::SmoothAR1, sol, N::Integer)
    fitted = SmoothAR1(m.spec, m.nT, _params_like(m.spec, parameters(m), sol.u))
    history = FitHistory(;
        converged=successful_retcode(sol.retcode),
        iterations=sol.stats.iterations,
        logtots=[_ar1_loglikelihood(sol.objective, N)],
        solution=sol)
    return fitted, history
end

"""
    fit_mle(m::SmoothAR1, y, z; n2t, K, solver, adtype, ensemble, solve_kwargs) -> (m, history)

Maximum likelihood fit of one periodic AR(1) per hidden state, given the state sequence `z`.

`parameters(m).ρ` is `(2d+1) × K`, column `k` holding the coefficients of state `k`, so
`curve(m, :ρ)` is `K × T` and `AR1(m)` carries `K × T` curves. A model built with
`SmoothAR1(d, T)` rather than `SmoothAR1(d, T; K)` starts every state from the same column.

`history.retcode` is the `K` return codes. Pass `ensemble = EnsembleThreads()` to run the `K`
fits in parallel. A state that `z` never visits has no data and keeps its starting value.
"""
function fit_mle(m::SmoothAR1, y::AbstractVector, z::AbstractVector{<:Integer};
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer},
    K=_nstates(m, z),
    solver=default_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;))
    spec, T = m.spec, m.nT
    design = PeriodicDesign(spec, 1:T)
    stats = [ar1_sufficient_statistics(y, n2t, T, [n for n = 2:length(y) if z[n] == k]) for k = 1:K]
    starts = _starting_points(spec, parameters(m), K)

    base = OptimizationProblem(OptimizationFunction(ar1_nll, adtype), starts[1], (stats[1], design))
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
    history = FitHistory(;
        converged=all(s -> successful_retcode(s.retcode), sol.u),
        iterations=maximum(s -> s.stats.iterations, sol.u),
        logtots=[_ar1_loglikelihood(sum(s.objective for s in sol.u), N)],
        solution=sol.u)
    return fitted, history
end

"""
    loglikelihood(m::SmoothAR1, y; n2t) -> Float64

The conditional loglikelihood of `y` under `m`, given its first observation.

`ar1_nll` drops the `-N/2·log(2π)` that does not depend on the coefficients; this puts it back,
so the number is comparable with anything else. `N` counts *transitions*: the first observation
conditions the rest rather than contributing a term.
"""
function StatsAPI.loglikelihood(m::SmoothAR1, y::AbstractVector;
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer})
    stats = ar1_sufficient_statistics(y, n2t, m.nT)
    design = PeriodicDesign(m.spec, 1:m.nT)
    return _ar1_loglikelihood(ar1_nll(parameters(m), (stats, design)), sum(stats.N))
end

_ar1_loglikelihood(nll, N::Integer) = -nll - N * log(2π) / 2

"""
    fit(SeasonalFit, m::SmoothAR1, y, args...; n2t, kwargs...) -> SeasonalFit

Fit a periodic AR(1) and keep the diagnostics. The same estimator and the same keywords as
[`fit_mle`](@ref), so `f.model` and `f.history` are what that would have returned; see
[`SeasonalFit`](@ref) for what the extra tier buys.
"""
function StatsAPI.fit(::Type{SeasonalFit}, m::SmoothAR1, y::AbstractVector, args...;
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer}, kwargs...)
    fitted, history = fit_mle(m, y, args...; n2t, kwargs...)
    # `nobs` of a conditional likelihood is the number of terms in it, not the number of points.
    N = sum(ar1_sufficient_statistics(y, n2t, m.nT).N)
    return SeasonalFit(fitted, history, N, loglikelihood(history))
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
