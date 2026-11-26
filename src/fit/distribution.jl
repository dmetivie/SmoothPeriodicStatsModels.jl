# # Fitting a seasonal distribution, one per hidden state
#
# One solve per state, of the weighted negative loglikelihood divided by the total weight of the
# state: an average, which leaves the optimum where it is and keeps the first gradient of order
# one, so that the line search cannot push a log link into underflow on its first step. An
# additional loss is divided by the same total, so that it counts as added to the summed
# negative loglikelihood.

"""
    fit_mle(m::SmoothDistribution, y[, w]; n2t, past, solver, adtype, additional_loss, solve_kwargs) -> SmoothDistribution

Maximum likelihood fit of the seasonal distribution `m` to the observations `y`, `y[n]` at period
index `n2t[n]`, starting from `parameters(m)`. Returns the fitted model.

- `w` are the state weights, `OneHot(z, K)` or an `N × K` matrix, required when `K > 1`; for a
  model of one state, a vector of float weights. Weights are finite and nonnegative, and not all
  zero. A state of zero weight keeps its start, with a warning.
- `additional_loss(phi, θ, p)` is added to the negative loglikelihood of each state, with
  `part = :distribution` and `index = k`, see
  [`additional_loss`](@ref SmoothPeriodicStatsModels.additional_loss).
- `adtype = NoAutoDiff()` with a derivative-free `solver` fits a family whose constructor cannot
  take dual numbers.
- `past` is accepted and unused: the observations are independent given the state.
"""
function fit_mle(m::SmoothDistribution, y::AbstractVector{<:Real}; n2t::AbstractVector{<:Integer},
    past=nothing, kwargs...)
    K = ncolumns(m.family.spec, m.ps)
    K == 1 || throw(ArgumentError("this model has $K states: pass the weights, OneHot(z, K) or an N × K matrix"))
    return first(_fit_distribution(m, y, nothing, n2t; kwargs...))
end

function fit_mle(m::SmoothDistribution, y::AbstractVector{<:Real}, w; n2t::AbstractVector{<:Integer},
    past=nothing, kwargs...)
    return first(_fit_distribution(m, y, w, n2t; kwargs...))
end

"""
    fit(SeasonalFit, m::SmoothDistribution, y[, w]; n2t, solver, kwargs...) -> SeasonalFit

Fit with [`fit_mle`](@ref), same keywords, and return a [`SeasonalFit`](@ref). `nobs` is the
number of observations of positive weight, and the loglikelihood is weighted by `w`.
"""
function StatsAPI.fit(::Type{SeasonalFit}, m::SmoothDistribution, y::AbstractVector{<:Real}, w...;
    n2t::AbstractVector{<:Integer}, past=nothing, kwargs...)
    length(w) <= 1 || throw(ArgumentError("`fit` takes the observations and at most one array of weights"))
    K = ncolumns(m.family.spec, m.ps)
    isempty(w) && K > 1 && throw(ArgumentError("this model has $K states: pass the weights, OneHot(z, K) or an N × K matrix"))
    fitted, history = _fit_distribution(m, y, isempty(w) ? nothing : only(w), n2t; kwargs...)
    W = isempty(w) ? nothing : _state_weights(only(w), length(y), K)
    nobs = W === nothing ? length(y) : count(n -> any(>(0), view(W, n, :)), axes(W, 1))
    return SeasonalFit(fitted, history, nobs, loglikelihood(history))
end

# The fit and its history. The objective of state `k` is `PeriodicNLL` over the observations of
# positive weight in that state, divided by their total weight.
function _fit_distribution(m::SmoothDistribution, y::AbstractVector{<:Real}, w, n2t::AbstractVector{<:Integer};
    solver=_required_solver(), adtype=default_adtype(), additional_loss=nothing, solve_kwargs=(;))
    fam, T = m.family, m.nT
    loss = _resolve_loss(additional_loss)
    spec = fam.spec
    K, N = ncolumns(spec, m.ps), length(y)
    _check_n2t(n2t, N, T)
    W = w === nothing ? nothing : _state_weights(w, N, K)
    design = PeriodicDesign(spec, 1:T)
    phi = _FamilyPhi(spec, design)

    ps = copy(m.ps)
    sols = []
    logtot = 0.0
    empty = Int[]
    for k = 1:K
        keep = W === nothing ? collect(1:N) : [n for n = 1:N if W[n, k] > 0]
        if isempty(keep)
            push!(empty, k)
            continue
        end
        for n in keep
            isfinite(y[n]) || throw(ArgumentError("`y[$n] = $(y[n])` has a positive weight in state $k; the observations must be finite"))
        end
        groups = [Int[] for _ = 1:T]
        for n in keep
            push!(groups[n2t[n]], n)
        end
        # Unit weights are left out of the objective, so that a known path gives the unweighted
        # likelihood of its observations exactly.
        wk = W === nothing || W isa OneHot ? nothing : view(W, :, k)
        total = wk === nothing ? length(keep) : sum(wk[n] for n in keep)
        nll = PeriodicNLL(fam, design, y, groups, wk)
        start = column(spec, m.ps, k)
        _check_loss(loss, phi, start, :distribution, k, T)
        sol = solve_unconstrained(_StateNLL(nll, total, loss, phi, k, T), start, NullParameters();
            solver, adtype, solve_kwargs, return_sol=true)
        _set_column!(ps, spec, k, sol.u)
        push!(sols, sol)
        logtot -= nll(sol.u) + _loss_float(loss, phi, sol.u, :distribution, k, T)
    end
    isempty(empty) || @warn "SmoothDistribution fit: state(s) $empty have no weight and keep their starting coefficients"
    fitted = SmoothDistribution(fam, T, ps)
    history = FitHistory(;
        converged=all(s -> successful_retcode(s.retcode), sols),
        iterations=isempty(sols) ? 0 : maximum(s -> s.stats.iterations, sols),
        logtots=[logtot],
        solution=[s for s in sols], penalized=loss !== nothing)
    return fitted, history
end

# The objective of one state: the average negative loglikelihood, and the additional loss over the
# total weight.
struct _StateNLL{N,W,L,PH}
    nll::N
    total::W
    loss::L
    phi::PH
    index::Int
    T::Int
end
(o::_StateNLL{<:Any,<:Any,Nothing})(θ, _) = o.nll(θ) / o.total
(o::_StateNLL)(θ, _) = o.nll(θ) / o.total + _loss_value(o.loss, o.phi, θ, :distribution, o.index, o.T) / o.total

# Write the flat coefficients `θ` of state `k` into its column of every block.
function _set_column!(ps, spec, k, θ)
    for (j, nm) in enumerate(paramnames(spec))
        view(getproperty(ps, nm), :, k) .= view(θ, blockrange(spec, j))
    end
    return ps
end
