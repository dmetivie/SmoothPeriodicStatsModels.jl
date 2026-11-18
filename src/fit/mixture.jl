# # Fitting a seasonal mixture
#
# `fit_mle(m::SmoothSeasonalMixture, y; n2t) -> (m, history)`, the convention of every fit in
# this package. The EM loop itself is `ExpectationMaximization`'s; what is below is the seasonal
# M-step it calls and the wrapper that packs a `SmoothSeasonalMixture` into the shape that
# driver wants and unpacks the result.

"""
    fit_mle(m::SmoothSeasonalMixture, y[, w]; n2t, solver, kwargs...) -> (m, history)

Fit a seasonal mixture by EM: both the mixing weights and the component parameters vary with
the period index `t = n2t[n]`, each through its own basis expansion.

Starts from `parameters(m)` and returns the fitted mixture and a [`FitHistory`](@ref).

Keywords: `maxiter = 100`, `tol = 1e-3`, `rtol`, `robust`, `display` (`:none`, `:iter` or
`:final`), `method` (`ClassicEM()` or `StochasticEM(rng)`), and the solver settings `solver`,
`adtype`, `solve_kwargs` of the M-step. `w` gives each observation a weight. Any family with a
[`seasonal_family`](@ref) declaration can be fitted.
"""
function fit_mle(m::SmoothSeasonalMixture, y::AbstractVector, weights...;
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer},
    display=:none, maxiter=100, tol=1e-3, rtol=nothing, robust=false, method=ClassicEM(),
    solver=_required_solver(), adtype=default_adtype(), solve_kwargs=(;))
    K = ncomponents(m)
    ps = parameters(m)

    # The driver works in the loose coefficient layout the M-step kernels were written for:
    # `θ_α` row-per-category, `θ_Y` one vector per component. `ps` is the same numbers.
    θ_α = permutedims(collect(ps.α))
    θ_Y = [collect(float.(component_coefs(m, k))) for k = 1:K]

    _, θ_α, θ_Y, h = _fit_seasonal_mixture(mixtures(m), θ_α, θ_Y, y, n2t, weights...;
        families=m.components, display, maxiter, tol, rtol, robust, method, solver, adtype,
        solve_kwargs)

    fitted = SmoothSeasonalMixture(m.components, m.spec_α, m.nT, copy(ps))
    parameters(fitted).α .= permutedims(θ_α)
    for k = 1:K
        component_coefs(fitted, k) .= θ_Y[k]
    end
    history = FitHistory(;
        converged=h["converged"], iterations=h["iterations"], logtots=h["logtots"])
    return fitted, history
end

"""
    fit(SeasonalFit, m::SmoothSeasonalMixture, y, args...; n2t, solver, kwargs...) -> SeasonalFit

Fit a seasonal mixture with [`fit_mle`](@ref), same keywords, and return a
[`SeasonalFit`](@ref), whose `bic` compares numbers of components or degrees.
"""
function StatsAPI.fit(::Type{SeasonalFit}, m::SmoothSeasonalMixture, y::AbstractVector, args...;
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer}, kwargs...)
    fitted, history = fit_mle(m, y, args...; n2t, kwargs...)
    return SeasonalFit(fitted, history, length(y), loglikelihood(history))
end

# The loop itself. Not exported: `fit_mle(::SmoothSeasonalMixture, …)` is the entry point, and
# this is the loose-coefficient form it is written in.
#
# `θ_α::AbstractMatrix`, not `AbstractArray`: `ExpectationMaximization`'s multi-start
# `fit_mle(::AbstractArray{<:MixtureModel}, ::AbstractVecOrMat, weights...)` also matches, and
# only the narrower second argument makes this strictly more specific than that one.
function _fit_seasonal_mixture(mix::AbstractArray{<:MixtureModel}, θ_α::AbstractMatrix,
    θ_Y::AbstractVector, y::AbstractVecOrMat, n2t::AbstractVector, weights...;
    families, display=:none, maxiter=100, tol=1e-3, rtol=nothing, robust=false,
    method=ClassicEM(), solver=_required_solver(), adtype=default_adtype(), solve_kwargs=(;))
    @argcheck maxiter >= 0

    K, T = ncomponents(mix[1]), size(mix, 1)
    # TODO deal with K=1 case
    @argcheck K > 1

    # Initial parameters. `θ_Y` is a vector of vectors, so `copy` would leave the inner vectors
    # shared with the caller's: the refitted coefficients are returned as fresh vectors instead.
    θ_α = copy(θ_α)

    dists = copy(hcat(components.(mix)...))
    types = typeof.(dists[:, 1])
    designs = map(f -> PeriodicDesign(f.spec, 1:T), families)

    # separate the array per component to preserve types
    comps = [SeasonalFamily(convert(Vector{types[k, 1]}, dists[k, :]),
        collect(float(θ_Y[k])), families[k], designs[k]) for k = 1:K]

    α = SeasonalWeights(hcat(probs.(mix)...), reshape(θ_α, K - 1, :),
        trig_basis(T, (size(θ_α, 2) - 1) ÷ 2), (; solver, adtype, solve_kwargs))

    sample = SeasonalSample(y, n2t, T)

    history = em_fit_mle!(α, comps, sample, weights..., method;
        display=display, maxiter=maxiter, atol=tol, rtol=rtol, robust=robust)

    θ_α[:, :] = reshape(α.θ, size(θ_α))
    return [MixtureModel([comps[k].dists[t] for k = 1:K], [α.α[k, t] for k = 1:K]) for t = 1:T],
    θ_α, [comps[k].θ for k = 1:K], history
end

"""
    M_step!(α::SeasonalWeights, dists, y::SeasonalSample, γ, ::ClassicEM)

M-step of the seasonal mixture: refit the weights' multinomial-logit coefficients, then each
component's coefficients, from the `N × K` posteriors `γ`.
"""
function M_step!(α::SeasonalWeights, dists, y::SeasonalSample, γ, ::ClassicEM)
    K, T = size(α.α)
    cfg = α.cfg

    s_γ = zeros(T, K)
    s_γα!(s_γ, γ, y.n_in_t)
    update_α!(α.α, α.θ, s_γ, α.basis; cfg.solver, cfg.adtype, cfg.solve_kwargs)

    for k = 1:K
        c = dists[k]
        c.dists[:], θₖ = update_dist(c.fam, c.design, c.θ, view(γ, :, k), y.y, y.n_in_t;
            cfg.solver, cfg.adtype, cfg.solve_kwargs)
        c.θ[:] = θₖ
    end
end

"""
    M_step!(α::SeasonalWeights, dists, y::SeasonalSample, γ, w, ::ClassicEM)

The same M-step with per-observation weights `w`; `γ` is overwritten.
"""
function M_step!(α::SeasonalWeights, dists, y::SeasonalSample, γ, w, ::ClassicEM)
    γ .*= w
    M_step!(α, dists, y, γ, ClassicEM())
end

"""
    M_step!(α::SeasonalWeights, dists, y::SeasonalSample, cat, ::StochasticEM)

Stochastic-EM M-step: `cat[k]` holds the observations assigned to component `k`, used as
posteriors of 0 and 1 in the `ClassicEM` M-step. A trailing `w` adds per-observation weights.
"""
function M_step!(α::SeasonalWeights, dists, y::SeasonalSample, cat::AbstractVector{<:AbstractVector{<:Integer}}, ::StochasticEM)
    γ = zeros(length(y), size(α.α, 1))
    for (k, cₖ) in enumerate(cat)
        γ[cₖ, k] .= 1
    end
    M_step!(α, dists, y, γ, ClassicEM())
end

function M_step!(α::SeasonalWeights, dists, y::SeasonalSample, cat::AbstractVector{<:AbstractVector{<:Integer}}, w, ::StochasticEM)
    γ = zeros(length(y), size(α.α, 1))
    for (k, cₖ) in enumerate(cat)
        γ[cₖ, k] .= 1
    end
    M_step!(α, dists, y, γ, w, ClassicEM())
end

"""
    update_dist(fam, design, θ, γ, observations, n_in_t; solver, adtype, solve_kwargs)

M-step for one seasonal family: a weighted maximum likelihood fit. It refits the coefficients
`θ` of the family `fam` to the observations weighted by their posteriors `γ`, and returns
`(distributions, θ)`, the family's `nT` distributions and its coefficients. A mixture component
and an emission block of a [`SmoothPeriodicHMM`](@ref) are fitted this way. `design` is the family's bases over the period and
`n_in_t[t]` lists the observations at period index `t`.

The generic method maximizes the weighted loglikelihood, so any family with a
[`seasonal_family`](@ref) declaration works. A method for a specific family can replace it, for
example to use sufficient statistics, as the `Exponential` method does:

```julia
function SmoothPeriodicStatsModels.update_dist(
        fam::TimeVaryingFamily{SmoothPeriodicStatsModels.DistConstructor{Rayleigh}},
        design::PeriodicDesign, θ, γ, obs, n_in_t; kwargs...)
    # ... a solve against Σ γ and Σ γ y² per period index, then
    return distributions(fam, design, θ), θ
end
```
"""
function update_dist(fam::TimeVaryingFamily, design::PeriodicDesign, θ, γ, observations, n_in_t;
    solver=_required_solver(), adtype=default_adtype(), solve_kwargs=(;))
    @argcheck length(γ) == size(observations, 1)
    total = sum(γ)
    θ = collect(float.(θ))
    total > 0 || return distributions(fam, design, θ), θ   # a component with no weight left
    nll = PeriodicNLL(fam, design, observations, n_in_t, γ ./ total)
    θ = solve_unconstrained(nll, θ, NullParameters(); solver, adtype, solve_kwargs)
    return distributions(fam, design, θ), θ
end

# ## Why both M-steps minimize an *average* rather than a sum
#
# Dividing by the total posterior weight leaves the argmin where it was and makes the gradient
# independent of `N`. The summed objective does not: at `N = 6000` the first gradient of a
# seasonal `Gamma` component has norm ≈ 2500, so `LBFGS`'s first trial step is ~2500 in
# coefficient space, `exp(η)` underflows to exactly `0.0`, and `Gamma`'s constructor throws
# `DomainError` — before the line search has a chance to back off. Normalized, the same
# gradient has norm 0.6 and the solve reaches the same optimum.

# The sufficient-statistic specialization. The type pins the parametrization exactly — one
# parameter named `θ` behind a `LogLink` — so `exponential_nll`, which is the Exponential
# loglikelihood regrouped per period index, is guaranteed to be the right objective.
const SeasonalExponential = TimeVaryingFamily{DistConstructor{Exponential},
    <:ParamSpec{(:θ,),<:Tuple,Tuple{LogLink}}}

"""
    update_dist(fam::SeasonalExponential, design, θ, γ, observations, n_in_t; kwargs...)

M-step for a seasonal `Exponential` family with a `LogLink` scale, by sufficient statistics.
Per period index, these are the summed posterior weight and the posterior-weighted sum of the
observations.
"""
function update_dist(fam::SeasonalExponential, design::PeriodicDesign, θ, γ, observations, n_in_t;
    solver=_required_solver(), adtype=default_adtype(), solve_kwargs=(;))
    @argcheck length(γ) == size(observations, 1)
    T = design.nT
    total = sum(γ)
    θ = collect(float.(θ))
    total > 0 || return distributions(fam, design, θ), θ

    # sufficient statistics per period index: summed weight, and weighted sum of observations,
    # both per unit of total weight -- `exponential_nll` is linear in them, so this is the same
    # averaging the generic method does, for the same reason.
    s_π = [sum(γ[n] for n in n_in_t[t]; init=0.0) / total for t = 1:T]
    s_πy = [sum(γ[n] * observations[n] for n in n_in_t[t]; init=0.0) / total for t = 1:T]

    θ = solve_unconstrained(exponential_nll, θ, (s_π, s_πy, design.designs[1]);
        solver, adtype, solve_kwargs)
    return distributions(fam, design, θ), θ
end

"""
    s_γα!(s, γ, n_in_t)

Aggregate the posteriors per period index: `s[t, k] = Σ γ[n, k]` over the observations
`n ∈ n_in_t[t]`.
"""
function s_γα!(s_ξ, ξ, n_in_t)
    T, K = size(s_ξ, 1), size(s_ξ, 2)
    for t = 1:T
        for k = 1:K
            s_ξ[t, k] = sum(ξ[n, k] for n in n_in_t[t]; init=0)
        end
    end
end

"""
    update_α!(α, θ, s_γ, basis; solver, adtype, solve_kwargs)

M-step for the seasonal mixture weights: refit the multinomial-logit coefficients `θ` against the
aggregated posteriors `s_γ`, and write the resulting weights into `α`. Same objective as the HMM
transition M-step, see [`multinomial_logit_nll`](@ref).
"""
function update_α!(
    α::AbstractArray{F},
    θ::AbstractArray{F},
    s_γ::AbstractMatrix,
    basis::AbstractMatrix;
    solver=_required_solver(), adtype=default_adtype(), solve_kwargs=(;)
) where {F}
    T, K = size(s_γ)
    u = solve_unconstrained(multinomial_logit_nll, collect(vec(θ)), (s_γ, basis, K);
        solver, adtype, solve_kwargs)
    θ[:, :] = reshape(u, K - 1, :)
    [α[k, t] = exp(polynomial_trigo(t / T, θ[k, :])) for k = 1:K-1, t = 1:T]
    [α[K, t] = 1 for t = 1:T] # last column is 1/normalization (one could do otherwise)
    normalization_polynomial = [1 + sum(α[k, t] for k = 1:K-1) for t = 1:T]
    [α[k, t] /= normalization_polynomial[t] for k = 1:K, t = 1:T]
end

# The E-step and the driver loop come from `ExpectationMaximization`; the seasonal part of the
# E-step is the `loglikelihoods!` method in `src/seasonal_mixture.jl`.
