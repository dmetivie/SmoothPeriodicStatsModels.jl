# `θ_α::AbstractMatrix`, not `AbstractArray`: `ExpectationMaximization`'s multi-start
# `fit_mle(::AbstractArray{<:MixtureModel}, ::AbstractVecOrMat, weights...)` also matches, and only
# the narrower second argument makes this method strictly more specific than that one.
"""
    fit_mle(mix, θ_α, θ_Y, y, n2t; kwargs...) -> (mixtures, θ_α, θ_Y, history)

Fit a seasonal mixture by EM: both the mixing weights and the component parameters vary with the
period index `t`, through trigonometric polynomials.

`mix` is the `T`-long vector of initial mixtures, `θ_α` the `(K-1) × (2dα+1)` multinomial-logit
coefficients of the weights, `θ_Y[k]` the coefficients of component `k` (degrees may differ per
component), `y` the observations and `n2t[n] ∈ 1:T` the period index of observation `n`.

Returns the refitted `T`-long vector of `MixtureModel`s, the refitted coefficients, and the
`history` `Dict` with `"converged"`, `"iterations"` and `"logtots"`.

`ExpectationMaximization`'s own keywords are accepted here too: `rtol`,
`method = StochasticEM()`, and a positional vector of per-observation weights after `n2t`.

To fit a component family other than `Exponential`, add a method of [`update_dist`](@ref); see the
README.
"""
function fit_mle(mix::AbstractArray{<:MixtureModel}, θ_α::AbstractMatrix, θ_Y::AbstractVector, y::AbstractVecOrMat, n2t::AbstractVector, weights...;
    display=:none, maxiter=100, tol=1e-3, rtol=nothing, robust=false, method=ClassicEM(),
    solver=default_solver(), adtype=default_adtype(), solve_kwargs=(;))
    @argcheck maxiter >= 0

    K, T = ncomponents(mix[1]), size(mix, 1)
    # TODO deal with K=1 case
    @argcheck K > 1

    # Initial parameters. `θ_Y` is a vector of vectors, so `copy` would leave the inner vectors
    # shared with the caller's: the refitted coefficients are returned as fresh vectors instead.
    θ_α = copy(θ_α)

    dists = copy(hcat(components.(mix)...))
    types = typeof.(dists[:, 1])
    # separate the array per component to preserve types
    comps = [SeasonalFamily(convert(Vector{types[k, 1]}, dists[k, :]), collect(float(θ_Y[k]))) for k = 1:K]

    basis_Y = [trig_basis(T, (length(θ_Y[k]) - 1) ÷ 2) for k = 1:K]
    α = SeasonalWeights(hcat(probs.(mix)...), reshape(θ_α, K - 1, :),
        trig_basis(T, (size(θ_α, 2) - 1) ÷ 2),
        (; solver, adtype, solve_kwargs, basis_Y))

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
component's own coefficients, from the posteriors `γ`.

`γ` may alias the log-likelihood matrix the E-step wrote, so it is only ever read here.
"""
function M_step!(α::SeasonalWeights, dists, y::SeasonalSample, γ, ::ClassicEM)
    K, T = size(α.α)
    cfg = α.cfg

    s_γ = zeros(T, K)
    s_γα!(s_γ, γ, y.n_in_t)
    update_α!(α.α, α.θ, s_γ, α.basis; cfg.solver, cfg.adtype, cfg.solve_kwargs)

    for k = 1:K
        dists[k].dists[:], θₖ = update_dist(dists[k].dists, dists[k].θ, view(γ, :, k), y.y,
            y.n_in_t, cfg.basis_Y[k]; cfg.solver, cfg.adtype, cfg.solve_kwargs)
        dists[k].θ[:] = θₖ
    end
end

"""
    M_step!(α::SeasonalWeights, dists, y::SeasonalSample, γ, w, ::ClassicEM)

Same M-step with per-observation weights `w`, folded into the posteriors `γ`, which is scratch
here.
"""
function M_step!(α::SeasonalWeights, dists, y::SeasonalSample, γ, w, ::ClassicEM)
    γ .*= w
    M_step!(α, dists, y, γ, ClassicEM())
end

"""
    M_step!(α::SeasonalWeights, dists, y::SeasonalSample, cat, ::StochasticEM)

Stochastic-EM M-step: `cat[k]` holds the observations the S-step assigned to component `k`, so the
posteriors are the hard assignment, and the seasonal refits are then the `ClassicEM` ones.
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
    update_dist(dist, θ_Y, γ, observations, n_in_t, basis; solver, adtype, solve_kwargs)

M-step for one seasonal `Exponential` component: refit its trigonometric coefficients against the
posterior weights `γ`, and return the refitted distributions together with the coefficients.

This is the extension point for new emission families: `fit_mle` dispatches here on the component
type, so a method for your own distribution is enough.

A method owns four family-specific decisions:

1. which sufficient statistics to aggregate per period index `t` (here `s_π` and `s_πy`);
2. the layout of `θ_Y`, one block of `size(basis, 2)` coefficients per seasonal parameter;
3. the objective, any `(θ, p) -> Number` — a plain function, handed to [`solve_unconstrained`](@ref);
4. the link mapping `basis * θ` back onto the parameter domain (here `exp`, for a positive scale).

`dist` is the `T`-long vector of the component's current distributions, `γ[n]` the posterior weight
of observation `n`, and `n_in_t[t]` the observation indices sharing period index `t`. Return the
refitted vector and the coefficients. See the README for a worked `Rayleigh` example.
"""
function update_dist(dist::AbstractVector{F} where {F<:Exponential}, θ_Y::AbstractArray, γ::AbstractVector, observations::AbstractArray, n_in_t, basis::AbstractMatrix;
    solver=default_solver(), adtype=default_adtype(), solve_kwargs=(;))
    @argcheck size(γ, 1) == size(observations, 1)
    dist = copy(dist)
    θ_Y = copy(θ_Y)
    T = size(dist, 1)

    # sufficient statistics per period index: summed weight, and weighted sum of observations
    s_π = [sum(γ[n] for n in n_in_t[t]; init=0.0) for t = 1:T]
    s_πy = [sum(γ[n] * observations[n] for n in n_in_t[t]; init=0.0) for t = 1:T]

    θ_Y[:] = solve_unconstrained(exponential_nll, collect(θ_Y[:]), (s_π, s_πy, basis);
        solver, adtype, solve_kwargs)

    p = [exp(polynomial_trigo(t / T, θ_Y[:])) for t = 1:T]
    dist[:] = Exponential.(p)
    return dist, θ_Y
end

# Uppdate γ
"""
    s_γα!(s_ξ, ξ, n_in_t)
Give an aggregated weight to each category `k` for each "day" `t` which is the sum of the weight `ξ`
"""
function s_γα!(s_ξ, ξ, n_in_t)
    T, K = size(s_ξ, 1), size(s_ξ, 2)
    for t = 1:T
        for k = 1:K
            s_ξ[t, k] = sum(ξ[n, k] for n in n_in_t[t]; init=0)
        end
    end
end

# Uppdate α
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
    solver=default_solver(), adtype=default_adtype(), solve_kwargs=(;)
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
