# # The second tier: a fit with its diagnostics attached
#
# `fit_mle(model, data...)` returns `(model, history)`, which is what every sequence-model
# package in this ecosystem returns and what you want nine times out of ten. It deliberately
# does not remember the data, so it cannot answer `aic` or `bic`.
#
# `SeasonalFit` is the other tier, borrowed from `Copulas.jl`: `fit(SeasonalFit, model, data…)`
# runs the same estimator and keeps the inference metadata — the sample size and the attained
# loglikelihood — so the model-selection questions have answers. It is a
# `StatsAPI.StatisticalModel`, which is what makes `aic`, `aicc` and `bic` free.

"""
    SeasonalFit(model, history, nobs, loglikelihood)
    fit(SeasonalFit, model, data...; solver, kwargs...) -> SeasonalFit

A seasonal fit together with what is needed to judge it.

`fit(SeasonalFit, m0, data...; kwargs...)` runs the same estimator as `fit_mle(m0, data...;
kwargs...)` and keeps the sample size and the attained loglikelihood, so that `loglikelihood`,
`dof`, `nobs`, `aic`, `aicc` and `bic` are defined.

```julia
m, history = fit_mle(m0, y; n2t, solver)
f = fit(SeasonalFit, m0, y; n2t)        # the same estimator, diagnostics kept

f.model          # the fitted model, as `fit_mle` returns it
f.history        # the `FitHistory`, as `fit_mle` returns it
coef(f, :ρ)      # one parameter's coefficients
curve(f, :σ)     # σ(t) over one period
rand(f, 10_000)  # samples from the fitted model
bic(f)           # to choose a degree, or a number of states
```
"""
struct SeasonalFit{M,H,L} <: StatisticalModel
    model::M
    history::H
    nobs::Int
    loglikelihood::L
end

"""
    coef(f::SeasonalFit) -> ComponentArray
    coef(f::SeasonalFit, name::Symbol)

The fitted coefficients: the whole container, or one parameter's block. `coef(f)` is
`parameters(f.model)`, so `coef(f).ρ` works too.
"""
StatsAPI.coef(f::SeasonalFit) = parameters(f.model)
StatsAPI.coef(f::SeasonalFit, name::Symbol) = getproperty(parameters(f.model), name)

parameters(f::SeasonalFit) = parameters(f.model)

TimeVaryingParameters.curve(f::SeasonalFit, name::Symbol, times...; kwargs...) =
    curve(f.model, name, times...; kwargs...)

StatsAPI.params(f::SeasonalFit, t) = params(f.model, t)
(f::SeasonalFit)(t) = f.model(t)

# A length, or a vector followed by the other arguments of the model's own `rand`. A single
# `args...` method would be ambiguous with `Random`'s `rand` methods.
rand(rng::AbstractRNG, f::SeasonalFit, N::Integer; kwargs...) = rand(rng, f.model, N; kwargs...)
rand(rng::AbstractRNG, f::SeasonalFit, x::AbstractVector, args...; kwargs...) =
    rand(rng, f.model, x, args...; kwargs...)
rand(f::SeasonalFit, N::Integer; kwargs...) = rand(GLOBAL_RNG, f, N; kwargs...)
rand(f::SeasonalFit, x::AbstractVector, args...; kwargs...) = rand(GLOBAL_RNG, f, x, args...; kwargs...)

"""
    loglikelihood(f::SeasonalFit) -> Real

The loglikelihood the fit attained.
"""
StatsAPI.loglikelihood(f::SeasonalFit) = f.loglikelihood

"""
    nobs(f::SeasonalFit) -> Int

The number of terms in the fitted loglikelihood: the number of transitions for an AR(1), of
observations for a mixture.
"""
StatsAPI.nobs(f::SeasonalFit) = f.nobs

"""
    dof(f::SeasonalFit) -> Int

The number of fitted parameters: the coefficients, plus the `K - 1` free initial probabilities of
a hidden Markov model. A periodic AR(1) has `3(2d+1)`, `K` times that when one is fitted per
hidden state. Parameters of a process the fit conditions on rather than estimates, such as a
given state sequence, are not counted.
"""
StatsAPI.dof(f::SeasonalFit) = nparams(f.model) + _ninitial(f.model)

# The free initial probabilities a fit estimates besides the coefficients; the HMMs have `K - 1`.
_ninitial(m) = 0

converged(f::SeasonalFit) = converged(f.history)

@inline function Base.getproperty(f::SeasonalFit, name::Symbol)
    if name === :retcode || name === :objective || name === :stats
        return getproperty(getfield(f, :history), name)
    end
    return getfield(f, name)
end

Base.propertynames(f::SeasonalFit) = (fieldnames(SeasonalFit)...,
    filter(!=(:solution), _solver_properties(getfield(f, :history)))...)

# v3 returned bare tuples here, so `θμ, θρ, θσ = fit_mle_AR1(...)` was the idiom, and v4's own
# `fit_mle` returns a 2-tuple. Destructuring calls `iterate`, which would otherwise be an
# unexplained `MethodError`.
@noinline function Base.iterate(::SeasonalFit, state...)
    throw(ArgumentError("""
    a `SeasonalFit` is not a tuple.

        m, history = fit_mle(m0, y; n2t, solver)   # the two-value return
        f          = fit(SeasonalFit, m0, y; n2t)
        f.model, f.history                 # the same two, off the fit
        coef(f, :μ), coef(f, :ρ), coef(f, :σ)
    """))
end

function Base.show(io::IO, ::MIME"text/plain", f::SeasonalFit)
    print(io, "SeasonalFit of ")
    model = sprint(show, MIME"text/plain"(), f.model; context=io)
    print(io, model, endswith(model, '\n') ? "" : "\n")
    show(io, MIME"text/plain"(), f.history)
    print(io, "\n  loglikelihood ", round(loglikelihood(f); digits=2),
        "   dof ", dof(f), "   nobs ", nobs(f), "   bic ", round(bic(f); digits=2))
    return
end
