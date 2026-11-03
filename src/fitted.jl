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
    fit(SeasonalFit, model, data...; kwargs...) -> SeasonalFit

A seasonal fit together with what is needed to judge it.

```julia
m, history = fit_mle(m0, y; n2t)        # the usual call: model in, model out
f = fit(SeasonalFit, m0, y; n2t)        # the same estimator, diagnostics kept

f.model          # the fitted `SmoothAR1`, as `fit_mle` would have returned
f.history        # as `fit_mle` would have returned
coef(f, :ρ)      # that parameter's coefficients
curve(f, :σ)     # σ(t) over one period
rand(f, 10_000)  # it behaves like the model it fitted
bic(f)           # choose a degree, or a number of states
```

`loglikelihood`, `dof` and `nobs` are the whole `StatisticalModel` interface that `aic`, `aicc`
and `bic` are built from, so stating those three is enough.
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

TimeVaryingParameters.curve(f::SeasonalFit, name::Symbol, times...) =
    curve(f.model, name, times...)

StatsAPI.params(f::SeasonalFit, t) = params(f.model, t)
(f::SeasonalFit)(t) = f.model(t)

rand(rng::AbstractRNG, f::SeasonalFit, args...; kwargs...) = rand(rng, f.model, args...; kwargs...)
rand(f::SeasonalFit, args...; kwargs...) = rand(GLOBAL_RNG, f.model, args...; kwargs...)

StatsAPI.loglikelihood(f::SeasonalFit) = f.loglikelihood
StatsAPI.nobs(f::SeasonalFit) = f.nobs

"""
    dof(f::SeasonalFit) -> Int

The number of fitted coefficients — `3(2d+1)` for a periodic AR(1), `K` times that when one is
fitted per hidden state. Parameters of a process the fit conditions on rather than estimates,
such as a given state sequence, are not counted.
"""
StatsAPI.dof(f::SeasonalFit) = nparams(f.model)

converged(f::SeasonalFit) = converged(f.history)

@inline function Base.getproperty(f::SeasonalFit, name::Symbol)
    if name === :retcode || name === :objective || name === :stats
        return getproperty(getfield(f, :history), name)
    end
    return getfield(f, name)
end

Base.propertynames(::SeasonalFit) =
    (fieldnames(SeasonalFit)..., :retcode, :objective, :stats)

# v3 returned bare tuples here, so `θμ, θρ, θσ = fit_mle_AR1(...)` was the idiom, and v4's own
# `fit_mle` returns a 2-tuple. Destructuring calls `iterate`, which would otherwise be an
# unexplained `MethodError`.
@noinline function Base.iterate(::SeasonalFit, state...)
    throw(ArgumentError("""
    a `SeasonalFit` is not a tuple.

        m, history = fit_mle(m0, y; n2t)   # the two-value return
        f          = fit(SeasonalFit, m0, y; n2t)
        f.model, f.history                 # the same two, off the fit
        coef(f, :μ), coef(f, :ρ), coef(f, :σ)
    """))
end

function Base.show(io::IO, ::MIME"text/plain", f::SeasonalFit)
    print(io, "SeasonalFit of ")
    show(io, MIME"text/plain"(), f.model)
    show(io, MIME"text/plain"(), f.history)
    print(io, "\n  loglikelihood ", round(loglikelihood(f); digits=2),
        "   dof ", dof(f), "   nobs ", nobs(f), "   bic ", round(bic(f); digits=2))
    return
end
