# # A seasonal mixture, expressed for ExpectationMaximization.jl
#
# The EM loop for a seasonal mixture is the ordinary one: the only thing that is periodic is the
# *parametrization* of the weights and of the components. So rather than carry a second EM driver,
# this package states the parametrization as three types and hands the loop to
# `ExpectationMaximization.fit_mle!`.
#
# That works because its driver never indexes `α` itself — `α` only travels into `E_step!` and
# `M_step!`, both of which dispatch, and its built-in `M_step!` leaves `α` untyped. A method typed
# on the wrappers below is therefore strictly more specific and wins, with no change to that
# package. The algorithm really is `ClassicEM`; only the parameters are seasonal.
#
# `loglikelihoods!` lives here because evaluating a likelihood needs no solver. The matching
# `M_step!` lives in the fitting extension, next to the `update_*` functions it calls.

"""
    SeasonalSample(y, n2t, n_in_t)

The sample of a seasonal fit: the observations together with the period index of each one.

`n2t[n] ∈ 1:T` is the period index of observation `n`, and `n_in_t[t]` the indices of the
observations sharing period index `t` — the inverse map, built once because every M-step
aggregates over it.

A view, not a copy: `y` is stored as given. Subtyping `AbstractVector` is what makes
`ExpectationMaximization` accept it as a sample and count it with `size_sample`.
"""
struct SeasonalSample{V<:AbstractVector} <: AbstractVector{Float64}
    y::V
    n2t::Vector{Int}
    n_in_t::Vector{Vector{Int}}
end

SeasonalSample(y::AbstractVector, n2t::AbstractVector, T::Integer) =
    SeasonalSample(y, collect(Int, n2t), [findall(==(t), n2t) for t = 1:T])

Base.size(s::SeasonalSample) = size(s.y)
Base.getindex(s::SeasonalSample, n) = s.y[n]

"""
    SeasonalWeights(α, θ, basis, cfg)

The seasonal mixing weights of a mixture, in the shape `ExpectationMaximization` expects a weight
vector to have.

`α` is `K × T`: `α[k, t]` is the weight of component `k` at period index `t`. `θ` holds the
`(K-1) × (2d+1)` multinomial-logit coefficients that generate it, with component `K` as the
reference, and `basis` is the `T × (2d+1)` design matrix from [`trig_basis`](@ref).

`cfg` carries what the M-step needs and cannot otherwise reach: the solver settings and the
components' own bases. `ExpectationMaximization.M_step!` receives only `(α, dists, y, γ, method)`
and `ClassicEM` is a singleton, so this struct is the one place left to put them.

Indexing it returns the time-averaged weight of a component, so that it still reads as the
length-`K` vector of weights its supertype promises.
"""
mutable struct SeasonalWeights{C} <: AbstractVector{Float64}
    α::Matrix{Float64}
    θ::Matrix{Float64}
    basis::Matrix{Float64}
    cfg::C
end

Base.size(a::SeasonalWeights) = (size(a.α, 1),)
Base.getindex(a::SeasonalWeights, k::Integer) = sum(view(a.α, k, :)) / size(a.α, 2)

"""
    SeasonalFamily(dists, θ)

One component of a seasonal mixture: the `T` distributions it takes over a period, and the
trigonometric coefficients `θ` that generate them.

It subtypes `Distribution` only because `ExpectationMaximization.fit_mle!` types its components as
`AbstractVector{F} where {F<:Distribution}`. Nothing ever calls `logpdf`, `rand` or `params` on
the family itself — only on `dists[t]` — so it implements none of that interface.
"""
struct SeasonalFamily{D<:Distribution} <: Distribution{Univariate,Continuous}
    dists::Vector{D}
    θ::Vector{Float64}
end

"""
    loglikelihoods!(LL, dists, α::SeasonalWeights, y::SeasonalSample)

E-step hook: `LL[n, k] = log αₖ(tₙ) + log fₖ,ₜₙ(yₙ)`, the weight and the component both taken at
the period index of observation `n`.
"""
function loglikelihoods!(LL::AbstractMatrix, dists, α::SeasonalWeights, y::SeasonalSample)
    for k in eachindex(dists)
        dₖ = dists[k].dists
        @inbounds for (n, t) in enumerate(y.n2t)
            LL[n, k] = log(α.α[k, t]) + logpdf(dₖ[t], y.y[n])
        end
    end
    return LL
end
