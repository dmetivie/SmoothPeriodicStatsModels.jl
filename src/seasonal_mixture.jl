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
observations sharing period index `t`. `y` is stored as given, not copied.
"""
struct SeasonalSample{T,V<:AbstractVector{T}} <: AbstractVector{T}
    y::V
    n2t::Vector{Int}
    n_in_t::Vector{Vector{Int}}
end

# `n2t` and `n_in_t` are deliberately not parametric: they are period indices, built here rather
# than taken from the caller, so there is nothing for a type parameter to vary over.
SeasonalSample(y::AbstractVector, n2t::AbstractVector, nT::Integer) =
    SeasonalSample{eltype(y),typeof(y)}(y, collect(Int, n2t), [findall(==(t), n2t) for t = 1:nT])

Base.size(s::SeasonalSample) = size(s.y)
Base.getindex(s::SeasonalSample, n) = s.y[n]

"""
    SeasonalWeights(α, θ, basis, cfg)

The seasonal mixing weights of a mixture, in the shape `ExpectationMaximization` expects a weight
vector to have.

`α` is `K × T`: `α[k, t]` is the weight of component `k` at period index `t`. `θ` holds the
`(K-1) × (2d+1)` multinomial-logit coefficients that generate it, with component `K` as the
reference, and `basis` is the `T × (2d+1)` design matrix from [`trig_basis`](@ref).

`cfg` holds what the M-step needs: the solver settings and the components' own bases.

Indexing it returns the time-averaged weight of a component.
"""
mutable struct SeasonalWeights{T,C} <: AbstractVector{T}
    α::Matrix{T}
    θ::Matrix{T}
    basis::Matrix{T}
    cfg::C
end

# The three matrices come from different places — the initial mixtures, the caller's coefficients
# and `trig_basis` — so they are promoted to a common element type rather than assumed to share one.
function SeasonalWeights(α::AbstractMatrix, θ::AbstractMatrix, basis::AbstractMatrix, cfg)
    T = promote_type(eltype(α), eltype(θ), eltype(basis))
    return SeasonalWeights{T,typeof(cfg)}(Matrix{T}(α), Matrix{T}(θ), Matrix{T}(basis), cfg)
end

Base.size(a::SeasonalWeights) = (size(a.α, 1),)
Base.getindex(a::SeasonalWeights, k::Integer) = sum(view(a.α, k, :)) / size(a.α, 2)

"""
    SeasonalFamily(dists, θ)

One component of a seasonal mixture: the `T` distributions it takes over a period, and the
trigonometric coefficients `θ` that generate them.

It implements no part of the `Distribution` interface itself: `logpdf`, `rand` and `params` are
called on `dists[t]`.
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
