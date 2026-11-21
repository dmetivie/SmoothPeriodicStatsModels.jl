# # Time-varying families and their objectives
#
# A `TimeVaryingFamily` is a `ParamSpec` plus a distribution constructor, so that evaluating
# it at a time yields an ordinary `Distributions.jl` object. Everything that does not depend
# on time -- `logpdf`, `rand`, `quantile`, `cdf` -- then comes from there.
#
# Two objectives are provided. `PointwiseNLL` is generic and handles any `ts`.
# `PeriodicNLL` is for the case this package exists to serve: `ts` takes only `nT` distinct
# values, so the design is `nT × p` rather than `N × p`, the linear predictors are `nparams`
# matrix-vector products, and only `nT` distributions are built instead of `N`.


"""
    TimeVaryingFamily(dist, spec::ParamSpec)

A distribution whose parameters vary with time. `dist` is any callable taking the
parameters in the order `spec` names them; `fam(t, θ)` returns what it builds.

```julia
fam = TimeVaryingFamily(Normal, ParamSpec(
    μ = TrigBasis(T, d) => IdentityLink(),
    σ = TrigBasis(T, d) => LogLink(),
))
d = fam(t, θ)        # a Normal
logpdf(d, y); rand(d)
```

To skip `Distributions`' argument validation, or to reorder arguments, pass a closure:
`TimeVaryingFamily((μ, σ) -> Normal(μ, σ; check_args = false), spec)`.
"""
struct TimeVaryingFamily{D, S <: ParamSpec}
    dist::D
    spec::S
end

# Storing a bare `Normal` in the `dist` field would store a `UnionAll`, which inference
# cannot see through: the objective then returns `Any` and runs an order of magnitude
# slower. Holding the type as a *type parameter* of a singleton keeps the call concrete,
# so `TimeVaryingFamily(Normal, spec)` is as fast as hand-writing the closure.
struct DistConstructor{D} end

@inline (::DistConstructor{D})(args...) where {D} = D(args...)

Base.show(io::IO, ::DistConstructor{D}) where {D} = print(io, D)

TimeVaryingFamily(D::Type, spec::ParamSpec) = TimeVaryingFamily(DistConstructor{D}(), spec)

spec(f::TimeVaryingFamily) = f.spec
ncoef(f::TimeVaryingFamily) = ncoef(f.spec)
paramnames(f::TimeVaryingFamily) = paramnames(f.spec)

@inline (f::TimeVaryingFamily)(t, θ::AbstractVector) = f.dist(parameters_at(f.spec, θ, t)...)

function Base.show(io::IO, ::MIME"text/plain", f::TimeVaryingFamily)
    println(io, "TimeVaryingFamily(", f.dist, ") with")
    show(io, MIME"text/plain"(), f.spec)
    return
end

# ## The periodic design

"""
    PeriodicDesign(spec, times)

The design matrices of `spec` evaluated once at `times`, the distinct times of a periodic
problem (for a seasonal model of period `T`, `times = 1:T`).

Row `t` of design `j` is parameter `j`'s basis at `times[t]`, so `designs[j] * βⱼ` is the
whole seasonal cycle of its linear predictor in one matrix-vector product.
"""
struct PeriodicDesign{S <: ParamSpec, M <: Tuple}
    spec::S
    designs::M
    nT::Int
end

function PeriodicDesign(spec::ParamSpec, times::AbstractVector)
    @argcheck !isempty(times)
    designs = map_params(spec) do j
        design_matrix(spec.bases[j], times)
    end
    return PeriodicDesign(spec, designs, length(times))
end

nparams(d::PeriodicDesign) = nparams(d.spec)
ncoef(d::PeriodicDesign) = ncoef(d.spec)

"""
    linear_predictors(design, θ) -> Tuple{Vararg{Vector}}

One `nT`-long vector of linear predictors per parameter: `designs[j] * βⱼ`.
"""
@inline function linear_predictors(d::PeriodicDesign, θ::AbstractVector)
    return map_params(d.spec) do j
        d.designs[j] * coefs(d.spec, θ, j)
    end
end

"""
    distributions(fam, design, θ) -> Vector

The `nT` distributions of `fam` over one period. Builds the linear predictors with one
matrix-vector product per parameter, then applies the links.
"""
function distributions(f::TimeVaryingFamily, d::PeriodicDesign, θ::AbstractVector)
    η = linear_predictors(d, θ)
    return [dist_at(f, η, t) for t in 1:d.nT]
end

@inline function dist_at(f::TimeVaryingFamily, η::Tuple, t::Integer)
    return f.dist(map_params(f.spec) do j
        apply_link(f.spec.links[j], @inbounds η[j][t])
    end...)
end

# ## Grouping observations by period index

"""
    group_by_time(n2t, nT) -> Vector{Vector{Int}}

`groups[t]` lists the observation indices whose period index is `t`. Computed once and
reused across solver iterations.
"""
function group_by_time(n2t::AbstractVector{<:Integer}, nT::Integer)
    @argcheck all(t -> 1 <= t <= nT, n2t) "every entry of `n2t` must lie in 1:$(nT)"
    groups = [Int[] for _ in 1:nT]
    for (n, t) in enumerate(n2t)
        push!(groups[t], n)
    end
    return groups
end

# ## Objectives
#
# Both are callable as `(θ, p)`, the signature `Optimization.jl` expects; `p` is unused
# because the data is captured in the object.

"""
    PeriodicNLL(fam, design, y, groups[, weights])

Weighted negative loglikelihood of a periodic model:
`-Σₜ Σ_{n ∈ groups[t]} wₙ · logpdf(dₜ, yₙ)`.

Each solver iteration costs `nparams` matrix-vector products and `nT` distribution
constructions, not `N` of them.
"""
struct PeriodicNLL{F <: TimeVaryingFamily, D <: PeriodicDesign, Y, G, W}
    fam::F
    design::D
    y::Y
    groups::G
    weights::W
end

PeriodicNLL(fam, design, y, groups) = PeriodicNLL(fam, design, y, groups, nothing)

@inline obsweight(::Nothing, n) = true
@inline obsweight(w, n) = @inbounds w[n]

function (o::PeriodicNLL)(θ::AbstractVector, _p = nothing)
    η = linear_predictors(o.design, θ)
    nll = zero(eltype(first(η)))
    @inbounds for t in 1:o.design.nT
        g = o.groups[t]
        isempty(g) && continue
        d = dist_at(o.fam, η, t)
        acc = zero(nll)
        for n in g
            acc += obsweight(o.weights, n) * logpdf(d, o.y[n])
        end
        nll -= acc
    end
    return nll
end

"""
    PointwiseNLL(fam, y, ts[, weights])

Weighted negative loglikelihood evaluated one observation at a time,
`-Σₙ wₙ · logpdf(fam(tsₙ, θ), yₙ)`. Works for any `ts`; [`PeriodicNLL`](@ref) is far
cheaper when `ts` repeats.
"""
struct PointwiseNLL{F <: TimeVaryingFamily, Y, X, W}
    fam::F
    y::Y
    ts::X
    weights::W
end

PointwiseNLL(fam, y, ts) = PointwiseNLL(fam, y, ts, nothing)

function (o::PointwiseNLL)(θ::AbstractVector, _p = nothing)
    nll = zero(promote_type(eltype(θ), Float64))
    @inbounds for n in eachindex(o.y)
        nll -= obsweight(o.weights, n) * logpdf(o.fam(o.ts[n], θ), o.y[n])
    end
    return nll
end

"""
    PointwiseLoss(spec, loss, y, ts[, weights])

The M-estimation escape hatch: minimises `Σₙ wₙ · loss(θ(tsₙ), yₙ)` for an arbitrary
`loss(params::Tuple, y) -> Number`, with no distribution involved.
"""
struct PointwiseLoss{S <: ParamSpec, L, Y, X, W}
    spec::S
    loss::L
    y::Y
    ts::X
    weights::W
end

PointwiseLoss(spec, loss, y, ts) = PointwiseLoss(spec, loss, y, ts, nothing)

function (o::PointwiseLoss)(θ::AbstractVector, _p = nothing)
    total = zero(promote_type(eltype(θ), Float64))
    @inbounds for n in eachindex(o.y)
        total += obsweight(o.weights, n) * o.loss(parameters_at(o.spec, θ, o.ts[n]), o.y[n])
    end
    return total
end
