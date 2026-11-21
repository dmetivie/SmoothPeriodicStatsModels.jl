# # Parameter specifications
#
# A `ParamSpec` says, for each named parameter of a model, which basis expands it and which
# link maps the expansion onto its domain:
#
#     θⱼ(t) = linkⱼ⁻¹( Bⱼ(t) · βⱼ )
#
# The coefficients of every parameter live end to end in one flat vector, which is what a
# solver wants. `coefs` hands back the block for one name as a `view`, so the layout replaces
# index arithmetic like `u[1:nc], u[nc+1:2nc], u[2nc+1:3nc]` without copying.

# A link is either a `LinkFunctions.Link` or any callable playing the role of `g⁻¹`.
apply_link(l::Link, η) = linkinv(l, η)
apply_link(l, η) = l(η)

# `g` itself, needed only to turn a scalar starting *value* into a starting *coefficient*.
invert_link(l::Link, θ) = linkfun(l, θ)
invert_link(l, θ) = throw(ArgumentError(
    "a scalar `init` needs a `LinkFunctions.Link` to invert; $(l) is a plain callable, " *
    "so pass that parameter's starting coefficients as a vector instead"))

"""
    ParamSpec(; name = basis => link, ...)

The parametrization of a time-varying model: one basis and one link per named parameter.

```julia
spec = ParamSpec(
    μ = TrigBasis(T, d) => IdentityLink(),
    ρ = TrigBasis(T, d) => BoundedLink(-1, 1),
    σ = TrigBasis(T, d) => LogLink(),
)
```

A link is a `LinkFunctions.Link` or any callable mapping the linear predictor onto the
parameter domain. Coefficients are stored flat, in the order the names are given;
[`coefs`](@ref) retrieves one block and [`ncoef`](@ref) the total length.
"""
struct ParamSpec{names, B <: Tuple, L <: Tuple, N}
    bases::B
    links::L
    offsets::NTuple{N, Int}   # offsets[j]:offsets[j+1]-1 is block j; length == nparams + 1
end

function ParamSpec(; kwargs...)
    nt = NamedTuple(kwargs)
    @argcheck !isempty(nt) "a `ParamSpec` needs at least one parameter"
    entries = Tuple(nt)
    for (nm, e) in zip(keys(nt), entries)
        e isa Pair || throw(ArgumentError(
            "parameter `$(nm)` must be given as `basis => link`, got $(typeof(e))"))
        first(e) isa AbstractBasis || throw(ArgumentError(
            "the left of `$(nm) = basis => link` must be an `AbstractBasis`, got $(typeof(first(e)))"))
    end
    bases = map(first, entries)
    links = map(last, entries)
    offsets = (1, (cumsum(map(nbasis, bases)) .+ 1)...)
    return ParamSpec{keys(nt), typeof(bases), typeof(links), length(offsets)}(bases, links, offsets)
end

"""
    paramnames(spec) -> Tuple{Vararg{Symbol}}

The parameter names, in coefficient order.
"""
paramnames(::ParamSpec{names}) where {names} = names

"""
    nparams(spec) -> Int

How many parameters the spec describes -- `length(paramnames(spec))`, *not* the number of
coefficients, which is [`ncoef`](@ref).

Deliberately not exported: downstream packages use `nparams(model)` for the number of free
coefficients, following `HMMBase`, and two meanings on one generic is a trap. Reach for it as
`TimeVaryingParameters.nparams` or use `length(paramnames(spec))`.
"""
nparams(::ParamSpec{names}) where {names} = length(names)

"""
    ncoef(spec) -> Int

Total length of the flat coefficient vector.
"""
ncoef(spec::ParamSpec) = spec.offsets[end] - 1

# Unroll over parameters at compile time: `names` is a type parameter, so `Val` keeps the
# resulting tuple concretely typed.
@inline map_params(f, ::ParamSpec{names}) where {names} = ntuple(f, Val(length(names)))

bases(spec::ParamSpec) = spec.bases
links(spec::ParamSpec) = spec.links

# `names` is a compile-time tuple of `Symbol`, so this folds away.
@inline function blockindex(::ParamSpec{names}, name::Symbol) where {names}
    j = findfirst(==(name), names)
    j === nothing && throw(ArgumentError("no parameter `$(name)`; the spec has $(names)"))
    return j
end

"""
    blockrange(spec, name::Symbol) -> UnitRange{Int}
    blockrange(spec, j::Integer) -> UnitRange{Int}

The indices one parameter's coefficients occupy in the flat vector.
"""
@inline blockrange(spec::ParamSpec, j::Integer) = spec.offsets[j]:(spec.offsets[j + 1] - 1)
@inline blockrange(spec::ParamSpec, name::Symbol) = blockrange(spec, blockindex(spec, name))

"""
    coefs(spec, θ, name::Symbol) -> SubArray
    coefs(spec, θ, j::Integer) -> SubArray

The coefficient block of one parameter, as a view into the flat vector `θ`.
"""
@inline coefs(spec::ParamSpec, θ::AbstractVector, j::Integer) =
    view(θ, blockrange(spec, j))
@inline coefs(spec::ParamSpec, θ::AbstractVector, name::Symbol) =
    coefs(spec, θ, blockindex(spec, name))

"""
    parameters_at(spec, θ, t) -> Tuple

Every parameter of the spec evaluated at time `t`, in coefficient order: the tuple
`(link₁⁻¹(B₁(t)·β₁), …)`.

This allocates one small vector per basis. The periodic path in [`PeriodicDesign`](@ref)
evaluates every `t` at once instead, with a matrix-vector product per parameter.
"""
@inline function parameters_at(spec::ParamSpec, θ::AbstractVector, t)
    return map_params(spec) do j
        b = spec.bases[j]
        β = coefs(spec, θ, j)
        η = zero(promote_type(eltype(θ), basis_eltype(b, typeof(t))))
        x = eval_basis(b, t)
        @inbounds for k in eachindex(β, x)
            η += x[k] * β[k]
        end
        apply_link(spec.links[j], η)
    end
end

"""
    initial_coefficients(spec; init...) -> Vector{Float64}

Flat starting vector. Each parameter is given either its starting coefficients as a vector,
or a single starting *value*: the coefficients that make the parameter that constant.

A scalar is accepted when the parameter's basis can express a constant: it starts with a
`ConstantBasis` (the value goes on the intercept), or with a [`BSplineBasis`](@ref) or
[`CyclicBSplineBasis`](@ref) (every spline coefficient takes the value, the columns summing to
one). Otherwise it is rejected.
"""
function initial_coefficients(spec::ParamSpec; init...)
    nt = NamedTuple(init)
    names = paramnames(spec)
    @argcheck Set(keys(nt)) == Set(names) "`init` must name exactly $(names), got $(keys(nt))"
    θ = zeros(Float64, ncoef(spec))
    for (j, nm) in enumerate(names)
        v = nt[nm]
        β = coefs(spec, θ, j)
        if v isa Number
            set_constant!(β, spec.bases[j], invert_link(spec.links[j], v)) || throw(ArgumentError(
                "parameter `$(nm)` got the scalar start $(v), but its basis $(spec.bases[j]) " *
                "starts with neither a `ConstantBasis` nor a spline basis, so it has no " *
                "coefficients that make it a constant; " *
                "pass a coefficient vector of length $(nbasis(spec.bases[j])) instead"))
        else
            @argcheck length(v) == nbasis(spec.bases[j])
            β .= v
        end
    end
    return θ
end

# Write into the zeroed `β` the coefficients that make the expansion the constant `η`, and say
# whether this basis can. A combined basis does it through its first term, the rest left at zero.
set_constant!(β, ::AbstractBasis, η) = false
set_constant!(β, ::ConstantBasis, η) = (β[1] = η; true)
set_constant!(β, ::Union{BSplineBasis, CyclicBSplineBasis}, η) = (fill!(β, η); true)
set_constant!(β, b::CombinedBasis, η) =
    set_constant!(view(β, 1:nbasis(first(b.bases))), first(b.bases), η)

function Base.show(io::IO, ::MIME"text/plain", spec::ParamSpec)
    println(io, "ParamSpec with ", ncoef(spec), " coefficients:")
    for (j, nm) in enumerate(paramnames(spec))
        println(io, "  ", rpad(nm, 8), " ", spec.bases[j], "  =>  ", spec.links[j],
            "   [", spec.offsets[j], ":", spec.offsets[j + 1] - 1, "]")
    end
    return
end
