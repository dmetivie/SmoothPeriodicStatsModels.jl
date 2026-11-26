# # An additional loss term on a fit
#
# The convention is NeuralPDE.jl's `additional_loss(phi, θ, p)`: a fit takes
# `additional_loss = f` and adds `f(phi, θ, p)` to the negative loglikelihood of every M-step
# subproblem. `phi(t, θ)` evaluates the subproblem's seasonal parameters on their natural scale,
# so each kind of subproblem has its own `phi` type below, which also names the coefficients that
# `f` is given. The loss reaches the optimizer inside a `_PenalizedObjective`; `nothing` stays
# `nothing` all the way down, so that an unpenalized fit runs the code it always did.

"""
    additional_loss

The keyword `additional_loss = f` of the fits adds `f(phi, θ, p)`, a finite real, to the negative
loglikelihood; it mirrors NeuralPDE.jl's `additional_loss`.

`f` is called once per subproblem of the fit, with `θ` its coefficients, `p = (; part, index, T)`
the subproblem and the period, and `phi(t, θ)` its seasonal parameters on their natural scale at
the period index `t`, an integer or a vector of integers (then a vector, a `NamedTuple` of
vectors or a `K × length(t)` matrix). `part` and `index` are in this table:

| fit                                   | `part`          | `index`   | `θ`                                      | `phi(t, θ)`                            |
|:--------------------------------------|:----------------|:----------|:-----------------------------------------|:---------------------------------------|
| `SmoothAR1`                           | `:AR`           | `k`       | state `k`'s blocks `μ`, `ρ`, `σ`         | `(μ, ρ, σ)`                            |
| `SmoothDistribution`                  | `:distribution` | `k`       | state `k`'s blocks, named by parameter   | the family's parameters, a `NamedTuple` |
| `SmoothSeasonalMixture`, the weights  | `:mixture`      | `nothing` | blocks `logit1`, …, `logit(K-1)`         | the `K` weights                        |
| `SmoothSeasonalMixture`, a component  | `:component`    | `k`       | component `k`'s blocks                   | the family's parameters                |
| Bernoulli HMMs, an emission           | `:B`            | `(k,s,h)` | `θᴮ[k,s,h,:]`: state, site, memory       | the success probability                |
| `SmoothPeriodicHMM`, an emission      | `:B`            | `(k,b)`   | blocks of `emission_coefs(m, k, b)`      | the family's parameters                |
| every HMM, the transitions out of `k` | `:A`            | `k`       | blocks `logit1`, …, `logit(K-1)`         | the `K` transition probabilities       |
| `SmoothARPeriodicHMMSpatial`, range   | `:R`            | `k`       | `θᴿ[k,:]`                                | the range                              |

A model of one state has `k = 1`. The objective is the summed negative loglikelihood plus `f`,
whatever average a fit minimizes inside, so `f` is on the loglikelihood scale. `f` must be finite
at the starting coefficients of every subproblem, or the fit throws an `ArgumentError`; elsewhere a
non-finite value is left to the line search. `history.logtots`
and `loglikelihood(history)` then hold the penalized loglikelihood, and `aic`, `aicc` and `bic` of
a [`SeasonalFit`](@ref) throw. The range of the spatial HMM is fitted on its composite
loglikelihood. [`RidgePenalty`](@ref), [`DifferencePenalty`](@ref) and [`CurvePenalty`](@ref) are
ready-made terms.
"""
function additional_loss end

"""
    RidgePenalty(λ)

The term `λ * sum(abs2, θ)` on the coefficients of every subproblem, `λ ≥ 0`, for the
[`additional_loss`](@ref SmoothPeriodicStatsModels.additional_loss) keyword of the fits.
"""
struct RidgePenalty{T<:Real}
    λ::T
    function RidgePenalty{T}(λ) where {T<:Real}
        (isfinite(λ) && λ >= 0) || throw(ArgumentError("the ridge weight `λ` is finite and nonnegative, not $λ"))
        return new{T}(λ)
    end
end
RidgePenalty(λ::T) where {T<:Real} = RidgePenalty{T}(λ)

(p::RidgePenalty)(phi, θ, context) = p.λ * sum(abs2, θ)

Base.show(io::IO, p::RidgePenalty) = print(io, "RidgePenalty(", p.λ, ")")

"""
    DifferencePenalty(λ; order = 2, cyclic = true)

The term `λ * sum(abs2, D θ)` on the `order`-th differences of the coefficients of each block, the
P-spline penalty of a smooth curve, for the
[`additional_loss`](@ref SmoothPeriodicStatsModels.additional_loss) keyword of the fits. With
`cyclic` the differences wrap around the end of the block.

It is meant for B-spline coefficients, such as those of a `CyclicBSplineBasis`; a ridge on the
higher harmonics, or a [`CurvePenalty`](@ref), is the analogue for a `TrigBasis`.
A block no longer than `order` has no differences and no penalty.
"""
struct DifferencePenalty{T<:Real}
    λ::T
    order::Int
    cyclic::Bool
    function DifferencePenalty{T}(λ, order, cyclic) where {T<:Real}
        (isfinite(λ) && λ >= 0) || throw(ArgumentError("the weight `λ` is finite and nonnegative, not $λ"))
        order >= 1 || throw(ArgumentError("the order of the differences is at least 1, not $order"))
        return new{T}(λ, order, cyclic)
    end
end
DifferencePenalty(λ::T; order::Integer=2, cyclic::Bool=true) where {T<:Real} =
    DifferencePenalty{T}(λ, Int(order), cyclic)

(p::DifferencePenalty)(phi, θ::ComponentVector, context) =
    p.λ * sum(name -> _roughness(getproperty(θ, name), p.order, p.cyclic), propertynames(θ))
(p::DifferencePenalty)(phi, θ::AbstractVector, context) = p.λ * _roughness(θ, p.order, p.cyclic)

Base.show(io::IO, p::DifferencePenalty) =
    print(io, "DifferencePenalty(", p.λ, "; order = ", p.order, ", cyclic = ", p.cyclic, ")")

"""
    CurvePenalty(λ; order = 2)

The term `λ * sum(abs2, D phi(1:T, θ))` on the cyclic differences of the `order`-th order of each
seasonal parameter over the period, for the
[`additional_loss`](@ref SmoothPeriodicStatsModels.additional_loss) keyword of the fits.

It penalizes the roughness of the curves themselves, so it works with any basis.
"""
struct CurvePenalty{T<:Real}
    λ::T
    order::Int
    function CurvePenalty{T}(λ, order) where {T<:Real}
        (isfinite(λ) && λ >= 0) || throw(ArgumentError("the weight `λ` is finite and nonnegative, not $λ"))
        order >= 1 || throw(ArgumentError("the order of the differences is at least 1, not $order"))
        return new{T}(λ, order)
    end
end
CurvePenalty(λ::T; order::Integer=2) where {T<:Real} = CurvePenalty{T}(λ, Int(order))

(p::CurvePenalty)(phi, θ, context) = p.λ * _curve_roughness(phi(1:context.T, θ), p.order)

Base.show(io::IO, p::CurvePenalty) = print(io, "CurvePenalty(", p.λ, "; order = ", p.order, ")")

# The sum of squared differences of order `order` of a vector; cyclic differences wrap around.
function _roughness(x::AbstractVector, order::Integer, cyclic::Bool=true)
    d = x
    for _ = 1:order
        d = cyclic ? d .- circshift(d, -1) : diff(d)
    end
    return sum(abs2, d; init=zero(float(eltype(x))))
end

# A curve is a vector, a `NamedTuple` of them, or a matrix with one curve per row.
_curve_roughness(x::AbstractVector, order) = _roughness(x, order)
_curve_roughness(x::NamedTuple, order) = sum(v -> _curve_roughness(v, order), x)
_curve_roughness(x::AbstractMatrix, order) = sum(r -> _roughness(r, order), eachrow(x))

# ## `phi`: the seasonal parameters of one kind of subproblem

# A Bernoulli success probability, `logistic(basis[t, :] ⋅ θ)`.
struct _BernoulliPhi{M<:AbstractMatrix}
    basis::M
end
(φ::_BernoulliPhi)(t::Integer, θ) = logistic(_row_dot(φ.basis, t, θ))
(φ::_BernoulliPhi)(t::AbstractVector{<:Integer}, θ) = logistic.(φ.basis[t, :] * θ)
_coefficients(::_BernoulliPhi, u) = u

# The spatial range, `exp(basis[t, :] ⋅ θ)`.
struct _RangePhi{M<:AbstractMatrix}
    basis::M
end
(φ::_RangePhi)(t::Integer, θ) = exp(_row_dot(φ.basis, t, θ))
(φ::_RangePhi)(t::AbstractVector{<:Integer}, θ) = exp.(φ.basis[t, :] * θ)
_coefficients(::_RangePhi, u) = u

# The parameters of a `ParamSpec`, a `NamedTuple` with their links applied.
struct _FamilyPhi{S<:ParamSpec,D<:PeriodicDesign,A<:Tuple}
    spec::S
    design::D
    axes::A
end
_FamilyPhi(spec::ParamSpec, design::PeriodicDesign) =
    _FamilyPhi(spec, design, getaxes(seasonal_params(spec)))

function (φ::_FamilyPhi)(t, θ)
    spec = φ.spec
    names = paramnames(spec)
    values = ntuple(j -> _link_at(spec.links[j], φ.design.designs[j], t, coefs(spec, θ, j)), length(names))
    return NamedTuple{names}(values)
end
_link_at(link, X, t::Integer, β) = apply_link(link, _row_dot(X, t, β))
_link_at(link, X, t::AbstractVector{<:Integer}, β) = apply_link.(Ref(link), X[t, :] * β)
_coefficients(φ::_FamilyPhi, u) = ComponentVector(u, φ.axes)

# A softmax over `K` categories, the last the reference at zero: the mixing weights and one row of
# transition probabilities. `θ` has one block per free category.
struct _SoftmaxPhi{M<:AbstractMatrix}
    basis::M
    K::Int
end

function (φ::_SoftmaxPhi)(t::Integer, θ)
    η = [_row_dot(φ.basis, t, β) for β in _category_blocks(θ, φ.K)]
    return _softmax_with_reference(η)
end
function (φ::_SoftmaxPhi)(t::AbstractVector{<:Integer}, θ)
    cols = [_softmax_with_reference([_row_dot(φ.basis, ti, β) for β in _category_blocks(θ, φ.K)]) for ti in t]
    return reduce(hcat, cols)
end

# The blocks of `θ`, one per free category: its named blocks, or the rows of the flat vector the
# optimizer holds, which lists the categories fastest.
_category_blocks(θ::ComponentVector, K) = [getproperty(θ, name) for name in propertynames(θ)]
function _category_blocks(θ::AbstractVector, K)
    M = reshape(θ, K - 1, :)
    return [view(M, l, :) for l = 1:K-1]
end

function _softmax_with_reference(η::AbstractVector)
    shift = max(zero(eltype(η)), maximum(η))
    e = [exp(x - shift) for x in η]
    r = exp(-shift)
    total = r + sum(e)
    return vcat(e ./ total, r / total)
end

function _coefficients(φ::_SoftmaxPhi, u)
    M = reshape(u, φ.K - 1, :)
    names = ntuple(l -> Symbol(:logit, l), φ.K - 1)
    return ComponentVector(NamedTuple{names}(ntuple(l -> M[l, :], φ.K - 1)))
end

_row_dot(X::AbstractMatrix, t::Integer, β) = sum(X[t, j] * β[j] for j in eachindex(β))

# ## The loss of one subproblem, checked

_resolve_loss(::Nothing) = nothing
function _resolve_loss(f)
    _takes_three(f) || throw(ArgumentError(
        "`additional_loss` is a function `additional_loss(phi, θ, p)`, or `nothing`; got a $(typeof(f))"))
    return f
end

# Whether some method of `f` takes three positional arguments. `hasmethod` would refuse a method
# that restricts the type of `θ`.
_takes_three(f) = any(m -> m.isva ? m.nargs - 2 <= 3 : m.nargs - 1 == 3, methods(f))

# `f(phi, θ, p)` for the coefficients `u` of one subproblem, a real number or an error that says
# which call returned what. Inside a solve a non-finite value is the solver's to handle, as the
# loglikelihood's own is when a line search overshoots; `_check_loss` refuses it at the start.
function _loss_value(f, phi, u, part::Symbol, index, T)
    v = f(phi, _coefficients(phi, u), (; part, index, T))
    v isa Real || _loss_error(v, part, index)
    return v
end

@noinline function _loss_error(v, part, index)
    shown = v isa Real && hasfield(typeof(v), :value) ? getfield(v, :value) : v   # a dual number's value
    throw(ArgumentError("`additional_loss` returned $(repr(shown)) for part = $(repr(part)), index = $(repr(index)); it returns a finite real number"))
end

# The objective of one subproblem with its loss: `f(u, p.data)` plus the loss over `p.scale`, the
# total weight of a fit that minimizes an average and 1 for a sum. `p.index` says which
# subproblem it is, so that one objective serves an ensemble of them.
struct _PenalizedObjective{F,L,PH}
    f::F
    loss::L
    phi::PH
    part::Symbol
    T::Int
end
(o::_PenalizedObjective)(u, p) = o.f(u, p.data) + _loss_value(o.loss, o.phi, u, o.part, p.index, o.T) / p.scale

_penalized(f, ::Nothing, phi, part, T) = f
_penalized(f, loss, phi, part, T) = _PenalizedObjective(f, loss, phi, part, Int(T))

_penalized_data(::Nothing, data, index, scale=1.0) = data
_penalized_data(loss, data, index, scale=1.0) = (; data, index, scale)

# The loss at the coefficients `u` as a finite float, 0 without one.
_loss_float(::Nothing, phi, u, part, index, T) = 0.0
function _loss_float(loss, phi, u, part, index, T)
    v = _loss_value(loss, phi, u, part, index, T)
    isfinite(v) || _loss_error(v, part, index)
    return float(v)
end

# The check every subproblem makes before its solve: the loss is finite at its starting point.
_check_loss(::Nothing, phi, u, part, index, T) = nothing
_check_loss(loss, phi, u, part, index, T) = (_loss_float(loss, phi, u, part, index, T); nothing)
