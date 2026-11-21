# # Links
#
# Links come from `LinkFunctions.jl`; this file adds the one it is missing.

"""
    IntervalLink(a, b)

A parameter confined to the open interval `(a, b)`:

    θ = a + (b - a) · logistic(η),    η = log((θ - a) / (b - θ))

the same map as `LinkFunctions.BoundedLink(a, b)`, written so that it saturates rather than
overflows. `BoundedLink` evaluates `(a + b·exp(η)) / (1 + exp(η))`, which is `Inf / Inf` --
`NaN`, and a `NaN` gradient with it -- as soon as `η > 709`, somewhere a line search can
step. The two agree to within rounding wherever `BoundedLink` is finite.

`IntervalLink(-1, 1)` is the stationarity constraint of an AR(1) coefficient.
"""
struct IntervalLink{T <: Real} <: Link
    a::T
    b::T

    # Inner, so that it cannot be bypassed by the constructor Julia generates by default.
    function IntervalLink(a::T, b::T) where {T <: Real}
        @argcheck a < b
        return new{T}(a, b)
    end
end

IntervalLink(a::Real, b::Real) = IntervalLink(promote(a, b)...)

LinkFunctions.linkfun(l::IntervalLink, θ::Real) = logit((θ - l.a) / (l.b - l.a))
LinkFunctions.linkinv(l::IntervalLink, η::Real) = l.a + (l.b - l.a) * logistic(η)

LinkFunctions.eta_theta(l::IntervalLink, θ::Real) = (l.b - l.a) / ((θ - l.a) * (l.b - θ))
LinkFunctions.eta2_theta2(l::IntervalLink, θ::Real) =
    ((l.a - l.b) * (l.a + l.b - 2θ)) / ((θ - l.a) * (l.b - θ))^2

# `logistic(η) * logistic(-η)` rather than `p * (1 - p)`: the second factor cancels to zero
# in floating point once `p` rounds to 1, the first does not.
LinkFunctions.theta_eta(l::IntervalLink, η::Real) = (l.b - l.a) * logistic(η) * logistic(-η)
LinkFunctions.theta2_eta2(l::IntervalLink, η::Real) =
    (l.b - l.a) * logistic(η) * logistic(-η) * tanh(-η / 2)

Base.show(io::IO, l::IntervalLink) = print(io, "IntervalLink(", l.a, ", ", l.b, ")")
