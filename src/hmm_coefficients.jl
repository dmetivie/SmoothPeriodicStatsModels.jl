# # From coefficients to the arrays of a periodic HMM
#
# Every parameter of the Bernoulli HMMs is a basis expansion over the period index `t = 1:T`,
# `X[t, :] ⋅ θ` with `X = design_matrix(basis, 1:T)`, pushed through a softmax for the transition
# rows (the last state being the reference), a logistic for the success probabilities, and an
# exponential for the spatial range. `TrigBasis(T, d)` is the default and gives trigonometric
# polynomials of degree `d`; any basis of TimeVaryingParameters works the same way.

# `X[t, :] ⋅ θ`
function _linear_predictor(X::AbstractMatrix, t::Integer, θ::AbstractVector)
    η = zero(promote_type(eltype(X), eltype(θ)))
    @inbounds for i in eachindex(θ)
        η += X[t, i] * θ[i]
    end
    return η
end

# Row `k` of `A[:, :, t]`: the softmax of `X[t, :] ⋅ θᴬ[k, l, :]` over `l < K`, state `K` the
# reference. With one state, `A` is all ones.
function _transitions!(A::AbstractArray{<:Real,3}, θᴬ::AbstractArray{<:Real,3}, X::AbstractMatrix)
    K, T = size(A, 1), size(A, 3)
    for k = 1:K, t = 1:T
        A[k, K, t] = 1
        for l = 1:K-1
            A[k, l, t] = exp(_linear_predictor(X, t, view(θᴬ, k, l, :)))
        end
        s = sum(view(A, k, :, t))
        for l = 1:K
            A[k, l, t] /= s
        end
    end
    return A
end

# `B[k, t, j, h]`, the law of site `j` in state `k` with memory category `h`: a Bernoulli whose
# success probability is the logistic of `X[t, :] ⋅ θᴮ[k, j, h, :]`.
function _emissions!(B::AbstractArray{<:Bernoulli,4}, θᴮ::AbstractArray{<:Real,4}, X::AbstractMatrix)
    K, T, D, size_order = size(B)
    for k = 1:K, t = 1:T, j = 1:D, h = 1:size_order
        B[k, t, j, h] = Bernoulli(logistic(_linear_predictor(X, t, view(θᴮ, k, j, h, :))))
    end
    return B
end

# The same with site-dependent memory: `B[k, t, j][h]` from `θᴮ[k, j][h]`.
function _emissions!(B::AbstractArray{<:AbstractVector{<:Bernoulli},3}, θᴮ::AbstractMatrix, X::AbstractMatrix)
    K, T, D = size(B)
    for k = 1:K, t = 1:T, j = 1:D
        for (h, θh) in enumerate(θᴮ[k, j])
            B[k, t, j][h] = Bernoulli(logistic(_linear_predictor(X, t, θh)))
        end
    end
    return B
end

# The degree `d` of a trigonometric polynomial with `n = 2d + 1` coefficients: the default basis
# of a coefficient array given without one.
function _trig_order(n::Integer, name)
    isodd(n) || throw(ArgumentError(
        "`$name` has $n coefficients per expansion, which is not the 2d + 1 of a trigonometric " *
        "polynomial; pass its basis"))
    return (n - 1) ÷ 2
end

function _check_basis(basis, n::Integer, name)
    nbasis(basis) == n || throw(ArgumentError(
        "the basis of `$name` has $(nbasis(basis)) columns, and `$name` holds $n coefficients per expansion"))
    return true
end

# The degree of a trigonometric basis, `TrigBasis(T, d)`.
_trig_degree(::ConstantBasis) = 0
_trig_degree(b::CombinedBasis{<:Tuple{ConstantBasis,FourierBasis}}) = b.bases[2].order
_trig_degree(b) = throw(ArgumentError("$b is not a trigonometric basis, so it has no degree"))
