# # State weights
#
# The third argument of a fit is one convention for every model: an `N × K` matrix whose column
# `k` weights the observations of state `k`. `OneHot` is a state sequence read as such a matrix,
# so a known path and soft posteriors go through the same methods.

"""
    OneHot(z, K)

The state sequence `z` as an `N × K` matrix of `Bool`, row `n` being one at column `z[n]`.
Every `z[n]` must lie in `1:K`. `z` is not copied.

```julia
W = OneHot([1, 2, 2, 1], 2)
W[3, 2]                      # true
fit_mle(m, y, W; n2t, solver)   # one fit per state, given the states
```
"""
struct OneHot{V<:AbstractVector{<:Integer}} <: AbstractMatrix{Bool}
    z::V
    K::Int
    function OneHot(z::V, K::Integer) where {V<:AbstractVector{<:Integer}}
        Base.require_one_based_indexing(z)
        z isa AbstractVector{Bool} && throw(ArgumentError(
            "`z` is a vector of Bool; the states are the integers 1:K"))
        K >= 1 || throw(ArgumentError("`K` is the number of states, at least 1, not $K"))
        for (n, k) in enumerate(z)
            1 <= k <= K || throw(ArgumentError("`z[$n] = $k` is not a state of 1:$K"))
        end
        return new{V}(z, Int(K))
    end
end

Base.size(W::OneHot) = (length(W.z), W.K)

Base.@propagate_inbounds function Base.getindex(W::OneHot, n::Int, k::Int)
    @boundscheck checkbounds(W, n, k)
    return @inbounds W.z[n] == k
end

# The weights `w` of `N` observations and `K` states as an `N × K` matrix, checked; not copied.
function _state_weights(W::OneHot, N::Integer, K::Integer)
    @argcheck W.K == K "`OneHot` has $(W.K) states and the model $K"
    @argcheck length(W.z) == N "`OneHot` has $(length(W.z)) rows for $N observations"
    return W
end

function _state_weights(w::AbstractVector{<:AbstractFloat}, N::Integer, K::Integer)
    K == 1 || throw(ArgumentError(
        "a vector of weights fits a model of one state, and this one has $K: pass an N × K matrix, or OneHot(z, K)"))
    @argcheck length(w) == N "$(length(w)) weights for $N observations"
    _check_weight_values(w)
    return reshape(w, N, 1)
end

_state_weights(::AbstractVector{<:Integer}, ::Integer, ::Integer) = throw(ArgumentError(
    "integer weights look like a state path: pass OneHot(z, K) for states, float.(w) for weights"))

_state_weights(w::AbstractVector, ::Integer, ::Integer) = _bad_weights(w)

function _state_weights(W::AbstractMatrix{<:Real}, N::Integer, K::Integer)
    @argcheck size(W) == (N, K) "the weights are $(size(W, 1)) × $(size(W, 2)), and $N observations of $K states need $N × $K"
    _check_weight_values(W)
    return W
end

_state_weights(w, ::Integer, ::Integer) = _bad_weights(w)

@noinline _bad_weights(w) = throw(ArgumentError(
    "weights are `OneHot(z, K)`, an N × K matrix of reals, or, for a model of one state, a vector of floats; got a $(typeof(w))"))

function _check_weight_values(w::AbstractArray{<:Real})
    all(isfinite, w) || throw(ArgumentError("the weights must be finite"))
    all(>=(0), w) || throw(ArgumentError("the weights must be nonnegative"))
    any(>(0), w) || throw(ArgumentError("the weights are all zero"))
    return w
end

# A state `k` of `1:K`, and a state path of `N` such states.
function _check_state(k, K::Integer)
    k isa Integer && !(k isa Bool) && 1 <= k <= K ||
        throw(ArgumentError("`state = $k` is not a state of 1:$K"))
    return k
end

function _check_state_path(z, N::Integer, K::Integer)
    z isa AbstractVector{<:Integer} && !(z isa AbstractVector{Bool}) ||
        throw(ArgumentError("a state path is a vector of integers in 1:$K; got a $(typeof(z))"))
    @argcheck length(z) == N "the state path has $(length(z)) entries for $N steps"
    for (n, k) in enumerate(z)
        1 <= k <= K || throw(ArgumentError("`z[$n] = $k` is not a state of 1:$K"))
    end
    return z
end
