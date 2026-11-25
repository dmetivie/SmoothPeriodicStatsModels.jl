"""
    interleave2(args...) -> Vector

Merge vectors by alternating their elements.

```julia
interleave2([1, 2], [10, 20]) == [1, 10, 2, 20]
```
"""
interleave2(args...) = collect(Iterators.flatten(zip(args...)))

"""
    remaining(N::Int)

The indices `1:N`, or an empty `Vector{Int}` when `N ≤ 0`.
"""
remaining(N::Int) = N > 0 ? range(1, length=N) : Int64[]

# TODO: add check on length of β
# TODO: rm vectorize version ?

"""
    polynomial_trigo(t, β) -> Number

Trigonometric polynomial of degree `d = (length(β) - 1) ÷ 2` evaluated at `t`, where `t` is
expressed in **fractions of a period**: `t` and `t + 1` give the same value.

```math
β₁ + \\sum_{l=1}^{d} β_{2l} \\cos(2π l t) + β_{2l+1} \\sin(2π l t)
```

A degree-0 `β` gives the constant `β[1]`.
"""
function polynomial_trigo(t::Number, β)
    d = (length(β) - 1) ÷ 2
    # everything is shifted from 1 from usual notation due to array starting at 1
    return β[1] + sum(β[2*l] * cos(2π * l * t) + β[2*l+1] * sin(2π * l * t) for l = 1:d; init=zero(t))
end

function polynomial_trigo(t::AbstractArray{<:Number}, β)
    d = (length(β) - 1) ÷ 2
    # everything is shifted from 1 from usual notation due to array starting at 1
    return β[1] .+ sum(β[2*l] * cos.(2π * l * t) + β[2*l+1] * sin.(2π * l * t) for l = 1:d; init=zero(t))
end

"""
    μₜ(t, θ) -> Real

An unconstrained seasonal parameter: `polynomial_trigo(t, θ)`, with `t` in fractions of a period.
"""
μₜ(t, θ::AbstractArray) = polynomial_trigo(t, θ[:])

"""
    αₜ(t, θ) -> Real

A seasonal parameter in `(0, 1)`: `logistic(polynomial_trigo(t, θ))`, with `t` in fractions of a
period.
"""
αₜ(t, θ::AbstractArray) = logistic(polynomial_trigo(t, θ[:]))

"""
    σₜ(t, θ) -> Real

A positive seasonal parameter: `exp(polynomial_trigo(t, θ))`, with `t` in fractions of a period.
"""
σₜ(t, θ::AbstractArray) = exp(polynomial_trigo(t, θ[:]))

"""
    ρₜ(t, θ) -> Real

A seasonal parameter in `(-1, 1)`: `2logistic(polynomial_trigo(t, θ)) - 1`, with `t` in fractions
of a period.
"""
ρₜ(t, θ::AbstractArray) = 2logistic(polynomial_trigo(t, θ[:])) - 1

# ## Transition matrices
# Originally `PeriodicHiddenMarkovModels`', itself from HMMBase; neither keeps them, and the
# constructor checks and `randARPeriodicHMM` here still want them.

"""
    issquare(A) -> Bool

Whether `A` is a square matrix.
"""
issquare(A::AbstractMatrix) = size(A, 1) == size(A, 2)

"""
    istransmat(A) -> Bool

Whether `A` is square and each of its rows is a probability vector.
"""
istransmat(A::AbstractMatrix) = issquare(A) && all(isprobvec(A[i, :]) for i = 1:size(A, 1))

"""
    randtransmat([rng, ]prior::MultivariateDistribution) -> Matrix{Float64}
    randtransmat([rng, ]K::Integer, α = 1.0) -> Matrix{Float64}

Transition matrix whose rows are drawn from `prior`, or from `Dirichlet(K, α)`.

```julia
A = randtransmat(Dirichlet([0.1, 0.1, 0.1]))
A = randtransmat(4)
```
"""
function randtransmat(rng::AbstractRNG, prior::MultivariateDistribution)
    K = length(prior)
    A = Matrix{Float64}(undef, K, K)
    for i in OneTo(K)
        A[i, :] = rand(rng, prior)
    end
    @check istransmat(A)
    return A
end

randtransmat(rng::AbstractRNG, K::Integer, α=1.0) = randtransmat(rng, Dirichlet(K, α))
randtransmat(prior::MultivariateDistribution) = randtransmat(GLOBAL_RNG, prior)
randtransmat(K::Integer, args...) = randtransmat(GLOBAL_RNG, K, args...)


# ## `robust = true`: observations that no state can produce
#
# The loglikelihoods of one observation in every state, made usable by a forward pass.
#
# - If every state gives `-Inf`, the observation is impossible in all of them and the forward
#   pass would divide zero by zero. All states get `log(floatmin())`: the observation then says
#   nothing about the state, which its neighbours decide, and the loglikelihood of the data stays
#   finite and still moves with the parameters. `nextfloat(-Inf)` in its place would swamp every
#   other term of that sum, and a convergence test on the loglikelihood would stop a fit after
#   one iteration.
# - `-Inf` in some states only is left as it is: those states get probability zero, which is
#   right. A finite replacement could rank a state where the observation is impossible above one
#   where it is merely improbable.
# - `Inf` is capped at `log(floatmax())`.
function _clamp_inf!(logb::AbstractVector)
    T = float(eltype(logb))
    impossible = true
    @inbounds for i in eachindex(logb)
        v = logb[i]
        v == Inf && (logb[i] = log(floatmax(T)))
        impossible &= v == -Inf
    end
    impossible && fill!(logb, log(floatmin(T)))
    return logb
end

# The same for an `N × K` matrix, one observation per row.
function _clamp_inf!(LL::AbstractMatrix)
    foreach(_clamp_inf!, eachrow(LL))
    return LL
end

# Check that `n2t` gives a period index in `1:T` to each of `N` observations.
function _check_n2t(n2t::AbstractVector{<:Integer}, N::Integer, T::Integer)
    @argcheck length(n2t) == N "`n2t` has $(length(n2t)) entries for $N observations"
    isempty(n2t) || all(t -> 1 <= t <= T, n2t) || _period_index_error(n2t, T)
    return n2t
end

# Out of line, and a `throw` the compiler can see never returns: with a helper that might return,
# such as one ending in `@argcheck`, the loop of `ar1_sufficient_statistics` runs 70% slower.
@noinline _period_index_error(n2t, T) = throw(ArgumentError(
    "every entry of `n2t` must lie in 1:$T, and they run over $(extrema(n2t))"))

# ## `past`, what precedes the observations
#
# One keyword for every model, `past = (; Y = D × m, z = k₀)`: the last column of `Y` is the step
# before the first observation and `z` the hidden state there. A bare array is `(; Y = array)`.
_past_fields(::Nothing) = (Y=nothing, z=nothing)
_past_fields(Y::AbstractArray) = (Y=Y, z=nothing)
function _past_fields(past::NamedTuple)
    for k in keys(past)
        k in (:Y, :z) || throw(ArgumentError("`past` has the fields `Y` and `z`, not `$k`"))
    end
    return (Y=get(past, :Y, nothing), z=get(past, :z, nothing))
end
_past_fields(past) = throw(ArgumentError(
    "`past` is `(; Y = observations, z = state)`, or the observations alone; got a $(typeof(past))"))

# ## Several sequences laid end to end

# `seq_ends`, the last index of each sequence, checked against `N` observations.
function _check_seq_ends(seq_ends, N::Integer)
    seq_ends isa Union{AbstractVector{<:Integer},Tuple{Vararg{Integer}}} || throw(ArgumentError(
        "`seq_ends` is the last index of each sequence, a vector of integers; got a $(typeof(seq_ends))"))
    ends = collect(Int, seq_ends)
    isempty(ends) && throw(ArgumentError("`seq_ends` is empty"))
    first(ends) >= 1 && all(>(0), diff(ends)) ||
        throw(ArgumentError("`seq_ends` must increase strictly from at least 1, with no empty sequence; got $ends"))
    last(ends) == N || throw(ArgumentError("`seq_ends` ends at $(last(ends)), and there are $N observations"))
    return ends
end

# The indices of each sequence.
_seq_ranges(ends) = [(k == 1 ? 1 : ends[k-1] + 1):ends[k] for k in eachindex(ends)]

# The pasts of `n` sequences: a vector of them, or `nothing` for none.
function _pasts(past, n::Integer)
    past === nothing && return fill(nothing, n)
    past isa AbstractVector && !(past isa AbstractArray{<:Real}) || throw(ArgumentError(
        "with several sequences, `past` is a vector of their pasts, one per sequence"))
    length(past) == n || throw(ArgumentError("$(length(past)) pasts for $n sequences"))
    return past
end
