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

