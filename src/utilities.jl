"""
    Merge vectors with alternate elements
    For example
    ```julia
    x = [x₁, x₂]
    y = [y₁, y₂]
    interleave2(x, y) = [x₁, y₁, x₂, y₂]
    ```
"""
interleave2(args...) = collect(Iterators.flatten(zip(args...)))

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

A degree-0 `β` gives the constant `β[1]`. `polynomial_trigo(t, β, T)` takes `t` in time
steps and is `polynomial_trigo(t / T, β)`.
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

# Legacy spelling, kept because `polynomial_trigo` is exported. It takes `t` in units of time
# steps together with the period `T`, and forwards to the normalized form above. No `depwarn`:
# this is called once per time step inside the likelihoods, where that cost would show.
polynomial_trigo(t::Number, β, T) = polynomial_trigo(t / T, β)
polynomial_trigo(t::AbstractArray, β, T) = polynomial_trigo(t ./ T, β)

# Link functions mapping the trigonometric polynomial onto each parameter's domain.
# `t` is in fractions of a period, as for `polynomial_trigo`.
μₜ(t, θ::AbstractArray) = polynomial_trigo(t, θ[:]) # not constrained
αₜ(t, θ::AbstractArray) = 1 / (1 + exp(-polynomial_trigo(t, θ[:]))) # [0,1] parameter
σₜ(t, θ::AbstractArray) = exp(polynomial_trigo(t, θ[:])) # >0 parameter
ρₜ(t, θ::AbstractArray) = 2 / (1 + exp(-polynomial_trigo(t, θ[:]))) - 1 # [-1,1]

# Legacy spellings taking the period explicitly, as for `polynomial_trigo`
μₜ(t, θ::AbstractArray, T) = μₜ(t / T, θ)
ρₜ(t, θ::AbstractArray, T) = ρₜ(t / T, θ)
σₜ(t, θ::AbstractArray, T) = σₜ(t / T, θ)
