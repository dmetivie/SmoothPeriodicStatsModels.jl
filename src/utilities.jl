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
# TODO: rm vectorize version ? So far needed in Lsq.jl
# TODO: rm `T` version

function polynomial_trigo(t::Number, β, T)
    d = (length(β) - 1) ÷ 2
    if d == 0
        return β[1]
    else
        f = 2π / T
        # everything is shifted from 1 from usual notation due to array starting at 1
        return β[1] + sum(β[2*l] * cos(f * l * t) + β[2*l+1] * sin(f * l * t) for l = 1:d)
    end
end

function polynomial_trigo(t::AbstractArray, β, T)
    d = (length(β) - 1) ÷ 2
    if d == 0
        return fill(β[1], length(t))
    else
        f = 2π / T
        # everything is shifted from 1 from usual notation due to array starting at 1
        return β[1] .+ sum(β[2*l] * cos.(f * l * t) + β[2*l+1] * sin.(f * l * t) for l = 1:d)
    end
end

function polynomial_trigo(t::AbstractFloat, β)
    d = (length(β) - 1) ÷ 2
    # everything is shifted from 1 from usual notation due to array starting at 1
    return β[1] + sum(β[2*l] * cos(2π * l * t) + β[2*l+1] * sin(2π * l * t) for l = 1:d; init = zero(t))
end

function polynomial_trigo(t::AbstractArray{F}, β) where F<:AbstractFloat
    d = (length(β) - 1) ÷ 2
    # everything is shifted from 1 from usual notation due to array starting at 1
    return β[1] .+ sum(β[2*l] * cos.(2π * l * t) + β[2*l+1] * sin.(2π * l * t) for l = 1:d; init = zero(t))
end

μₜ(t, θ::AbstractArray, T) = polynomial_trigo(t, θ[:], T)
ρₜ(t, θ::AbstractArray, T) = 2 / (1 + exp(-polynomial_trigo(t, θ[:], T))) - 1
σₜ(t, θ::AbstractArray, T) = exp(polynomial_trigo(t, θ[:], T))

μₜ(t, θ::AbstractArray) = polynomial_trigo(t, θ[:]) # not constrained
αₜ(t, θ::AbstractArray) = 1 / (1 + exp(-polynomial_trigo(t, θ[:]))) # [0,1] parameter
σₜ(t, θ::AbstractArray) = exp(polynomial_trigo(t, θ[:])) # >0 parameter
ρₜ(t, θ::AbstractArray) = 2 / (1 + exp(-polynomial_trigo(t, θ[:]))) - 1 # [-1,1]
