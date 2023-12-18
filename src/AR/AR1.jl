#TODO look at the file [AR1_test.jl](C:\Users\metivier\.julia\dev\PeriodicAutoRegressive\src\AR1_test.jl) as there is some unfinished updates in the code.
#TODO add tests using [ar1_temperature.jl](C:\Users\metivier\Dropbox\PC (2)\Documents\Simulations\SWG\article_swg\ar1_temperature.jl)
#? Not sure the type is needed + it causes issues with integer
abstract type AbstractPeriodicAR{T} end

@doc raw"""
AR(1) process with finite markov chain
The process follows
```math
    y_t = \mu + \rho y_{t-1} + \epsilon_t
```
where ``\epsilon_t \sim N (0, \sigma^2)``
##### Arguments
- `N::Integer`: Number of points in markov process
- `|ρ::Real|<1` : Persistence parameter in AR(1) process
- `σ::Real` : Standard deviation of random component of AR(1) process
- `μ::Real(0.0)` : Mean of AR(1) process

"""
struct AR1{T} <: AbstractPeriodicAR{T}
    μ::T
    ρ::T
    σ::T
end
# https://stackoverflow.com/questions/56714992/julia-multiple-dispatch-for-mutable-struct-items

function rand(AR::AR1, N::Integer; y₁=rand(Normal(0, AR.σ[1])))
    T = size(AR.μ, 1)
    n2t = n_to_t(N, T)
    return rand(AR, n2t; y₁=y₁)
end

#TODO: add distribution into AR field
function rand(AR::AR1{<:AbstractVector}, n2t::AbstractVector{<:Integer}; y₁=rand(Normal(0, AR.σ[1])))
    N = length(n2t)
    y = zeros(eltype(y₁), N)
    y[1] = y₁
    for n = 2:N
        y[n] = AR.μ[n2t[n]] + AR.ρ[n2t[n]] * y[n-1] + rand(Normal(0, AR.σ[n2t[n]]))
    end
    return y
end

function rand(AR::AR1{<:AbstractMatrix}, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}; y₁=rand(Normal(0, AR.σ[1])))
    N = length(n2t)
    ϵ = randn(N)
    return rand(AR, ϵ, n2t, z; y₁=y₁)
end

function rand(AR::AR1{<:AbstractMatrix}, ϵ::AbstractVector, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}; y₁=rand(Normal(0, AR.σ[1])))
    @assert length(z) == length(n2t)
    @assert length(z) == length(ϵ) "got $(length(z)) == $(length(ϵ))"
    N = length(n2t)
    y = zeros(eltype(y₁), N)
    y[1] = y₁
    for n = 2:N
        y[n] = AR.μ[z[n], n2t[n]] + AR.ρ[z[n], n2t[n]] * y[n-1] + AR.σ[z[n], n2t[n]]*ϵ[n]
    end
    return y
end

# function rand2(ARs::AbstractArray{T}, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}, Σ; y₁=[AR.σ[1] for AR in ARs].*rand(MvNormal(Σ[1]))) where T<:AR1{<:AbstractMatrix}
#     @assert length(z) == length(n2t)
#     @assert size(ARs) == size(y₁)
#     N = length(n2t)
#     y = zeros(eltype(y₁), N, size(y₁)...)
#     ϵ = similar(y₁)
#     y[1, :] .= y₁
#     for n = 2:N
#         k = z[n]
#         tₙ = n2t[n]
#         C = GaussianCopula(Σ[k]) #! use cor2cov if you just have correlations matrix !!! (#? in practice does it make a difference?)
#         𝔇 = SklarDist(C, tuple(fill(Normal(), length(ARs))...)) 
#         @views ϵ .= rand(𝔇)
#         for j in eachindex(y₁)
#             y[n, j] = ARs[j].μ[k, tₙ] + ARs[j].ρ[k, tₙ] * y[n-1, j] + ARs[j].σ[k, tₙ] * ϵ[j]
#         end
#     end
#     return y
# end

function rand(ARs::AbstractArray{T}, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}, Σ; y₁=[AR.σ[1] for AR in ARs].*rand(MvNormal(Σ[1])), output_ϵ = false) where T<:AR1{<:AbstractMatrix}
    @assert length(z) == length(n2t)
    @assert size(ARs) == size(y₁)
    N = length(n2t)
    ϵ = similar(y₁, length(y₁), N)
    𝖀 = tuple(fill(Normal(), length(ARs))...)
    for (k, Σₖ) in enumerate(Σ)
        Nₖ = findall(z .== k)
        C = GaussianCopula(Σₖ) #! use cor2cov if you just have correlations matrix !!! (#? in practice does it make a difference?)
        𝔇 = SklarDist(C, 𝖀) 
        @views ϵ[:,Nₖ] .= rand(𝔇, length(Nₖ))
    end
    # for n = 1:N
    #     k = z[n]
    #     C = GaussianCopula(Σ[k]) #! use cor2cov if you just have correlations matrix !!! (#? in practice does it make a difference?)
    #     𝔇 = SklarDist(C, 𝖀) 
    #     @views ϵ[:,n] .= rand(𝔇)
    # end
    y = rand(ARs, ϵ, n2t, z, Σ; y₁=y₁)
    return output_ϵ ? (y, ϵ) : y
end

function rand(ARs::AbstractArray{T}, ϵ::AbstractMatrix, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}, Σ; y₁=[AR.σ[1] for AR in ARs].*rand(MvNormal(Σ[1]))) where T<:AR1{<:AbstractMatrix}
    @assert length(z) == length(n2t)
    @assert size(ARs) == size(y₁)
    N = length(n2t)
    y = zeros(eltype(y₁), N, size(y₁)...)
    for (j, AR) in enumerate(ARs)
        @views y[:, j] = rand(AR, ϵ[j, :], n2t, z, y₁ = y₁[j])
    end
    return permutedims(y)
end

function model_for_loglikelihood_AR1(d::Integer, T::Integer; silence=true)

    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)

    μ, ρ, σ = add_variables!(model, d, T)
    # println(model)
    # * Same initialization but will be changed later
    @NLparameter(model, N[t=1:T] == 1)
    @NLparameter(model, ∑yₜ²[t=1:T] == 1)
    @NLparameter(model, ∑yₜ₋₁²[t=1:T] == 1)
    @NLparameter(model, ∑yₜyₜ₋₁[t=1:T] == 1)
    @NLparameter(model, ∑yₜ[t=1:T] == 1)
    @NLparameter(model, ∑yₜ₋₁[t=1:T] == 1)

    @NLobjective(
        model, Max,
        -sum(
            (∑yₜ²[t] / 2 + N[t] / 2 * μ[t]^2 + ∑yₜ₋₁²[t] * ρ[t]^2 / 2 - ∑yₜyₜ₋₁[t] * ρ[t] - ∑yₜ[t] * μ[t] + ∑yₜ₋₁[t] * ρ[t] * μ[t]) / σ[t]^2
            +
            N[t] * log(σ[t])
            for t = 1:T)
    )

    # I don't know if it is the best but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:N] = N
    model[:∑yₜ²] = ∑yₜ²
    model[:∑yₜ₋₁²] = ∑yₜ₋₁²
    model[:∑yₜyₜ₋₁] = ∑yₜyₜ₋₁
    model[:∑yₜ] = ∑yₜ
    model[:∑yₜ₋₁] = ∑yₜ₋₁

    return model
end

"""
    initialvalue_optimize!(θ_μ::AbstractArray, θ_ρ::AbstractArray, θ_σ::AbstractArray, model::Model, observable; warm_start=true)
Fit the JuMP `model` with intial states `θ_μ`, `θ_ρ` and `θ_σ` and parameters observable (in my mind it should be a `Dict`).
In place modification of the parameters to the fitted values.

"""
function initialvalue_optimize!(θ_μ::AbstractArray, θ_ρ::AbstractArray, θ_σ::AbstractArray, model::Model, observable; warm_start=true)

    μ_jump = model[:μ_jump]
    ρ_jump = model[:ρ_jump]
    σ_jump = model[:σ_jump]

    # * Set the parameters in the JuMP model *#
    #TODO maybe let eachindex(observable[:N]) to something more generic
    for t in eachindex(observable[:N])
        set_value(model[:N][t], observable[:N][t])
        set_value(model[:∑yₜ²][t], observable[:∑yₜ²][t])
        set_value(model[:∑yₜ₋₁²][t], observable[:∑yₜ₋₁²][t])
        set_value(model[:∑yₜyₜ₋₁][t], observable[:∑yₜyₜ₋₁][t])
        set_value(model[:∑yₜ][t], observable[:∑yₜ][t])
        set_value(model[:∑yₜ₋₁][t], observable[:∑yₜ₋₁][t])
    end

    if warm_start
        # * Set the initial conditions * #
        set_start_value.(μ_jump, θ_μ[:])
        set_start_value.(ρ_jump, θ_ρ[:])
        set_start_value.(σ_jump, θ_σ[:])
    end

    optimize!(model)

    θ_μ[:] = value.(μ_jump)
    θ_ρ[:] = value.(ρ_jump)
    θ_σ[:] = value.(σ_jump)
end

logfYₜYₜ₋₁(yₙ, yₙ₋₁, μ, ρ, σ) = -(yₙ - ρ * yₙ₋₁ - μ)^2 / (2 * σ^2) - log(σ)

model_for_loglikelihood_AR1_full(y::AbstractVector, d::Integer, T::Integer; kwargs...) = model_for_loglikelihood_AR1_full(y, d, T, n_to_t(length(y), T); kwargs...)
model_for_loglikelihood_AR1_full(y::AbstractVector, z::AbstractVector{<:Integer}, d::Integer, T::Integer; kwargs...) = model_for_loglikelihood_AR1_full(y, z, d, T, n_to_t(length(y), T); kwargs...)

function model_for_loglikelihood_AR1_full(y::AbstractVector, d::Integer, T::Integer, n2t::AbstractVector{<:Integer}; silence=true)

    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)

    μ, ρ, σ = add_variables!(model, d, T)

    register(model, :logfYₜYₜ₋₁, 5, logfYₜYₜ₋₁; autodiff=true)

    @NLobjective(
        model, Max,
        sum(logfYₜYₜ₋₁(y[n], y[n-1], μ[n2t[n]], ρ[n2t[n]], σ[n2t[n]])
            for n = 2:size(y, 1))
    )
    optimize!(model)

    return value.(model[:μ_jump]), value.(model[:ρ_jump]), value.(model[:σ_jump])
end

function model_for_loglikelihood_AR1_full(y::AbstractVector, z::AbstractVector{<:Integer}, d::Integer, T::Integer, n2t::AbstractVector{<:Integer}; silence=true)

    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)

    K = maximum(z) # length(unique(z))

    @warn "K = $K. Keep in mind that if one category is not in z it will not be fitted"

    μ, ρ, σ = add_variables!(model, d, T, K)

    register(model, :logfYₜYₜ₋₁, 5, logfYₜYₜ₋₁; autodiff=true)

    @NLobjective(
        model, Max,
        sum(logfYₜYₜ₋₁(y[n], y[n-1], μ[n2t[n], z[n]], ρ[n2t[n], z[n]], σ[n2t[n], z[n]])
            for n = 2:size(y, 1))
    )
    optimize!(model)

    return value.(model[:μ_jump]), value.(model[:ρ_jump]), value.(model[:σ_jump])
end

function add_variables!(model, d::Integer, T::Integer)
    f = 2π / T
    cos_nj = [cos(f * j * t) for t = 1:T, j = 1:d]
    sin_nj = [sin(f * j * t) for t = 1:T, j = 1:d]
    trig = [[1; interleave2(cos_nj[t, :], sin_nj[t, :])] for t = 1:T]

    # * Polynomial μ(t) = P(t), -∞<μ<+∞ * #
    @variable(model, μ_jump[j=1:(2d+1)])
    μ = [@NLexpression(model, sum(trig[t][j] * μ_jump[j] for j = 1:(2d+1))) for t in 1:T]

    # * Polynomial ρ(t) = 2/(1+exp(-P(t)) - 1, -1<ρ<1 * #
    @variable(model, ρ_jump[j=1:(2d+1)])
    ρ = [@NLexpression(model, 2 / (1 + exp(-sum(trig[t][j] * ρ_jump[j] for j = 1:(2d+1)))) - 1) for t in 1:T]

    # * Polynomial σ(t) = exp(P(t)), 0<σ<+∞ * #
    @variable(model, σ_jump[j=1:(2d+1)])
    σ = [@NLexpression(model, exp(sum(trig[t][j] * σ_jump[j] for j = 1:(2d+1)))) for t in 1:T]
    return μ, ρ, σ
end

function add_variables!(model, d::Integer, T::Integer, K::Integer)
    f = 2π / T
    cos_nj = [cos(f * j * t) for t = 1:T, j = 1:d]
    sin_nj = [sin(f * j * t) for t = 1:T, j = 1:d]
    trig = [[1; interleave2(cos_nj[t, :], sin_nj[t, :])] for t = 1:T]

    # * Polynomial μ(t) = P(t), -∞<μ<+∞ * #
    @variable(model, μ_jump[j=1:(2d+1)])
    μ = [@NLexpression(model, sum(trig[t][j] * μ_jump[j, k] for j = 1:(2d+1))) for t in 1:T, k in 1:K]

    # * Polynomial ρ(t) = 2/(1+exp(-P(t)) - 1, -1<ρ<1 * #
    @variable(model, ρ_jump[j=1:(2d+1)])
    ρ = [@NLexpression(model, 2 / (1 + exp(-sum(trig[t][j] * ρ_jump[j, k] for j = 1:(2d+1)))) - 1) for t in 1:T, k in 1:K]

    # * Polynomial σ(t) = exp(P(t)), 0<σ<+∞ * #
    @variable(model, σ_jump[j=1:(2d+1)])
    σ = [@NLexpression(model, exp(sum(trig[t][j] * σ_jump[j, k] for j = 1:(2d+1)))) for t in 1:T, k in 1:K]
    return μ, ρ, σ
end