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

# # Fitting a periodic AR(1) by maximum likelihood

"""
    ar1_sufficient_statistics(y, n2t, T, ns = 2:length(y)) -> NamedTuple

Per-period sufficient statistics of the periodic AR(1) loglikelihood: for each period index `t`,
the number of transitions `N` and the sums `∑y²`, `∑y₋²`, `∑yy₋`, `∑y`, `∑y₋` taken over the
observations `n ∈ ns` with `n2t[n] == t`. `ns` defaults to every observation that has a
predecessor; restrict it to fit one hidden state at a time.

The loglikelihood depends on the data only through these, so this one `O(length(ns))` pass makes
every later objective evaluation `O(T)` — regardless of how long the series is.
"""
function ar1_sufficient_statistics(y::AbstractVector, n2t::AbstractVector{<:Integer}, T::Integer, ns=2:length(y))
    N = zeros(Int, T)
    ∑y², ∑y₋², ∑yy₋, ∑y, ∑y₋ = (zeros(T) for _ = 1:5)
    for n in ns
        n == 1 && continue # the first observation has no predecessor
        t = n2t[n]
        N[t] += 1
        ∑y²[t] += y[n]^2
        ∑y₋²[t] += y[n-1]^2
        ∑yy₋[t] += y[n] * y[n-1]
        ∑y[t] += y[n]
        ∑y₋[t] += y[n-1]
    end
    return (; N, ∑y², ∑y₋², ∑yy₋, ∑y, ∑y₋)
end

"""
    ar1_nll(θ, (stats, basis)) -> Number

Negative loglikelihood of a periodic AR(1), from the sufficient statistics of
[`ar1_sufficient_statistics`](@ref).

`θ` stacks the trigonometric coefficients of the three parameters, `[θμ; θρ; θσ]`, each of length
`size(basis, 2)`. They are mapped onto their domains by the usual links: `μ(t) = P(t)` is free,
`ρ(t) = 2 / (1 + exp(-P(t))) - 1` lies in `(-1, 1)` so that the process is stationary, and
`σ(t) = exp(P(t))` is positive.
"""
function ar1_nll(θ, (stats, basis))
    T, nc = size(basis)
    Pμ = basis * view(θ, 1:nc)
    Pρ = basis * view(θ, nc+1:2nc)
    logσ = basis * view(θ, 2nc+1:3nc)
    nll = zero(eltype(Pμ))
    for t = 1:T
        μ = Pμ[t]
        ρ = 2 / (1 + exp(-Pρ[t])) - 1
        quad = stats.∑y²[t] / 2 + stats.N[t] / 2 * μ^2 + stats.∑y₋²[t] * ρ^2 / 2 -
               stats.∑yy₋[t] * ρ - stats.∑y[t] * μ + stats.∑y₋[t] * ρ * μ
        nll += quad * exp(-2logσ[t]) + stats.N[t] * logσ[t]
    end
    return nll
end

