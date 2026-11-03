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

function rand(rng::AbstractRNG, AR::AR1, N::Integer; y₁=rand(rng, Normal(0, AR.σ[1])))
    T = size(AR.μ, 1)
    n2t = n_to_t(N, T)
    return rand(rng, AR, n2t; y₁=y₁)
end

rand(AR::AR1, N::Integer; kwargs...) = rand(GLOBAL_RNG, AR, N; kwargs...)

# The innovation is `Normal(0, σ(t))` and not a field of `AR1`, deliberately. Making it one is
# only half the job: [`ar1_nll`](@ref) is the *Gaussian* closed form — its quadratic form is
# `Σₙ (yₙ - μ(t) - ρ(t) yₙ₋₁)² / 2` regrouped per period index — so a configurable noise law
# would be a field that the package's only AR(1) fit silently ignores. It waits on a generic
# conditional-likelihood objective, which needs a notion of covariate (the lagged observation)
# that `TimeVaryingParameters` does not have yet.
function rand(rng::AbstractRNG, AR::AR1{<:AbstractVector}, n2t::AbstractVector{<:Integer}; y₁=rand(rng, Normal(0, AR.σ[1])))
    N = length(n2t)
    y = zeros(eltype(y₁), N)
    y[1] = y₁
    for n = 2:N
        y[n] = AR.μ[n2t[n]] + AR.ρ[n2t[n]] * y[n-1] + rand(rng, Normal(0, AR.σ[n2t[n]]))
    end
    return y
end

rand(AR::AR1{<:AbstractVector}, n2t::AbstractVector{<:Integer}; kwargs...) = rand(GLOBAL_RNG, AR, n2t; kwargs...)

function rand(rng::AbstractRNG, AR::AR1{<:AbstractMatrix}, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}; y₁=rand(rng, Normal(0, AR.σ[1])))
    N = length(n2t)
    ϵ = randn(rng, N)
    return rand(rng, AR, ϵ, n2t, z; y₁=y₁)
end

rand(AR::AR1{<:AbstractMatrix}, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}; kwargs...) = rand(GLOBAL_RNG, AR, n2t, z; kwargs...)

function rand(rng::AbstractRNG, AR::AR1{<:AbstractMatrix}, ϵ::AbstractVector, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}; y₁=rand(rng, Normal(0, AR.σ[1])))
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

rand(AR::AR1{<:AbstractMatrix}, ϵ::AbstractVector, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}; kwargs...) = rand(GLOBAL_RNG, AR, ϵ, n2t, z; kwargs...)

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

function rand(rng::AbstractRNG, ARs::AbstractArray{T}, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}, Σ; y₁=[AR.σ[1] for AR in ARs].*rand(rng, MvNormal(Σ[1])), output_ϵ = false) where T<:AR1{<:AbstractMatrix}
    @assert length(z) == length(n2t)
    @assert size(ARs) == size(y₁)
    N = length(n2t)
    ϵ = similar(y₁, length(y₁), N)
    𝖀 = tuple(fill(Normal(), length(ARs))...)
    for (k, Σₖ) in enumerate(Σ)
        Nₖ = findall(z .== k)
        C = GaussianCopula(Σₖ) #! use cor2cov if you just have correlations matrix !!! (#? in practice does it make a difference?)
        𝔇 = SklarDist(C, 𝖀) 
        @views ϵ[:,Nₖ] .= rand(rng, 𝔇, length(Nₖ))
    end
    # for n = 1:N
    #     k = z[n]
    #     C = GaussianCopula(Σ[k]) #! use cor2cov if you just have correlations matrix !!! (#? in practice does it make a difference?)
    #     𝔇 = SklarDist(C, 𝖀) 
    #     @views ϵ[:,n] .= rand(𝔇)
    # end
    y = rand(rng, ARs, ϵ, n2t, z, Σ; y₁=y₁)
    return output_ϵ ? (y, ϵ) : y
end

rand(ARs::AbstractArray{T}, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}, Σ; kwargs...) where {T<:AR1{<:AbstractMatrix}} = rand(GLOBAL_RNG, ARs, n2t, z, Σ; kwargs...)

function rand(rng::AbstractRNG, ARs::AbstractArray{T}, ϵ::AbstractMatrix, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}, Σ; y₁=[AR.σ[1] for AR in ARs].*rand(rng, MvNormal(Σ[1]))) where T<:AR1{<:AbstractMatrix}
    @assert length(z) == length(n2t)
    @assert size(ARs) == size(y₁)
    N = length(n2t)
    y = zeros(eltype(y₁), N, size(y₁)...)
    for (j, AR) in enumerate(ARs)
        @views y[:, j] = rand(rng, AR, ϵ[j, :], n2t, z, y₁ = y₁[j])
    end
    return permutedims(y)
end

rand(ARs::AbstractArray{T}, ϵ::AbstractMatrix, n2t::AbstractVector{<:Integer}, z::AbstractVector{<:Integer}, Σ; kwargs...) where {T<:AR1{<:AbstractMatrix}} = rand(GLOBAL_RNG, ARs, ϵ, n2t, z, Σ; kwargs...)

# # Fitting a periodic AR(1) by maximum likelihood

"""
    ar1_spec(d, T) -> ParamSpec

The parametrization of a periodic AR(1): the three seasonal parameters of

    y[n] = μ(t) + ρ(t) y[n-1] + σ(t) ε[n],   t = n2t[n]

as trigonometric polynomials of degree `d` over a period of `T` steps, each pushed through the
link that keeps it in its domain — `μ` free, `ρ ∈ (-1, 1)` so that the process is stationary,
`σ > 0`.

```julia
ParamSpec(
    μ = TrigBasis(T, d) => IdentityLink(),
    ρ = TrigBasis(T, d) => IntervalLink(-1, 1),
    σ = TrigBasis(T, d) => LogLink(),
)
```

The coefficients lie end to end in that order, `2d+1` of them each — the layout
[`SmoothAR1`](@ref) names and [`ar1_nll`](@ref) reads.
"""
ar1_spec(d::Integer, T::Integer) = ParamSpec(
    μ=TrigBasis(T, d) => IdentityLink(),
    ρ=TrigBasis(T, d) => IntervalLink(-1, 1),
    σ=TrigBasis(T, d) => LogLink(),
)

"""
    SmoothAR1(d, T; K = nothing)
    SmoothAR1(spec::ParamSpec, nT, ps)

A periodic AR(1) whose three parameters are smooth functions of the period index: the structure
(`spec`, the period `nT`) held apart from the coefficients (`ps`), as `Lux.jl` holds a layer
apart from its parameters.

```julia
m  = SmoothAR1(2, 366)                 # degree 2 over a year, coefficients zeroed
ps = parameters(m)                     # ComponentArray(μ = …, ρ = …, σ = …)
ps.ρ .= θρ                             # named, and an `AbstractVector` for the solver

m, history = fit_mle(m, y; n2t)        # same type in, same type out
curve(m, :σ)                           # σ(t) over one period
AR1(m)                                 # the evaluated model, to `rand` from
```

`K` gives one *column* of coefficients per hidden state, so `parameters(m).ρ` is `(2d+1) × K`
and `curve(m, :ρ)` is `K × T` — the layout [`rand`](@ref) wants for a state-dependent AR(1).
Pass `spec` directly to use a parametrization other than [`ar1_spec`](@ref); `σ`'s link has to
stay `LogLink()`, which is the one thing [`ar1_nll`](@ref) assumes.
"""
struct SmoothAR1{S<:ParamSpec,P<:AbstractVector}
    spec::S
    nT::Int
    ps::P
end

SmoothAR1(spec::ParamSpec, nT::Integer; K=nothing) =
    SmoothAR1(spec, Int(nT), K === nothing ? seasonal_params(spec) : seasonal_params(spec, K))

SmoothAR1(d::Integer, T::Integer; kwargs...) = SmoothAR1(ar1_spec(d, T), T; kwargs...)

parameters(m::SmoothAR1) = m.ps

"""
    curve(m::SmoothAR1, name::Symbol, times = 1:m.nT)

One parameter over `times`, links applied: `curve(m, :σ)` is `σ(t)` for `t = 1:nT`. A
`K × length(times)` matrix when the model holds one column of coefficients per hidden state.
"""
TimeVaryingParameters.curve(m::SmoothAR1, name::Symbol, times=1:m.nT) =
    seasonal_curve(m.spec, m.ps, name, times)

"""
    params(m::SmoothAR1, t) -> (μ, ρ, σ)

The three parameters at one period index. Defined for a single model; with one column of
coefficients per hidden state use [`curve`](@ref), which returns a `K × T` matrix.
"""
function StatsAPI.params(m::SmoothAR1, t)
    @argcheck ncolumns(m.spec, m.ps) == 1 "`params` reads one coefficient vector, and this model holds $(ncolumns(m.spec, m.ps)); use `curve(m, name)`"
    return parameters_at(m.spec, m.ps, t)
end

(m::SmoothAR1)(t) = params(m, t)

"""
    AR1(m::SmoothAR1) -> AR1

The evaluated model: the three curves of `m` over `1:nT`, links applied. This is what `rand`
takes — `SmoothAR1` carries the coefficients, `AR1` carries the numbers they generate.
"""
AR1(m::SmoothAR1) = AR1(curve(m, :μ), curve(m, :ρ), curve(m, :σ))

rand(rng::AbstractRNG, m::SmoothAR1, args...; kwargs...) = rand(rng, AR1(m), args...; kwargs...)
rand(m::SmoothAR1, args...; kwargs...) = rand(GLOBAL_RNG, AR1(m), args...; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", m::SmoothAR1)
    print(io, "SmoothAR1 over a period of ", m.nT)
    K = ncolumns(m.spec, m.ps)
    K > 1 && print(io, ", ", K, " states")
    println(io, ", ", nparams(m), " coefficients")
    show(io, MIME"text/plain"(), m.spec)
    return
end

"""
    ar1_sufficient_statistics(y, n2t, T, ns = 2:length(y)) -> NamedTuple

Per-period sufficient statistics of the periodic AR(1) loglikelihood: for each period index `t`,
the number of transitions `N` and the sums `∑y²`, `∑y₋²`, `∑yy₋`, `∑y`, `∑y₋` taken over the
observations `n ∈ ns` with `n2t[n] == t`. `ns` defaults to every observation that has a
predecessor; restrict it to fit one hidden state at a time.
"""
function ar1_sufficient_statistics(y::AbstractVector, n2t::AbstractVector{<:Integer}, T::Integer, ns=2:length(y))
    @argcheck length(n2t) >= length(y) "`n2t` gives the period index of every observation, so it needs $(length(y)) entries, not $(length(n2t))"
    # Without this a mismatch surfaces as a `BoundsError` on the accumulators, which says nothing
    # about the cause. The usual causes are a period that does not match the index (day-of-year
    # values against a `T` smaller than 366) and a 0-based index. The loop checks each entry it
    # reads, rather than a pass of its own over `n2t`; `n2t[1]`, which it skips, is checked here,
    # being the entry that gives a 0-based index away on a series no longer than one period.
    isempty(n2t) || 1 <= first(n2t) <= T || _period_index_error(n2t, T)
    N = zeros(Int, T)
    ∑y², ∑y₋², ∑yy₋, ∑y, ∑y₋ = (zeros(T) for _ = 1:5)
    for n in ns
        n == 1 && continue # the first observation has no predecessor
        t = n2t[n]
        1 <= t <= T || _period_index_error(n2t, T)
        yₙ, yₙ₋₁ = y[n], y[n-1]
        @inbounds begin # `t` is checked just above and every accumulator has length `T`
            N[t] += 1
            ∑y²[t] += yₙ^2
            ∑y₋²[t] += yₙ₋₁^2
            ∑yy₋[t] += yₙ * yₙ₋₁
            ∑y[t] += yₙ
            ∑y₋[t] += yₙ₋₁
        end
    end
    return (; N, ∑y², ∑y₋², ∑yy₋, ∑y, ∑y₋)
end

# Out of line, and a `throw` the compiler can see never returns: with a helper that might return,
# such as one ending in `@argcheck`, the loop above runs 70% slower.
@noinline _period_index_error(n2t, T) = throw(ArgumentError(
    "every entry of `n2t` must lie in 1:$T, and they run over $(extrema(n2t))"))

"""
    ar1_nll(θ, (stats, design)) -> Number

Negative loglikelihood of a periodic AR(1), from the sufficient statistics of
[`ar1_sufficient_statistics`](@ref).

`θ` is the flat coefficient vector of [`ar1_spec`](@ref) and `design` the `PeriodicDesign` of
that spec over `1:T`, so the three linear predictors cost one matrix-vector product each and the
remaining work is `O(T)` however long the series is.

Reducing the data to sufficient statistics is what makes this cheaper than the generic
`PeriodicNLL`, and it is tied to the Gaussian likelihood: the quadratic form below is
`Σₙ (yₙ - μ - ρ yₙ₋₁)² / 2` regrouped per period index. The links come from `design`, except
that `σ` is read on the log scale directly rather than exponentiated and logged back.
"""
function ar1_nll(θ, (stats, design))
    spec = design.spec
    ημ, ηρ, ηlogσ = linear_predictors(design, θ)
    nll = zero(eltype(ημ))
    @inbounds for t in eachindex(ημ)
        μ = apply_link(spec.links[1], ημ[t])
        ρ = apply_link(spec.links[2], ηρ[t])
        logσ = ηlogσ[t]
        quad = stats.∑y²[t] / 2 + stats.N[t] / 2 * μ^2 + stats.∑y₋²[t] * ρ^2 / 2 -
               stats.∑yy₋[t] * ρ - stats.∑y[t] * μ + stats.∑y₋[t] * ρ * μ
        nll += quad * exp(-2logσ) + stats.N[t] * logσ
    end
    return nll
end

