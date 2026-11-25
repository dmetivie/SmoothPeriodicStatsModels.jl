# # The site-dependent periodic HMM with its coefficients attached
#
# The same pair as [`SmoothARPeriodicHMM`](@ref) — coefficients plus the evaluation they
# generate — for the variant where each site carries its own autoregressive order.

"""
    SmoothsARPeriodicHMM(a, θᴬ, θᴮ, T; basis_A, basis_B)

A periodic autoregressive Bernoulli HMM with site-dependent memory, given by its coefficients,
with the layout and the bases of the coefficient method of [`sARPeriodicHMM`](@ref).

```julia
m  = SmoothsARPeriodicHMM(a, θᴬ, θᴮ, T)
parameters(m)                  # (A = θᴬ, B = θᴮ)
m, history = fit_mle(m, Y, Y_past; n2t, solver)
sARPeriodicHMM(m)              # the evaluated model
```

`parameters(m)` is a `NamedTuple`: `θᴮ[k, j]` holds one coefficient vector per memory category of
site `j`, and the sites may differ in order.
"""
struct SmoothsARPeriodicHMM{A,B,S<:sARPeriodicHMM,BA,BB,C} <: HMMs.AbstractHMM
    θᴬ::A
    θᴮ::B
    stored::S    # the evaluated model, a cache (see `_current`); concrete, as every accessor reads it
    nT::Int
    basis_A::BA
    basis_B::BB
    cfg::C
    stored_from::Base.RefValue{NamedTuple{(:A, :B),Tuple{A,B}}}   # the coefficients `stored` was evaluated from
end

# `stored` evaluated from `θᴬ` and `θᴮ` as they are now.
SmoothsARPeriodicHMM(θᴬ, θᴮ, stored, nT, basis_A, basis_B, cfg) =
    SmoothsARPeriodicHMM(θᴬ, θᴮ, stored, nT, basis_A, basis_B, cfg, Ref(deepcopy((A=θᴬ, B=θᴮ))))

function SmoothsARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:Real,3},
    θᴮ::AbstractMatrix, T::Integer;
    basis_A=TrigBasis(T, _trig_order(size(θᴬ, 3), "θᴬ")),
    basis_B=TrigBasis(T, _trig_order(length(first(first(θᴮ))), "θᴮ")), cfg=nothing)
    θᴬ = collect(float.(θᴬ))
    θᴮ = map(θ -> [collect(float.(θh)) for θh in θ], θᴮ)   # a copy, in floats the fit can write
    stored = sARPeriodicHMM(a, θᴬ, θᴮ, T; basis_A, basis_B)
    return SmoothsARPeriodicHMM(θᴬ, θᴮ, stored, Int(T), basis_A, basis_B, cfg)
end

degrees(m::SmoothsARPeriodicHMM) = (_trig_degree(m.basis_A), _trig_degree(m.basis_B))

# `:p` is a `K × D` matrix whose entry `[k, j]` is the `2^memory_j × length(times)` matrix of site
# `j` in state `k`, the sites differing in memory.
function TimeVaryingParameters.curve(m::SmoothsARPeriodicHMM, name::Symbol, times=1:m.nT; state=:)
    name === :A && return _select_state(_curve_A(m.θᴬ, m.basis_A, times), state)
    name === :p && return _select_state(_curve_p(m.θᴮ, m.basis_B, times), state)
    throw(ArgumentError("`$name` is not a curve of this model; its curves are `:A` and `:p`"))
end

parameters(m::SmoothsARPeriodicHMM) = (A=m.θᴬ, B=m.θᴮ)
nparams(m::SmoothsARPeriodicHMM) =
    length(m.θᴬ) + sum(length(θh) for θ in m.θᴮ for θh in θ)

"""
    sARPeriodicHMM(m::SmoothsARPeriodicHMM) -> sARPeriodicHMM

The `sARPeriodicHMM` with the `A` and `B` arrays the coefficients of `m` generate.
"""
sARPeriodicHMM(m::SmoothsARPeriodicHMM) = copy(_current(m))

size(m::SmoothsARPeriodicHMM, dim=:) = size(m.stored, dim)

# Rewrite the evaluated arrays from the coefficients, in place.
function _evaluate!(m::SmoothsARPeriodicHMM)
    T = m.nT
    _transitions!(m.stored.A, m.θᴬ, design_matrix(m.basis_A, 1:T))
    _emissions!(m.stored.B, m.θᴮ, design_matrix(m.basis_B, 1:T))
    return m
end

# One method per signature, as for `SmoothARPeriodicHMM`.
rand(rng::AbstractRNG, m::SmoothsARPeriodicHMM, N::Integer; kwargs...) = rand(rng, _current(m), N; kwargs...)
rand(rng::AbstractRNG, m::SmoothsARPeriodicHMM, n2t::_IndexVector; kwargs...) =
    rand(rng, _current(m), n2t; kwargs...)
rand(rng::AbstractRNG, m::SmoothsARPeriodicHMM, z::_IndexVector, n2t::_IndexVector; kwargs...) =
    rand(rng, _current(m), z, n2t; kwargs...)
rand(m::SmoothsARPeriodicHMM, N::Integer; kwargs...) = rand(GLOBAL_RNG, m, N; kwargs...)
rand(m::SmoothsARPeriodicHMM, n2t::_IndexVector; kwargs...) = rand(GLOBAL_RNG, m, n2t; kwargs...)
rand(m::SmoothsARPeriodicHMM, z::_IndexVector, n2t::_IndexVector; kwargs...) =
    rand(GLOBAL_RNG, m, z, n2t; kwargs...)

loglikelihoods(m::SmoothsARPeriodicHMM, args...; kwargs...) = loglikelihoods(_current(m), args...; kwargs...)
viterbi(m::SmoothsARPeriodicHMM, args...; kwargs...) = viterbi(_current(m), args...; kwargs...)
complete_loglikelihood(m::SmoothsARPeriodicHMM, args...; kwargs...) =
    complete_loglikelihood(_current(m), args...; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", m::SmoothsARPeriodicHMM)
    K, D, T, size_orders = size(m.stored)
    println(io, "SmoothsARPeriodicHMM: ", K, " states, ", D, " sites, period ", T,
        ", memory per site ", Int.(log2.(size_orders)))
    println(io, "  transitions  ", m.basis_A, ", emissions ", m.basis_B)
    print(io, "  ", nparams(m), " coefficients")
    return
end
