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
    stored::S    # concrete: every accessor reads it, once per observation
    nT::Int
    basis_A::BA
    basis_B::BB
    cfg::C
end

function SmoothsARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:Real,3},
    θᴮ::AbstractMatrix, T::Integer;
    basis_A=TrigBasis(T, _trig_order(size(θᴬ, 3), "θᴬ")),
    basis_B=TrigBasis(T, _trig_order(length(first(first(θᴮ))), "θᴮ")), cfg=nothing)
    θᴬ, θᴮ = collect(float.(θᴬ)), deepcopy(θᴮ)
    stored = sARPeriodicHMM(a, θᴬ, θᴮ, T; basis_A, basis_B)
    return SmoothsARPeriodicHMM(θᴬ, θᴮ, stored, Int(T), basis_A, basis_B, cfg)
end

degrees(m::SmoothsARPeriodicHMM) = (_trig_degree(m.basis_A), _trig_degree(m.basis_B))

parameters(m::SmoothsARPeriodicHMM) = (A=m.θᴬ, B=m.θᴮ)
nparams(m::SmoothsARPeriodicHMM) =
    length(m.θᴬ) + sum(length(θh) for θ in m.θᴮ for θh in θ)

"""
    sARPeriodicHMM(m::SmoothsARPeriodicHMM) -> sARPeriodicHMM

The `sARPeriodicHMM` with the `A` and `B` arrays the coefficients of `m` generate.
"""
sARPeriodicHMM(m::SmoothsARPeriodicHMM) = m.stored

size(m::SmoothsARPeriodicHMM, dim=:) = size(m.stored, dim)

HMMs.initialization(m::SmoothsARPeriodicHMM) = HMMs.initialization(m.stored)
HMMs.transition_matrix(m::SmoothsARPeriodicHMM, c::Tuple) = HMMs.transition_matrix(m.stored, c)
HMMs.obs_distributions(m::SmoothsARPeriodicHMM, c::Tuple) = HMMs.obs_distributions(m.stored, c)
HMMs.obs_logdensities!(logb::AbstractVector, m::SmoothsARPeriodicHMM, obs, c::Tuple; kwargs...) =
    HMMs.obs_logdensities!(logb, m.stored, obs, c; kwargs...)

# Rewrite the evaluated arrays from the current coefficients, in place.
function _refresh!(m::SmoothsARPeriodicHMM)
    K, T, D = size(m.stored.B, 1), m.nT, size(m.stored.B, 3)
    _transitions!(m.stored.A, m.θᴬ, design_matrix(m.basis_A, 1:T))
    X = design_matrix(m.basis_B, 1:T)
    for k = 1:K, t = 1:T, j = 1:D
        Bt = m.stored.B[k, t, j]
        for (h, θh) in enumerate(m.θᴮ[k, j])
            Bt[h] = Bernoulli(logistic(_linear_predictor(X, t, θh)))
        end
    end
    return m
end

# One method per signature, as for `SmoothARPeriodicHMM`.
rand(rng::AbstractRNG, m::SmoothsARPeriodicHMM, N::Integer; kwargs...) = rand(rng, m.stored, N; kwargs...)
rand(rng::AbstractRNG, m::SmoothsARPeriodicHMM, n2t::_IndexVector; kwargs...) =
    rand(rng, m.stored, n2t; kwargs...)
rand(rng::AbstractRNG, m::SmoothsARPeriodicHMM, z::_IndexVector, n2t::_IndexVector; kwargs...) =
    rand(rng, m.stored, z, n2t; kwargs...)
rand(m::SmoothsARPeriodicHMM, N::Integer; kwargs...) = rand(GLOBAL_RNG, m, N; kwargs...)
rand(m::SmoothsARPeriodicHMM, n2t::_IndexVector; kwargs...) = rand(GLOBAL_RNG, m, n2t; kwargs...)
rand(m::SmoothsARPeriodicHMM, z::_IndexVector, n2t::_IndexVector; kwargs...) =
    rand(GLOBAL_RNG, m, z, n2t; kwargs...)

loglikelihoods(m::SmoothsARPeriodicHMM, args...; kwargs...) = loglikelihoods(m.stored, args...; kwargs...)
viterbi(m::SmoothsARPeriodicHMM, args...; kwargs...) = viterbi(m.stored, args...; kwargs...)
complete_loglikelihood(m::SmoothsARPeriodicHMM, args...; kwargs...) =
    complete_loglikelihood(m.stored, args...; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", m::SmoothsARPeriodicHMM)
    K, D, T, size_orders = size(m.stored)
    println(io, "SmoothsARPeriodicHMM: ", K, " states, ", D, " sites, period ", T,
        ", memory per site ", Int.(log2.(size_orders)))
    println(io, "  transitions  ", m.basis_A, ", emissions ", m.basis_B)
    print(io, "  ", nparams(m), " coefficients")
    return
end
