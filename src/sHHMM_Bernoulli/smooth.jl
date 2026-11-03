# # The site-dependent periodic HMM with its coefficients attached
#
# The same pair as [`SmoothARPeriodicHMM`](@ref) — coefficients plus the evaluation they
# generate — for the variant where each site carries its own autoregressive order.

"""
    SmoothsARPeriodicHMM(a, θᴬ, θᴮ, T)

A periodic autoregressive Bernoulli HMM with site-dependent memory, carrying the coefficients
that generate it.

```julia
m  = SmoothsARPeriodicHMM(a, θᴬ, θᴮ, T)
parameters(m)                  # (A = θᴬ, B = θᴮ)
m, history = fit_mle(m, Y, Y_past; n2t)
sARPeriodicHMM(m)              # the evaluated twin
```

Unlike its sibling, `parameters(m)` is a `NamedTuple` and not a `ComponentArray`: `θᴮ[k, j]` is
a vector of coefficient vectors, one per memory category of site `j`, and the sites may differ
in order — a ragged shape that no flat container holds. Everything else, including the
`(model, history)` return, is the same.
"""
struct SmoothsARPeriodicHMM{A,B,S<:sARPeriodicHMM,C} <: HMMs.AbstractHMM
    θᴬ::A
    θᴮ::B
    stored::S    # concrete: every accessor reads it, once per observation
    nT::Int
    cfg::C
end

function SmoothsARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:Real,3},
    θᴮ::AbstractMatrix, T::Integer; cfg=nothing)
    θᴬ, θᴮ = collect(float.(θᴬ)), deepcopy(θᴮ)
    return SmoothsARPeriodicHMM(θᴬ, θᴮ, Trig2ARPeriodicHMM(a, θᴬ, θᴮ, T), Int(T), cfg)
end

parameters(m::SmoothsARPeriodicHMM) = (A=m.θᴬ, B=m.θᴮ)
nparams(m::SmoothsARPeriodicHMM) =
    length(m.θᴬ) + sum(length(θh) for θ in m.θᴮ for θh in θ)

"""
    sARPeriodicHMM(m::SmoothsARPeriodicHMM) -> sARPeriodicHMM

The evaluated twin: the `A` and `B` arrays the coefficients generate.
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
    if K > 1
        A = m.stored.A
        for k = 1:K, l = 1:K-1, t = 1:T
            A[k, l, t] = exp(polynomial_trigo(t / T, view(m.θᴬ, k, l, :)))
        end
        for k = 1:K, t = 1:T
            A[k, K, t] = 1
        end
        for k = 1:K, t = 1:T
            s = sum(view(A, k, :, t))
            for l = 1:K
                A[k, l, t] /= s
            end
        end
    end
    for k = 1:K, t = 1:T, j = 1:D
        m.stored.B[k, t, j][:] .=
            Bernoulli.([logistic(polynomial_trigo(t / T, θh)) for θh in m.θᴮ[k, j]])
    end
    return m
end

rand(rng::AbstractRNG, m::SmoothsARPeriodicHMM, args...; kwargs...) =
    rand(rng, m.stored, args...; kwargs...)
rand(m::SmoothsARPeriodicHMM, args...; kwargs...) = rand(GLOBAL_RNG, m.stored, args...; kwargs...)

loglikelihoods(m::SmoothsARPeriodicHMM, args...; kwargs...) = loglikelihoods(m.stored, args...; kwargs...)
viterbi(m::SmoothsARPeriodicHMM, args...; kwargs...) = viterbi(m.stored, args...; kwargs...)
complete_loglikelihood(m::SmoothsARPeriodicHMM, args...; kwargs...) =
    complete_loglikelihood(m.stored, args...; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", m::SmoothsARPeriodicHMM)
    K, D, T, size_orders = size(m.stored)
    println(io, "SmoothsARPeriodicHMM: ", K, " states, ", D, " sites, period ", T,
        ", memory per site ", Int.(log2.(size_orders)))
    print(io, "  ", nparams(m), " coefficients")
    return
end
