# # The spatial periodic HMM with its coefficients attached
#
# The same pair as [`SmoothARPeriodicHMM`](@ref) — coefficients plus the evaluation they
# generate — for the model whose sites are tied together by a latent Gaussian field.

"""
    SmoothARPeriodicHMMSpatial(a, θᴬ, θᴮ, θᴿ, T, h; QMC_m = 30)

A periodic autoregressive spatial HMM carrying the coefficients that generate it.

```julia
m  = SmoothARPeriodicHMMSpatial(a, θᴬ, θᴮ, θᴿ, T, distance)
ps = parameters(m)                 # ComponentArray(A = θᴬ, B = θᴮ, R = θᴿ)
ps.R                               # K × (2d+1): the range's own coefficients

m, history = fit_mle(m, Y, Y_past; n2t)
ARPeriodicHMMSpatial(m)            # the evaluated twin, to simulate from and plot
curve(m, :R)                       # K × T, the spatial range over the period
```

All three coefficient blocks are rectangular — `K×(K-1)×(2d+1)`, `K×D×2^order×(2d+1)` and
`K×(2d+1)` — so unlike [`SmoothsARPeriodicHMM`](@ref) this one does hold them in a
`ComponentArray`. The three links are the package's usual ones: a multinomial logit for the
transitions, a logistic for the marginal rain probabilities, and `exp` for the range, which must
stay positive.

`fit_mle` returns the **best** iterate rather than the last; see its docstring.
"""
struct SmoothARPeriodicHMMSpatial{P<:AbstractVector,S,C} <: HMMs.AbstractHMM
    ps::P
    stored::S
    nT::Int
    cfg::C
end

function SmoothARPeriodicHMMSpatial(a::AbstractVector, θᴬ::AbstractArray{<:Real,3},
    θᴮ::AbstractArray{<:Real,4}, θᴿ::AbstractArray{<:Real,2}, T::Integer,
    h::AbstractMatrix; QMC_m::Integer=30, cfg=nothing)
    ps = ComponentArray(A=collect(float.(θᴬ)), B=collect(float.(θᴮ)), R=collect(float.(θᴿ)))
    stored = Trig2ARPeriodicHMMSpatial(a, ps.A, ps.B, ps.R, T, h; QMC_m)
    return SmoothARPeriodicHMMSpatial(ps, stored, Int(T), cfg)
end

parameters(m::SmoothARPeriodicHMMSpatial) = m.ps

"""
    ARPeriodicHMMSpatial(m::SmoothARPeriodicHMMSpatial) -> ARPeriodicHMMSpatial

The evaluated twin: the `A`, `B` and `R` arrays the coefficients generate.
"""
ARPeriodicHMMSpatial(m::SmoothARPeriodicHMMSpatial) = m.stored

size(m::SmoothARPeriodicHMMSpatial, dim=:) = size(m.stored, dim)

degrees(m::SmoothARPeriodicHMMSpatial) = ((size(m.ps.A, 3) - 1) ÷ 2,
    (size(m.ps.B, 4) - 1) ÷ 2, (size(m.ps.R, 2) - 1) ÷ 2)

HMMs.initialization(m::SmoothARPeriodicHMMSpatial) = HMMs.initialization(m.stored)
HMMs.transition_matrix(m::SmoothARPeriodicHMMSpatial, c::Tuple) =
    HMMs.transition_matrix(m.stored, c)
HMMs.obs_distributions(m::SmoothARPeriodicHMMSpatial, c::Tuple) =
    HMMs.obs_distributions(m.stored, c)
HMMs.obs_logdensities!(logb::AbstractVector, m::SmoothARPeriodicHMMSpatial, obs, c::Tuple; kwargs...) =
    HMMs.obs_logdensities!(logb, m.stored, obs, c; kwargs...)

"""
    curve(m::SmoothARPeriodicHMMSpatial, :R) -> Matrix

The spatial range `R[k, t]` over one period, one row per hidden state.
"""
function TimeVaryingParameters.curve(m::SmoothARPeriodicHMMSpatial, name::Symbol)
    name === :R || throw(ArgumentError(
        "`$name` is not a curve of this model; `:R` is the spatial range, and the transition " *
        "and emission arrays are `ARPeriodicHMMSpatial(m).A` and `.B`"))
    return m.stored.R
end

# Rewrite the evaluated arrays from the current coefficients, in place.
function _refresh!(m::SmoothARPeriodicHMMSpatial)
    K, T = size(m.stored.A, 1), m.nT
    if K > 1
        A = m.stored.A
        for k = 1:K, l = 1:K-1, t = 1:T
            A[k, l, t] = exp(polynomial_trigo(t / T, view(m.ps.A, k, l, :)))
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
    D, size_order = size(m.ps.B, 2), size(m.ps.B, 3)
    for k = 1:K, t = 1:T, j = 1:D, hh = 1:size_order
        m.stored.B[k, t, j, hh] = logistic(polynomial_trigo(t / T, view(m.ps.B, k, j, hh, :)))
    end
    for k = 1:K, t = 1:T
        m.stored.R[k, t] = exp(polynomial_trigo(t / T, view(m.ps.R, k, :)))
    end
    return m
end

rand(rng::AbstractRNG, m::SmoothARPeriodicHMMSpatial, args...; kwargs...) =
    rand(rng, m.stored, args...; kwargs...)
rand(m::SmoothARPeriodicHMMSpatial, args...; kwargs...) =
    rand(GLOBAL_RNG, m.stored, args...; kwargs...)

loglikelihoods(m::SmoothARPeriodicHMMSpatial, args...; kwargs...) =
    loglikelihoods(m.stored, args...; kwargs...)
viterbi(m::SmoothARPeriodicHMMSpatial, args...; kwargs...) = viterbi(m.stored, args...; kwargs...)
complete_loglikelihood(m::SmoothARPeriodicHMMSpatial, args...; kwargs...) =
    complete_loglikelihood(m.stored, args...; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", m::SmoothARPeriodicHMMSpatial)
    K, D, T, size_order = size(m.stored)
    dA, dB, dR = degrees(m)
    println(io, "SmoothARPeriodicHMMSpatial: ", K, " states, ", D, " sites, period ", T,
        ", memory ", Int(log2(size_order)))
    println(io, "  transitions  degree ", dA, ", ", length(m.ps.A), " coefficients")
    println(io, "  emissions    degree ", dB, ", ", length(m.ps.B), " coefficients")
    print(io, "  range        degree ", dR, ", ", length(m.ps.R),
        " coefficients, QMC budget ", m.stored.QMC_m)
    return
end
