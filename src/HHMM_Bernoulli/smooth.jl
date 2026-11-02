# # The periodic Bernoulli HMM with its coefficients attached
#
# `ARPeriodicHMM` stores the *evaluated* arrays `A[k, l, t]` and `B[k, t, j, h]`. What generates
# them is a pair of trigonometric polynomials — `θᴬ` through a multinomial logit, `θᴮ` through a
# logistic — and until v4 those travelled loose beside the model, which is why fitting one
# returned six values.
#
# `SmoothARPeriodicHMM` is the pair: the coefficients `ps` and the evaluation they generate,
# held together. It is the `HiddenMarkovModels.AbstractHMM` the Baum-Welch loop iterates on, and
# the evaluated twin is one field away.

"""
    SmoothARPeriodicHMM(a, θᴬ, θᴮ, T)
    SmoothARPeriodicHMM(hmm::ARPeriodicHMM, θᴬ, θᴮ)

A periodic autoregressive Bernoulli HMM carrying the coefficients that generate it.

```julia
m  = SmoothARPeriodicHMM(a, θᴬ, θᴮ, T)
ps = parameters(m)              # ComponentArray(A = θᴬ, B = θᴮ)
ps.B[k, j, h, :]                # one emission's trigonometric coefficients

m, history = fit_mle(m, Y, Y_past; n2t)    # same type in, same type out
ARPeriodicHMM(m)                # the evaluated twin, to simulate from
```

`A(t)` is the transition matrix of the step *leaving* period `t`, this package's convention;
[`SeasonalControls`](@ref) explains how that meets `HiddenMarkovModels`'.

The `cfg` field is what the M-step needs beyond the parameters — the solver settings and the
observation index sets, which depend on the data and are therefore attached by `fit_mle` rather
than by the constructor.
"""
struct SmoothARPeriodicHMM{P<:AbstractVector,C} <: HMMs.AbstractHMM
    ps::P
    stored::ARPeriodicHMM
    nT::Int
    cfg::C
end

function SmoothARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:Real,3},
    θᴮ::AbstractArray{<:Real,4}, T::Integer; cfg=nothing)
    ps = ComponentArray(A=collect(float.(θᴬ)), B=collect(float.(θᴮ)))
    return SmoothARPeriodicHMM(ps, Trig2ARPeriodicHMM(a, ps.A, ps.B, T), Int(T), cfg)
end

parameters(m::SmoothARPeriodicHMM) = m.ps

"""
    ARPeriodicHMM(m::SmoothARPeriodicHMM) -> ARPeriodicHMM

The evaluated twin: the `A` and `B` arrays the coefficients generate. This is what `rand`,
`viterbi` and the plots take — `SmoothARPeriodicHMM` carries the coefficients, this carries the
numbers they produce.
"""
ARPeriodicHMM(m::SmoothARPeriodicHMM) = m.stored

"""
    degrees(m::SmoothARPeriodicHMM) -> (dA, dB)

The trigonometric degrees of the transition and emission coefficients.
"""
degrees(m::SmoothARPeriodicHMM) =
    ((size(m.ps.A, 3) - 1) ÷ 2, (size(m.ps.B, 4) - 1) ÷ 2)

size(m::SmoothARPeriodicHMM, dim=:) = size(m.stored, dim)

# The three accessors forward to the evaluated twin, which `_refresh!` keeps in step with the
# coefficients. Evaluating `A` and `B` on every call instead would redo the whole period's
# trigonometry once per observation.
HMMs.initialization(m::SmoothARPeriodicHMM) = HMMs.initialization(m.stored)
HMMs.transition_matrix(m::SmoothARPeriodicHMM, c::Tuple) = HMMs.transition_matrix(m.stored, c)
HMMs.obs_distributions(m::SmoothARPeriodicHMM, c::Tuple) = HMMs.obs_distributions(m.stored, c)
HMMs.obs_logdensities!(logb::AbstractVector, m::SmoothARPeriodicHMM, obs, c::Tuple; kwargs...) =
    HMMs.obs_logdensities!(logb, m.stored, obs, c; kwargs...)

# Rewrite the evaluated arrays from the current coefficients, in place: the M-step changes `ps`
# and every accessor reads `stored`.
function _refresh!(m::SmoothARPeriodicHMM)
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
    for k = 1:K, t = 1:T, j = 1:D, h = 1:size_order
        m.stored.B[k, t, j, h] = Bernoulli(logistic(polynomial_trigo(t / T, view(m.ps.B, k, j, h, :))))
    end
    return m
end

rand(rng::AbstractRNG, m::SmoothARPeriodicHMM, args...; kwargs...) =
    rand(rng, m.stored, args...; kwargs...)
rand(m::SmoothARPeriodicHMM, args...; kwargs...) = rand(GLOBAL_RNG, m.stored, args...; kwargs...)

loglikelihoods(m::SmoothARPeriodicHMM, args...; kwargs...) = loglikelihoods(m.stored, args...; kwargs...)
viterbi(m::SmoothARPeriodicHMM, args...; kwargs...) = viterbi(m.stored, args...; kwargs...)
complete_loglikelihood(m::SmoothARPeriodicHMM, args...; kwargs...) =
    complete_loglikelihood(m.stored, args...; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", m::SmoothARPeriodicHMM)
    K, D, T, size_order = size(m.stored)
    dA, dB = degrees(m)
    println(io, "SmoothARPeriodicHMM: ", K, " states, ", D, " sites, period ", T,
        ", memory ", Int(log2(size_order)))
    println(io, "  transitions  degree ", dA, ", ", length(m.ps.A), " coefficients")
    print(io, "  emissions    degree ", dB, ", ", length(m.ps.B), " coefficients")
    return
end
