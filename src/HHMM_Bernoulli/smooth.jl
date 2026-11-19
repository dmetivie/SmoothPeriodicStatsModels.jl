# # The periodic Bernoulli HMM with its coefficients attached
#
# `ARPeriodicHMM` stores the *evaluated* arrays `A[k, l, t]` and `B[k, t, j, h]`. What generates
# them is a pair of basis expansions over the period index — `θᴬ` through a multinomial logit,
# `θᴮ` through a logistic — and until v4 those travelled loose beside the model, which is why
# fitting one returned six values.
#
# `SmoothARPeriodicHMM` is the pair: the coefficients `ps` and the evaluation they generate,
# held together. It is the `HiddenMarkovModels.AbstractHMM` the Baum-Welch loop iterates on, and
# the evaluated twin is one field away.

"""
    SmoothARPeriodicHMM(a, θᴬ, θᴮ, T; basis_A, basis_B)

A periodic autoregressive Bernoulli HMM given by its coefficients. The coefficients have the
layout and the bases of the coefficient method of [`ARPeriodicHMM`](@ref): trigonometric polynomials by default, any
TimeVaryingParameters basis otherwise. It is a `HiddenMarkovModels.AbstractHMM` whose control
at step `n` is `(n2t[n], memory categories, n)`, see [`SeasonalControls`](@ref).

```julia
m  = SmoothARPeriodicHMM(a, θᴬ, θᴮ, T)
ps = parameters(m)              # ComponentArray(A = θᴬ, B = θᴮ)
ps.B[k, j, h, :]                # one emission's coefficients

m, history = fit_mle(m, Y, Y_past; n2t, solver)
ARPeriodicHMM(m)                # the evaluated model
rand(m, n2t; Y_past, seq = true) # simulate
viterbi(m, Y, Y_past; n2t)      # decode

m = SmoothARPeriodicHMM(a, θᴬ, θᴮ, T; basis_B = CyclicBSplineBasis(T, 8))  # splines for B
```

`rand`, `viterbi`, `loglikelihoods`, `complete_loglikelihood` and `size` act as on
`ARPeriodicHMM(m)`. The bases are the fields `basis_A` and `basis_B`.
"""
struct SmoothARPeriodicHMM{P<:AbstractVector,S<:ARPeriodicHMM,BA,BB,C} <: HMMs.AbstractHMM
    ps::P
    stored::S    # the evaluated model, a cache (see `_current`); concrete, as every accessor reads it
    nT::Int
    basis_A::BA
    basis_B::BB
    cfg::C
    stored_from::Base.RefValue{P}   # the coefficients `stored` was evaluated from
end

# `stored` evaluated from `ps` as they are now.
SmoothARPeriodicHMM(ps, stored, nT, basis_A, basis_B, cfg) =
    SmoothARPeriodicHMM(ps, stored, nT, basis_A, basis_B, cfg, Ref(deepcopy(ps)))

function SmoothARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:Real,3},
    θᴮ::AbstractArray{<:Real,4}, T::Integer;
    basis_A=TrigBasis(T, _trig_order(size(θᴬ, 3), "θᴬ")),
    basis_B=TrigBasis(T, _trig_order(size(θᴮ, 4), "θᴮ")), cfg=nothing)
    ps = ComponentArray(A=collect(float.(θᴬ)), B=collect(float.(θᴮ)))
    stored = ARPeriodicHMM(a, ps.A, ps.B, T; basis_A, basis_B)
    return SmoothARPeriodicHMM(ps, stored, Int(T), basis_A, basis_B, cfg)
end

parameters(m::SmoothARPeriodicHMM) = m.ps

"""
    ARPeriodicHMM(m::SmoothARPeriodicHMM) -> ARPeriodicHMM

The `ARPeriodicHMM` with the `A` and `B` arrays the coefficients of `m` generate.
"""
ARPeriodicHMM(m::SmoothARPeriodicHMM) = copy(_current(m))

"""
    degrees(m) -> Tuple

The degrees of the trigonometric bases of a smooth HMM: `(dA, dB)`, and `(dA, dB, dR)` for the
spatial model. Throws for a basis that is not a `TrigBasis`.
"""
degrees(m::SmoothARPeriodicHMM) = (_trig_degree(m.basis_A), _trig_degree(m.basis_B))

size(m::SmoothARPeriodicHMM, dim=:) = size(m.stored, dim)

# Rewrite the evaluated arrays from the coefficients, in place.
function _evaluate!(m::SmoothARPeriodicHMM)
    T = m.nT
    _transitions!(m.stored.A, m.ps.A, design_matrix(m.basis_A, 1:T))
    _emissions!(m.stored.B, m.ps.B, design_matrix(m.basis_B, 1:T))
    return m
end

# One method per signature: a single `args...` method would be ambiguous with the `rand` methods
# of `Random` and of `HiddenMarkovModels`.
const _IndexVector = AbstractVector{<:Integer}
rand(rng::AbstractRNG, m::SmoothARPeriodicHMM, N::Integer; kwargs...) = rand(rng, _current(m), N; kwargs...)
rand(rng::AbstractRNG, m::SmoothARPeriodicHMM, n2t::_IndexVector; kwargs...) =
    rand(rng, _current(m), n2t; kwargs...)
rand(rng::AbstractRNG, m::SmoothARPeriodicHMM, z::_IndexVector, n2t::_IndexVector; kwargs...) =
    rand(rng, _current(m), z, n2t; kwargs...)
rand(m::SmoothARPeriodicHMM, N::Integer; kwargs...) = rand(GLOBAL_RNG, m, N; kwargs...)
rand(m::SmoothARPeriodicHMM, n2t::_IndexVector; kwargs...) = rand(GLOBAL_RNG, m, n2t; kwargs...)
rand(m::SmoothARPeriodicHMM, z::_IndexVector, n2t::_IndexVector; kwargs...) =
    rand(GLOBAL_RNG, m, z, n2t; kwargs...)

loglikelihoods(m::SmoothARPeriodicHMM, args...; kwargs...) = loglikelihoods(_current(m), args...; kwargs...)
viterbi(m::SmoothARPeriodicHMM, args...; kwargs...) = viterbi(_current(m), args...; kwargs...)
complete_loglikelihood(m::SmoothARPeriodicHMM, args...; kwargs...) =
    complete_loglikelihood(_current(m), args...; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", m::SmoothARPeriodicHMM)
    K, D, T, size_order = size(m.stored)
    println(io, "SmoothARPeriodicHMM: ", K, " states, ", D, " sites, period ", T,
        ", memory ", Int(log2(size_order)))
    println(io, "  transitions  ", m.basis_A, ", ", length(m.ps.A), " coefficients")
    print(io, "  emissions    ", m.basis_B, ", ", length(m.ps.B), " coefficients")
    return
end
