# # The spatial periodic HMM with its coefficients attached
#
# The same pair as [`SmoothARPeriodicHMM`](@ref) — coefficients plus the evaluation they
# generate — for the model whose sites are tied together by a latent Gaussian field.

"""
    SmoothARPeriodicHMMSpatial(a, θᴬ, θᴮ, θᴿ, T, h; QMC_m = 30, basis_A, basis_B, basis_R)

A periodic autoregressive spatial HMM given by its coefficients. These are `θᴬ`
(`K × (K-1) × nA`, multinomial logit) for the transitions, `θᴮ` (`K × D × 2^order × nB`, logistic) for the marginal
success probabilities, and `θᴿ` (`K × nR`, `exp`) for the spatial range of each state, with the
bases of the coefficient method of [`ARPeriodicHMMSpatial`](@ref), trigonometric by default.
`h` is the `D × D` distance matrix between sites and `QMC_m` the number of quasi-Monte-Carlo
points of the likelihood.

```julia
m  = SmoothARPeriodicHMMSpatial(a, θᴬ, θᴮ, θᴿ, T, distance)
ps = parameters(m)                 # ComponentArray(A = θᴬ, B = θᴮ, R = θᴿ)

m, history = fit_mle(m, Y, Y_past; n2t, solver)
ARPeriodicHMMSpatial(m)            # the evaluated model
curve(m, :R)                       # K × T, the spatial range over the period
```

`fit_mle` returns the iterate with the highest loglikelihood, not the last one.
"""
struct SmoothARPeriodicHMMSpatial{P<:AbstractVector,S,BA,BB,BR,C} <: HMMs.AbstractHMM
    ps::P
    stored::S
    nT::Int
    basis_A::BA
    basis_B::BB
    basis_R::BR
    cfg::C
end

function SmoothARPeriodicHMMSpatial(a::AbstractVector, θᴬ::AbstractArray{<:Real,3},
    θᴮ::AbstractArray{<:Real,4}, θᴿ::AbstractArray{<:Real,2}, T::Integer,
    h::AbstractMatrix; QMC_m::Integer=30,
    basis_A=TrigBasis(T, _trig_order(size(θᴬ, 3), "θᴬ")),
    basis_B=TrigBasis(T, _trig_order(size(θᴮ, 4), "θᴮ")),
    basis_R=TrigBasis(T, _trig_order(size(θᴿ, 2), "θᴿ")), cfg=nothing)
    ps = ComponentArray(A=collect(float.(θᴬ)), B=collect(float.(θᴮ)), R=collect(float.(θᴿ)))
    stored = ARPeriodicHMMSpatial(a, ps.A, ps.B, ps.R, T, h; QMC_m, basis_A, basis_B, basis_R)
    return SmoothARPeriodicHMMSpatial(ps, stored, Int(T), basis_A, basis_B, basis_R, cfg)
end

parameters(m::SmoothARPeriodicHMMSpatial) = m.ps

"""
    ARPeriodicHMMSpatial(m::SmoothARPeriodicHMMSpatial) -> ARPeriodicHMMSpatial

The `ARPeriodicHMMSpatial` with the `A`, `B` and `R` arrays the coefficients of `m` generate.
"""
ARPeriodicHMMSpatial(m::SmoothARPeriodicHMMSpatial) = _current(m)

size(m::SmoothARPeriodicHMMSpatial, dim=:) = size(m.stored, dim)

degrees(m::SmoothARPeriodicHMMSpatial) =
    (_trig_degree(m.basis_A), _trig_degree(m.basis_B), _trig_degree(m.basis_R))

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
    return _current(m).R
end

# Rewrite the evaluated arrays from the current coefficients, in place.
function _refresh!(m::SmoothARPeriodicHMMSpatial)
    K, T = size(m.stored.A, 1), m.nT
    _transitions!(m.stored.A, m.ps.A, design_matrix(m.basis_A, 1:T))
    XB, XR = design_matrix(m.basis_B, 1:T), design_matrix(m.basis_R, 1:T)
    D, size_order = size(m.ps.B, 2), size(m.ps.B, 3)
    for k = 1:K, t = 1:T, j = 1:D, hh = 1:size_order
        m.stored.B[k, t, j, hh] = logistic(_linear_predictor(XB, t, view(m.ps.B, k, j, hh, :)))
    end
    for k = 1:K, t = 1:T
        m.stored.R[k, t] = exp(_linear_predictor(XR, t, view(m.ps.R, k, :)))
    end
    return m
end

# One method per signature, as for `SmoothARPeriodicHMM`.
rand(rng::AbstractRNG, m::SmoothARPeriodicHMMSpatial, N::Integer; kwargs...) =
    rand(rng, _current(m), N; kwargs...)
rand(rng::AbstractRNG, m::SmoothARPeriodicHMMSpatial, n2t::_IndexVector; kwargs...) =
    rand(rng, _current(m), n2t; kwargs...)
rand(rng::AbstractRNG, m::SmoothARPeriodicHMMSpatial, z::_IndexVector, n2t::_IndexVector; kwargs...) =
    rand(rng, _current(m), z, n2t; kwargs...)
rand(m::SmoothARPeriodicHMMSpatial, N::Integer; kwargs...) = rand(GLOBAL_RNG, m, N; kwargs...)
rand(m::SmoothARPeriodicHMMSpatial, n2t::_IndexVector; kwargs...) = rand(GLOBAL_RNG, m, n2t; kwargs...)
rand(m::SmoothARPeriodicHMMSpatial, z::_IndexVector, n2t::_IndexVector; kwargs...) =
    rand(GLOBAL_RNG, m, z, n2t; kwargs...)

loglikelihoods(m::SmoothARPeriodicHMMSpatial, args...; kwargs...) =
    loglikelihoods(_current(m), args...; kwargs...)
viterbi(m::SmoothARPeriodicHMMSpatial, args...; kwargs...) = viterbi(_current(m), args...; kwargs...)
complete_loglikelihood(m::SmoothARPeriodicHMMSpatial, args...; kwargs...) =
    complete_loglikelihood(_current(m), args...; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", m::SmoothARPeriodicHMMSpatial)
    K, D, T, size_order = size(m.stored)
    println(io, "SmoothARPeriodicHMMSpatial: ", K, " states, ", D, " sites, period ", T,
        ", memory ", Int(log2(size_order)))
    println(io, "  transitions  ", m.basis_A, ", ", length(m.ps.A), " coefficients")
    println(io, "  emissions    ", m.basis_B, ", ", length(m.ps.B), " coefficients")
    print(io, "  range        ", m.basis_R, ", ", length(m.ps.R),
        " coefficients, QMC budget ", m.stored.QMC_m)
    return
end
