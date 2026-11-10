# # The periodic HMMs, as `HiddenMarkovModels.jl` models
#
# A periodic HMM is a *controlled* HMM whose control is the period index: `HiddenMarkovModels`
# documents exactly that case, and once a model answers its three questions —
# `initialization`, `transition_matrix` and `obs_distributions` — the whole library applies to
# it. `logdensityof`, `forward`, `forward_backward`, `viterbi` and `baum_welch` then come from
# there instead of being written here.
#
# For the autoregressive models the control carries one more thing: the memory category of
# every site, which the observations themselves determine. That is what [`SeasonalControls`](@ref)
# holds.

"""
    SeasonalControls(n2t, lag_cat)

The control sequence of a periodic autoregressive HMM: `(n2t[n], lag_cat[:, n], n, n2t[n-1])`
at step `n`.

`n2t[n] ∈ 1:T` is the period index and `lag_cat[j, n]` the memory category of site `j`, the
`1 + ` binary encoding of its `order` previous values that [`conditional_to`](@ref) builds.
The third element is the observation index, used by [`CachedEmissions`](@ref). The fourth is
the period index of the previous observation, which the transition into `n` leaves; at `n = 1`,
where no transition arrives, it is `n2t[1]`.

It is the `control_seq` to pass to `HiddenMarkovModels` functions such as `forward_backward`
or `logdensityof`. The tuples are built on access, not stored.
"""
struct SeasonalControls{L<:AbstractMatrix{<:Integer},E} <: AbstractVector{E}
    n2t::Vector{Int}
    lag_cat::L
end

function SeasonalControls(n2t::AbstractVector{<:Integer}, lag_cat::AbstractMatrix{<:Integer})
    @argcheck length(n2t) == size(lag_cat, 2) "`n2t` has $(length(n2t)) entries and `lag_cat` $(size(lag_cat, 2)) columns; both index the observations"
    E = Tuple{Int,typeof(view(lag_cat, :, 1)),Int,Int}
    return SeasonalControls{typeof(lag_cat),E}(collect(Int, n2t), lag_cat)
end

"""
    SeasonalControls(Y, Y_past; n2t)
    SeasonalControls(Ys, Y_pasts; n2t)

The control sequence of the `D × N` observations `Y` with initial observations `Y_past`, the
memory categories computed with [`conditional_to`](@ref). With vectors of sequences, the
controls of all of them end to end, in the layout of the multi-sequence [`fit_mle`](@ref); give
`HiddenMarkovModels` the matching `seq_ends = cumsum(size.(Ys, 2))`.
"""
SeasonalControls(Y::AbstractMatrix{<:Bool}, Y_past::AbstractArray; n2t::AbstractVector{<:Integer}) =
    SeasonalControls(n2t, conditional_to(Y, Y_past))

Base.size(c::SeasonalControls) = (length(c.n2t),)
Base.@propagate_inbounds Base.getindex(c::SeasonalControls, n::Integer) =
    (c.n2t[n], view(c.lag_cat, :, n), Int(n), c.n2t[max(n - 1, 1)])

# ## The three accessors
#
# ### Which period index the transition matrix is read at
#
# This package, following Touron, indexes `A` by the period the step *leaves*: the complete
# loglikelihood multiplies `A[z[n], z[n+1], n2t[n]]`, whatever `n2t[n+1]` is. `HiddenMarkovModels`
# asks instead for the matrix of the step that *arrives* at a control — `forward_backward!` builds
# `ξ[n]` with `transition_matrix(hmm, control_seq[n + 1])`, its convention since v0.6. So each
# control also carries the period index of the observation before it, and the accessor reads `A`
# there. Stored `θᴬ` keeps its published meaning.
#
# Reading `A` one period index before the arriving one, `mod1(t - 1, T)`, would agree only when
# `n2t` steps by one, and it need not: with `T = 366` and February 29 at index 60, a non-leap year
# goes from 59 to 61 in one day, and that step leaves February 28.

HMMs.initialization(hmm::ARPeriodicHMM) = hmm.a

"""
    transition_matrix(hmm::ARPeriodicHMM, control) -> AbstractMatrix

The transition matrix of the step arriving at `control`, as `HiddenMarkovModels` asks for it:
`A[:, :, s]`, where `s = control[4]` is the period index of the previous observation, the one
the step leaves. The elements of a [`SeasonalControls`](@ref) carry it. Also defined for
`sARPeriodicHMM` and `ARPeriodicHMMSpatial`.
"""
HMMs.transition_matrix(hmm::ARPeriodicHMM, c::Tuple) = view(hmm.A, :, :, _leaving(c))

# The period index the transition into a control leaves.
_leaving(c::NTuple{4,Any}) = c[4]
_leaving(::Tuple) = throw(ArgumentError(
    "a control of a periodic HMM is `(t, h, n, s)`, with `s` the period index of the previous " *
    "observation, which the transition leaves; build the controls with `SeasonalControls`"))

"""
    obs_distributions(hmm::ARPeriodicHMM, (t, h)) -> Vector

One distribution per hidden state at period index `t` with memory categories `h`: the product
over sites of `B[k, t, j, h[j]]`, which is the conditional independence this model assumes.
"""
function HMMs.obs_distributions(hmm::ARPeriodicHMM, (t, h)::Tuple)
    K, D = size(hmm.B, 1), size(hmm.B, 3)
    return [product_distribution([hmm.B[k, t, j, h[j]] for j = 1:D]) for k = 1:K]
end

"""
    obs_logdensities!(logb, hmm::ARPeriodicHMM, obs, (t, h))

Fill `logb[k]` with the loglikelihood of `obs` in state `k`, without building distributions.
Also defined for `sARPeriodicHMM`.
"""
function HMMs.obs_logdensities!(logb::AbstractVector{T}, hmm::ARPeriodicHMM, obs,
    (t, h)::Tuple; error_if_not_finite::Bool=true) where {T}
    _obs_logdensities!(logb, hmm.B, obs, t, h)
    error_if_not_finite && @argcheck maximum(logb) < typemax(T)
    return nothing
end

# The inner loop, the same arithmetic as `loglikelihoods!`. This was once a function barrier
# against `ARPeriodicHMM.B` being abstractly typed; it moved the dynamic dispatch out of the
# `N·K·D` `logpdf` calls but still paid one per observation. `B`'s type is a parameter of the
# model now, so the whole path is static.
function _obs_logdensities!(logb::AbstractVector{T}, B::AbstractArray{F,4}, obs,
    t::Integer, h) where {T,F<:UnivariateDistribution}
    K, D = size(B, 1), size(B, 3)
    @inbounds for k in OneTo(K)
        ll = zero(T)
        for j in OneTo(D)
            ll += logpdf(B[k, t, j, h[j]], obs[j])
        end
        logb[k] = ll
    end
    return nothing
end

# ## The site-dependent variant
#
# Identical in structure — the control is the same pair, the emission is the same product over
# sites — except that each site carries its own autoregressive order, so `B[k, t, j]` is a
# vector indexed by that site's memory category rather than a slice of a rectangular array.

HMMs.initialization(hmm::sARPeriodicHMM) = hmm.a

HMMs.transition_matrix(hmm::sARPeriodicHMM, c::Tuple) = view(hmm.A, :, :, _leaving(c))

function HMMs.obs_distributions(hmm::sARPeriodicHMM, (t, h)::Tuple)
    K, D = size(hmm.B, 1), size(hmm.B, 3)
    return [product_distribution([hmm.B[k, t, j][h[j]] for j = 1:D]) for k = 1:K]
end

function HMMs.obs_logdensities!(logb::AbstractVector{T}, hmm::sARPeriodicHMM, obs,
    (t, h)::Tuple; error_if_not_finite::Bool=true) where {T}
    _obs_logdensities_sites!(logb, hmm.B, obs, t, h)
    error_if_not_finite && @argcheck maximum(logb) < typemax(T)
    return nothing
end

# The inner loop, as `_obs_logdensities!` above.
function _obs_logdensities_sites!(logb::AbstractVector{T}, B::AbstractArray{V,3}, obs,
    t::Integer, h) where {T,V<:AbstractVector{<:UnivariateDistribution}}
    K, D = size(B, 1), size(B, 3)
    @inbounds for k in OneTo(K)
        ll = zero(T)
        for j in OneTo(D)
            ll += logpdf(B[k, t, j][h[j]], obs[j])
        end
        logb[k] = ll
    end
    return nothing
end

# ## The spatial variant
#
# Here the sites are *not* conditionally independent: the emission is a latent Gaussian field
# with an exponential kernel of range `R[k, t]`, so one observation's likelihood is a
# `D`-dimensional orthant probability, estimated by quasi-Monte Carlo. Two consequences.
#
# First, the emission is evaluated in a batched, threaded sweep rather than one control at a
# time — see [`CachedEmissions`](@ref), which is what the observation index in the control is
# for. The accessors below ignore that index.
#
# Second, the QMC budget lives on the model (`hmm.QMC_m`), since these signatures have nowhere
# to put a keyword.

HMMs.initialization(hmm::ARPeriodicHMMSpatial) = hmm.a

HMMs.transition_matrix(hmm::ARPeriodicHMMSpatial, c::Tuple) = view(hmm.A, :, :, _leaving(c))

"""
    obs_distributions(hmm::ARPeriodicHMMSpatial, (t, h, _...)) -> Vector{SpatialBernoulli}

One [`SpatialBernoulli`](@ref) per hidden state: marginal success probabilities
`B[k, t, j, h[j]]` tied together by a latent Gaussian field whose correlation is
`exp(-distance / R[k, t])`.
"""
function HMMs.obs_distributions(hmm::ARPeriodicHMMSpatial, c::Tuple)
    t, h = c[1], c[2]
    K, D = size(hmm.B, 1), size(hmm.B, 3)
    return [SpatialBernoulli(hmm.R[k, t], hmm.B[CartesianIndex.(k, t, 1:D, h)], hmm.h,
        expkernel.(hmm.h; range=hmm.R[k, t])) for k = 1:K]
end

"""
    obs_logdensities!(logb, hmm::ARPeriodicHMMSpatial, obs, (t, h, _...))

Fill `logb[k]` with the loglikelihood of `obs` in state `k`, the orthant probability that
[`loglikelihoods!`](@ref) computes, for one observation. For many observations,
`loglikelihoods!` is faster: it runs over observations in parallel.
"""
function HMMs.obs_logdensities!(logb::AbstractVector{T}, hmm::ARPeriodicHMMSpatial, obs,
    c::Tuple; error_if_not_finite::Bool=true) where {T}
    t, h = c[1], c[2]
    K, D = size(hmm.B, 1), size(hmm.B, 3)
    zerosvec = zeros(D)
    finite_bounds = zeros(D)
    @inbounds for k in OneTo(K)
        finite_bounds .= quantile.(Normal(), hmm.B[CartesianIndex.(k, t, 1:D, h)])
        a = ifelse.(obs .== 0, finite_bounds, -Inf)
        b = ifelse.(obs .== 1, finite_bounds, Inf)
        hy = mvnormcdf(zerosvec, expkernel.(hmm.h; range=hmm.R[k, t]), a, b; m=D * hmm.QMC_m)
        logb[k] = log(hy[1])
    end
    error_if_not_finite && @argcheck maximum(logb) < typemax(T)
    return nothing
end

# `rand` is this package's own and stays that way. `HiddenMarkovModels` draws from
# `obs_distributions(hmm, control)` with the control sequence fixed before the first draw,
# which cannot work here: the memory category of site `j` at step `n` is a function of the
# observation this very simulation is about to produce. Rather than let that return quietly
# wrong draws, say so.
@noinline function Base.rand(::AbstractRNG,
    ::Union{ARPeriodicHMM,sARPeriodicHMM,ARPeriodicHMMSpatial},
    ::SeasonalControls{<:Any,<:Tuple})
    throw(ArgumentError("""
    an autoregressive periodic HMM cannot be simulated from a fixed control sequence: the
    memory categories in `SeasonalControls` are a function of the observations, which do not
    exist yet. Use this package's own

        z, y = rand(hmm, n2t; Y_past = …, z_ini = …, seq = true)

    which encodes each observation's memory as it goes. `SeasonalControls` is for inference,
    where the observations are given."""))
end

# ## Batched emissions
#
# Every model here computes its emission loglikelihoods for all `N` observations at once —
# `loglikelihoods!` fills an `N × K` matrix, and for the spatial model it does so in parallel.
# `HiddenMarkovModels`' recursions are necessarily sequential and ask for one control at a
# time, which for the spatial emission costs an order of magnitude (measured 8–10× at
# `D = 10..20`, `N = 300`, 28 threads; caching the `D × D` kernel recovers none of it, because
# `mvnormcdf` dominates). This type is how both are had at once.

"""
    CachedEmissions(model, LL)

An `AbstractHMM` that takes `initialization`, `transition_matrix` and `obs_distributions` from
`model` but reads its emission loglikelihoods from `LL`, an `N × K` matrix computed beforehand
(for example with [`loglikelihoods!`](@ref)). Row `n` is found from the observation index, the
third element of each [`SeasonalControls`](@ref) control. `LL` is not updated when `model`
changes.
"""
struct CachedEmissions{M,L<:AbstractMatrix} <: HMMs.AbstractHMM
    model::M
    LL::L
end

HMMs.initialization(c::CachedEmissions) = HMMs.initialization(c.model)
HMMs.transition_matrix(c::CachedEmissions, ctl::Tuple) = HMMs.transition_matrix(c.model, ctl)
HMMs.obs_distributions(c::CachedEmissions, ctl::Tuple) = HMMs.obs_distributions(c.model, ctl)

function HMMs.obs_logdensities!(logb::AbstractVector, c::CachedEmissions, obs, ctl::Tuple;
    error_if_not_finite::Bool=true)
    logb .= view(c.LL, ctl[3], :)
    return nothing
end

"""
    viterbi(hmm, Ys, Y_pasts; n2t, kwargs...) -> Vector{Vector{Int}}

The most likely hidden state path of each of several independent sequences: `Ys[i]` is the
`D × Nᵢ` sequence `i`, `Y_pasts[i]` its initial observations and `n2t[i]` its period indices, by
default `n_to_t(Nᵢ, T)`. Other keywords are passed to the single-sequence `viterbi`.
"""
function viterbi(hmm::Union{ARPeriodicHMM,sARPeriodicHMM,ARPeriodicHMMSpatial},
    Ys::AbstractVector{<:AbstractMatrix{<:Bool}}, Y_pasts::AbstractVector;
    n2t=[n_to_t(size(Y, 2), size(hmm.B, 2)) for Y in Ys], kwargs...)
    @argcheck length(Ys) == length(Y_pasts) == length(n2t) "$(length(Ys)) sequences, $(length(Y_pasts)) initial conditions and $(length(n2t)) period-index vectors"
    return [viterbi(hmm, Y, Y_past; n2t=n2tᵢ, kwargs...) for (Y, Y_past, n2tᵢ) in zip(Ys, Y_pasts, n2t)]
end
