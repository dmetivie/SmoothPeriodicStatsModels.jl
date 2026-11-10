# # Several sequences, and the loglikelihood of the smooth HMMs
#
# Here rather than in the fitting extension: a likelihood needs no solver. The fits take their
# data through `_sequences` too, so every entry point lays several sequences out the same way.

# The observations of one or several independent sequences, laid end to end: `Y`, the memory
# categories, the period indices, and `seq_ends`, the last index of each sequence.
function _sequences(Y::AbstractArray{<:Bool}, Y_past, n2t::AbstractVector{<:Integer})
    @argcheck size(Y, 2) == length(n2t) "`Y` has $(size(Y, 2)) observations and `n2t` $(length(n2t)) entries"
    return Y, conditional_to(Y, Y_past), n2t, (size(Y, 2),)
end

function _sequences(Ys::AbstractVector{<:AbstractMatrix{<:Bool}}, Y_pasts::AbstractVector,
    n2ts::AbstractVector{<:AbstractVector{<:Integer}})
    @argcheck !isempty(Ys) "no sequence to fit"
    @argcheck length(Ys) == length(Y_pasts) == length(n2ts) "$(length(Ys)) sequences, $(length(Y_pasts)) initial conditions and $(length(n2ts)) period-index vectors"
    for (i, (Y, n2t)) in enumerate(zip(Ys, n2ts))
        @argcheck size(Y, 2) == length(n2t) "sequence $i has $(size(Y, 2)) observations and its `n2t` $(length(n2t)) entries"
    end
    Y = reduce(hcat, Ys)
    lag_cat = reduce(hcat, [conditional_to(Yᵢ, Y_pastᵢ) for (Yᵢ, Y_pastᵢ) in zip(Ys, Y_pasts)])
    return Y, lag_cat, reduce(vcat, n2ts), cumsum([size(Yᵢ, 2) for Yᵢ in Ys])
end

const _SmoothHMM = Union{SmoothARPeriodicHMM,SmoothsARPeriodicHMM,SmoothARPeriodicHMMSpatial}

function SeasonalControls(Ys::AbstractVector{<:AbstractMatrix{<:Bool}}, Y_pasts::AbstractVector;
    n2t::AbstractVector{<:AbstractVector{<:Integer}})
    _, lag_cat, n2t_all, _ = _sequences(Ys, Y_pasts, n2t)
    return SeasonalControls(n2t_all, lag_cat)
end

"""
    loglikelihood(m::SmoothARPeriodicHMM, Y, Y_past; n2t) -> Float64

The marginal loglikelihood of `Y` under `m`, from `HiddenMarkovModels`' forward pass.
"""
function StatsAPI.loglikelihood(m::SmoothARPeriodicHMM, Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool}; n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer})
    return _loglikelihood(m, _sequences(Y, Y_past, n2t)...)
end

"""
    loglikelihood(m::SmoothsARPeriodicHMM, Y, Y_past; n2t) -> Float64

The marginal loglikelihood of `Y` under `m`, from `HiddenMarkovModels`' forward pass.
"""
function StatsAPI.loglikelihood(m::SmoothsARPeriodicHMM, Y::AbstractMatrix{<:Bool},
    Y_past::AbstractVector; n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer})
    return _loglikelihood(m, _sequences(Y, Y_past, n2t)...)
end

"""
    loglikelihood(m::SmoothARPeriodicHMMSpatial, Y, Y_past; n2t) -> Float64

The marginal loglikelihood of `Y` under `m`. Stochastic: the emission is a quasi-Monte-Carlo
estimate, so repeated calls differ. Raise the model's `QMC_m` to tighten it.
"""
function StatsAPI.loglikelihood(m::SmoothARPeriodicHMMSpatial, Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool}; n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer})
    return _loglikelihood(m, _sequences(Y, Y_past, n2t)...)
end

"""
    loglikelihood(m, Ys, Y_pasts; n2t) -> Float64

The loglikelihood of several independent sequences under a smooth periodic HMM: the sum of the
loglikelihoods of each, with the layout of the multi-sequence [`fit_mle`](@ref).
"""
function StatsAPI.loglikelihood(m::_SmoothHMM, Ys::AbstractVector{<:AbstractMatrix{<:Bool}},
    Y_pasts::AbstractVector; n2t=[n_to_t(size(Y, 2), m.nT) for Y in Ys])
    return _loglikelihood(m, _sequences(Ys, Y_pasts, n2t)...)
end

function _loglikelihood(m::Union{SmoothARPeriodicHMM,SmoothsARPeriodicHMM}, Y, lag_cat, n2t, seq_ends)
    controls = SeasonalControls(n2t, lag_cat)
    return HMMs.logdensityof(m, [view(Y, :, n) for n in axes(Y, 2)], controls; seq_ends)
end

function _loglikelihood(m::SmoothARPeriodicHMMSpatial, Y, lag_cat, n2t, seq_ends)
    N, K = size(Y, 2), size(m.stored.A, 1)
    controls = SeasonalControls(n2t, lag_cat)
    obs = [view(Y, :, n) for n = 1:N]
    cached = CachedEmissions(m, zeros(N, K))
    loglikelihoods!(cached.LL, m.stored, Y, lag_cat; n2t=n2t)
    return HMMs.logdensityof(cached, obs, controls; seq_ends)
end
