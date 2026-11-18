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

# The same for a model without memory: no initial observations, and a vector is one row.
function _sequences(Y::AbstractVecOrMat{<:Real}, n2t::AbstractVector{<:Integer})
    Y = _as_rows(Y)
    @argcheck size(Y, 2) == length(n2t) "`Y` has $(size(Y, 2)) observations and `n2t` $(length(n2t)) entries"
    return Y, n2t, (size(Y, 2),)
end

function _sequences(Ys::AbstractVector{<:AbstractVecOrMat{<:Real}}, n2ts::AbstractVector{<:AbstractVector{<:Integer}})
    @argcheck !isempty(Ys) "no sequence to fit"
    @argcheck length(Ys) == length(n2ts) "$(length(Ys)) sequences and $(length(n2ts)) period-index vectors"
    for (i, (Y, n2t)) in enumerate(zip(Ys, n2ts))
        @argcheck _ncols(Y) == length(n2t) "sequence $i has $(_ncols(Y)) observations and its `n2t` $(length(n2t)) entries"
    end
    return reduce(hcat, map(_as_rows, Ys)), reduce(vcat, n2ts), cumsum([_ncols(Y) for Y in Ys])
end

const _SmoothHMM = Union{SmoothARPeriodicHMM,SmoothsARPeriodicHMM,SmoothARPeriodicHMMSpatial}

function SeasonalControls(Ys::AbstractVector{<:AbstractMatrix{<:Bool}}, Y_pasts::AbstractVector;
    n2t::AbstractVector{<:AbstractVector{<:Integer}})
    _, lag_cat, n2t_all, _ = _sequences(Ys, Y_pasts, n2t)
    return SeasonalControls(n2t_all, lag_cat)
end

# The evaluated twins of the smooth Bernoulli HMMs: a likelihood is defined on both forms.
const _EvaluatedHMM = Union{ARPeriodicHMM,sARPeriodicHMM,ARPeriodicHMMSpatial}

_period(m::_SmoothHMM) = m.nT
_period(hmm::_EvaluatedHMM) = size(hmm.B, 2)
_evaluated(m::_SmoothHMM) = m.stored
_evaluated(hmm::_EvaluatedHMM) = hmm

"""
    loglikelihood(m::SmoothARPeriodicHMM, Y, Y_past; n2t) -> Float64

The marginal loglikelihood of `Y` under `m`, from `HiddenMarkovModels`' forward pass. Also
defined for `ARPeriodicHMM`.
"""
function StatsAPI.loglikelihood(m::Union{SmoothARPeriodicHMM,ARPeriodicHMM}, Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool}; n2t=n_to_t(size(Y, 2), _period(m))::AbstractVector{<:Integer})
    return _loglikelihood(m, _sequences(Y, Y_past, n2t)...)
end

"""
    loglikelihood(m::SmoothsARPeriodicHMM, Y, Y_past; n2t) -> Float64

The marginal loglikelihood of `Y` under `m`, from `HiddenMarkovModels`' forward pass. Also
defined for `sARPeriodicHMM`.
"""
function StatsAPI.loglikelihood(m::Union{SmoothsARPeriodicHMM,sARPeriodicHMM}, Y::AbstractMatrix{<:Bool},
    Y_past::AbstractVector; n2t=n_to_t(size(Y, 2), _period(m))::AbstractVector{<:Integer})
    return _loglikelihood(m, _sequences(Y, Y_past, n2t)...)
end

"""
    loglikelihood(m::SmoothARPeriodicHMMSpatial, Y, Y_past; n2t) -> Float64

The marginal loglikelihood of `Y` under `m`. Stochastic: the emission is a quasi-Monte-Carlo
estimate, so repeated calls differ. Raise the model's `QMC_m` to tighten it. Also defined for
`ARPeriodicHMMSpatial`.
"""
function StatsAPI.loglikelihood(m::Union{SmoothARPeriodicHMMSpatial,ARPeriodicHMMSpatial},
    Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool};
    n2t=n_to_t(size(Y, 2), _period(m))::AbstractVector{<:Integer})
    return _loglikelihood(m, _sequences(Y, Y_past, n2t)...)
end

"""
    loglikelihood(m, Ys, Y_pasts; n2t) -> Float64

The loglikelihood of several independent sequences under a periodic Bernoulli HMM, smooth or
evaluated: the sum of the loglikelihoods of each, with the layout of the multi-sequence
[`fit_mle`](@ref).
"""
function StatsAPI.loglikelihood(m::Union{_SmoothHMM,_EvaluatedHMM}, Ys::AbstractVector{<:AbstractMatrix{<:Bool}},
    Y_pasts::AbstractVector; n2t=[n_to_t(size(Y, 2), _period(m)) for Y in Ys])
    return _loglikelihood(m, _sequences(Ys, Y_pasts, n2t)...)
end

function _loglikelihood(m::Union{SmoothARPeriodicHMM,SmoothsARPeriodicHMM,ARPeriodicHMM,sARPeriodicHMM},
    Y, lag_cat, n2t, seq_ends)
    controls = SeasonalControls(n2t, lag_cat)
    return HMMs.logdensityof(m, [view(Y, :, n) for n in axes(Y, 2)], controls; seq_ends)
end

"""
    loglikelihood(m::SmoothPeriodicHMM, Y; n2t) -> Float64
    loglikelihood(m::SmoothPeriodicHMM, Ys; n2t) -> Float64

The marginal loglikelihood of `Y` (`D × N`, or a vector when `D = 1`) under `m`, from
`HiddenMarkovModels`' forward pass; for several independent sequences, the sum over them. Also
defined for `PeriodicHMMDists`.
"""
function StatsAPI.loglikelihood(m::Union{SmoothPeriodicHMM,PeriodicHMMDists}, Y::AbstractVecOrMat{<:Real};
    n2t=n_to_t(_ncols(Y), _period(m))::AbstractVector{<:Integer})
    return _loglikelihood(m, _sequences(Y, n2t)...)
end

function StatsAPI.loglikelihood(m::Union{SmoothPeriodicHMM,PeriodicHMMDists},
    Ys::AbstractVector{<:AbstractVecOrMat{<:Real}}; n2t=[n_to_t(_ncols(Y), _period(m)) for Y in Ys])
    return _loglikelihood(m, _sequences(Ys, n2t)...)
end

function _loglikelihood(m::Union{SmoothPeriodicHMM,PeriodicHMMDists}, Y, n2t, seq_ends)
    @argcheck size(Y, 1) == size(m, 2) "the observations have $(size(Y, 1)) rows and the emission blocks cover $(size(m, 2))"
    return HMMs.logdensityof(m, [view(Y, :, n) for n in axes(Y, 2)], SeasonalControls(n2t); seq_ends)
end

function _loglikelihood(m::Union{SmoothARPeriodicHMMSpatial,ARPeriodicHMMSpatial}, Y, lag_cat, n2t, seq_ends)
    hmm = _evaluated(m)
    N, K = size(Y, 2), size(hmm.A, 1)
    controls = SeasonalControls(n2t, lag_cat)
    obs = [view(Y, :, n) for n = 1:N]
    cached = CachedEmissions(hmm, zeros(N, K))
    loglikelihoods!(cached.LL, hmm, Y, lag_cat; n2t=n2t)
    return HMMs.logdensityof(cached, obs, controls; seq_ends)
end
