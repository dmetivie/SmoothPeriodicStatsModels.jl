# # Several sequences, and the loglikelihood of the smooth HMMs
#
# Here rather than in the fitting extension: a likelihood needs no solver. The fits take their
# data through `_sequences` too, so every entry point lays several sequences out the same way.

# The observations of one or several independent sequences of the model `m`, laid end to end:
# `Y`, the memory categories, the period indices, and `seq_ends`, the last index of each sequence.
function _sequences(m, Y::AbstractArray{<:Bool}, Y_past, n2t::AbstractVector{<:Integer})
    @argcheck size(Y, 2) == length(n2t) "`Y` has $(size(Y, 2)) observations and `n2t` $(length(n2t)) entries"
    return Y, _lag_cat(m, Y, Y_past), n2t, (size(Y, 2),)
end

function _sequences(m, Ys::AbstractVector{<:AbstractMatrix{<:Bool}}, Y_pasts::AbstractVector,
    n2ts::AbstractVector{<:AbstractVector{<:Integer}})
    @argcheck !isempty(Ys) "no sequence to fit"
    @argcheck length(Ys) == length(Y_pasts) == length(n2ts) "$(length(Ys)) sequences, $(length(Y_pasts)) initial conditions and $(length(n2ts)) period-index vectors"
    for (i, (Y, n2t)) in enumerate(zip(Ys, n2ts))
        @argcheck size(Y, 2) == length(n2t) "sequence $i has $(size(Y, 2)) observations and its `n2t` $(length(n2t)) entries"
    end
    Y = reduce(hcat, Ys)
    lag_cat = reduce(hcat, [_lag_cat(m, Yᵢ, Y_pastᵢ) for (Yᵢ, Y_pastᵢ) in zip(Ys, Y_pasts)])
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

# `Y_past` holds the observations before `Y`, as many per site as the memory of `hmm`: a
# `D × order` matrix, or for `sARPeriodicHMM` one vector of `order_j` values per site. Read with
# another memory, the categories would be silently wrong.
function _check_Y_past(hmm::Union{ARPeriodicHMM,ARPeriodicHMMSpatial}, Y_past)
    D, order = size(hmm, 2), Int(log2(size(hmm, 4)))
    @argcheck size(Y_past) == (D, order) "`Y_past` is $(size(Y_past)) instead of $((D, order)): $D sites, memory $order"
    return Y_past
end

function _check_Y_past(hmm::sARPeriodicHMM, Y_past)
    D, orders = size(hmm, 2), Int.(log2.(size(hmm, 4)))
    @argcheck length(Y_past) == D "`Y_past` has $(length(Y_past)) sites instead of $D"
    @argcheck length.(Y_past) == orders "`Y_past` has lengths $(length.(Y_past)) instead of the memory orders $orders"
    return Y_past
end

_check_Y_past(m::_SmoothHMM, Y_past) = _check_Y_past(m.stored, Y_past)
_check_Y_past(::Nothing, Y_past) = Y_past   # no model: the memory is that of `Y_past`

# The memory categories of `Y` after `Y_past`, checked against the memory of `hmm`.
_lag_cat(hmm, Y, Y_past) = conditional_to(Y, _check_Y_past(hmm, Y_past))

function SeasonalControls(Ys::AbstractVector{<:AbstractMatrix{<:Bool}}, Y_pasts::AbstractVector;
    n2t::AbstractVector{<:AbstractVector{<:Integer}})
    _, lag_cat, n2t_all, _ = _sequences(nothing, Ys, Y_pasts, n2t)
    return SeasonalControls(n2t_all, lag_cat)
end

# The evaluated twins of the smooth Bernoulli HMMs: a likelihood is defined on both forms.
const _EvaluatedHMM = Union{ARPeriodicHMM,sARPeriodicHMM,ARPeriodicHMMSpatial}

_period(m::_SmoothHMM) = m.nT
_period(hmm::_EvaluatedHMM) = size(hmm.B, 2)
_evaluated(m::_SmoothHMM) = _current(m)
_evaluated(hmm::_EvaluatedHMM) = hmm

const _AnySmoothHMM = Union{_SmoothHMM,SmoothPeriodicHMM}

# ## The evaluated model, a cache
#
# `stored` holds the arrays the coefficients generate, so that the accessors, read once per
# observation, need not evaluate the bases. A write into `parameters(m)` does not reach it:
# `stored_from` keeps a copy of the coefficients `stored` was evaluated from, and `_current`
# evaluates it anew when they differ. The lock keeps two threads from rewriting it at once.

const _CACHE_LOCK = ReentrantLock()

# Evaluate `stored` from the coefficients, in place, and record them.
function _refresh!(m::_AnySmoothHMM)
    _evaluate!(m)
    m.stored_from[] = deepcopy(parameters(m))
    return m
end

# The evaluated model of the coefficients as they are now: the cache, brought up to date.
function _current(m::_AnySmoothHMM)
    Base.@lock _CACHE_LOCK isequal(m.stored_from[], parameters(m)) || _refresh!(m)
    return m.stored
end

# `initialization` and `obs_distributions` bring the cache up to date. HiddenMarkovModels.jl's
# algorithms call `initialization` at the start of every sequence, so `transition_matrix` and
# `obs_logdensities!`, called once per observation, read the cache without the comparison.
HMMs.initialization(m::_AnySmoothHMM) = HMMs.initialization(_current(m))
HMMs.transition_matrix(m::_AnySmoothHMM, c::Tuple) = HMMs.transition_matrix(m.stored, c)
HMMs.obs_distributions(m::_AnySmoothHMM, c::Tuple) = HMMs.obs_distributions(_current(m), c)
HMMs.obs_logdensities!(logb::AbstractVector, m::_AnySmoothHMM, obs, c::Tuple; kwargs...) =
    HMMs.obs_logdensities!(logb, m.stored, obs, c; kwargs...)

# A fixed control sequence goes to the evaluated twin, which refuses it for the autoregressive
# models (see `hmm_interface.jl`) and draws from it for `SmoothPeriodicHMM`.
Base.rand(rng::AbstractRNG, m::_AnySmoothHMM, c::SeasonalControls{<:Any,<:Tuple}) =
    rand(rng, _current(m), c)

_ninitial(m::_AnySmoothHMM) = length(HMMs.initialization(m)) - 1

"""
    loglikelihood(m::SmoothARPeriodicHMM, Y, Y_past; n2t) -> Float64

The marginal loglikelihood of `Y` under `m`, from `HiddenMarkovModels`' forward pass. Also
defined for `ARPeriodicHMM`.
"""
function StatsAPI.loglikelihood(m::Union{SmoothARPeriodicHMM,ARPeriodicHMM}, Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool}; n2t=n_to_t(size(Y, 2), _period(m))::AbstractVector{<:Integer})
    return _loglikelihood(m, _sequences(m, Y, Y_past, n2t)...)
end

"""
    loglikelihood(m::SmoothsARPeriodicHMM, Y, Y_past; n2t) -> Float64

The marginal loglikelihood of `Y` under `m`, from `HiddenMarkovModels`' forward pass. Also
defined for `sARPeriodicHMM`.
"""
function StatsAPI.loglikelihood(m::Union{SmoothsARPeriodicHMM,sARPeriodicHMM}, Y::AbstractMatrix{<:Bool},
    Y_past::AbstractVector; n2t=n_to_t(size(Y, 2), _period(m))::AbstractVector{<:Integer})
    return _loglikelihood(m, _sequences(m, Y, Y_past, n2t)...)
end

"""
    loglikelihood(m::SmoothARPeriodicHMMSpatial, Y, Y_past; n2t, rng = nothing) -> Float64

The marginal loglikelihood of `Y` under `m`. Stochastic: the emission is a quasi-Monte-Carlo
estimate, so repeated calls differ, unless `rng` seeds it. Raise the model's `QMC_m` to tighten
it. Also defined for `ARPeriodicHMMSpatial`.
"""
function StatsAPI.loglikelihood(m::Union{SmoothARPeriodicHMMSpatial,ARPeriodicHMMSpatial},
    Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool};
    n2t=n_to_t(size(Y, 2), _period(m))::AbstractVector{<:Integer}, rng=nothing)
    return _loglikelihood(m, _sequences(m, Y, Y_past, n2t)...; rng)
end

"""
    loglikelihood(m, Ys, Y_pasts; n2t) -> Float64

The loglikelihood of several independent sequences under a periodic Bernoulli HMM, smooth or
evaluated: the sum of the loglikelihoods of each, with the layout of the multi-sequence
[`fit_mle`](@ref).
"""
function StatsAPI.loglikelihood(m::Union{_SmoothHMM,_EvaluatedHMM}, Ys::AbstractVector{<:AbstractMatrix{<:Bool}},
    Y_pasts::AbstractVector; n2t=[n_to_t(size(Y, 2), _period(m)) for Y in Ys], kwargs...)
    return _loglikelihood(m, _sequences(m, Ys, Y_pasts, n2t)...; kwargs...)
end

function _loglikelihood(m::Union{SmoothARPeriodicHMM,SmoothsARPeriodicHMM,ARPeriodicHMM,sARPeriodicHMM},
    Y, lag_cat, n2t, seq_ends)
    controls = SeasonalControls(n2t, lag_cat)
    return HMMs.logdensityof(m, [view(Y, :, n) for n in axes(Y, 2)], controls; seq_ends)
end

"""
    loglikelihood(m::SmoothPeriodicHMM, Y; n2t, past, seq_ends) -> Float64
    loglikelihood(m::SmoothPeriodicHMM, Ys; n2t, past) -> Float64

The marginal loglikelihood of `Y` (`D × N`, or a vector when `D = 1`) under `m`, from
`HiddenMarkovModels`' forward pass; for several independent sequences, `Ys` or `Y` with
`seq_ends`, the sum over them. `past` is checked and unused, the model having no memory; a
`past.z` is refused. Also defined for `PeriodicHMMDists`.
"""
function StatsAPI.loglikelihood(m::Union{SmoothPeriodicHMM,PeriodicHMMDists}, Y::AbstractVecOrMat{<:Real};
    n2t=n_to_t(_ncols(Y), _period(m))::AbstractVector{<:Integer}, past=nothing, seq_ends=nothing)
    Yc, n2tc, ends = _sequences(Y, n2t)
    seq_ends === nothing || (ends = _check_seq_ends(seq_ends, size(Yc, 2)))
    _memoryless_pasts(m, past, seq_ends === nothing ? nothing : length(ends))
    return _loglikelihood(m, Yc, n2tc, ends)
end

function StatsAPI.loglikelihood(m::Union{SmoothPeriodicHMM,PeriodicHMMDists},
    Ys::AbstractVector{<:AbstractVecOrMat{<:Real}}; n2t=[n_to_t(_ncols(Y), _period(m)) for Y in Ys],
    past=nothing)
    _memoryless_pasts(m, past, length(Ys))
    return _loglikelihood(m, _sequences(Ys, n2t)...)
end

# Check the pasts of a model without memory, one or one per sequence, none with a `z`.
function _memoryless_pasts(m, past, nseq)
    past === nothing && return nothing
    for p in (nseq === nothing ? (past,) : _pasts(past, nseq))
        last(_hmm_past(m, p; required=false)) === nothing || _no_past_z()
    end
    return nothing
end

function _loglikelihood(m::Union{SmoothPeriodicHMM,PeriodicHMMDists}, Y, n2t, seq_ends)
    @argcheck size(Y, 1) == size(m, 2) "the observations have $(size(Y, 1)) rows and the emission blocks cover $(size(m, 2))"
    return HMMs.logdensityof(m, [view(Y, :, n) for n in axes(Y, 2)], SeasonalControls(n2t); seq_ends)
end

function _loglikelihood(m::Union{SmoothARPeriodicHMMSpatial,ARPeriodicHMMSpatial}, Y, lag_cat, n2t, seq_ends;
    rng=nothing)
    hmm = _evaluated(m)
    N, K = size(Y, 2), size(hmm.A, 1)
    controls = SeasonalControls(n2t, lag_cat)
    obs = [view(Y, :, n) for n = 1:N]
    cached = CachedEmissions(hmm, zeros(N, K))
    loglikelihoods!(cached.LL, hmm, Y, lag_cat; n2t=n2t, rng)
    return HMMs.logdensityof(cached, obs, controls; seq_ends)
end

# # `past`, `seq_ends` and the keyword forms
#
# `past = (; Y, z)` replaces the positional `Y_past`: `Y` holds the observations before the
# first, at least as many as the memory (the last ones are used), and `z` the hidden state of
# the step before the first. `rand` and `viterbi` use `z`; the fits and the marginal
# loglikelihood start each sequence from the initial distribution, which HiddenMarkovModels.jl
# gives no per-sequence hook to replace, and refuse it.

# The memory of the sites: one order, or one per site.
_memory(hmm::Union{ARPeriodicHMM,ARPeriodicHMMSpatial}) = Int(log2(size(hmm, 4)))
_memory(hmm::sARPeriodicHMM) = Int.(log2.(size(hmm, 4)))
_memory(m::_SmoothHMM) = _memory(m.stored)

# `(Y_past, z₀)` from `past`, `Y_past` in the layout of the positional methods. With `required`,
# a model with memory needs `past.Y`; otherwise it defaults to zeros, as `rand` always did.
function _hmm_past(m, past; required::Bool)
    (; Y, z) = _past_fields(past)
    z === nothing || _check_state(z, size(m, 1))
    return _past_observations(m, Y, required), z
end

_past_observations(m::_SmoothHMM, Y, required::Bool) = _past_observations(m.stored, Y, required)

function _past_observations(hmm::Union{ARPeriodicHMM,ARPeriodicHMMSpatial}, Y, required::Bool)
    D, order = size(hmm, 2), _memory(hmm)
    if Y === nothing
        required && order > 0 && _missing_past(D, order)
        return zeros(Bool, D, order)
    end
    Y isa AbstractMatrix{<:Bool} || throw(ArgumentError(
        "`past.Y` is a $D × m matrix of Bool with m ≥ $order, the observations before the first; got a $(typeof(Y))"))
    size(Y, 1) == D || throw(ArgumentError("`past.Y` has $(size(Y, 1)) rows for $D sites"))
    size(Y, 2) >= order || throw(ArgumentError("`past.Y` has $(size(Y, 2)) columns, fewer than the memory $order"))
    return Y[:, (end-order+1):end]
end

function _past_observations(hmm::sARPeriodicHMM, Y, required::Bool)
    D, orders = size(hmm, 2), _memory(hmm)
    if Y === nothing
        required && any(>(0), orders) && _missing_past(D, maximum(orders))
        return [zeros(Bool, o) for o in orders]
    elseif Y isa AbstractMatrix{<:Bool}
        size(Y, 1) == D || throw(ArgumentError("`past.Y` has $(size(Y, 1)) rows for $D sites"))
        size(Y, 2) >= maximum(orders) || throw(ArgumentError(
            "`past.Y` has $(size(Y, 2)) columns, fewer than the largest memory $(maximum(orders))"))
        return [Y[j, (end-orders[j]+1):end] for j = 1:D]
    elseif Y isa AbstractVector{<:AbstractVector{<:Bool}}
        length(Y) == D || throw(ArgumentError("`past.Y` has $(length(Y)) sites instead of $D"))
        for j = 1:D
            length(Y[j]) >= orders[j] || throw(ArgumentError(
                "`past.Y[$j]` has $(length(Y[j])) values, fewer than the memory $(orders[j]) of site $j"))
        end
        return [Y[j][(end-orders[j]+1):end] for j = 1:D]
    end
    throw(ArgumentError("`past.Y` is a D × m matrix of Bool, or one vector of Bool per site; got a $(typeof(Y))"))
end

# A model without memory: `past.Y` is checked and unused.
function _past_observations(hmm::Union{PeriodicHMMDists,SmoothPeriodicHMM}, Y, ::Bool)
    Y === nothing && return nothing
    D = size(hmm, 2)
    Y isa AbstractVecOrMat{<:Real} && size(_as_rows(Y), 1) == D || throw(ArgumentError(
        "`past.Y` holds observations of the $D rows of `Y`; got a $(typeof(Y)) of size $(size(Y))"))
    return nothing
end

@noinline _missing_past(D, order) = throw(ArgumentError(
    "this model has memory $order: pass `past = (; Y = Y_past)`, the $D × $order observations before the first"))

@noinline _no_past_z() = throw(ArgumentError(
    "`past.z`, the state before the first observation, is read by `rand` and `viterbi`; the fits and the marginal loglikelihood start each sequence from the initial distribution and do not take it"))

# `_sequences` for the keyword forms: the data, its memory categories, the period indices, the
# sequence ends, and the `past.z` of each sequence.
function _keyword_sequences(m, Y::AbstractMatrix{<:Bool}, past, n2t::AbstractVector{<:Integer}, seq_ends)
    if seq_ends === nothing
        Y_past, z0 = _hmm_past(m, past; required=true)
        return _sequences(m, Y, Y_past, n2t)..., [z0]
    end
    @argcheck size(Y, 2) == length(n2t) "`Y` has $(size(Y, 2)) observations and `n2t` $(length(n2t)) entries"
    ends = _check_seq_ends(seq_ends, size(Y, 2))
    ranges = _seq_ranges(ends)
    pasts = [_hmm_past(m, p; required=true) for p in _pasts(past, length(ranges))]
    lag_cat = reduce(hcat, [_lag_cat(m, view(Y, :, r), first(p)) for (r, p) in zip(ranges, pasts)])
    return Y, lag_cat, n2t, ends, last.(pasts)
end

function _keyword_sequences(m, Ys::AbstractVector{<:AbstractMatrix{<:Bool}}, past,
    n2ts::AbstractVector{<:AbstractVector{<:Integer}})
    pasts = [_hmm_past(m, p; required=true) for p in _pasts(past, length(Ys))]
    return _sequences(m, Ys, first.(pasts), n2ts)..., last.(pasts)
end

"""
    loglikelihood(m, Y; n2t, past, seq_ends) -> Float64
    loglikelihood(m, Ys; n2t, past) -> Float64

The marginal loglikelihood of a periodic Bernoulli HMM, smooth or evaluated. `past = (; Y =
Y_past)` holds the observations before `Y`, at least as many as the memory; with `seq_ends`,
`Y` is several sequences end to end and `past` a vector of their pasts. `Ys` is a vector of
sequences, with vectors `n2t` and `past`. The spatial model also takes `rng`.
"""
function StatsAPI.loglikelihood(m::Union{_SmoothHMM,_EvaluatedHMM}, Y::AbstractMatrix{<:Bool};
    n2t=n_to_t(size(Y, 2), _period(m))::AbstractVector{<:Integer}, past=nothing, seq_ends=nothing,
    kwargs...)
    Yc, lag_cat, n2tc, ends, z0s = _keyword_sequences(m, Y, past, n2t, seq_ends)
    all(isnothing, z0s) || _no_past_z()
    return _loglikelihood(m, Yc, lag_cat, n2tc, ends; kwargs...)
end

function StatsAPI.loglikelihood(m::Union{_SmoothHMM,_EvaluatedHMM}, Ys::AbstractVector{<:AbstractMatrix{<:Bool}};
    n2t=[n_to_t(size(Y, 2), _period(m)) for Y in Ys], past=nothing, kwargs...)
    Yc, lag_cat, n2tc, ends, z0s = _keyword_sequences(m, Ys, past, n2t)
    all(isnothing, z0s) || _no_past_z()
    return _loglikelihood(m, Yc, lag_cat, n2tc, ends; kwargs...)
end

"""
    loglikelihoods(hmm, Y; n2t, past, seq_ends, robust = false) -> Matrix

The `N × K` emission loglikelihoods of a periodic Bernoulli HMM, smooth or evaluated, the
observations before `Y` given as `past = (; Y = Y_past)`; `past.z` is accepted and unused. With
`seq_ends`, `Y` is several sequences end to end and `past` a vector of their pasts. The spatial
model also takes `QMC_m` and `rng`.
"""
function loglikelihoods(hmm::_EvaluatedHMM, Y::AbstractMatrix{<:Bool};
    n2t=n_to_t(size(Y, 2), _period(hmm))::AbstractVector{<:Integer}, past=nothing, seq_ends=nothing,
    robust=false, kwargs...)
    Yc, lag_cat, n2tc, _, _ = _keyword_sequences(hmm, Y, past, n2t, seq_ends)
    LL = Matrix{Float64}(undef, size(Yc, 2), size(hmm, 1))
    loglikelihoods!(LL, hmm, Yc, lag_cat; n2t=n2tc, kwargs...)
    robust && _clamp_inf!(LL)
    return LL
end

"""
    viterbi(hmm, Y; n2t, past, seq_ends, robust = false) -> Vector{Int}
    viterbi(hmm, Ys; n2t, past, robust = false) -> Vector{Vector{Int}}

The most likely hidden state path of a periodic HMM, smooth or evaluated. `past = (; Y = Y_past,
z = k₀)` gives the observations before `Y` and the state of the step before the first: the first
state then follows the row `A[k₀, :, mod1(n2t[1] - 1, T)]` rather than the initial distribution.
With `seq_ends`, `Y` is several sequences end to end, `past` a vector of their pasts, and the
paths come end to end. The spatial model also takes `QMC_m` and `rng`.
"""
function viterbi(hmm::_EvaluatedHMM, Y::AbstractMatrix{<:Bool};
    n2t=n_to_t(size(Y, 2), _period(hmm))::AbstractVector{<:Integer}, past=nothing, seq_ends=nothing,
    robust=false, kwargs...)
    Yc, lag_cat, n2tc, ends, z0s = _keyword_sequences(hmm, Y, past, n2t, seq_ends)
    LL = Matrix{Float64}(undef, size(Yc, 2), size(hmm, 1))
    loglikelihoods!(LL, hmm, Yc, lag_cat; n2t=n2tc, kwargs...)
    robust && _clamp_inf!(LL)
    return _viterbi_sequences(hmm.a, hmm.A, LL, n2tc, ends, z0s)
end

function viterbi(hmm::_EvaluatedHMM, Ys::AbstractVector{<:AbstractMatrix{<:Bool}};
    n2t=[n_to_t(size(Y, 2), _period(hmm)) for Y in Ys], past=nothing, kwargs...)
    @argcheck length(Ys) == length(n2t) "$(length(Ys)) sequences and $(length(n2t)) period-index vectors"
    return [viterbi(hmm, Y; n2t=n2tᵢ, past=pᵢ, kwargs...)
            for (Y, n2tᵢ, pᵢ) in zip(Ys, n2t, _pasts(past, length(Ys)))]
end

# The path of each sequence, end to end. A sequence that follows a known state `z₀` starts from
# `A[z₀, :, s]`, `s` the period index of the step before its first.
function _viterbi_sequences(a, A, LL, n2t, ends, z0s)
    z = Vector{Int}(undef, size(LL, 1))
    for (r, z0) in zip(_seq_ranges(ends), z0s)
        isempty(r) && continue
        a0 = z0 === nothing ? a : A[z0, :, mod1(n2t[first(r)] - 1, size(A, 3))]
        z[r] = viterbi(a0, A, LL[r, :]; n2t=n2t[r])
    end
    return z
end
