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

The control sequence of a periodic autoregressive HMM: `(n2t[n], lag_cat[:, n], n)` at step `n`.

`n2t[n] ∈ 1:T` is the period index and `lag_cat[j, n]` the memory category of site `j`, the
`1 + ` binary encoding of its `order` previous values that [`conditional_to`](@ref) builds.
The third element is the observation index, which the accessors ignore and
[`CachedEmissions`](@ref) uses to find its row.

Lazy on purpose: a year of 50 000 observations would otherwise be 50 000 allocated tuples, and
the memory categories are already laid out contiguously by site.
"""
struct SeasonalControls{L<:AbstractMatrix{<:Integer},E} <: AbstractVector{E}
    n2t::Vector{Int}
    lag_cat::L
end

function SeasonalControls(n2t::AbstractVector{<:Integer}, lag_cat::AbstractMatrix{<:Integer})
    @argcheck length(n2t) == size(lag_cat, 2) "`n2t` has $(length(n2t)) entries and `lag_cat` $(size(lag_cat, 2)) columns; both index the observations"
    E = Tuple{Int,typeof(view(lag_cat, :, 1)),Int}
    return SeasonalControls{typeof(lag_cat),E}(collect(Int, n2t), lag_cat)
end

"""
    SeasonalControls(Y, Y_past; n2t)

The control sequence implied by the observations: the memory categories are read off `Y` and
its initial condition `Y_past`, as every likelihood in this package does.
"""
SeasonalControls(Y::AbstractMatrix{<:Bool}, Y_past::AbstractMatrix{<:Bool};
    n2t=n_to_t(size(Y, 2), 0)::AbstractVector{<:Integer}) =
    SeasonalControls(n2t, conditional_to(Y, Y_past))

Base.size(c::SeasonalControls) = (length(c.n2t),)
Base.@propagate_inbounds Base.getindex(c::SeasonalControls, n::Integer) =
    (c.n2t[n], view(c.lag_cat, :, n), Int(n))

# ## The three accessors
#
# ### Why the transition matrix is read one period index back
#
# This package, following Touron, indexes `A` by the period the step *leaves*: the complete
# loglikelihood multiplies `A[z[n], z[n+1], n2t[n]]`. `HiddenMarkovModels` asks instead for the
# matrix of the step that *arrives* at a control — `forward_backward!` builds `ξ[n]` with
# `transition_matrix(hmm, control_seq[n + 1])`, and this has been its convention since v0.6.
#
# The two meet at `mod1(t - 1, T)`: for contiguous data `n2t[n+1] = mod1(n2t[n] + 1, T)`, so
# asking for period `n2t[n+1]` and shifting back by one gives `n2t[n]`, the matrix this package
# means. Stored `θᴬ` therefore keeps its published meaning and the `.jld` assets stay valid —
# the alternative was to redefine `A` and invalidate both.

HMMs.initialization(hmm::ARPeriodicHMM) = hmm.a

HMMs.transition_matrix(hmm::ARPeriodicHMM, (t, _)::Tuple) =
    view(hmm.A, :, :, mod1(t - 1, size(hmm.A, 3)))

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

Fill `logb[k]` with the loglikelihood of `obs` in state `k`, without building a distribution.

`HiddenMarkovModels` provides this as `logdensityof.(obs_distributions(hmm, control), obs)`,
which here would allocate `K` product distributions for *every one* of the `N` observations in
every forward pass — measured at roughly 3× the cost of the whole fit. The sum below is the
same arithmetic as [`loglikelihoods!`](@ref), which is where this model's inner loop has always
lived.

This overrides an internal of that package rather than part of its documented interface: if a
future version stops calling it, the generic path above still gives the same numbers, only
slower.
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

HMMs.transition_matrix(hmm::sARPeriodicHMM, (t, _)::Tuple) =
    view(hmm.A, :, :, mod1(t - 1, size(hmm.A, 3)))

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

HMMs.transition_matrix(hmm::ARPeriodicHMMSpatial, c::Tuple) =
    view(hmm.A, :, :, mod1(first(c) - 1, size(hmm.A, 3)))

"""
    obs_distributions(hmm::ARPeriodicHMMSpatial, (t, h, _...)) -> Vector{SpatialBernoulli}

One [`SpatialBernoulli`](@ref) per hidden state: marginal rain probabilities
`B[k, t, j, h[j]]` tied together by a latent Gaussian field whose correlation is
`exp(-distance / R[k, t])`.

The `D × D` kernel is rebuilt on every call, and since `HiddenMarkovModels` asks for the
distributions one observation at a time that is once per `(observation, state)`. The
four-argument constructor is used only to make that visible — the three-argument one computes
exactly the same `expkernel.(h)` inside itself.

It is left that way deliberately. Caching it on the model would mean `D²·K·T` floats kept in
step with `R`, which `update_R!` writes directly, and it buys about 1% here because
`mvnormcdf` dominates. The path that *does* matter for fitting never calls this: `fit_mle`
fills the whole `N × K` matrix with the threaded [`loglikelihoods!`](@ref), which precomputes
the kernels once per `(k, t)`.
"""
function HMMs.obs_distributions(hmm::ARPeriodicHMMSpatial, c::Tuple)
    t, h = c[1], c[2]
    K, D = size(hmm.B, 1), size(hmm.B, 3)
    return [SpatialBernoulli(hmm.R[k, t], hmm.B[CartesianIndex.(k, t, 1:D, h)], hmm.h,
        expkernel.(hmm.h; range=hmm.R[k, t])) for k = 1:K]
end

"""
    obs_logdensities!(logb, hmm::ARPeriodicHMMSpatial, obs, (t, h, _...))

Fill `logb[k]` with the loglikelihood of `obs` in state `k`, the same orthant probability
[`loglikelihoods!`](@ref) computes.

This is the correct but *sequential* path. The fitter does not use it: it fills the whole
`N × K` matrix with the threaded `loglikelihoods!` instead and substitutes that, because
`mvnormcdf` dominates and threading over observations is worth roughly an order of magnitude.
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
    ::SeasonalControls)
    throw(ArgumentError("""
    an autoregressive periodic HMM cannot be simulated from a fixed control sequence: the
    memory categories in `SeasonalControls` are a function of the observations, which do not
    exist yet. Use this package's own

        z, y = rand(hmm, n2t; y_ini = …, z_ini = …, seq = true)

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

An `AbstractHMM` that answers `initialization` and `transition_matrix` from `model` but reads
its emission loglikelihoods out of `LL`, an `N × K` matrix filled beforehand.

It is what lets the batched — and for the spatial model threaded — `loglikelihoods!` evaluate
the emission in one sweep while the forward–backward recursions, the transition marginals and
the multi-sequence bookkeeping still come from `HiddenMarkovModels`. The observation index in
[`SeasonalControls`](@ref) is the only way `obs_logdensities!` can find its row, which is why
the control carries one.

The cache can go stale, so keep it local: fill it, use it, and let it go. The spatial fitter
creates one inside a single call and never lets it escape.
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
