# # The fits of every model, before the extension that defines them is loaded
#
# `fit_mle` and `fit` are owned by Distributions and StatsAPI, so these stubs are methods of
# theirs, one per model type: less specific than every method of the extension, which therefore
# always wins once loaded, and a helpful error until then.

# The three Bernoulli HMMs share one stub: the extension has methods for their union.
for M in (:SmoothAR1, :SmoothSeasonalMixture, :SmoothDistribution, :_SmoothHMM, :SmoothPeriodicHMM,
    :OptimMLE)
    @eval fit_mle(m::$M, args...; kwargs...) = _unavailable(fit_mle, (m, args...), _OPTIMIZATION_EXT)
end

StatsAPI.fit(::Type{SeasonalFit}, m, args...; kwargs...) =
    _unavailable(fit, (SeasonalFit, m, args...), _OPTIMIZATION_EXT)

fit_mle_all_slices(m::SmoothPeriodicHMM, args...; kwargs...) =
    _unavailable(fit_mle_all_slices, (m, args...), _OPTIMIZATION_EXT)
fit_mle_all_slices(rng::AbstractRNG, m::SmoothPeriodicHMM, args...; kwargs...) =
    _unavailable(fit_mle_all_slices, (rng, m, args...), _OPTIMIZATION_EXT)
