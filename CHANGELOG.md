# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow
[Semantic Versioning](https://semver.org/).

## [4.0.0] - Unreleased

Version 4 gives every model the same shape: a model holds its coefficients,
`fit_mle(model, data...)` returns the fitted model and a history, and the hidden Markov models
are models of [HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl). The API
of version 3 is not kept.

### Added

- Models that hold their coefficients: `SmoothAR1`, `SmoothSeasonalMixture`,
  `SmoothARPeriodicHMM`, `SmoothsARPeriodicHMM` and `SmoothARPeriodicHMMSpatial`.
  `parameters(m)` returns the coefficients by name, `curve(m, name)` one parameter over the
  cycle, and `AR1(m)`, `mixtures(m)`, `ARPeriodicHMM(m)`, `sARPeriodicHMM(m)` and
  `ARPeriodicHMMSpatial(m)` the evaluated model.
- `SmoothPeriodicHMM`, a hidden Markov model without memory whose emission is any
  Distributions.jl law, given in blocks of rows: univariate laws, multivariate laws, and laws
  built from a copula. `PeriodicHMMDists` is its evaluated form.
- `fit_mle_all_slices` for `SmoothPeriodicHMM`: a starting point estimated from the data, one
  period index at a time.
- Bases other than trigonometric polynomials, from
  [TimeVaryingParameters.jl](https://github.com/dmetivie/TimeVaryingParameters.jl), such as
  `BSplineBasis` and `CyclicBSplineBasis`, and one basis and one link per parameter with
  `ParamSpec`.
- Seasonal families. Mixture components and HMM emissions can be `Exponential`, `Rayleigh`,
  `Gamma`, `Weibull`, `Normal`, `LogNormal`, `Bernoulli` or `Poisson`, a mixture can hold
  components of different families, and `seasonal_family` declares more.
- `fit(SeasonalFit, model, data...)` for model selection, with `loglikelihood`, `dof`, `nobs`,
  `aic`, `aicc` and `bic`.
- Several independent sequences in the HMM fits, likelihoods and `viterbi`.
- `loglikelihood` for every model, smooth or evaluated.
- `SeasonalControls`, the control sequence that makes the functions of HiddenMarkovModels.jl
  apply to the HMMs.
- `FitHistory`, returned by every fit.
- `robust = true` in the HMM fits, likelihoods and decoding, for observations that no state can
  produce.
- Documentation: tutorials for the AR(1), the mixture, the Bernoulli HMMs, the HMM with any
  emission and a non-periodic example on genes; guides on the data layout, fitting,
  HiddenMarkovModels.jl and seasonal families.

### Changed

- `fit_mle(model, data...; n2t, solver)` starts from the coefficients of `model` and returns
  `(model, history)`, for every model. It replaces `fit_mle_AR1` and the HMM fits that took and
  returned loose coefficient arrays.
- Every fit takes its optimization algorithm explicitly as `solver`: there is no default solver
  any more, and a fit without one stops with an error saying so. The fitting extension loads with
  Optimization.jl and SimpleNonlinearSolve; load the package of your solver beside them, for
  example OptimizationOptimJL for `solver = LBFGS()`. `default_solver` is removed, and so is the
  two-argument `OptimMLE(ℓ, θ₀)`.
- The Bernoulli success probability is `logistic(P(t))`: the emission coefficients `θᴮ` have the
  opposite sign of those of version 3.
- The hidden Markov models subtype `HiddenMarkovModels.AbstractHMM`, and the forward and
  backward passes of their fits are that package's.
- `Y_past`, the initial observations of the autoregressive HMMs, is the `D × order` observations
  before `Y` in time order, for `rand`, the fits, the likelihoods and `viterbi` alike.
- The fit of the spatial HMM returns the iterate with the highest loglikelihood,
  `history.best_iteration`, and the model holds its quasi-Monte-Carlo budget `QMC_m`.
- New dependencies: TimeVaryingParameters.jl, HiddenMarkovModels.jl, LinkFunctions.jl,
  ComponentArrays.jl and StatsAPI.jl. SciMLBase 3 is required.

### Removed

- `fit_mle_AR1`, its alias `model_for_loglikelihood_AR1_full`, `likelihoods`, and the `fit_mle!`
  methods.
- No longer exported, and reachable as `SmoothPeriodicStatsModels.name`: `bic_sel`, `nb_param`,
  `polynomial_trigo`, `OptimMLE`, `complete_loglikelihood`, `sort_wrt_ref!` and
  `norm_cdf_2d_vfast`.

### Fixed

- The HMM likelihoods and fits use the transition matrix of the period index a step leaves,
  also when `n2t` skips an index, such as 29 February in a non-leap year.
- `complete_loglikelihood` of the spatial HMM, which could not run.
