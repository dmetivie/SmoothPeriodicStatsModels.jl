# [Reference](@id reference)

```@meta
CurrentModule = SmoothPeriodicStatsModels
```

The API by task. `fit_mle`, `fit` and the functions of the starting-point and solver sections
are fitting routines: they need `using Optimization, SimpleNonlinearSolve`, and every fit takes
its optimization algorithm as `solver`.

```@contents
Pages = ["95-reference.md"]
Depth = 2:2
```

## Models

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> t in (SmoothAR1, SmoothSeasonalMixture, SmoothARPeriodicHMM, SmoothsARPeriodicHMM,
    SmoothARPeriodicHMMSpatial, SmoothPeriodicHMM, ar1_spec, seasonal_family, mixture_weight_spec)
```

## Coefficients and curves

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> t in (parameters, nparams, curve, params, coef, degrees,
    ncolumns, column, component_coefs, component_range, components, ncomponents,
    mixture_weights, mixtures, emission_coefs, emission_range)
```

## Evaluated models

The arrays the coefficients generate, one entry per position in the cycle, to plot or simulate
from.

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> t in (AR1, ARPeriodicHMM, sARPeriodicHMM, ARPeriodicHMMSpatial, PeriodicHMMDists,
    Trig2ARPeriodicHMM, Trig2ARPeriodicHMMSpatial, randARPeriodicHMM)
```

## Simulation

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> t === rand
```

## Fitting

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> t in (fit_mle, FitHistory, converged)
```

## Starting points for the HMMs

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> t in (fit_mle_all_slices, fit_θ, fit_θ!, fit_θᴬ!, fit_θᴮ!, fit_mle_RO)
```

## Likelihood and decoding

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> t in (loglikelihood, loglikelihoods, viterbi)
```

## Model selection

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> t in (SeasonalFit, fit, dof, nobs)
```

## HiddenMarkovModels.jl interface

See the [guide](@ref hmms-guide).

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> t in (SeasonalControls, SmoothPeriodicStatsModels.HMMs.transition_matrix,
    SmoothPeriodicStatsModels.HMMs.obs_distributions)
```

## Solvers

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> t in (default_nlls_solver, default_adtype, default_ensemble,
    solve_unconstrained, solve_nlls, fit_loss_optim)
```

## Trigonometric kernels and objectives

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> t in (αₜ, μₜ, ρₜ, σₜ, ar1_nll, ar1_sufficient_statistics)
```

## Re-exported

The bases, links and generic time-varying layer (`ParamSpec`, `TrigBasis`, `ConstantBasis`,
`FourierBasis`, `PolynomialBasis`, `IndicatorBasis`, `BSplineBasis`, `CyclicBSplineBasis`,
`IntervalLink`, `paramnames`, `ncoef`, `coefs`, `blockindex`, `blockrange`, `design_matrix`,
`nbasis`, `TimeVaryingFamily`, `PeriodicDesign`, `distributions`, `PeriodicNLL`,
`PointwiseNLL`, `group_by_time`, `TimeVaryingFit`) are those of
[TimeVaryingParameters.jl](https://github.com/arnaud-ma/TimeVaryingParameters.jl), not
registered yet and included as the submodule `SmoothPeriodicStatsModels.TimeVaryingParameters`;
the [bases guide](@ref bases-guide) describes them. The links
`IdentityLink`, `LogLink` and `LogitLink` come from
[LinkFunctions.jl](https://github.com/JuliaStats/LinkFunctions.jl), `ClassicEM` and
`StochasticEM` from [ExpectationMaximization.jl](https://github.com/dmetivie/ExpectationMaximization.jl),
and `n_to_t` from [PeriodicHiddenMarkovModels.jl](https://github.com/dmetivie/PeriodicHiddenMarkovModels.jl).

## Index

```@index
Pages = ["95-reference.md"]
```
