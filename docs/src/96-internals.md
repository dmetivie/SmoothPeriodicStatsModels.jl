# [Internals](@id internals)

```@meta
CurrentModule = SmoothPeriodicStatsModels
```

Documented functions and types that are not exported: the building blocks of the fits, for
reading or extending the code. They may change without notice.

```@autodocs
Modules = [SmoothPeriodicStatsModels, Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)]
Filter = t -> !(applicable(nameof, t) && Base.isexported(SmoothPeriodicStatsModels, nameof(t))) &&
    !(t in (rand, SmoothPeriodicStatsModels.HMMs.transition_matrix,
    SmoothPeriodicStatsModels.HMMs.obs_distributions))
```
