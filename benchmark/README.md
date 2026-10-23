# Benchmarks

`benchmarks.jl` defines a `SUITE` of [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl)
groups covering the trigonometric utilities, `rand`/`loglikelihoods`/`viterbi`/`fit_mle` for
`ARPeriodicHMM` and `ARPeriodicHMMSpatial`, the seasonal mixture (EM and direct maximization) and
the periodic AR(1).

Every `fit_mle` entry uses `tol = 0` together with a fixed `maxiter`, so each sample does exactly
the same amount of work: a change in convergence speed cannot be mistaken for a change in the cost
of an iteration.

## On a pull request

`.github/workflows/benchmark_action.yml` runs
[AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl) on every PR against
`master` and posts the comparison as a comment.

## Locally

Compare two revisions (`benchpkg` comes from AirspeedVelocity.jl):

```bash
julia -e 'using Pkg; Pkg.add("AirspeedVelocity")'   # provides the `benchpkg` script
benchpkg SmoothPeriodicStatsModels --rev=master,HEAD --bench-on=HEAD
```

Or just run the suite in this environment:

```julia
julia --project=benchmark -e '
    include("benchmark/benchmarks.jl")
    results = run(SUITE; verbose=true)
    foreach(println, sort(collect(BenchmarkTools.leaves(results)); by=first))'
```
