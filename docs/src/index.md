```@meta
CurrentModule = SmoothPeriodicStatsModels
```

# SmoothPeriodicStatsModels

Statistical models whose parameters change smoothly along a cycle, such as the days of a year,
fitted by maximum likelihood.

## Why smooth?

Weather is the motivating case. The chance that a day is rainy changes over the year, and so
do the chance that a dry spell ends, the spread of temperatures, and how often the large-scale
weather sits in one regime or another. A stochastic weather generator, which simulates
plausible weather series for impact studies, has to reproduce all of that seasonality.

The direct approach estimates each day of the year on its own: 366 sets of parameters, each
from the thirty-odd observations of that date in thirty years of data. The estimates are noisy
and jump from one day to the next. In a hidden Markov model it is worse: the weather regimes
found for 14 March need not be the regimes found for 15 March.

Here each parameter is a smooth function of the position in the cycle: a few coefficients of a
basis expansion, by default a trigonometric polynomial, pushed through a link that keeps the
parameter in its domain (a probability in ``(0, 1)``, a scale positive). Every observation
informs every coefficient, the curves are smooth by construction, and the hidden regimes keep
one meaning all year round.

```@example landing
using SmoothPeriodicStatsModels # hide
using Optimization, OptimizationOptimJL, SimpleNonlinearSolve, ForwardDiff # hide
using CairoMakie, Random # hide
Random.seed!(1) # hide
T, years = 366, 30 # hide
p(t) = 1 / (1 + exp(-(-0.4 + 0.7cos(2π * t / T) + 0.3sin(2π * t / T) + 0.2cos(4π * t / T)))) # hide
rainy = [rand() < p(t) for t in 1:T, _ in 1:years] # hide
per_day = vec(sum(rainy; dims=2)) ./ years # hide
θ = fit_θᴮ!(zeros(5), per_day) # hide
begin # hide
    fig = Figure(size=(800, 300)) # hide
    ax = Axis(fig[1, 1], xlabel="day of the year", ylabel="ℙ(rainy day)", limits=(1, T, 0, 1)) # hide
    scatter!(ax, 1:T, per_day; markersize=4, color=(:gray, 0.7), label="one estimate per day") # hide
    lines!(ax, 1:T, @.(1 / (1 + exp(-$(design_matrix(TrigBasis(T, 2), 1:T) * θ)))); linewidth=3, label="5 coefficients for the year") # hide
    lines!(ax, 1:T, p.(1:T); color=:black, linestyle=:dash, label="truth") # hide
    axislegend(ax; position=:ct, orientation=:horizontal, framevisible=false) # hide
    fig # hide
end # hide
```

*The chance of a rainy day at one station, from thirty simulated years. A trigonometric
polynomial of degree 2 has five coefficients for the whole year.*

The package was written for the multisite rain generator of
[Gobet, Métivier and Parey (2025)](https://doi.org/10.5194/ascmo-11-159-2025), built on the
seasonal hidden Markov models of [Touron (2019)](https://doi.org/10.1007/s11222-019-09854-4),
and serves the weather generators of
[StochasticWeatherGenerators.jl](https://github.com/dmetivie/StochasticWeatherGenerators.jl).
Nothing ties it to weather or to time: any data indexed by a position in a cycle fit, and with
a non-periodic basis the position need not even wrap around, as in the
[tutorial on genes](@ref TutoGenomics).

## Models

| Model | |
|---|---|
| [`SmoothAR1`](@ref) | autoregressive process of order 1 |
| [`SmoothSeasonalMixture`](@ref) | mixture with varying weights and component parameters |
| [`SmoothARPeriodicHMM`](@ref) | hidden Markov model with autoregressive Bernoulli emissions |
| [`SmoothsARPeriodicHMM`](@ref) | the same, each series with its own memory order |
| [`SmoothARPeriodicHMMSpatial`](@ref) | the same, with spatially correlated emissions |

Every model holds its coefficients and works the same way:

```julia
m, history = fit_mle(m0, data...; n2t)   # fit, starting from the coefficients of m0
f = fit(SeasonalFit, m0, data...; n2t)   # the same fit, with loglikelihood, aic and bic
parameters(m)                            # the coefficients, by name
curve(m, :σ)                             # one parameter over the cycle
rand(m, n2t)                             # simulate
```

[Reading a fitted model](@ref reading-a-model) lists, for each model, where its coefficients,
curves and evaluated parameters are.

`n2t[n]` is the position of observation `n` in the cycle of `T` steps. Multivariate
observations are `D × N`, one column per step, and the models with memory take the
observations before the first one as `Y_past`. The [data layout](@ref data-layout) page says
this in full, including several sequences and gaps.

## Installation

```julia
import Pkg
Pkg.add("SmoothPeriodicStatsModels")
```

The models, simulation and likelihoods need nothing else. Fitting needs a solver stack, loaded
with

```julia
using Optimization, OptimizationOptimJL, SimpleNonlinearSolve, ForwardDiff
```

## Where to go next

The tutorials use toy data and build each model from known coefficients, simulate it and fit it
back.

- [Seasonal AR(1)](@ref TutoAR1): fit, choose the degree, define your own parametrization.
- [Seasonal mixture](@ref TutoMixture): two regimes, components of different families.
- [Periodic hidden Markov model](@ref TutoHMM): fit, find a starting point, decode, correlate
  the series in space.
- [Chromatin states along genes](@ref TutoGenomics): a B-spline basis over a position that is
  not periodic, and many short sequences.

The guides cover what the tutorials only use:

- [Data layout](@ref data-layout): positions, initial conditions, several sequences and gaps.
- [Fitting](@ref fitting-guide): solver options, starting points, model selection, saving a fit.
- [HiddenMarkovModels.jl](@ref hmms-guide): the HMMs as models of that package.

For real data, the tutorials of
[StochasticWeatherGenerators.jl](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/)
fit these models to French weather stations.

## Contributors

```@raw html
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
```
