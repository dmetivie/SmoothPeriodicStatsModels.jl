# SmoothPeriodicStatsModels

[![Build Status](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://dmetivie.github.io/SmoothPeriodicStatsModels.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://dmetivie.github.io/SmoothPeriodicStatsModels.jl/dev)
[![Docs workflow Status](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![All Contributors](https://img.shields.io/github/all-contributors/dmetivie/SmoothPeriodicStatsModels.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

Statistical models whose parameters change smoothly along a cycle, such as the days of a year,
fitted by maximum likelihood:

- an autoregressive process of order 1;
- mixtures with varying weights and component parameters;
- hidden Markov models with autoregressive Bernoulli emissions, optionally correlated in space;
- hidden Markov models with any `Distributions.jl` emission laws, univariate or multivariate.

Each parameter is a basis expansion, a trigonometric polynomial by default, pushed through a link
that keeps it in its domain. Estimating each day of the year on its own gives noisy curves. One
estimate per month or season, the usual remedy, gives curves that jump at every change of month,
and the months still have to be glued back together. In a hidden Markov model, either way, the
meaning of the regimes drifts from one day or month to the next. A few coefficients per curve
avoid all three problems.

The package was written for the multisite rain generator of
[Gobet, Métivier and Parey (2025)](https://doi.org/10.5194/ascmo-11-159-2025), built on the
seasonal hidden Markov models of [Touron (2019)](https://doi.org/10.1007/s11222-019-09854-4),
and serves the weather generators of
[StochasticWeatherGenerators.jl](https://github.com/dmetivie/StochasticWeatherGenerators.jl).

## Installation

```julia
import Pkg
Pkg.add("SmoothPeriodicStatsModels")
```

The fitting routines need Optimization.jl and SimpleNonlinearSolve, `using Optimization,
SimpleNonlinearSolve`, and every fit takes its optimization algorithm explicitly as `solver`,
from the package of your choice.

## Example

A hidden Markov model with three states, from dry to wet, drives six binary series, each
remembering its previous value, over a cycle of 100 steps. Every curve is a trigonometric
polynomial of degree 1, so three coefficients: an intercept, a cosine and a sine.

```julia
using SmoothPeriodicStatsModels
using Optimization, SimpleNonlinearSolve   # loads the fitting routines
using OptimizationOptimJL                  # the solver passed to the fit, LBFGS()
using Random

K, D, order, T, d = 3, 6, 1, 100, 1
θᴬ = zeros(K, K - 1, 2d + 1)        # transitions, through a softmax: state K is the reference
for k in 1:K, l in 1:K-1
    θᴬ[k, l, :] = k == l ? [2.0, 0.5, 0.0] : [-1.0, 0.0, 0.5]   # states persist
end
θᴮ = zeros(K, D, 2^order, 2d + 1)   # emissions, through a logistic; h = 2 after a 1
for k in 1:K, j in 1:D, h in 1:2^order
    θᴮ[k, j, h, :] = [-2.0 + 2(k - 1) + (h - 1), 0.8, 0.2j]
end
truth = SmoothARPeriodicHMM(fill(1 / K, K), θᴬ, θᴮ, T)

# simulate 50 cycles; Y_past holds the observations before the first one
Random.seed!(1)
n2t = n_to_t(50T, T)
Y_past = zeros(Bool, D, order)
z, Y = rand(truth, n2t; Y_past, seq=true)

# fit back, starting from flat curves at the true intercepts
θᴬ₀, θᴮ₀ = zeros(size(θᴬ)), zeros(size(θᴮ))
θᴬ₀[:, :, 1], θᴮ₀[:, :, :, 1] = θᴬ[:, :, 1], θᴮ[:, :, :, 1]
m0 = SmoothARPeriodicHMM(fill(1 / K, K), θᴬ₀, θᴮ₀, T)
m, history = fit_mle(m0, Y, Y_past; n2t, solver=LBFGS())

parameters(m).B              # the fitted emission coefficients
ARPeriodicHMM(m).A           # the K × K × T transition matrices they give
viterbi(m, Y, Y_past; n2t)   # the most likely state sequence
```

The [documentation](https://dmetivie.github.io/SmoothPeriodicStatsModels.jl/dev) has a tutorial
per model, guides to the data layout, fitting and HiddenMarkovModels.jl, and the reference. The
[tutorials of StochasticWeatherGenerators.jl](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/)
fit these models to French weather stations.

## How to Cite

If you use SmoothPeriodicStatsModels.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl/blob/main/CITATION.cff).

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dmetivie"><img src="https://avatars.githubusercontent.com/u/46794064?v=4&s=100" width="100px;" alt="David Métivier"/><br /><sub><b>David Métivier</b></sub></a><br /><a href="https://github.com/dmetivie/SmoothPeriodicStatsModels.jl/commits?author=dmetivie" title="Code">💻</a> <a href="https://github.com/dmetivie/SmoothPeriodicStatsModels.jl/commits?author=dmetivie" title="Documentation">📖</a> <a href="#research-dmetivie" title="Research">🔬</a> <a href="#maintenance-dmetivie" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/caroline-cognot"><img src="https://github.com/caroline-cognot.png?size=100" width="100px;" alt="Caroline Cognot"/><br /><sub><b>Caroline Cognot</b></sub></a><br /><a href="https://github.com/dmetivie/SmoothPeriodicStatsModels.jl/commits?author=caroline-cognot" title="Code">💻</a> <a href="#research-caroline-cognot" title="Research">🔬</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
