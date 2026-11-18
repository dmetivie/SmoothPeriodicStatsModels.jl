using Markdown#hide
import Pkg#src
Pkg.activate(joinpath(@__DIR__, "..", "docs"))#src

md"""
# [Seasonal mixture](@id TutoMixture)

A mixture whose weights and component parameters vary smoothly along a cycle of ``T`` steps:

```math
y_n \sim \sum_{k=1}^K \alpha_k(t_n)\, f_k\big(\,\cdot\,;\ \theta_k(t_n)\big),
```

where

- ``t_n \in 1{:}T`` is the period index of observation ``n`` in the cycle (`n2t[n]`, see the
  [data layout](@ref data-layout));
- ``f_k(\,\cdot\,; \theta)`` is the density of component ``k``, a law of a given family (here
  exponential) with parameter ``\theta``;
- ``\alpha_k(t) \geq 0`` is the weight of component ``k`` at period index ``t``, with
  ``\sum_k \alpha_k(t) = 1``.

The weights and the component parameters are trigonometric polynomials of degree ``d``,
``P(t) = c_0 + \sum_{\ell=1}^{d} c_{2\ell-1} \cos(2\pi \ell t/T) + c_{2\ell} \sin(2\pi \ell t/T)``,
through a link: a softmax for the weights, with component ``K`` as the reference, and a log for
a scale,

```math
\alpha_k(t) = \frac{e^{P^\alpha_k(t)}}{\sum_{l=1}^{K} e^{P^\alpha_l(t)}}, \quad P^\alpha_K = 0,
\qquad \theta_k(t) = \exp\big(P^\theta_k(t)\big).
```

As a toy example, take waiting times that come from one of two regimes, short or long. How often
each regime occurs, and how long its waits are, both change along a cycle of 50 steps.
"""

using SmoothPeriodicStatsModels
using Optimization, SimpleNonlinearSolve # loads the fitting routines
using OptimizationOptimJL # the optimization algorithm, LBFGS
using Distributions, Random
using CairoMakie

solver = LBFGS()   # every fit below takes its optimization algorithm explicitly

md"""
## A model with known coefficients

`SmoothSeasonalMixture(Exponential, K, d, T; init)` is that mixture with exponential
components, ``f_k(y; \theta) = e^{-y/\theta}/\theta``, ``\theta`` being the scale. `init` gives
each component's starting scale.

The coefficients sit in `parameters(m)`: `parameters(m).α` holds those of ``P^\alpha_k``, one
column per component but the reference ``K``, and `component_coefs(m, k)` those of
``P^\theta_k``.
"""

T = 50
truth = SmoothSeasonalMixture(Exponential, 2, 1, T; init=[1.0, 5.0])
parameters(truth).α .= [0.3, 1.2, -0.6]
component_coefs(truth, 1) .= [0.0, 0.4, 0.0]
component_coefs(truth, 2) .= [log(6.0), -0.3, 0.3]
truth

md"""
`mixture_weights(m)` is the `K × T` matrix of weights, and `curve(m, :θ)` the `K × T` matrix of
scales (`θ` is the name of `Exponential`'s parameter).
"""

function plot_mixture(models, labels)
    fig = Figure(size=(900, 280))
    ax1 = Axis(fig[1, 1], title="weight of the short regime", xlabel="t")
    ax2 = Axis(fig[1, 2], title="scale of each regime", xlabel="t", yscale=log10)
    for (m, label, style) in zip(models, labels, (:solid, :dash))
        lines!(ax1, 1:T, mixture_weights(m)[1, :]; label, linestyle=style)
        θ = curve(m, :θ)
        lines!(ax2, 1:T, θ[1, :]; label="short, " * label, linestyle=style)
        lines!(ax2, 1:T, θ[2, :]; label="long, " * label, linestyle=style)
    end
    axislegend(ax1)
    Legend(fig[1, 3], ax2)
    return fig
end

plot_mixture([truth], ["truth"])

md"""
## Simulate and fit

`rand(m, N)` draws `N` independent observations, observation `n` at period index
`n_to_t(N, T)[n]`. `m(t)` is the `MixtureModel` at one period index.
"""

Random.seed!(2)
N = 20_000
y = rand(truth, N)
truth(10)

md"""
`fit_mle` runs the EM algorithm from the coefficients of the model it is given. Starting
components must differ, here by their scales.
"""

m0 = SmoothSeasonalMixture(Exponential, 2, 1, T; init=[0.5, 3.0])
m, history = fit_mle(m0, y; solver, maxiter=1000)
history

#-

plot_mixture([truth, m], ["truth", "fit"])

md"""
The fitted coefficients are `parameters(m).α` for the weights and `component_coefs(m, k)` for
component `k`. The long regime:
"""

(fit=round.(component_coefs(m, 2); digits=2), truth=round.(component_coefs(truth, 2); digits=2))

md"""
## Components of different families

Each component carries its own family and degree. A list of `distribution => degree` pairs
builds such a mixture; for a family with several parameters, `init` takes a `NamedTuple`. Here
the long regime is modelled by a `Gamma` law, which contains the exponential as the case of shape
1.
"""

m_gamma = SmoothSeasonalMixture([Exponential => 1, Gamma => 1], T;
    init=[0.5, (α=1.5, θ=3.0)])
f_gamma = fit(SeasonalFit, m_gamma, y; solver, maxiter=1000)
f_exp = fit(SeasonalFit, m0, y; solver, maxiter=1000)
[(model="Exponential + Exponential", dof=dof(f_exp), bic=round(bic(f_exp); digits=1)),
    (model="Exponential + Gamma", dof=dof(f_gamma), bic=round(bic(f_gamma); digits=1))]

md"""
The data are a mixture of exponentials, and the BIC prefers the model without the extra shape
parameters. `components(m)[k][t]` is component `k` at period index `t`; here is the range of the
fitted `Gamma` shape over the cycle:
"""

extrema(shape.(components(f_gamma.model)[2]))

md"""
`seasonal_family` lists the families available as components (`Exponential`, `Rayleigh`,
`Gamma`, `Weibull`, `Normal`, `LogNormal`, `Bernoulli`, `Poisson`); its docstring shows how to
add one.
"""
