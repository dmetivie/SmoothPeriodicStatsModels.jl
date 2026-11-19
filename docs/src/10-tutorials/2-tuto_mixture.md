```@codeblocks
line_counter = :continue
```

```@meta
EditURL = "../../../examples/tuto_mixture.jl"
```

````@example 2-tuto_mixture
using Markdown#hide
````

# [Seasonal mixture](@id TutoMixture)

We observe independent real observations ``Y_1, \dots, Y_N``. Observation ``n`` falls at period
index ``t_n \in \{1, \dots, T\}`` of a cycle of ``T`` steps (`n2t[n]`, see the
[data layout](@ref data-layout)). Each comes from one of ``K`` components, drawn with
probabilities that vary along the cycle: with ``Z_n \in \{1, \dots, K\}`` the unobserved
component of observation ``n``,

```math
\mathbb{P}(Z_n = k) = \alpha_k(t_n), \qquad
Y_n \mid Z_n = k \ \sim\ f_k\big(\,\cdot\,;\ \theta_k(t_n)\big),
```

so that ``Y_n`` has density ``y \mapsto \sum_{k=1}^K \alpha_k(t_n)\, f_k\big(y;\ \theta_k(t_n)\big)``,
where

- ``\alpha_k(t) \geq 0`` is the weight of component ``k`` at period index ``t``, with
  ``\sum_{k=1}^K \alpha_k(t) = 1``;
- ``f_k(\,\cdot\,; \theta)`` is the density of component ``k``, a law of a given family with
  parameter ``\theta``; here the exponential law of scale ``\theta``,
  ``f_k(y; \theta) = e^{-y/\theta}/\theta`` for ``y \geq 0``;
- ``\theta_k(t)`` is the parameter of component ``k`` at period index ``t``.

The weights and the parameters are trigonometric polynomials of degree ``d`` (any other
[basis](@ref bases-guide) can replace them),

```math
P(t) = c_0 + \sum_{\ell=1}^{d} \left( c_{2\ell-1} \cos\frac{2\pi \ell t}{T} + c_{2\ell} \sin\frac{2\pi \ell t}{T} \right),
```

pushed through a link: a softmax for the weights, with one polynomial ``P^\alpha_k`` for each
``1 \le k < K`` and component ``K`` as the reference, ``P^\alpha_K = 0``; and a log for the
scales, one polynomial ``P^\theta_k`` per component:

```math
\alpha_k(t) = \frac{e^{P^\alpha_k(t)}}{\sum_{l=1}^{K} e^{P^\alpha_l(t)}}, \qquad
\theta_k(t) = \exp\big(P^\theta_k(t)\big), \qquad k \in \{1, \dots, K\}.
```

As a toy example, take waiting times that come from one of two regimes, short or long. How often
each regime occurs, and how long its waits are, both change along a cycle of 50 steps.

````@example 2-tuto_mixture
using SmoothPeriodicStatsModels
using Optimization, SimpleNonlinearSolve # loads the fitting routines
using OptimizationOptimJL # the optimization algorithm, LBFGS
using Distributions, Random
using CairoMakie

solver = LBFGS()   # every fit below takes its optimization algorithm explicitly
````

## A model with known coefficients

`SmoothSeasonalMixture(Exponential, K, d, T; init)` is that mixture with exponential
components. `init` is required and gives each component a scale, constant over the cycle:
here ``\theta_1 = 1`` and ``\theta_2 = 6``. The constant term of ``P^\theta_k`` is then
``\log \theta_k``, its harmonics are zero, and the weights are equal.
[Starting values](@ref starting-values) gives the rule for every family.

The coefficients sit in `parameters(m)`: `parameters(m).α` holds those of ``P^\alpha_1, \dots,
P^\alpha_{K-1}``, one column per component but the reference ``K``, and
`component_coefs(m, k)` those of ``P^\theta_k``, ``(c_0, c_1, c_2)``.

````@example 2-tuto_mixture
T = 50
truth = SmoothSeasonalMixture(Exponential, 2, 1, T; init=[1.0, 6.0])
component_coefs(truth, 2)
````

The truth then sets the weights and the harmonics of the scales:

````@example 2-tuto_mixture
parameters(truth).α .= [0.3, 1.2, -0.6]
component_coefs(truth, 1)[2:3] .= [0.4, 0.0]
component_coefs(truth, 2)[2:3] .= [-0.3, 0.3]
truth
````

`mixture_weights(m)` is the `K × T` matrix of weights, and `curve(m, :θ)` the `K × T` matrix of
scales (`θ` is the name of `Exponential`'s parameter).

````@example 2-tuto_mixture
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
````

## Simulate and fit

`rand(m, N)` draws `N` independent observations, observation `n` at period index
`n_to_t(N, T)[n]`. `m(t)` is the `MixtureModel` at one period index.

````@example 2-tuto_mixture
Random.seed!(2)
N = 20_000
y = rand(truth, N)
truth(10)
````

`fit_mle` runs the EM algorithm from the coefficients of the model it is given. The starting
components must differ (otherwise the EM will not be able to distinguish the two identical states), here by their scales: 0.5 for the short waits, 3 for the long ones. Their
curves start flat and the weights equal.

````@example 2-tuto_mixture
m0 = SmoothSeasonalMixture(Exponential, 2, 1, T; init=[0.5, 3.0])
m, history = fit_mle(m0, y; solver, maxiter=1000)
history
````

````@example 2-tuto_mixture
plot_mixture([truth, m], ["truth", "fit"])
````

The fitted coefficients are `parameters(m).α` for the weights and `component_coefs(m, k)` for
component `k`. The long regime:

````@example 2-tuto_mixture
(fit=round.(component_coefs(m, 2); digits=2), truth=round.(component_coefs(truth, 2); digits=2))
````

## Components of different families

Each component carries its own family and degree. A list of `distribution => degree` pairs
builds such a mixture; for a family with several parameters, `init` takes a `NamedTuple`. Here
the long regime is modelled by a `Gamma` law, which contains the exponential as the case of shape
1.

````@example 2-tuto_mixture
m_gamma = SmoothSeasonalMixture([Exponential => 1, Gamma => 1], T;
    init=[0.5, (α=1.5, θ=3.0)])
f_gamma = fit(SeasonalFit, m_gamma, y; solver, maxiter=1000)
f_exp = fit(SeasonalFit, m0, y; solver, maxiter=1000)
[(model="Exponential + Exponential", dof=dof(f_exp), bic=round(bic(f_exp); digits=1)),
    (model="Exponential + Gamma", dof=dof(f_gamma), bic=round(bic(f_gamma); digits=1))]
````

The data are a mixture of exponentials, and the BIC prefers the model without the extra shape
parameters. `components(m)[k][t]` is component `k` at period index `t`; here is the range of the
fitted `Gamma` shape over the cycle:

````@example 2-tuto_mixture
extrema(shape.(components(f_gamma.model)[2]))
````

`seasonal_family` lists the families available as components (`Exponential`, `Rayleigh`,
`Gamma`, `Weibull`, `Normal`, `LogNormal`, `Bernoulli`, `Poisson`); its docstring shows how to
add one.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

