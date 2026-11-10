```@meta
EditURL = "../../../examples/tuto_mixture.jl"
```

````@example 2-tuto_mixture
using Markdown#hide
````

# [Seasonal mixture](@id TutoMixture)

A mixture whose weights and component parameters vary smoothly along a cycle of ``T`` steps:

```math
y_n \sim \sum_{k=1}^K \alpha_k(t_n)\, f_k\big(\,\cdot\,;\ \theta_k(t_n)\big),
```

with ``t_n \in 1{:}T`` the position of observation ``n`` in the cycle.

As a toy example, take waiting times that come from one of two regimes, short or long. How often
each regime occurs, and how long its waits are, both change along a cycle of 50 steps.

````@example 2-tuto_mixture
using SmoothPeriodicStatsModels
using Optimization, OptimizationOptimJL, SimpleNonlinearSolve, ForwardDiff # loads the fitting routines
using Distributions, Random
using CairoMakie
````

## A model with known coefficients

`SmoothSeasonalMixture(Exponential, K, d, T; init)` is a mixture of `K` exponential laws whose
scales, and the mixing weights, are trigonometric polynomials of degree `d` (through a `log` for
the scales and a softmax for the weights). `init` gives each component's starting scale.

The coefficients sit in `parameters(m)`: `α` for the weights (component `K` is the reference, so
there is one column per other component), and `component_coefs(m, k)` for component `k`.

````@example 2-tuto_mixture
T = 50
truth = SmoothSeasonalMixture(Exponential, 2, 1, T; init=[1.0, 5.0])
parameters(truth).α .= [0.3, 1.2, -0.6]
component_coefs(truth, 1) .= [0.0, 0.4, 0.0]
component_coefs(truth, 2) .= [log(6.0), -0.3, 0.3]
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

`rand(m, N)` draws `N` independent observations, observation `n` at position
`n_to_t(N, T)[n]` in the cycle. `m(t)` is the `MixtureModel` at one position.

````@example 2-tuto_mixture
Random.seed!(2)
N = 20_000
y = rand(truth, N)
truth(10)
````

`fit_mle` runs the EM algorithm from the coefficients of the model it is given. Starting
components must differ, here by their scales.

````@example 2-tuto_mixture
m0 = SmoothSeasonalMixture(Exponential, 2, 1, T; init=[0.5, 3.0])
m, history = fit_mle(m0, y; maxiter=1000)
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
f_gamma = fit(SeasonalFit, m_gamma, y; maxiter=1000)
f_exp = fit(SeasonalFit, m0, y; maxiter=1000)
[(model="Exponential + Exponential", dof=dof(f_exp), bic=round(bic(f_exp); digits=1)),
    (model="Exponential + Gamma", dof=dof(f_gamma), bic=round(bic(f_gamma); digits=1))]
````

The data are a mixture of exponentials, and the BIC prefers the model without the extra shape
parameters. `components(m)[k][t]` is component `k` at position `t`; here is the range of the
fitted `Gamma` shape over the cycle:

````@example 2-tuto_mixture
extrema(shape.(components(f_gamma.model)[2]))
````

`seasonal_family` lists the families available as components (`Exponential`, `Rayleigh`,
`Gamma`, `Weibull`, `Normal`, `LogNormal`); its docstring shows how to add one.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

