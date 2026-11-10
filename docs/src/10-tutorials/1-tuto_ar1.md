```@meta
EditURL = "../../../examples/tuto_ar1.jl"
```

````@example 1-tuto_ar1
using Markdown#hide
````

# [Seasonal AR(1)](@id TutoAR1)

A periodic autoregressive process of order 1,

```math
y_n = \mu(t_n) + \rho(t_n)\, y_{n-1} + \sigma(t_n)\, \epsilon_n, \qquad \epsilon_n \sim \mathcal{N}(0, 1),
```

where ``t_n \in 1{:}T`` is the position of step ``n`` in a cycle of ``T`` steps. Each parameter is
a smooth function of ``t``: a trigonometric polynomial pushed through a link that keeps it in its
domain, ``\rho \in (-1, 1)`` and ``\sigma > 0``.

This toy example simulates such a process with a known seasonality, fits it back, and chooses
the degree of the polynomials.

````@example 1-tuto_ar1
using SmoothPeriodicStatsModels
using Optimization, OptimizationOptimJL, SimpleNonlinearSolve, ForwardDiff # loads the fitting routines
using Random
using CairoMakie
````

## A model with known coefficients

`SmoothAR1(d, T)` is a model whose three parameters are trigonometric polynomials of degree `d`
over a cycle of `T` steps. Its coefficients start at zero. `parameters(m)` returns them as a
`ComponentArray` with one block per parameter: the intercept, then a cosine and a sine
coefficient per harmonic.

````@example 1-tuto_ar1
T = 40
truth = SmoothAR1(1, T)
ps = parameters(truth)
ps.μ .= [1.0, 0.8, -0.4]
ps.ρ .= [0.5, 0.6, 0.2]
ps.σ .= [-0.7, 0.3, 0.0]
truth
````

`curve(m, name)` evaluates one parameter over the cycle, link applied.

````@example 1-tuto_ar1
function plot_curves(models, labels)
    fig = Figure(size=(900, 260))
    for (i, nm) in enumerate((:μ, :ρ, :σ))
        ax = Axis(fig[1, i], title=string(nm), xlabel="t")
        for (m, label) in zip(models, labels)
            lines!(ax, 1:T, curve(m, nm); label)
        end
        i == 3 && length(models) > 1 && axislegend(ax, position=:ct)
    end
    return fig
end

plot_curves([truth], ["truth"])
````

## Simulate and fit

`rand(m, N; y₁)` simulates `N` steps from the initial value `y₁`. Step `n` sits at position
`n2t[n]` in the cycle, with `n2t = n_to_t(N, T)` by default (`1, 2, …, T, 1, 2, …`).

````@example 1-tuto_ar1
Random.seed!(1)
N = 20_000
y = rand(truth, N; y₁=0.0)
````

`fit_mle(model, y)` starts from the coefficients of `model` and returns the fitted model together
with a [`FitHistory`](@ref).

````@example 1-tuto_ar1
m, history = fit_mle(SmoothAR1(1, T), y)
history
````

````@example 1-tuto_ar1
plot_curves([truth, m], ["truth", "fit"])
````

The fitted coefficients, block by block; `parameters(m).σ` is the block of `σ`:

````@example 1-tuto_ar1
parameters(m)
````

`AR1(m)` gives the curves themselves, and `rand(m, N)` simulates from the fitted model.

## Choosing the degree

`fit(SeasonalFit, model, y)` runs the same fit and keeps what model selection needs, so
`loglikelihood`, `dof`, `nobs`, `aic` and `bic` apply to it.

````@example 1-tuto_ar1
fits = [fit(SeasonalFit, SmoothAR1(d, T), y) for d in 0:3]
[(degree=d, dof=dof(f), bic=round(bic(f); digits=1)) for (d, f) in zip(0:3, fits)]
````

The BIC is lowest at the degree the data were simulated with.

## A parametrization of your own

The three parameters need not share a basis. Any `ParamSpec` naming `μ`, `ρ` and `σ`, in that
order and with `σ` behind a `LogLink()`, defines a model. Here `ρ` is constant (a degree-0
`TrigBasis` is just an intercept) and `μ` has degree 2.

````@example 1-tuto_ar1
spec = ParamSpec(
    μ=TrigBasis(T, 2) => IdentityLink(),
    ρ=TrigBasis(T, 0) => IntervalLink(-1, 1),
    σ=TrigBasis(T, 1) => LogLink(),
)
f2 = fit(SeasonalFit, SmoothAR1(spec, T), y)
(ρ=coef(f2, :ρ), dof=dof(f2), bic=round(bic(f2); digits=1))
````

The true `ρ` varies over the cycle, and the BIC says so: this model is worse than the degree-1
fit above.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

