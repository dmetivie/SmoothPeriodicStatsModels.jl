using Markdown#hide
import Pkg#src
Pkg.activate(joinpath(@__DIR__, "..", "docs"))#src

md"""
# [Seasonal AR(1)](@id TutoAR1)

A periodic autoregressive process of order 1,

```math
y_n = \mu(t_n) + \rho(t_n)\, y_{n-1} + \sigma(t_n)\, \epsilon_n, \qquad \epsilon_n \sim \mathcal{N}(0, 1),
```

where ``t_n \in 1{:}T`` is the period index of step ``n`` in a cycle of ``T`` steps (`n2t[n]`, see
the [data layout](@ref data-layout)). Each parameter is a trigonometric polynomial of degree
``d``,

```math
P(t) = c_0 + \sum_{\ell=1}^{d} \left( c_{2\ell-1} \cos\frac{2\pi \ell t}{T} + c_{2\ell} \sin\frac{2\pi \ell t}{T} \right),
```

pushed through a link that keeps it in its domain:

```math
\mu(t) = P_\mu(t), \qquad
\rho(t) = 2\,\operatorname{logistic}\big(P_\rho(t)\big) - 1 \in (-1, 1), \qquad
\sigma(t) = \exp\big(P_\sigma(t)\big) > 0,
```

with ``\operatorname{logistic}(x) = 1/(1 + e^{-x})``.

This toy example simulates such a process with a known seasonality, fits it back, and chooses
the degree of the polynomials.
"""

using SmoothPeriodicStatsModels
using Optimization, SimpleNonlinearSolve # loads the fitting routines
using OptimizationOptimJL # the optimization algorithm, LBFGS
using Random
using CairoMakie

solver = LBFGS()   # every fit below takes its optimization algorithm explicitly

md"""
## A model with known coefficients

`SmoothAR1(d, T)` is a model whose three parameters are trigonometric polynomials of degree `d`
over a cycle of `T` steps. Its coefficients start at zero. `parameters(m)` returns them as a
`ComponentArray` with one block per parameter: `parameters(m).μ` is ``(c_0, c_1, \dots, c_{2d})``
of ``P_\mu``, the intercept then a cosine and a sine coefficient per harmonic.
"""

T = 40
truth = SmoothAR1(1, T)
ps = parameters(truth)
ps.μ .= [1.0, 0.8, -0.4]   # μ(t) = 1 + 0.8 cos(2πt/T) - 0.4 sin(2πt/T)
ps.ρ .= [0.5, 0.6, 0.2]
ps.σ .= [-0.7, 0.3, 0.0]   # σ(t) = exp(-0.7 + 0.3 cos(2πt/T))
truth

md"""
`curve(m, name)` evaluates one parameter over the cycle, link applied.
"""

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

md"""
## Simulate and fit

`rand(m, N; y₁)` simulates `N` steps from the initial value `y₁`. Step `n` has period index
`n2t[n]`, with `n2t = n_to_t(N, T)` by default (`1, 2, …, T, 1, 2, …`).
"""

Random.seed!(1)
N = 20_000
y = rand(truth, N; y₁=0.0)

md"""
`fit_mle(model, y; solver)` starts from the coefficients of `model` and returns the fitted model together
with a [`FitHistory`](@ref).
"""

m, history = fit_mle(SmoothAR1(1, T), y; solver)
history

#-

plot_curves([truth, m], ["truth", "fit"])

md"""
The fitted coefficients, block by block; `parameters(m).σ` is the block of `σ`:
"""

parameters(m)

md"""
`AR1(m)` gives the curves themselves, and `rand(m, N)` simulates from the fitted model.

## Choosing the degree

`fit(SeasonalFit, model, y; solver)` runs the same fit and keeps what model selection needs, so
`loglikelihood`, `dof`, `nobs`, `aic` and `bic` apply to it.
"""

fits = [fit(SeasonalFit, SmoothAR1(d, T), y; solver) for d in 0:3]
[(degree=d, dof=dof(f), bic=round(bic(f); digits=1)) for (d, f) in zip(0:3, fits)]

md"""
The BIC is lowest at the degree the data were simulated with.

## A parametrization of your own

The three parameters need not share a basis. Any `ParamSpec` naming `μ`, `ρ` and `σ`, in that
order and with `σ` behind a `LogLink()`, defines a model. Here `ρ` is constant (a degree-0
`TrigBasis` is just an intercept) and `μ` has degree 2.
"""

spec = ParamSpec(
    μ=TrigBasis(T, 2) => IdentityLink(),
    ρ=TrigBasis(T, 0) => IntervalLink(-1, 1),
    σ=TrigBasis(T, 1) => LogLink(),
)
f2 = fit(SeasonalFit, SmoothAR1(spec, T), y; solver)
(ρ=coef(f2, :ρ), dof=dof(f2), bic=round(bic(f2); digits=1))

md"""
The true `ρ` varies over the cycle, and the BIC says so: this model is worse than the degree-1
fit above.
"""
