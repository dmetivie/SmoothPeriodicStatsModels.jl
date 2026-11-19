# [Seasonal families](@id families-guide)

```@meta
CurrentModule = SmoothPeriodicStatsModels
```

A seasonal family is a distribution whose parameters vary with the period index: one basis and
one link per parameter, in the order the distribution's constructor takes them. Families are the
components of a [`SmoothSeasonalMixture`](@ref) and the emission laws of a
[`SmoothPeriodicHMM`](@ref). These are declared:

| Family | Parameters and links |
|:--|:--|
| `Exponential` | scale `θ`, log |
| `Rayleigh` | scale `σ`, log |
| `Gamma` | shape `α` and scale `θ`, log |
| `Weibull` | shape `α` and scale `θ`, log |
| `Normal` | mean `μ`, identity; standard deviation `σ`, log |
| `LogNormal` | `μ`, identity; `σ`, log |
| `Bernoulli` | success probability `p`, logit |
| `Poisson` | mean `λ`, log |

In a model constructor, `Normal => 2` stands for `seasonal_family(Normal, TrigBasis(T, 2))`: each
parameter a trigonometric polynomial of degree 2. `seasonal_family(Normal, basis)` takes any
other [basis](@ref bases-guide), such as `CyclicBSplineBasis(T, 8)`.

## Declaring a custom family

One method of [`seasonal_family`](@ref) declares a family. For a `Pareto` law with both
parameters positive:

```julia
using SmoothPeriodicStatsModels, Distributions

SmoothPeriodicStatsModels.seasonal_family(::Type{Pareto}, basis) =
    TimeVaryingFamily(Pareto, ParamSpec(α=basis => LogLink(), θ=basis => LogLink()))
```

The method is declared on the type without its parameters, `Pareto` and not
`Pareto{Float64}`. The family can then be a mixture component or an HMM emission:

```julia
m0 = SmoothSeasonalMixture([Exponential => 1, Pareto => 1], T; init=[1.0, (α=2.0, θ=1.0)])
m, history = fit_mle(m0, y; n2t, solver=LBFGS())

h0 = SmoothPeriodicHMM(Pareto, 2, 1, T; init=[(α=2.0, θ=1.0), (α=4.0, θ=1.0)])
h, history = fit_mle(h0, y; n2t, solver=LBFGS())
```

The M-step then maximizes the weighted loglikelihood of the family directly: one matrix-vector
product per parameter and `T` distribution constructions per iteration, not one per
observation.

## One basis per parameter, and multivariate laws

`seasonal_family` gives every parameter the same basis. A `TimeVaryingFamily` built directly
takes one basis and one link per parameter, and any function that builds the law from them.
Here a bivariate normal law, with a constant second standard deviation:

```julia
b = TrigBasis(T, 1)
bivariate = TimeVaryingFamily(
    (μ1, μ2, σ1, σ2, ρ) -> MvNormal([μ1, μ2], [σ1^2 ρ*σ1*σ2; ρ*σ1*σ2 σ2^2]),
    ParamSpec(μ1=b => IdentityLink(), μ2=b => IdentityLink(), σ1=b => LogLink(),
        σ2=TrigBasis(T, 0) => LogLink(), ρ=b => IntervalLink(-1, 1)))
```

Every parameter is a scalar. A multivariate law is an emission block of a `SmoothPeriodicHMM`,
covering as many rows of the observations as its length; the [tutorial](@ref TutoHMMDists) fits
this law and one built from a copula. A seasonal mixture takes univariate families.

## A faster M-step

Some families reduce to sufficient statistics per period index, and the M-step then never
touches the observations inside the solver. That is a method of `update_dist`, the weighted fit
of one family, written for it. For the seasonal `Rayleigh`, whose scale is `S(t) = exp(P(t))`
and whose log-density is `log y - 2 log S - y² / 2S²`:

```julia
using SmoothPeriodicStatsModels, Distributions
const SPSM = SmoothPeriodicStatsModels

rayleigh_nll(θ, (s_γ, s_γy², basis)) =
    sum(2 * s_γ[t] * P + s_γy²[t] * exp(-2P) / 2 for (t, P) in enumerate(basis * θ))

function SPSM.update_dist(fam::TimeVaryingFamily{SPSM.DistConstructor{Rayleigh}},
    design::PeriodicDesign, θ, γ, obs, n_in_t; solver, adtype, solve_kwargs)
    T, total = design.nT, sum(γ)
    s_γ = [sum(γ[n] for n in n_in_t[t]; init=0.0) / total for t = 1:T]
    s_γy² = [sum(γ[n] * obs[n]^2 for n in n_in_t[t]; init=0.0) / total for t = 1:T]
    θ = solve_unconstrained(rayleigh_nll, collect(float.(θ)), (s_γ, s_γy², design.designs[1]);
        solver, adtype, solve_kwargs)
    return distributions(fam, design, θ), θ
end
```

Dividing by the total weight leaves the optimum unchanged and keeps the gradient independent of
the number of observations, which keeps the first steps of `LBFGS` from pushing the link into
underflow.

Nothing else changes: `fit_mle` is the same call and reaches the same estimator, for a mixture
and for a `SmoothPeriodicHMM` alike. The test suite checks that the two routes agree to `1e-8`
on the coefficients. `Exponential` has such a method built in, and its EM runs about 3.5 times
faster than through the generic M-step.

The emissions of the Bernoulli hidden Markov models are not pluggable this way: their memory
encoding makes the Bernoulli family part of the model.
