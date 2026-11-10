# [Seasonal families](@id families-guide)

```@meta
CurrentModule = SmoothPeriodicStatsModels
```

A component of a [`SmoothSeasonalMixture`](@ref) can be any distribution whose parameters this
package knows how to vary along the cycle: one basis and one link per parameter, in the order
the distribution's constructor takes them. `Exponential`, `Rayleigh`, `Gamma`, `Weibull`,
`Normal` and `LogNormal` are declared already.

## Declaring a family

One method of [`seasonal_family`](@ref) declares a family. For a `Pareto` law with both
parameters positive:

```julia
using SmoothPeriodicStatsModels, Distributions

SmoothPeriodicStatsModels.seasonal_family(::Type{Pareto}, basis) =
    TimeVaryingFamily(Pareto, ParamSpec(α=basis => LogLink(), θ=basis => LogLink()))
```

The method is declared on the type without its parameters, `Pareto` and not
`Pareto{Float64}`. The family can then be a component:

```julia
m0 = SmoothSeasonalMixture([Exponential => 1, Pareto => 1], T; init=[1.0, (α=2.0, θ=1.0)])
m, history = fit_mle(m0, y; n2t)
```

The M-step then maximizes the weighted loglikelihood of the family directly: one matrix-vector
product per parameter and `T` distribution constructions per iteration, not one per
observation.

## A faster M-step

Some families reduce to sufficient statistics per position in the cycle, and the M-step then
never touches the observations inside the solver. That is a method of `update_dist`, the M-step
of one component, written for the family. For the seasonal `Rayleigh`, whose scale is
`S(t) = exp(P(t))` and whose log-density is `log y - 2 log S - y² / 2S²`:

```julia
using SmoothPeriodicStatsModels, Distributions
const SPSM = SmoothPeriodicStatsModels

rayleigh_nll(θ, (s_γ, s_γy², basis)) =
    sum(2 * s_γ[t] * P + s_γy²[t] * exp(-2P) / 2 for (t, P) in enumerate(basis * θ))

function SPSM.update_dist(fam::TimeVaryingFamily{SPSM.DistConstructor{Rayleigh}},
    design::PeriodicDesign, θ, γ, obs, n_in_t;
    solver=default_solver(), adtype=default_adtype(), solve_kwargs=(;))
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

Nothing else changes: `fit_mle` is the same call and reaches the same estimator. The test suite
checks that the two routes agree to `1e-8` on the coefficients. `Exponential` has such a method
built in, and its EM runs about 3.5 times faster than through the generic M-step.

The emissions of the hidden Markov models are not pluggable this way: their memory encoding
makes the Bernoulli family part of the model.
