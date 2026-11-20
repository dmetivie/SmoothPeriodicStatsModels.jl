# [Data layout](@id data-layout)

```@meta
CurrentModule = SmoothPeriodicStatsModels
```

## Period indices

Every model runs over a cycle of `T` steps, and every function that takes data also takes
the period index of each observation, its position in the cycle: observation `n` has period
index `n2t[n] ∈ 1:T`. The default, `n_to_t(N, T)`, is `1, 2, …, T, 1, 2, …`, which suits data
that start at period index 1 and have no gap.

For daily data the period index is the day of the year. With `T = 366`, February 29 keeps its own
index, 60, and non-leap years skip it:

```@example data
using Dates
day_of_cycle(d::Date) = dayofyear(d) + (!isleapyear(d) && month(d) > 2)
day_of_cycle.([Date(2001, 2, 28), Date(2001, 3, 1), Date(2004, 2, 29), Date(2004, 3, 1)])
```

`StochasticWeatherGenerators.jl` calls this function `dayofyear_Leap`. Skipping an index is
fine: the step from observation `n` to `n + 1` uses the transition probabilities of period index
`n2t[n]`, the one the step leaves, whatever `n2t[n + 1]` is. In a non-leap year the step from
February 28 to March 1 uses the transitions of February 28.

## Observations

- The AR(1) and the mixture take a vector `y`, one entry per observation.
- The Bernoulli hidden Markov models take a `D × N` matrix of `Bool`, one column per step and one
  row per series (a station, a device, a chromatin mark): `Y[j, n]` is series `j` at step `n`.
- [`SmoothPeriodicHMM`](@ref) takes a `D × N` matrix of real numbers, or a vector when there is a
  single row. A multivariate emission law covers several consecutive rows.

## Initial conditions

In the autoregressive HMMs, the emission of series `j` at step `n` depends on its own `order`
previous values, so the first steps need values from before the data. These are `Y_past`, the
`D × order` observations before `Y` in time order, so that `[Y_past Y]` is the series.

- The fits, the likelihoods and `viterbi` take it as an argument: `fit_mle(m, Y, Y_past; n2t, solver)`.
- `rand(m, n2t; Y_past)` takes it as a keyword, zeros by default, and draws the `Y` that
  follows it.

With real data, the first `order` steps can serve: `Y_past = Y[:, 1:order]`, and the model is
fitted to `Y[:, order+1:end]` with `n2t[order+1:end]`. A model without memory takes a
`D × 0` matrix.

In [`SmoothsARPeriodicHMM`](@ref) the series have different orders, so `Y_past` is a vector:
`Y_past[j]` holds the `order_j` previous values of series `j`.

[`SmoothPeriodicHMM`](@ref) has no memory and takes no `Y_past`: `fit_mle(m, Y; n2t, solver)`.

## [Starting values: `init`](@id starting-values)

No model guesses the parameters of its laws: there are no default values. The models built from
families, [`SmoothSeasonalMixture`](@ref) and [`SmoothPeriodicHMM`](@ref), require `init`, one
starting value per component or hidden state, in one of three forms:

| `init[k]` | for | meaning |
|:--|:--|:--|
| a number | a family with one parameter | that parameter, `1.0` for `Exponential` is the scale ``\theta = 1`` |
| a `NamedTuple` | any family | every parameter by name, `(α=1.5, θ=3.0)` for `Gamma` |
| a vector | any family | the coefficients themselves, link scale, in the order of the family's parameters |

A number or a `NamedTuple` gives a *flat* start, the parameter constant over the cycle: the link
is inverted and the result put on the constant term, the harmonics being zero. For an
`Exponential` with trigonometric polynomials of degree 1, `init = [1.0, 5.0]` gives the
coefficients `[log(1.0), 0, 0]` and `[log(5.0), 0, 0]`. On a spline basis, every coefficient
takes the inverted value instead, the spline functions summing to one (see
[Bases](@ref bases-guide)).

The starting values must differ: two identical components or states get identical posterior
probabilities, and EM cannot tell them apart. Their order is the order of the fitted
components or states, so `init` also says which is which. Values on either side of what the
data show work well, for example a short and a long scale around the mean waiting time, or a
dry and a wet state around the mean temperature. In an HMM with blocks, `init[k]` holds one
value per block.

What `init` does not cover starts from zero coefficients:

| model | starts at | so that |
|:--|:--|:--|
| `SmoothSeasonalMixture` | weight coefficients zero | the weights are equal, ``1/K`` |
| `SmoothPeriodicHMM` | `θᴬ = 0`, `a = fill(1/K, K)` | the transitions and the initial distribution are uniform |
| `SmoothAR1(d, T)` | all coefficients zero | ``\mu = 0``, ``\rho = 0``, ``\sigma = 1`` |

The Bernoulli HMMs take no `init`: they are built from their coefficients, `θᴬ` and `θᴮ`, which
are the start of a fit. The [fitting guide](@ref fitting-guide) lists ways to choose a starting
point, including one from the data.

## Several sequences

A sequence is a run of consecutive steps of the hidden chain. Data that the chain does not
connect are several sequences:

- independent units, such as the genes of the [genomics tutorial](@ref TutoGenomics);
- a series with gaps: a missing day breaks the chain, so each run of consecutive days is a
  sequence.

A period index the calendar skips, such as February 29 above, is not a gap.

The HMM functions take vectors: `Ys[i]` is the `D × Nᵢ` sequence `i`, `Y_pasts[i]` its initial
condition and `n2ts[i]` its period indices.

```julia
loglikelihood(m, Ys, Y_pasts; n2t=n2ts)       # the sum loglikelihoods over the sequences
m, history = fit_mle(m0, Ys, Y_pasts; n2t=n2ts, solver) # maximize the sum of loglikelihoods
viterbi(m, Ys, Y_pasts; n2t=n2ts)                        # one state path per sequence
fit(SeasonalFit, m0, Ys, Y_pasts; n2t=n2ts, solver)      # nobs is the total number of steps
```

The sequences share all the coefficients. Each starts afresh: its first hidden state is drawn
from the initial distribution, which the fit estimates from the first step of every sequence,
and its memory starts from its own `Y_past`. `rand` draws one sequence at a time. For
`SmoothPeriodicHMM`, drop the `Y_pasts`: `fit_mle(m0, Ys; n2t=n2ts, solver)`.

### Handling missing data: splitting a series at its gaps

Three years of daily records from three stations, with two gaps:

```@example data
using SmoothPeriodicStatsModels, Random
Random.seed!(1)
K, D, order, T = 2, 3, 1, 366
m = SmoothARPeriodicHMM([0.5, 0.5], randn(K, K - 1, 3), randn(K, D, 2^order, 3), T)

dates = collect(Date(2000, 1, 1):Day(1):Date(2002, 12, 31))
Y = rand(m, day_of_cycle.(dates); Y_past=zeros(Bool, D, order))
missing_days = (Date(2000, 6, 10) .<= dates .<= Date(2000, 6, 20)) .| (dates .== Date(2001, 11, 2))
dates, Y = dates[.!missing_days], Y[:, .!missing_days]
size(Y)
```

The runs of consecutive days:

```@example data
function runs(dates)
    breaks = findall(!=(Day(1)), diff(dates))
    starts, stops = [1; breaks .+ 1], [breaks; length(dates)]
    return [a:b for (a, b) in zip(starts, stops)]
end
runs(dates)
```

Each run gives its first `order` days as the initial condition and is fitted on the rest:

```@example data
pieces = filter(r -> length(r) > order, runs(dates))
Y_pasts = [Y[:, r[1:order]] for r in pieces]
Ys = [Y[:, r[order+1:end]] for r in pieces]
n2ts = [day_of_cycle.(dates[r[order+1:end]]) for r in pieces]
loglikelihood(m, Ys, Y_pasts; n2t=n2ts)
```

```@example data
length.(viterbi(m, Ys, Y_pasts; n2t=n2ts))
```
