# [Fitting](@id fitting-guide)

```@meta
CurrentModule = SmoothPeriodicStatsModels
```

The fitting routines need the solver stack:

```julia
using Optimization, OptimizationOptimJL, SimpleNonlinearSolve, ForwardDiff
```

## Model in, model out

`fit_mle(m0, data...; n2t, kwargs...)` starts from the coefficients of `m0`, leaves `m0`
unchanged, and returns the fitted model and a [`FitHistory`](@ref):

| | |
|:--|:--|
| `history.converged` | whether the stopping rule was met |
| `history.iterations` | the number of iterations |
| `history.logtots` | the loglikelihood after each iteration |
| `history.best_iteration` | the iteration with the highest loglikelihood |
| `history.ps_iters` | with `ps_iters = true`, the coefficients before the first iteration and after each one |

The AR(1) is fitted by one solve, or one per hidden state, and its history also holds what the
solver returned: `solution`, `retcode`, `objective` and `stats`. The mixtures and HMMs are
fitted by EM, which solves inside every iteration; their history keeps none of those solves.

## [Reading a fitted model](@id reading-a-model)

A fitted model gives its parameters at three levels: the coefficients, the curves they generate
over the cycle, and the evaluated model, which stores the parameters at every position.

| Model | coefficients | curves | evaluated model |
|:--|:--|:--|:--|
| `SmoothAR1` | `parameters(m).μ`, `.ρ`, `.σ` | `curve(m, :σ)` | `AR1(m)`, or `m(t)` at one position |
| `SmoothSeasonalMixture` | `parameters(m).α`, `component_coefs(m, k)` | `mixture_weights(m)`, `curve(m, :θ)` | `mixtures(m)`, or `m(t)` |
| `SmoothARPeriodicHMM` | `parameters(m).A`, `.B` | | `ARPeriodicHMM(m)` |
| `SmoothsARPeriodicHMM` | `parameters(m).A`, `.B[k, j][h]` | | `sARPeriodicHMM(m)` |
| `SmoothARPeriodicHMMSpatial` | `parameters(m).A`, `.B`, `.R` | `curve(m, :R)` | `ARPeriodicHMMSpatial(m)` |

The coefficients have the layout the model was built with, for example `θᴬ` and `θᴮ` for an
HMM. In `ARPeriodicHMM(m)`, `a` is the initial distribution, `A[k, l, t]` the probability of
going from state `k` to `l` when leaving position `t`, and `B[k, t, j, h]` the `Bernoulli` law of
series `j` in state `k` at position `t` with memory category `h`; `succprob.(ARPeriodicHMM(m).B)`
gives the probabilities.

A block of `parameters(m)` is a view into the model: `copy(parameters(m).A)` gives an array of
its own.

## Options

Every fit takes the settings of the solver its M-step calls:

- `solver`: any [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) algorithm,
  `LBFGS()` by default;
- `adtype`: the automatic differentiation backend, `AutoForwardDiff()` by default;
- `ensemble`: `EnsembleSerial()` by default. `EnsembleThreads()` or `EnsembleDistributed()`
  solve in parallel the independent problems of an M-step, one per state, series and memory
  category;
- `solve_kwargs`: a `NamedTuple` passed on to `solve`, such as `(maxiters = 100,)`.

The EM fits also take `maxiter`, `tol`, `display` (`:none`, `:iter` or `:final`) and `robust`.
The docstring of each [`fit_mle`](@ref) method lists the rest.

```julia
fit_mle(m0, Y, Y_past; n2t, maxiter=500, tol=1e-6, display=:final,
    ensemble=EnsembleThreads(), solve_kwargs=(maxiters=200,))
```

## Starting points

The EM fits find a local maximum of the likelihood, and the starting point decides which one,
and with it what each hidden state means. Three ways to choose it:

- from what is known, as in the tutorials: which state is the active or the wet one, with flat
  curves;
- for the Bernoulli HMMs, the slice method: fit a mixture at each position of the cycle,
  ignoring the dynamics, then fit smooth coefficients to the result. The
  [HMM tutorial](@ref TutoHMMStart) shows it;
- several starts, keeping the fit with the highest `loglikelihood(history)`.

## Model selection

`fit(SeasonalFit, m0, data...; n2t, kwargs...)` runs the same fit and keeps what model selection
needs, so `loglikelihood`, `dof`, `nobs`, `aic`, `aicc` and `bic` apply to it. To choose a
number of hidden states, for example:

```julia
fits = [fit(SeasonalFit, start(K), Y, Y_past; n2t) for K in 2:5]
bic.(fits)
```

`dof` is the number of coefficients. `nobs` is the number of terms of the likelihood: the
transitions for the AR(1), the observations for the mixture, and the observed steps, summed over
the sequences, for the HMMs.

## One AR(1) per hidden state

`fit_mle(m, y, z; n2t)` fits one AR(1) per hidden state given the state sequence `z`, for
example a temperature given the weather regimes an HMM decoded. Build `m` with
`SmoothAR1(d, T; K)`: `parameters(m).ρ` is then `(2d+1) × K`, and `curve(m, :ρ)` is `K × T`.

## Saving a fit

The coefficients are a `ComponentArray`, which [JLD.jl](https://github.com/JuliaIO/JLD.jl)
cannot write, and `parameters(m).A` is a view into it. Save plain copies and rebuild the model:

```julia
using JLD
JLD.save("fit.jld", "a", ARPeriodicHMM(m).a, "θ", map(copy, NamedTuple(parameters(m))),
    "history", history)

saved = JLD.load("fit.jld")
m = SmoothARPeriodicHMM(saved["a"], saved["θ"].A, saved["θ"].B, T)
```

The history saves as it is: its `ps_iters` are plain arrays already.
