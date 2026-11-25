# [Fitting](@id fitting-guide)

```@meta
CurrentModule = SmoothPeriodicStatsModels
```

Fitting needs the [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) ecosystem. The
models, simulation and likelihoods work without it, so the fitting methods live in two package
extensions, each loaded with its package:

```julia
using Optimization          # every fit
using SimpleNonlinearSolve  # fit_θ, fit_θ!, fit_θᴬ!, fit_θᴮ! and initialize
```

Without them, a fit raises an error that names the package to load.

!!! warning "Every fit needs an explicit `solver`"
    There is no default optimization algorithm: every fit takes one as `solver`, from the
    package you choose. For example, with
    `LBFGS` from OptimizationOptimJL:

    ```julia
    using OptimizationOptimJL
    m, history = fit_mle(m0, y; n2t, solver=LBFGS())
    ```

`SimpleNonlinearSolve` serves only `fit_θ` and its variants, which fit coefficients by least
squares to curves already evaluated at each period index, as in the slice method of the
Bernoulli HMMs; they keep `SimpleGaussNewton()` as their default. ForwardDiff.jl, which the
default automatic differentiation, `adtype = AutoForwardDiff()`, uses, is a dependency of this
package. The fits themselves maximize likelihoods through Optimization.jl; `adtype =
NoAutoDiff()` fits without derivatives, with a derivative-free `solver`.

Any solver or automatic differentiation backend of Optimization.jl works, once its package is
loaded:

```julia
using ReverseDiff
fit_mle(m0, y; n2t, solver=LBFGS(), adtype=AutoReverseDiff())   # reverse-mode differentiation
fit_mle(m0, y; n2t, solver=NelderMead())                        # derivative-free, OptimizationOptimJL
```

Box constraints are not currently supported: the fits build their optimization problems without
bounds. They should not be needed: the links keep every parameter in its domain, so the
coefficients are free and the problems unconstrained. A solver that requires bounds, such as those
of OptimizationBBO or `Fminbox`, therefore does not apply.

## Model in, model out

`m, history = fit_mle(m0, data...; n2t, solver, kwargs...)` starts from the coefficients of the model `m0`, leaves `m0` unchanged, and returns the fitted model `m` and a [`FitHistory`](@ref) `history` with the following fields:

| | |
|:--|:--|
| `history.converged` | whether the stopping rule was met |
| `history.iterations` | the number of iterations |
| `history.logtots` | the loglikelihood after each iteration |
| `history.best_iteration` | the iteration with the highest loglikelihood, `0` for the starting model |
| `loglikelihood(history)` | the loglikelihood of the returned model |
| `history.ps_iters` | with `ps_iters = true`, the coefficients before the first iteration and after each one |

## How the models are fitted

The models fall in two groups, which take different options.

- **The AR(1) maximizes its loglikelihood directly**, in one call of the solver, or one per hidden
  state when the state sequence is given, `fit_mle(m, y, z)`. Its history also holds what the
  solver returned: `solution`, `retcode`, `objective` and `stats`.
- **The mixtures and the HMMs are fitted by EM**, an iteration of two steps:
  - the E-step computes the posterior probability of each component or hidden state given the
    data, with no solver: the `E_step!` of ExpectationMaximization.jl for the mixtures, and the
    forward-backward algorithm of HiddenMarkovModels.jl, `forward_backward!`, for the HMMs;
  - the M-step maximizes the expected loglikelihood: the weights or the transitions, and each
    component or emission, each by its own weighted solve.

  EM stops when the loglikelihood changes by less than `tol`. Its history records the
  loglikelihood of each iteration, and none of the solves. The loop around the two steps is
  that of ExpectationMaximization.jl for the mixtures. For the HMMs it is this package's rather
  than `HiddenMarkovModels.baum_welch`: it records the loglikelihood after each M-step, and for
  the spatial HMM returns the best iterate.

## [Reading a fitted model](@id reading-a-model)

A fitted model `m` gives its parameters at three levels: the coefficients, the curves they generate
over the cycle, and the evaluated model, which stores the parameters at every period index.

| Model | coefficients | curves | evaluated model |
|:--|:--|:--|:--|
| `SmoothAR1` | `parameters(m).μ`, `.ρ`, `.σ` | `curve(m, :σ)` | `AR1(m)`, or `m(t)` at one period index |
| `SmoothSeasonalMixture` | `parameters(m).α`, `component_coefs(m, k)` | `mixture_weights(m)`, `curve(m, :θ)` | `mixtures(m)`, or `m(t)` |
| `SmoothARPeriodicHMM` | `parameters(m).A`, `.B` | | `ARPeriodicHMM(m)` |
| `SmoothsARPeriodicHMM` | `parameters(m).A`, `.B[k, j][h]` | | `sARPeriodicHMM(m)` |
| `SmoothARPeriodicHMMSpatial` | `parameters(m).A`, `.B`, `.R` | `curve(m, :R)` | `ARPeriodicHMMSpatial(m)` |
| `SmoothPeriodicHMM` | `parameters(m).A`, `emission_coefs(m, k, b)` | `curve(m, :μ; block)` | `PeriodicHMMDists(m)` |

The coefficients have the layout the model was built with, for example `θᴬ` and `θᴮ` for an
HMM. In `ARPeriodicHMM(m)`, `a` is the initial distribution, `A[k, l, t]` the probability of
going from state `k` to `l` when leaving period index `t`, and `B[k, t, j, h]` the `Bernoulli` law of
series `j` in state `k` at period index `t` with memory category `h`; `succprob.(ARPeriodicHMM(m).B)`
gives the probabilities. `PeriodicHMMDists(m)` has the same `a` and `A`, and `B[b][k, t]` is the
law of the rows of block `b` in state `k` at period index `t`.

A block of `parameters(m)` is a view into the model: `copy(parameters(m).A)` gives an array of
its own.

## Options

### The solver, for every fit

The solver settings are those of `solve` in Optimization.jl. They apply to the solve of the AR(1),
and to every solve of every M-step of an EM fit:

- `solver`: any Optimization.jl algorithm, required, for example `LBFGS()` from
  OptimizationOptimJL;
- `adtype`: the automatic differentiation backend, `AutoForwardDiff()` by default;
- `solve_kwargs`: a `NamedTuple` passed on to `solve`, such as `(maxiters = 100,)`.

`ensemble` runs independent solves in parallel: `EnsembleSerial()` by default, `EnsembleThreads()`
or `EnsembleDistributed()` otherwise. The independent solves are the per-state fits of the AR(1)
given `z`, and the M-step problems of the HMMs: one per state, series and memory category, or one
per state and block for a `SmoothPeriodicHMM`. The single fit of the AR(1) has nothing to run in
parallel, and the mixture solves its components one after the other: neither takes `ensemble`.

### EM, for the mixtures and the HMMs

| keyword | | default |
|:--|:--|:--|
| `maxiter` | the maximum number of iterations | `100` |
| `tol` | stop when the loglikelihood changes by less than this | `1e-3` |
| `display` | `:none`, `:iter` or `:final` | `:none` |
| `robust` | see [Impossible observations](@ref impossible-observations) | `false` |
| `ps_iters` | keep the coefficients of every iteration, HMMs only | `false` |

The mixture also takes `rtol`, a relative tolerance, and `method`, `ClassicEM()` or
`StochasticEM()`. The spatial HMM also takes `QMC_E`, `maxiters_R`, `tdist` and `wp`. The
docstring of each [`fit_mle`](@ref) method lists them all.

```julia
fit_mle(m0, Y, Y_past; n2t, solver=LBFGS(), maxiter=500, tol=1e-6, display=:final,
    ensemble=EnsembleThreads(), solve_kwargs=(maxiters=200,))
```

### [Impossible observations: `robust`](@id impossible-observations)

An observation that the model finds impossible in every state, a negative value for an
exponential law, makes the forward pass of an HMM fit stop on a `NaN` loglikelihood.
`robust = true` keeps the fit going:

- an observation with density zero in every state gets the same small loglikelihood in all of
  them, `log(floatmin())` ≈ -708: it then says nothing about the state, which its neighbours
  decide;
- an infinite loglikelihood is capped at `log(floatmax())`;
- the transition probabilities are kept off zero.

An observation that is impossible in some states only needs no help: those states get posterior
probability zero, with or without `robust`. The loglikelihoods in `history.logtots` include the
value given to the impossible observations. `loglikelihoods` and `viterbi` take `robust` too. For
the seasonal mixture, `robust` is that of ExpectationMaximization.jl.

`robust` keeps the E-step finite, not the M-step. An emission fit still reads the observation
through its family's density, so if no parameter value makes the observation possible (a
negative value for a `Gamma` law), that state's fit cannot move: the data or the family have to
change.

The emission fit of each state of a [`SmoothPeriodicHMM`](@ref) also leaves out the observations
whose posterior probability in that state is at most `1e-10`, with or without `robust`. They do
not change the estimate, but that state's law can give them a density of zero in floating point,
as a copula law does far in the tail of a marginal, and the state's fit would stop there.

## Starting points

The EM fits find a local maximum of the likelihood, and the starting point decides which one,
and with it what each hidden state means. Three ways to choose it:

- from what is known, as in the tutorials: which state is the active or the wet one, with flat
  curves; [Starting values](@ref starting-values) says how `init` becomes coefficients and what
  starts at zero;
- for the HMMs, the slice method: [`fit_mle_all_slices`](@ref) fits a mixture at each period
  index, ignoring the dynamics, and smooth coefficients are then fitted to the result. The
  tutorials show it for the [Bernoulli HMMs](@ref TutoHMMStart) and for the
  [HMM with any emission](@ref TutoHMMDistsStart);
- several starts, keeping the fit with the highest `loglikelihood(history)`.

## Model selection

`fit(SeasonalFit, m0, data...; n2t, solver, kwargs...)` runs the same fit and keeps what model selection
needs, so `loglikelihood`, `dof`, `nobs`, `aic`, `aicc` and `bic` apply to it. To choose a
number of hidden states, for example:

```julia
fits = [fit(SeasonalFit, start(K), Y, Y_past; n2t, solver=LBFGS()) for K in 2:5]
bic.(fits)
```

`dof` is the number of coefficients, plus the `K - 1` free initial probabilities for the HMMs.
`nobs` is the number of terms of the likelihood: the
transitions for the AR(1), the observations for the mixture, and the observed steps, summed over
the sequences, for the HMMs.

## One AR(1) per hidden state

`fit_mle(m, y, z; n2t, solver)` fits one AR(1) per hidden state given the state sequence `z`, for
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
