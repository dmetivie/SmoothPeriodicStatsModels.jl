# [From 4.2 to 5.0](@id api-42-50)

```@meta
CurrentModule = SmoothPeriodicStatsModels
```

Version 4.2 adds one form of every fitting and simulation call, the form version 5.0 keeps.
The forms of version 4 still work in 4.2 and give the same numbers. A program written with the
4.2 forms runs on 5.0 unchanged.

| Call | Version 4 | 4.2 | 5.0 |
|---|---|---|---|
| AR(1) given the states | `fit_mle(m, y, z)` | `fit_mle(m, y, OneHot(z, K))`, added; `z` unchanged | `OneHot` only; an integer vector throws |
| State weights | none | `fit_mle(m, y, W)` with an `N × K` matrix, added | unchanged |
| Initial observations of an HMM | `fit_mle(m, Y, Y_past)` | `fit_mle(m, Y; past = (; Y = Y_past))`, added | `past` only |
| Several sequences | `fit_mle(m, Ys, Y_pasts)` | `fit_mle(m, Ys; past = Y_pasts)` or `fit_mle(m, Y; seq_ends, past)`, added | keyword forms only |
| Decoding | `viterbi(m, Y, Y_past)`, `viterbi(m, Ys, Y_pasts)` | `viterbi(m, Y; past, seq_ends)`, added, with `past.z` | keyword forms only |
| Simulation of an HMM | `rand(m, n2t; z_ini, Y_past, seq = true) -> (z, Y)` | `rand(m, n2t; z, past) -> (; z, Y)`, added | keyword form only |
| Simulation of an AR(1) | `rand(m, n2t, z; y₁)` | `rand(m, n2t; z, past)`, added | keyword form only |
| Emission loglikelihoods | `loglikelihoods(m, Y, Y_past)` | `loglikelihoods(m, y; n2t, past) -> N × K`, added for every model | keyword form only |
| Emissions given the states | `fit_mle_RO(Y, Y_past, z, n2t, d, K, T)` | `fit_mle(SmoothARPeriodicHMM(K, D, T; memory, season = d), Y, OneHot(z, K); past, n2t, fixed = (:A,))`, added | removed |
| Starting point of an HMM | `fit_mle_all_slices` then `fit_θ!` | `initialize(m0, Y; past, n2t, method = SliceEM(window, n_random, dirichlet))`, added | `initialize` only |
| Seasonal distribution | `PeriodicNLL` and `fit_loss_optim` by hand | `SmoothDistribution(D, T; K, season)`, added | unchanged |
| Coefficients of a model | `parameters(m)` | `coef(m)`, added | `coef` only |
| `n2t` | defaults to `n_to_t(N, T)` | unchanged; required by `SmoothDistribution` | required |
| Objective of the fits | summed | summed for the models of version 4; averaged for `SmoothDistribution` | averaged |
| Return of `fit_mle` | `(m, history)` | unchanged; the model alone for `SmoothDistribution` and the fit given state weights | the model alone; `fit(SeasonalFit, …)` for the history |

`past = (; Y, z)` is one keyword for every model. `Y` holds the observations before the first,
at least as many as the memory, of which the last are used; `z` is the hidden state of the step
before the first. `rand` and `viterbi` read `z`. The fits and the marginal loglikelihood refuse
it, every sequence starting from the initial distribution.
