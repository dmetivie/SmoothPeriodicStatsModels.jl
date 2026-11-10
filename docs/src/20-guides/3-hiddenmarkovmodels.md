# [HiddenMarkovModels.jl](@id hmms-guide)

```@meta
CurrentModule = SmoothPeriodicStatsModels
```

The hidden Markov models of this package are models of
[HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl): `SmoothARPeriodicHMM`,
`SmoothsARPeriodicHMM`, `SmoothARPeriodicHMMSpatial` and their evaluated twins all subtype
`HiddenMarkovModels.AbstractHMM`. The forward and backward passes of every fit and likelihood
here are that package's.

## A periodic HMM is a controlled HMM

HiddenMarkovModels.jl lets the transitions and the emissions depend on a *control*, a known
input at each step; its documentation calls this
[time dependency](https://gdalle.github.io/HiddenMarkovModels.jl/stable/examples/temporal/). A
periodic autoregressive HMM is the case where the control at step `n` is

```
(t, h, n, s) = (n2t[n], lag_cat[:, n], n, n2t[n-1])
```

- `t`, the position in the cycle, chooses the emission laws;
- `h[j]`, the memory category of series `j`, encodes its `order` previous values, which the
  emission of series `j` depends on;
- `n` is the index of the step;
- `s` is the position of the previous step. HiddenMarkovModels.jl asks for the transition
  matrix of the step that *arrives* at a control, while here `A[:, :, t]` is the step that
  *leaves* position `t`, so the control carries `s`.

[`SeasonalControls`](@ref) builds that control sequence from the data, without storing the
tuples:

```@example hmms
using SmoothPeriodicStatsModels, Random
import HiddenMarkovModels as HMMs
Random.seed!(2)
K, D, order, T = 2, 3, 1, 20
m = SmoothARPeriodicHMM([0.5, 0.5], randn(K, K - 1, 3), randn(K, D, 2^order, 3), T)

n2t = n_to_t(200, T)
Y_past = zeros(Bool, D, order)
Y = rand(m, n2t; Y_past)
controls = SeasonalControls(Y, Y_past; n2t)
controls[2]
```

The accessors of HiddenMarkovModels.jl read the model at a control:

```@example hmms
HMMs.transition_matrix(m, controls[2]) == ARPeriodicHMM(m).A[:, :, 1]
```

```@example hmms
HMMs.obs_distributions(m, controls[2])   # one law of the D series per state
```

## What applies

The inference functions of HiddenMarkovModels.jl take the observations as a vector, one entry per
step, and the controls:

```@example hmms
obs = [Y[:, n] for n in axes(Y, 2)]
HMMs.logdensityof(m, obs, controls) ≈ loglikelihood(m, Y, Y_past; n2t)
```

`forward_backward` gives the posterior probabilities of the states, `viterbi` the most likely
path, `forward` the filtered probabilities. For several sequences, lay them end to end and pass
`seq_ends`, the index of the last step of each:

```@example hmms
Ys = [rand(m, 1:T; Y_past) for _ in 1:5]
Y_pasts = fill(Y_past, 5)
n2ts = fill(1:T, 5)

obs = [Y[:, n] for Y in Ys for n in axes(Y, 2)]
controls = SeasonalControls(Ys, Y_pasts; n2t=n2ts)
seq_ends = cumsum(size.(Ys, 2))
γ, logL = HMMs.forward_backward(m, obs, controls; seq_ends)
logL   # one loglikelihood per sequence
```

Two functions do not apply:

- `baum_welch`. Fit with [`fit_mle`](@ref), which runs Baum-Welch with the forward-backward of
  HiddenMarkovModels.jl and the M-step of this package, and takes the same data layout.
- `rand`. Its controls are fixed in advance, so the memory categories would come from given data
  rather than from the values just drawn. `rand(m, n2t; Y_past)` simulates the model.

For the spatial model, every emission loglikelihood is a quasi-Monte-Carlo estimate. The
functions of HiddenMarkovModels.jl work, but they evaluate one step at a time, while
`loglikelihood` and the fits evaluate all steps at once, in parallel.

## PeriodicHiddenMarkovModels.jl

[PeriodicHiddenMarkovModels.jl](https://github.com/dmetivie/PeriodicHiddenMarkovModels.jl) is
the non-smooth relative: one transition matrix and one set of emission laws per position, any
emission distribution, no autoregressive memory. It suits data with many observations at every
position. Here the parameters are smooth functions of the position, which needs far fewer
observations and keeps the meaning of the states over the cycle, at the price of Bernoulli
emissions only.
