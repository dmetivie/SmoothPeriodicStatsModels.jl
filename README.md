# SmoothPeriodicStatsModels

[![Build Status](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Coming soon!

Basically the similar to [PeriodicHiddenMarkovModels.jl](https://github.com/dmetivie/PeriodicHiddenMarkovModels.jl) but where smoothness and identifiability are enforced (up to a global index relabeling).
The HMM part will be moved at some point to [PeriodicHiddenMarkovModels.jl](https://github.com/dmetivie/PeriodicHiddenMarkovModels.jl).

So fat the HMM needs a big rebase + it only works with Bernoulli emission distribution.

This is inspired by seasonal Hidden Markov Model, see [A. Touron (2019)](https://link.springer.com/article/10.1007/s11222-019-09854-4).
It is not only for Hidden Markov Chain but also for Mixture, Auto Regressive in [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl).

## Example

### Set up

```julia
Random.seed!(2020)
K = 3 # Number of Hidden states
T = 11 # Period
N = 49_586 # Length of observation
D = 6 # dimension of observed variables
autoregressive_order = 1

size_order = 2^autoregressive_order
degree_of_P = 1
size_degree_of_P = 2*degree_of_P + 1 

trans_θ = 4*(rand(K, K - 1, size_degree_of_P) .- 1/2)
Bernoulli_θ = 2*(rand(K, D, size_order, size_degree_of_P) .- 1/2)
hmm = Trig2HierarchicalPeriodicHMM([1/3, 1/6, 1/2], trans_θ, Bernoulli_θ, T)
```

### Simulations

```julia
z_ini = 1
y_past = rand(Bool, autoregressive_order, D)
n2t = SmoothPeriodicStatsModels.n_to_t(N,T)
z, y = rand(hmm, n2t; y_ini=y_past, z_ini=z_ini, seq=true)
 ```

### Fit

```julia
trans_θ_guess = rand(K, K-1, size_degree_of_P)
Bernoulli_θ_guess = zeros(K, D, size_order, size_degree_of_P)
trans_θ_guess[:,:,1] .= trans_θ[:,:,1] # cheating on initial guess to recover very good mle maxima
Bernoulli_θ_guess[:,:,:,1] = Bernoulli_θ[:,:,:,1]
hmm_guess = Trig2HierarchicalPeriodicHMM([1/4, 1/4, 1/2], trans_θ_guess, Bernoulli_θ_guess, T)

@time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_guess, trans_θ_guess, Bernoulli_θ_guess, y, y_past, maxiter=10000, robust=true; display=:iter, silence=true, tol=1e-3, θ_iters=true, n2t=n2t);
# EM converged in 317 iterations, logtot = -194299.4177103428
# FitMLE SHMM (Baum Welch): 72.532794 seconds (344.65 M allocations: 27.641 GiB, 3.59% gc time)
```

### Plots

```julia
using LaTeXStrings, Plots
begin
    pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing) for k in 1:K]
    for k in 1:K
        [plot!(pA[k], hmm.A[k, l, :], c=l, label=L"Q_{%$(k)\to %$(l)}", legend=:topleft) for l in 1:K]
        [plot!(pA[k], hmm_fit.A[k, l, :], c=l, label=:none, legend=:topleft, s = :dot) for l in 1:K]

        hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
        ylims!(0,1)
    end
    pallA = plot(pA..., size=(1000, 500))
end

begin
    mm = 1
    pB = [plot() for j in 1:D]
    for j in 1:D
        [plot!(pB[j], succprob.(hmm.B[k, :, j, mm]), c=k, label=:none) for k in 1:K]
        [plot!(pB[j], succprob.(hmm_fit.B[k, :, j, mm]), c=k, label=:none, s = :dot) for k in 1:K]

        hline!(pB[j], [0.5], c=:black, label=:none, s=:dot)
        ylims!(pB[j], (0, 1))
    end
    pallB = plot(pB...)
end

begin
    mm = 2
    pB = [plot() for j in 1:D]
    for j in 1:D
        [plot!(pB[j], succprob.(hmm.B[k, :, j, mm]), c=k, label=:none) for k in 1:K]
        [plot!(pB[j], succprob.(hmm_fit.B[k, :, j, mm]), c=k, label=:none, s = :dot) for k in 1:K]

        hline!(pB[j], [0.5], c=:black, label=:none, s=:dot)
        ylims!(pB[j], (0, 1))
    end
    pallB = plot(pB...)
end
```

![Transition matrix](img/Q_test.svg)
![Emmission distribution](img/nu_test_1.svg)
![Emmission distribution](img/nu_test_2.svg)
