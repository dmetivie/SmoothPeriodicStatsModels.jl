```@codeblocks
line_counter = :continue
```

```@meta
EditURL = "../../../examples/tuto_hmm_dists.jl"
```

````@example 4-tuto_hmm_dists
using Markdown#hide
````

# [Hidden Markov model with any emission](@id TutoHMMDists)

[`SmoothPeriodicHMM`](@ref) is a hidden Markov model whose emission in each hidden state is any
`Distributions.jl` law, with parameters that vary smoothly along the cycle.

We observe a sequence ``Y_1, \dots, Y_N`` of vectors of ``\mathbb{R}^D``; in the first example
``D = 1``, a single real series. Step ``n`` falls at period index ``t_n \in \{1, \dots, T\}`` of
a cycle of ``T`` steps (`n2t[n]`, see the [data layout](@ref data-layout)). The law of ``Y_n``
depends on ``t_n`` and on a hidden state ``Z_n \in \{1, \dots, K\}``, a Markov chain whose
transition probabilities also depend on ``t_n``:

```math
\mathbb{P}(Z_1 = k) = a_k, \qquad
\mathbb{P}(Z_{n+1} = l \mid Z_n = k) = A_{kl}(t_n), \qquad
Y_n \mid Z_n = k \ \sim\ f\big(\,\cdot\,;\ \theta_k(t_n)\big).
```

Given the hidden states, the ``Y_n`` are independent: the model has no memory. Here

- ``a = (a_1, \dots, a_K)`` is the initial distribution of the hidden states;
- ``A(t)`` is the ``K \times K`` transition matrix used when leaving period index ``t``, its rows
  summing to 1;
- ``f(\,\cdot\,; \theta)`` is the emission law, here a normal law of mean ``\mu`` and standard
  deviation ``\sigma``, ``\theta = (\mu, \sigma)``;
- ``\theta_k(t) = \big(\mu_k(t), \sigma_k(t)\big)`` is its parameter in state ``k`` at period
  index ``t``.

Each curve is a trigonometric polynomial of degree ``d`` (any other [basis](@ref bases-guide)
can replace it),

```math
P(t) = c_0 + \sum_{\ell=1}^{d} \left( c_{2\ell-1} \cos\frac{2\pi \ell t}{T} + c_{2\ell} \sin\frac{2\pi \ell t}{T} \right),
```

pushed through a link that keeps it in its domain. The transitions go through a softmax, as in
the [HMM tutorial](@ref TutoHMM), with one polynomial ``P^A_{kl}`` for each
``k \in \{1, \dots, K\}`` and ``1 \le l < K``, state ``K`` being the reference,
``P^A_{kK} = 0``. The mean goes through the identity and the standard deviation through a log,
one polynomial ``P^\mu_k`` and ``P^\sigma_k`` per state:

```math
A_{kl}(t) = \frac{e^{P^A_{kl}(t)}}{\sum_{l'=1}^{K} e^{P^A_{kl'}(t)}}, \qquad
\mu_k(t) = P^\mu_k(t), \qquad \sigma_k(t) = \exp\big(P^\sigma_k(t)\big), \qquad
k, l \in \{1, \dots, K\}.
```

As a toy example, take daily temperatures at one station over twenty years, in two states: a
*dry* and a *wet* weather regime. The data are simulated.

````@example 4-tuto_hmm_dists
using SmoothPeriodicStatsModels
using Optimization, SimpleNonlinearSolve # loads the fitting routines
using OptimizationOptimJL # the optimization algorithm, LBFGS
using Distributions, Random, Statistics
using CairoMakie

solver = LBFGS()   # every fit below takes its optimization algorithm explicitly
````

## One series

`SmoothPeriodicHMM(families, a, θᴬ, θᴮ, T)` builds the model from its coefficients:

- `families`, one per block of rows of the observations: here one row, a `Normal` whose
  parameters are trigonometric polynomials of degree 1, written `Normal => 1`;
- `θᴬ[k, l, :]`, the coefficients of the transition from ``k`` to ``l``, state ``K`` being the
  reference of the softmax;
- `θᴮ[k, b]`, the coefficients of block ``b`` in state ``k``: those of ``P^\mu_k``, then those of
  ``P^\sigma_k``.

````@example 4-tuto_hmm_dists
T, K = 365, 2
θᴬ = zeros(K, K - 1, 3)
θᴬ[1, 1, :] = [1.5, 0.5, 0.0]     # dry → dry: high probability, highest in winter
θᴬ[2, 1, :] = [-1.0, 0.0, 0.3]    # wet → dry
θᴮ = reshape([
        [16.0, -8.0, -3.0, log(2.5), 0.0, 0.0],   # dry: μ(t) = 16 - 8cos(2πt/T) - 3sin(2πt/T), σ = 2.5
        [9.0, -6.0, -2.0, log(2.0), 0.0, 0.0],    # wet
    ], K, 1)
truth = SmoothPeriodicHMM([Normal => 1], [0.5, 0.5], θᴬ, θᴮ, T)
````

`rand(m, n2t; seq = true)` returns the state path and the `D × N` observations, here `1 × N`.

````@example 4-tuto_hmm_dists
Random.seed!(1)
years = 20
n2t = n_to_t(years * T, T)
z, Y = rand(truth, n2t; seq=true)
y = vec(Y)

begin
    fig = Figure(size=(900, 260))
    ax = Axis(fig[1, 1], xlabel="day", ylabel="temperature")
    for (k, label, color) in ((1, "dry regime", :darkorange), (2, "wet regime", :steelblue))
        days = findall(==(k), z[1:2T])
        scatter!(ax, days, y[days]; markersize=4, color, label)
    end
    Legend(fig[1, 2], ax, framevisible=false)
    fig
end
````

## Fit

`fit_mle(m, y; solver, n2t)` runs the Baum–Welch algorithm from the coefficients of `m`. The start
`SmoothPeriodicHMM(Normal, K, d, T; init)` has flat curves, a starting `(μ, σ)` per state and
uniform transitions; the states must start apart. Here the dry state starts at a mean of 15 and
the wet one at 8, on either side of the mean temperature of the data, about 13.
[Starting values](@ref starting-values) gives the rule for every family.

````@example 4-tuto_hmm_dists
m0 = SmoothPeriodicHMM(Normal, K, 1, T; init=[(μ=15.0, σ=3.0), (μ=8.0, σ=3.0)])
m, history = fit_mle(m0, y; solver, n2t)
history
````

`curve(m, :μ)` is the `K × T` matrix of means, one row per state, and `curve(m, :σ)` that of the
standard deviations.

````@example 4-tuto_hmm_dists
function plot_curves(models, labels, names)
    fig = Figure(size=(900, 280))
    for (i, name) in enumerate(names)
        cs = [curve(mod, name) for mod in models]
        ax = Axis(fig[1, i], title=string(name), xlabel="day",
            limits=(nothing, (min(0, minimum(minimum, cs)), nothing)))
        for (c, label, style) in zip(cs, labels, (:solid, :dash))
            lines!(ax, 1:T, c[1, :]; color=:darkorange, linestyle=style, label="dry, " * label)
            lines!(ax, 1:T, c[2, :]; color=:steelblue, linestyle=style, label="wet, " * label)
        end
        i == length(names) && Legend(fig[1, i+1], ax, framevisible=false)
    end
    return fig
end

plot_curves([truth, m], ["truth", "fit"], (:μ, :σ))
````

`emission_coefs(m, k, b)` holds the coefficients of block `b` in state `k`, and
`parameters(m).A` the transition coefficients. The dry state, against the truth:

````@example 4-tuto_hmm_dists
(fit=round.(emission_coefs(m, 1, 1); digits=2), truth=θᴮ[1, 1])
````

`viterbi(m, y; n2t)` is the most likely state path:

````@example 4-tuto_hmm_dists
mean(viterbi(m, y; n2t) .== z)
````

## [A starting point from the data](@id TutoHMMDistsStart)

The fit above started from a guess that knew roughly where each state sits. Without such
knowledge, [`fit_mle_all_slices`](@ref) builds a start from the data, as the slice method does
for the [Bernoulli HMMs](@ref TutoHMMStart). At each period index `t` it fits a mixture of the
`K` emission laws to the observations of a window around `t`, here eleven days, ignoring the
dynamics. The states are put in the same order at every `t`, by default by decreasing mean of
the first row, so state 1 is the warm one all year. The estimates are then smoothed into the
bases of the model, and the transition coefficients are fitted to the steps between the most
likely states.

The model it is given sets the structure, and its laws are one of the starts of each mixture.
Here it knows nothing, with two states one degree apart:

````@example 4-tuto_hmm_dists
blind = SmoothPeriodicHMM(Normal, K, 1, T; init=[(μ=13.0, σ=5.0), (μ=12.0, σ=5.0)])
m_slice, slices = fit_mle_all_slices(Xoshiro(1), blind, y; solver, n2t, Yₜ_extanted=-5:5, n_random_ini=3, slices=true)
````

`slices = true` also returns the estimates at each period index, as a `PeriodicHMMDists`. The
dots are those estimates, the lines the smooth start, and the dashed lines the truth:

````@example 4-tuto_hmm_dists
begin
    fig = Figure(size=(700, 260))
    ax = Axis(fig[1, 1], title="μ", xlabel="day")
    for (k, color) in zip(1:K, (:darkorange, :steelblue))
        scatter!(ax, 1:T, mean.(slices.B[1][k, :]); color, markersize=3)
        lines!(ax, 1:T, curve(m_slice, :μ)[k, :]; color)
        lines!(ax, 1:T, curve(truth, :μ)[k, :]; color, linestyle=:dash)
    end
    fig
end
````

The transitions of such a start are rough: they are counted between the most likely states of
fits that ignore the dynamics, so an observation given to the wrong state counts as two
switches, and the states look less persistent than they are. Baum–Welch corrects that. From this
start, against the fit from the guess:

````@example 4-tuto_hmm_dists
m_from_slices, history_slices = fit_mle(m_slice, y; solver, n2t)
(from_guess=(loglikelihood=loglikelihood(history), iterations=history.iterations),
    from_slices=(loglikelihood=loglikelihood(history_slices), iterations=history_slices.iterations))
````

## Several series in blocks

The observations can have several rows, split into blocks: each family covers one row if its law
is univariate, several if it is multivariate. Given the state, the blocks are independent: with
``Y_{b,n}`` the rows of block ``b`` at step ``n``, the law of ``Y_n`` given ``Z_n = k`` has
density

```math
y \mapsto \prod_{b} f_b\big(y_b;\ \theta_{kb}(t_n)\big),
```

where ``f_b`` is the law of block ``b`` and ``\theta_{kb}(t)`` its parameter in state ``k`` at
period index ``t``. A multivariate law fits as long as each of its parameters is a scalar. Here
``D = 3``: the daily minimum and maximum temperatures ``Y_{1,n}`` and ``Y_{2,n}`` follow a
bivariate normal law whose correlation ``\rho_k(t) \in (-1, 1)`` varies over the year, and rain
occurrence ``Y_{3,n} \in \{0, 1\}`` a Bernoulli law of probability ``p_k(t)``:

```math
\begin{pmatrix} Y_{1,n} \\ Y_{2,n} \end{pmatrix} \Bigm| Z_n = k \sim
\mathcal{N}\left(\begin{pmatrix} \mu^{\min}_k \\ \mu^{\max}_k \end{pmatrix},
\begin{pmatrix} (\sigma^{\min}_k)^2 & \rho_k\,\sigma^{\min}_k\sigma^{\max}_k \\
\rho_k\,\sigma^{\min}_k\sigma^{\max}_k & (\sigma^{\max}_k)^2 \end{pmatrix}\right), \qquad
Y_{3,n} \mid Z_n = k \sim \operatorname{Bernoulli}(p_k),
```

every parameter being evaluated at ``t_n``.

A family not declared by [`seasonal_family`](@ref) is a `TimeVaryingFamily`: a function that
builds the law from its parameters, and a `ParamSpec` giving each parameter a basis and a link.
`IntervalLink(-1, 1)` keeps ``\rho`` in ``(-1, 1)``.

````@example 4-tuto_hmm_dists
b = TrigBasis(T, 1)
bivariate = TimeVaryingFamily(
    (μmin, μmax, σmin, σmax, ρ) -> MvNormal([μmin, μmax], [σmin^2 ρ*σmin*σmax; ρ*σmin*σmax σmax^2]),
    ParamSpec(μmin=b => IdentityLink(), μmax=b => IdentityLink(), σmin=b => LogLink(),
        σmax=b => LogLink(), ρ=b => IntervalLink(-1, 1)))
families = [bivariate, Bernoulli => 1]
````

The coefficients of each block follow the order of its parameters. For ``\rho``, the
coefficients are those of ``P^\rho_k``, with ``\rho_k(t) = 2\operatorname{logistic}(P^\rho_k(t)) - 1``;
for ``p``, ``p_k(t) = \operatorname{logistic}(P^p_k(t))``.

````@example 4-tuto_hmm_dists
θᴮ₂ = Matrix{Vector{Float64}}(undef, K, 2)
θᴮ₂[1, 1] = [8, -6, -2, 20, -9, -3, log(3), 0, 0, log(3.5), 0, 0, 1.5, 0.5, 0]   # dry
θᴮ₂[2, 1] = [7, -5, -2, 14, -7, -2, log(2.5), 0, 0, log(2.5), 0, 0, 0.8, 0, 0]   # wet
θᴮ₂[1, 2] = [-2.0, 0.5, 0.0]    # dry: rain is rare
θᴮ₂[2, 2] = [1.5, 0.3, 0.0]     # wet: rain is frequent
truth₂ = SmoothPeriodicHMM(families, [0.5, 0.5], θᴬ, θᴮ₂, T)
````

The observations are `3 × N`: rows 1 and 2 for the temperatures, row 3 for rain.

````@example 4-tuto_hmm_dists
z₂, Y₂ = rand(truth₂, n2t; seq=true)
Y₂[:, 1:5]
````

The start gives each state one starting value per block: a `NamedTuple` for the bivariate law, a
probability for rain.

````@example 4-tuto_hmm_dists
m0₂ = SmoothPeriodicHMM(families, K, T;
    init=[[(μmin=10.0, μmax=20.0, σmin=3.0, σmax=3.0, ρ=0.5), 0.2],
        [(μmin=8.0, μmax=14.0, σmin=3.0, σmax=3.0, ρ=0.5), 0.7]])
m₂, history₂ = fit_mle(m0₂, Y₂; solver, n2t)
history₂
````

````@example 4-tuto_hmm_dists
plot_curves([truth₂, m₂], ["truth", "fit"], (:ρ, :p))
````

Decoding works as for one series:

````@example 4-tuto_hmm_dists
mean(viterbi(m₂, Y₂; n2t) .== z₂)
````

`PeriodicHMMDists(m)` is the evaluated model: `a`, the transition matrices `A[:, :, t]`, and
`B[b][k, t]`, the law of block `b` in state `k` at period index `t`. The bivariate law of the dry state
on day 30:

````@example 4-tuto_hmm_dists
PeriodicHMMDists(m₂).B[1][1, 30]
````

## Dependence from a copula

A copula separates the marginal laws of a random vector from their dependence. By Sklar's
theorem, any joint law of ``(X_1, X_2)`` with continuous marginal cdfs ``F_1`` and ``F_2`` is

```math
\mathbb{P}(X_1 \le x_1,\ X_2 \le x_2) = C\big(F_1(x_1),\ F_2(x_2)\big),
```

where the copula ``C`` is a joint cdf on ``[0, 1]^2`` with uniform marginals. The bivariate
normal law above is the Gaussian copula with normal marginals; any marginals can be paired
instead. [Copulas.jl](https://github.com/lrnv/Copulas.jl) builds such a law as
`SklarDist(C, (law₁, law₂))`, an ordinary multivariate `Distribution`, so it can be a block like
any other.

As a proof of concept, take ``D = 2``: a temperature ``Y_{1,n}``, normal of mean ``\mu_k`` and
standard deviation ``\sigma_k``, and a wind speed ``Y_{2,n}``, exponential of mean ``\lambda_k``,
joined in each state by a Gaussian copula ``C_\rho`` of correlation ``\rho_k``:

```math
\mathbb{P}(Y_{1,n} \le x_1,\ Y_{2,n} \le x_2 \mid Z_n = k) =
C_{\rho_k}\big(F_{\mathcal{N}(\mu_k,\, \sigma_k)}(x_1),\ F_{\mathcal{E}(\lambda_k)}(x_2)\big),
```

where ``F_{\mathcal{N}(\mu, \sigma)}`` and ``F_{\mathcal{E}(\lambda)}`` are the normal and
exponential cdfs, every parameter being evaluated at ``t_n``. Each parameter has its own basis: the mean wind speed
is constant, a trigonometric polynomial of degree 0.

````@example 4-tuto_hmm_dists
using Copulas

copula_law = TimeVaryingFamily(
    (μ, σ, λ, ρ) -> SklarDist(GaussianCopula(2, ρ), (Normal(μ, σ), Exponential(λ))),
    ParamSpec(μ=b => IdentityLink(), σ=b => LogLink(), λ=TrigBasis(T, 0) => LogLink(),
        ρ=b => IntervalLink(-1, 1)))

θᴮ₃ = reshape([
        [16, -8, -3, log(2.5), 0, 0, log(3.0), -0.6, 0.4, 0],  # dry: calm, the warmer the calmer
        [9, -6, -2, log(2.0), 0, 0, log(5.0), 0.4, 0, 0],      # wet: windy
    ], K, 1)
truth₃ = SmoothPeriodicHMM([copula_law], [0.5, 0.5], θᴬ, θᴮ₃, T)
````

Ten years of data, then the fit from flat curves:

````@example 4-tuto_hmm_dists
n2t₃ = n_to_t(10T, T)
z₃, Y₃ = rand(truth₃, n2t₃; seq=true)
m0₃ = SmoothPeriodicHMM([copula_law], K, T;
    init=[(μ=15.0, σ=3.0, λ=3.0, ρ=0.0), (μ=8.0, σ=3.0, λ=4.0, ρ=0.0)])
m₃, history₃ = fit_mle(m0₃, Y₃; solver, n2t=n2t₃)
history₃
````

````@example 4-tuto_hmm_dists
plot_curves([truth₃, m₃], ["truth", "fit"], (:ρ,))
````

The density of a copula law goes through the marginal cdfs, so the fit differentiates them.
ForwardDiff does for `Normal`, `Exponential`, `Weibull` and `LogNormal`, but not for the `Gamma`
cdf in its shape: with such a marginal, pass `adtype = AutoFiniteDiff()` to `fit_mle`, after
`import FiniteDiff`. The parameters of the copula are scalars too, for example one correlation
in `GaussianCopula(d, ρ)`. An observation that every state finds impossible stops the fit unless
`robust = true`; the [fitting guide](@ref fitting-guide) says what that does.

`loglikelihood`, `fit(SeasonalFit, …)` for model selection, several sequences and the functions of
`HiddenMarkovModels.jl` work as for the other HMMs; see the [fitting guide](@ref fitting-guide)
and the [HiddenMarkovModels.jl guide](@ref hmms-guide).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

