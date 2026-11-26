# [Related approaches](@id related-approaches)

```@meta
CurrentModule = SmoothPeriodicStatsModels
```

This page compares the package with the regression models it builds on, GLMs, GAMs and GAMLSS,
and lists packages that fit hidden Markov models and mixtures with covariates.

## GLMs, GAMs and GAMLSS

For one parameter of independent data, a model of this package is a generalized linear model
(GLM) whose covariates are the [basis functions](@ref bases-guide) ``b_i(t_n)``. The probability
of rain on the [home page](@ref "Why smooth?"), a trigonometric polynomial of degree 2 through
the logistic function, is the logistic regression

```r
glm(rain ~ cos(2*pi*t/T) + sin(2*pi*t/T) + cos(4*pi*t/T) + sin(4*pi*t/T), family = binomial)
```

in R, or with [GLM.jl](https://github.com/JuliaStats/GLM.jl) on the columns of
`design_matrix(TrigBasis(T, 2), n2t)`, and has the same maximum-likelihood coefficients. This
package finds them with an [Optimization.jl](https://github.com/SciML/Optimization.jl) solver
instead of iteratively reweighted least squares. In the same way:

- `IndicatorBasis(month_of, 1:12)` is a month factor;
- `CyclicBSplineBasis(T, q)` is a cyclic regression spline, as `bs = "cc"` in
  [mgcv](https://cran.r-project.org/package=mgcv), without its penalty;
- a model whose parameters are all smooth, such as the mean and the standard deviation of a
  `Normal` or both parameters of a `Gamma`, is a distributional regression, as in
  [GAMLSS](https://cran.r-project.org/package=gamlss);
- each parameter takes its own basis, and bases add up, as the terms of a formula do.

### What this package adds

- **Hidden states.** The weights of a mixture, the transitions of a hidden Markov model and the
  emission of each state are smooth, and fitted together by EM. Each M-step is a weighted fit
  of the regressions above, weighted by the posterior probabilities of the states.
- **Parameters that are not the mean of a response:** the autoregressive coefficient
  ``\rho(t)`` of [`SmoothAR1`](@ref), through an interval link; each row of a transition
  matrix, through a softmax; the ranges of the spatial Gaussian copula of
  [`SmoothARPeriodicHMMSpatial`](@ref).
- **Any law of Distributions.jl.** A mixture component or an emission block can follow any law
  declared as a [seasonal family](@ref families-guide), one method of `seasonal_family`,
  including multivariate laws with a smooth correlation and copulas. GLM software is limited to
  exponential families, and GAMLSS mostly to univariate laws.
- **Any optimizer.** Every fit takes any [Optimization.jl](https://github.com/SciML/Optimization.jl)
  solver and automatic differentiation backend (see the [fitting guide](@ref fitting-guide)),
  where GLM and GAM software fix their algorithm.
- **Simulation.** `rand` draws from the fitted model along the cycle, as a weather generator
  needs.

### What GLMs and GAMs offer that this package does not (yet)

- **Estimated smoothness.** A GAM takes many basis functions and penalizes their roughness,
  with the amount of smoothing estimated by REML or GCV. This package penalizes through
  `additional_loss` (`DifferencePenalty` is the P-spline penalty, see the
  [fitting guide](@ref additional-loss-guide)), but the weight is yours to set and to compare by
  held-out loglikelihood: there is no automatic choice of it, and no `bic` for a penalized fit.
  Without a penalty, a curve is smooth because its basis is small, and the degree or the number
  of knots is chosen by comparing fits by BIC.
- **Covariates.** The parameters depend on the period index only, and on the previous
  observations in the models with memory. A GLM takes any covariate, such as a temperature or a
  climate index.
- **Inference on the coefficients.** There are no standard errors, confidence intervals or
  tests; the likelihood, AIC and BIC compare models.

## Hidden Markov models and mixtures with covariates

Several R packages let the parameters of a hidden Markov model or of a mixture depend on
covariates. With the basis functions of the period index as covariates, they fit periodic
models too.

- [depmixS4](https://cran.r-project.org/package=depmixS4): hidden Markov models whose
  transitions are multinomial logistic regressions on covariates, and whose emissions are GLMs.
- [momentuHMM](https://cran.r-project.org/package=momentuHMM): multivariate hidden Markov
  models for animal movement, with a `cosinor(cov, period)` term for cyclical covariates on the
  transitions.
- [hmmTMB](https://cran.r-project.org/package=hmmTMB): hidden Markov models with smoothing
  splines and random effects on the transitions and the emission parameters, fitted with
  Template Model Builder. It is the closest to this package, with penalized smoothness.
- [flexmix](https://cran.r-project.org/package=flexmix): mixtures of regressions fitted by EM,
  whose weights can depend on covariates (concomitant variables).

This package is built around the cycle: the period indices `n2t`, curves over
``1, \dots, T``, autoregressive Bernoulli emissions, spatially correlated emissions and
simulation, in Julia, on [HiddenMarkovModels.jl](@ref hmms-guide).
