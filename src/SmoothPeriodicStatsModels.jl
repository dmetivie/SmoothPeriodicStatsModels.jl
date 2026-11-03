module SmoothPeriodicStatsModels

# # Packages

# ## Utilities
using ArgCheck
using Base: OneTo
# ## Optimization
# The solver stack lives in `ext/SmoothPeriodicStatsModelsOptimizationExt.jl`: simulating from a
# model and evaluating its likelihood need none of it, and it is about half of this package's
# load time. `SciMLBase` stays here because `OptimMLE` and the retcode checks are light.
using SciMLBase: successful_retcode, AbstractSciMLSolution

# ## Multivariate
using Copulas

# Random and Distributions
using Distributions
using Random: AbstractRNG, GLOBAL_RNG, rand!

# ## Special function
using LogExpFunctions: logsumexp, log1pexp, logistic

# ## The time-varying parametrization
# `θⱼ(t) = linkⱼ⁻¹(Bⱼ(t) · βⱼ)` — one basis and one link per named parameter — is the thing
# every model here is built out of, and it lives in its own package. Imported by name rather
# than wholesale: `TimeVaryingParameters` also exports `fit`, `coef` and `params`, and `fit` in
# this namespace is PeriodicHiddenMarkovModels'.
using TimeVaryingParameters: TimeVaryingParameters,
    ParamSpec, TrigBasis, ConstantBasis, FourierBasis, PolynomialBasis, IndicatorBasis,
    PeriodicDesign, linear_predictors, design_matrix, nbasis,
    paramnames, ncoef, coefs, blockindex, blockrange, parameters_at, apply_link, curve,
    initial_coefficients,
    # the generic time-varying family and its weighted periodic likelihood: what makes a
    # seasonal mixture component declarable rather than hand-written
    TimeVaryingFamily, DistConstructor, PeriodicNLL, PointwiseNLL, distributions,
    group_by_time, TimeVaryingFit,
    IntervalLink   # the overflow-safe spelling of `LinkFunctions.BoundedLink`
using LinkFunctions: IdentityLink, LogLink, LogitLink
# `SeasonalFit` is a `StatisticalModel`, which is what gets `aic`, `aicc` and `bic` for free.
using StatsAPI: StatsAPI, StatisticalModel, fit, coef, dof, nobs, aic, aicc, bic
# One named container per model that is also a plain `AbstractVector`, so the same object reads
# as `ps.ρ` to a person and goes straight into an `OptimizationProblem`.
using ComponentArrays: ComponentArray

# Re-exported so that a seasonal family can be declared with nothing but this package loaded.
export ParamSpec, TrigBasis, ConstantBasis, FourierBasis, PolynomialBasis, IndicatorBasis
export IdentityLink, LogLink, LogitLink, IntervalLink
export paramnames, ncoef, coefs, blockindex, blockrange, curve, design_matrix, nbasis
# The generic time-varying layer, so that a downstream package can state a seasonal model and
# its likelihood with only this package loaded -- `StochasticWeatherGenerators.fit_TN` is one.
export TimeVaryingFamily, PeriodicDesign, distributions
export PeriodicNLL, PointwiseNLL, group_by_time, TimeVaryingFit

# ## EM
# The seasonal mixture is fitted by `ExpectationMaximization`'s driver; this package supplies only
# the parametrization, as the types in `seasonal_mixture.jl` and the two hooks they dispatch.
using ExpectationMaximization: ClassicEM, StochasticEM
# aliased because it is the EM driver, an implementation detail of the seasonal mixture, and not
# a name this package means to re-export beside `fit_mle`
using ExpectationMaximization: fit_mle! as em_fit_mle!
import ExpectationMaximization: loglikelihoods!, M_step!
export ClassicEM, StochasticEM

# ## HMM
# A periodic HMM is a controlled HMM whose control is the period index, which is a case
# `HiddenMarkovModels` documents and implements; `src/hmm_interface.jl` says so in three
# accessors and the library's inference and Baum-Welch then apply to these models.
import HiddenMarkovModels as HMMs

# From v0.2 on, `PeriodicHiddenMarkovModels` is itself a thin extension of `HiddenMarkovModels`,
# so all this package takes from it is `n_to_t` — shared rather than redefined, because both
# packages export that name and loading the two would otherwise make it ambiguous. The
# forward/backward passes, `viterbi`, `istransmat` and `randtransmat` that v0.1 also supplied
# are gone from it; the few still needed here are in `utilities.jl` and `viterbi.jl`.
using PeriodicHiddenMarkovModels: n_to_t

# Every `fit_mle` method in this package extends `Distributions`' generic, which is where the
# name belongs and where `fit_mle(::Type{D}, x)` already lives.
import Distributions: fit_mle

# # Overloaded functions
import Base: rand
import Base: ==, copy, size
using Base.Threads
# # Code
include("utilities.jl")

# ## Explicit parameters, and what a fit reports
include("params.jl")
export parameters, nparams, seasonal_params, seasonal_curve, ncolumns, column, FitHistory, converged

# ## The second tier, for model selection
include("fitted.jl")
# `params` and `loglikelihood` are `StatsAPI`'s, the same functions `Distributions` extends, so
# re-exporting them here cannot conflict with it.
export SeasonalFit, fit, coef, params
export loglikelihood, dof, nobs, aic, aicc, bic

# ## One solver convention shared by every fit in the package
include("fit_stubs.jl")
export default_solver, default_nlls_solver, default_adtype, default_ensemble
export solve_unconstrained, solve_nlls

# ## M-step objectives shared by the seasonal EM algorithms
include("trig_kernels.jl")

# ## The seasonal mixture, stated for ExpectationMaximization.jl
include("seasonal_mixture.jl")
export SmoothSeasonalMixture, seasonal_family, mixture_weight_spec, mixture_weights, mixtures
export component_coefs, component_range
export ncomponents, components

# ## Generic MLE problem solved with Optimization
include("optim_mle_type.jl")
export OptimMLE

# ## Auto-regressive problems
include("AR/AR1.jl")

# ## HMM problem
include("HHMM_Bernoulli/periodichmm.jl")
include("HHMM_Bernoulli/likelihoods.jl")
include("HHMM_Bernoulli/viterbi.jl")

include("HHMM_Bernoulli/trig_conversion.jl")
include("HHMM_Bernoulli/mle_slice.jl")
include("HHMM_Bernoulli/HMM_utilities.jl")

include("sHHMM_Bernoulli/periodichmm.jl")
include("sHHMM_Bernoulli/likelihoods.jl")
include("sHHMM_Bernoulli/viterbi.jl")

include("sHHMM_Bernoulli/trig_conversion.jl")
include("sHHMM_Bernoulli/mle_slice.jl")
include("sHHMM_Bernoulli/HMM_utilities.jl")

# for HMMSPA
using MvNormalCDF
using SpecialFunctions
include("HHMM_Bernoulli_spatial/fast_bivariate_cdf.jl")
include("HHMM_Bernoulli_spatial/SpatialBernoulli.jl")
include("HHMM_Bernoulli_spatial/periodichmmspa.jl")
include("HHMM_Bernoulli_spatial/likelihoods.jl")
include("HHMM_Bernoulli_spatial/viterbi.jl")
include("HHMM_Bernoulli_spatial/trig_conversion.jl")

# ### …as HiddenMarkovModels.jl models
# After all three HMM types: the accessors for every one of them are written in one place, so
# this has to see each type's definition.
include("hmm_interface.jl")
export SeasonalControls
include("HHMM_Bernoulli/smooth.jl")
include("sHHMM_Bernoulli/smooth.jl")
include("HHMM_Bernoulli_spatial/smooth.jl")
export SmoothARPeriodicHMM, SmoothsARPeriodicHMM, SmoothARPeriodicHMMSpatial, degrees
# for HMMMspa 
export ARPeriodicHMMSpatial,norm_cdf_2d_vfast,Trig2ARPeriodicHMMSpatial

export
    # periodichmm.jl
    ARPeriodicHMM,
    sort_wrt_ref!,
    randARPeriodicHMM,
    rand,
    # messages.jl
    # likelihoods.jl
    loglikelihoods,
    likelihoods,
    viterbi,
    complete_loglikelihood, nb_param, bic_sel,
    # trigonometric
    fit_θ!,
    fit_θ,
    fit_θᴬ!,
    fit_θᴮ!,
    polynomial_trigo,
    Trig2ARPeriodicHMM,
    fit_mle_RO,
    # fit slice
    fit_mle_all_slices

export sARPeriodicHMM

export AR1
export SmoothAR1, ar1_spec, ar1_sufficient_statistics, ar1_nll
export μₜ, ρₜ, σₜ, αₜ
export n_to_t
export fit_loss_optim, fit_mle



end
