"""
Fitting routines for SmoothPeriodicStatsModels.

This extension loads with Optimization.jl. Every fit takes its optimization algorithm explicitly,
as `solver`, from whichever package the user loads:

    using Optimization, OptimizationOptimJL   # for `solver = LBFGS()`, for example

ForwardDiff, for the default `adtype`, is a dependency of the parent package. The coefficient
fits `fit_θ*` are in the SimpleNonlinearSolve extension instead.

It adds the methods behind every `fit_*` name the package exports. The parent package keeps the
models themselves — construction, `rand`, likelihoods, viterbi — none of which need a solver.
"""
module SmoothPeriodicStatsModelsOptimizationExt

using SmoothPeriodicStatsModels
const SPSM = SmoothPeriodicStatsModels

# Declared in the parent (src/fit_stubs.jl) so that they exist and error helpfully when this
# extension is not loaded; `import` because we add the real methods to them.
import SmoothPeriodicStatsModels: default_adtype, default_ensemble, solve_unconstrained,
    fit_mle_RO, fit_loss_optim, update_B!, update_α!, update_dist, update_R!,
    fit_mle_all_slices

# The seasonal mixture's M-step: `M_step!` is ExpectationMaximization's, re-exposed through the
# parent, and the three types below carry the seasonal parametrization it operates on.
import SmoothPeriodicStatsModels: M_step!
using SmoothPeriodicStatsModels: SeasonalSample, SeasonalWeights, SeasonalFamily, em_fit_mle!,
    SmoothSeasonalMixture, seasonal_family, mixtures, mixture_weights, component_coefs,
    ClassicEM, StochasticEM

# Read-only: the models and the pure pieces the objectives are built from.
using SmoothPeriodicStatsModels: ARPeriodicHMM, sARPeriodicHMM, ARPeriodicHMMSpatial, AbstractMLE,
    OptimMLE, Trig2ARPeriodicHMM, Trig2ARPeriodicHMMSpatial,
    PeriodicDesign, SmoothAR1, SeasonalFit, FitHistory, seasonal_params, _params_like, ncolumns, column,
    SmoothARPeriodicHMM, SmoothsARPeriodicHMM, SmoothARPeriodicHMMSpatial,
    SeasonalControls, CachedEmissions, RobustEmissions, _clamp_inf!, degrees, _refresh!, _linear_predictor, _transitions!, _emissions!, design_matrix,
    _trig_order, _plain, _sequences, _SmoothHMM, _ar1_loglikelihood,
    SmoothPeriodicHMM, PeriodicHMMDists, emission_coefs, group_by_time, _ncols,
    linear_predictors, apply_link, _required_solver,
    TimeVaryingFamily, DistConstructor, PeriodicNLL, distributions, ParamSpec, TrigBasis,
    trig_basis, bernoulli_logit_nll, multinomial_logit_nll, exponential_nll, ar1_nll,
    bernoulli_residual, transition_residual, ar1_sufficient_statistics,
    polynomial_trigo, n_to_t, conditional_to, n_per_category, idx_observation_of_past_cat,
    loglikelihoods, loglikelihoods!, likelihoods!, complete_loglikelihood,
    pairwise_loglikelihood, pairwise_loglikelihood_memory,
    # the engine contract of 4.2: state weights, `past`, sequences
    OneHot, SmoothDistribution, _state_weights, _check_weight_values, _check_n2t, _hmm_past,
    _keyword_sequences, _no_past_z, _lag_cat, _ar1_with_past, _ar1_past, _past_fields,
    _check_seq_ends, _seq_ranges, _pasts, _memoryless_pasts, blockrange, paramnames, ncoef,
    # the additional loss of the fits
    _resolve_loss, _penalized, _penalized_data, _loss_value, _loss_float, _check_loss,
    _BernoulliPhi, _RangePhi, _FamilyPhi, _SoftmaxPhi

# The solver stack this extension is triggered by. ForwardDiff makes the default `adtype`,
# `AutoForwardDiff()`, usable; it is loaded here so that its Optimization extension is too.
using Optimization
using ForwardDiff: ForwardDiff
import HiddenMarkovModels as HMMs
using SciMLBase: successful_retcode, EnsembleProblem, EnsembleSerial, EnsembleThreads,
    EnsembleDistributed, remake, NullParameters, NoAD
using ADTypes: NoAutoDiff
using LinkFunctions: LogLink

# Shared by the fitting code and the parent package alike
using ArgCheck
using StatsAPI: StatsAPI
using Base: OneTo
using Base.Threads
using Random: AbstractRNG, GLOBAL_RNG, randperm, Xoshiro
using Distributions
using LogExpFunctions: logsumexp, log1pexp, logistic
import Distributions: fit_mle   # the generic every `fit_mle` method below extends
import StatsAPI: fit

include("../src/fit/solve_interface.jl")
include("../src/fit/optim_mle.jl")
include("../src/fit/model_loss.jl")
include("../src/fit/hmm.jl")
include("../src/fit/smooth_hmm.jl")
include("../src/fit/weighted_hmm.jl")
include("../src/fit/smooth_periodic_hmm.jl")
include("../src/fit/slice_periodic_hmm.jl")
include("../src/fit/shmm.jl")
include("../src/fit/add_sites.jl")
include("../src/fit/spatial.jl")
include("../src/fit/mixture.jl")
include("../src/fit/ar1.jl")
include("../src/fit/distribution.jl")

end # module
