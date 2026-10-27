"""
Fitting routines for SmoothPeriodicStatsModels.

This extension loads as soon as the solver stack is available:

    using Optimization, OptimizationOptimJL, SimpleNonlinearSolve, ForwardDiff

It adds the methods behind every `fit_*` name the package exports. The parent package keeps the
models themselves — construction, `rand`, likelihoods, viterbi — none of which need a solver.
"""
module SmoothPeriodicStatsModelsOptimizationExt

using SmoothPeriodicStatsModels
const SPSM = SmoothPeriodicStatsModels

# Declared in the parent (src/fit_stubs.jl) so that they exist and error helpfully when this
# extension is not loaded; `import` because we add the real methods to them.
import SmoothPeriodicStatsModels: default_solver, default_nlls_solver, default_adtype,
    default_ensemble, solve_unconstrained, solve_nlls,
    fit_θ, fit_θ!, fit_θᴬ!, fit_θᴮ!, fit_mle_AR1,
    fit_mle_RO, fit_loss_optim, update_A!, update_B!, update_α!, update_dist, update_R!

# The seasonal mixture's M-step: `M_step!` is ExpectationMaximization's, re-exposed through the
# parent, and the three types below carry the seasonal parametrization it operates on.
import SmoothPeriodicStatsModels: M_step!
using SmoothPeriodicStatsModels: SeasonalSample, SeasonalWeights, SeasonalFamily, em_fit_mle!

# Read-only: the models and the pure pieces the objectives are built from.
using SmoothPeriodicStatsModels: ARPeriodicHMM, sARPeriodicHMM, ARPeriodicHMMSpatial, AbstractMLE,
    OptimMLE, Trig2ARPeriodicHMM, Trig2ARPeriodicHMMSpatial,
    trig_basis, bernoulli_logit_nll, multinomial_logit_nll, exponential_nll, ar1_nll,
    bernoulli_residual, transition_residual, ar1_sufficient_statistics,
    polynomial_trigo, n_to_t, conditional_to, n_per_category, idx_observation_of_past_cat,
    loglikelihoods, loglikelihoods!, likelihoods!, complete_loglikelihood,
    pairwise_loglikelihood, pairwise_loglikelihood_memory

# The solver stack this extension is triggered by
using Optimization
using OptimizationOptimJL: LBFGS
using SimpleNonlinearSolve: SimpleGaussNewton
import ForwardDiff # `AutoForwardDiff()` is unusable unless ForwardDiff is loaded
using SciMLBase: successful_retcode, NonlinearFunction, NonlinearLeastSquaresProblem,
    EnsembleProblem, EnsembleSerial, EnsembleThreads, EnsembleDistributed, remake

# Shared by the fitting code and the parent package alike
using ArgCheck
using Base: OneTo
using Base.Threads
using Random: AbstractRNG, GLOBAL_RNG
using Distributions
using LogExpFunctions: logsumexp, log1pexp, logistic
using PeriodicHiddenMarkovModels: forwardlog!, backwardlog!, posteriors!, update_a!,
    istransmat, vec_maximum, viterbi
import PeriodicHiddenMarkovModels: fit_mle, fit_mle!

include("../src/fit/solve_interface.jl")
include("../src/fit/optim_mle.jl")
include("../src/fit/trig_conversion.jl")
include("../src/fit/hmm.jl")
include("../src/fit/shmm.jl")
include("../src/fit/add_sites.jl")
include("../src/fit/spatial.jl")
include("../src/fit/mixture.jl")
include("../src/fit/ar1.jl")

end # module
