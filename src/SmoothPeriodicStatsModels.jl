module SmoothPeriodicStatsModels

# # Packages

# ## Utilities
using ArgCheck
using Base: OneTo
# ## Optimization
# The solver stack lives in `ext/SmoothPeriodicStatsModelsOptimizationExt.jl`: simulating from a
# model and evaluating its likelihood need none of it, and it is about half of this package's
# load time. `SciMLBase` stays here because `OptimMLE` and the retcode checks are light.
using SciMLBase: successful_retcode

# ## Multivariate
using Copulas

# Random and Distributions
using Distributions
using Random: AbstractRNG, GLOBAL_RNG, rand!

# ## Special function
using LogExpFunctions: logsumexp!, logsumexp, log1pexp

# ## HMM
using PeriodicHiddenMarkovModels
using PeriodicHiddenMarkovModels: AbstractPeriodicHMM
using PeriodicHiddenMarkovModels: viterbi, istransmat, update_a!, vec_maximum
import PeriodicHiddenMarkovModels: forwardlog!, backwardlog!, viterbi, viterbi!, viterbilog!, posteriors!
import PeriodicHiddenMarkovModels: fit_mle!, fit_mle

# # Overloaded functions
import Base: rand
import Base: ==, copy, size
using Base.Threads
# # Code
include("utilities.jl")

# ## One solver convention shared by every fit in the package
include("legacy_kwargs.jl")
include("fit_stubs.jl")
export default_solver, default_nlls_solver, default_adtype, default_ensemble

# ## M-step objectives shared by the seasonal EM algorithms
include("trig_kernels.jl")

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
# for HMMMspa 
export ARPeriodicHMMSpatial,norm_cdf_2d_vfast,Trig2ARPeriodicHMMSpatial,fit_mle!

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
export fit_mle_AR1, ar1_sufficient_statistics, ar1_nll
export model_for_loglikelihood_AR1_full # legacy alias of `fit_mle_AR1`
export μₜ, ρₜ, σₜ, αₜ
export n_to_t
export fit_loss_optim, fit_mle



end
