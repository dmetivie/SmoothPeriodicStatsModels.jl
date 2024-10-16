module SmoothPeriodicStatsModels

# # Packages

# ## Utilities
using ArgCheck
using Base: OneTo
using ShiftedArrays: lead, lag
using Distributed
# ## Optimization
using JuMP, Ipopt
using Optimization, OptimizationMOI
using LsqFit

# ## Multivariate
using Copulas

# Random and Distributions
using Distributions
using Random: AbstractRNG, GLOBAL_RNG, rand!

# ## Special function
using LogExpFunctions: logsumexp!, logsumexp

# ## HMM
using PeriodicHiddenMarkovModels

using PeriodicHiddenMarkovModels: viterbi, istransmat, update_a!, vec_maximum
import PeriodicHiddenMarkovModels: forwardlog!, backwardlog!, viterbi, viterbi!, viterbilog!, posteriors!
import PeriodicHiddenMarkovModels: fit_mle!, fit_mle

# # Overloaded functions
import Distributions: fit_mle
import Base: rand
import Base: ==, copy, size

# # Code
include("utilities.jl")

# ## Generic MLE problem solved with Optimization
include("fit_mle_trig_Optim.jl")
export OptimMLE

# ## EM algorith for mixtures
include("Mixture/fit_mle_trig_EM.jl")

# ## Auto-regressive problems
include("AR/AR1.jl")

# ## HMM problem
include("HHMM_Bernoulli/periodichmm.jl")
include("HHMM_Bernoulli/likelihoods.jl")
include("HHMM_Bernoulli/viterbi.jl")

include("HHMM_Bernoulli/update_A_B_jump.jl")
include("HHMM_Bernoulli/trig_conversion.jl")
include("HHMM_Bernoulli/mle_slice.jl")
include("HHMM_Bernoulli/HMM_utilities.jl")

# For sites added after the HMM training
include("HHMM_Bernoulli/add_sites.jl")

export
    # periodichmm.jl
    HierarchicalPeriodicHMM,
    sort_wrt_ref!,
    randhierarchicalPeriodicHMM,
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
    Trig2HierarchicalPeriodicHMM,
    fit_mle_YY,
    # fit slice
    fit_mle_all_slices


export AR1
export model_for_loglikelihood_AR1, initialvalue_optimize!, model_for_loglikelihood_AR1_full
export μₜ, ρₜ, σₜ, αₜ
export n_to_t
export fit_loss_optim


end
