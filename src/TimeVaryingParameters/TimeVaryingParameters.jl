# # A vendored copy of TimeVaryingParameters.jl
#
# This folder is the `src/` of TimeVaryingParameters.jl, by arnaud-ma and contributors
# (https://github.com/arnaud-ma/TimeVaryingParameters.jl, MIT license, copied in `LICENSE` here),
# as it is on the `harden-for-spsm` branch of the fork
# https://github.com/dmetivie/TimeVaryingParameters.jl at commit 0594e38. That branch is the pull
# request https://github.com/arnaud-ma/TimeVaryingParameters.jl/pull/10, not merged yet, and the
# package is not registered. A package registered in General cannot depend on it, so it lives
# here as the submodule `SmoothPeriodicStatsModels.TimeVaryingParameters` in the meantime.
#
# Differences from that commit:
# - Its extension `OptimizationOptimJLExt` is not carried over. It set `DEFAULT_OPTIMIZER` to
#   `BFGS()` once OptimizationOptimJL and ForwardDiff were loaded. Here `fit(::TimeVaryingFamily,
#   …)` and `fit(::ParamSpec, …)` take `solver` explicitly, as every fit of this package does, and
#   the error of `default_optimizer` in `fit.jl` says so.
# - This header.
#
# A fix made here should also go to the fork's branch, so that the pull request stays this code.
#
# ## Switching back to the registered package
#
# Once the pull request is merged and a version of TimeVaryingParameters that contains it is
# registered in General:
# 1. Delete this folder.
# 2. In `src/SmoothPeriodicStatsModels.jl`, remove `include("TimeVaryingParameters/TimeVaryingParameters.jl")`
#    and write `using TimeVaryingParameters:` for `using .TimeVaryingParameters:`.
# 3. In `Project.toml`, add `TimeVaryingParameters = "c3441385-f229-4866-8817-bbca1834e4b8"` to
#    `[deps]`, with that version in `[compat]`. Remove `ADTypes` from `[deps]` and `[compat]`
#    unless something else in `src/` uses it by then.
# 4. Remove `test/timevaryingparameters.jl` and its `include` in `test/runtests.jl` (they are that
#    package's tests), the test dependencies added for them if nothing else uses them
#    (`LinearAlgebra`, `LinkFunctions`, `LogExpFunctions`), and `checkdocs_ignored_modules` in
#    `docs/make.jl`.
# 5. Update the notes about it in `docs/src/20-guides/4-bases.md`, `docs/src/95-reference.md`
#    and `CHANGELOG.md`.
#
# The names this package re-exports stay the same, and `SmoothPeriodicStatsModels.TimeVaryingParameters`
# still names the module. The types move from `SmoothPeriodicStatsModels.TimeVaryingParameters`
# to `TimeVaryingParameters`, though: models saved with JLD or Serialization, whose bases are such
# types, must be saved again.

"""
    TimeVaryingParameters

Statistical models whose parameters vary smoothly with time.

Each parameter is a basis expansion pushed through a link,

    θⱼ(t) = linkⱼ⁻¹( Bⱼ(t) · βⱼ )

so a model is a [`ParamSpec`](@ref) -- one basis and one link per named parameter -- and,
with a distribution constructor, a [`TimeVaryingFamily`](@ref) whose value at a time is an
ordinary `Distributions.jl` object.
"""
module TimeVaryingParameters

using ArgCheck: @argcheck
using LinkFunctions: LinkFunctions, Link, linkfun, linkinv
using LogExpFunctions: logistic, logit
using Distributions: Distributions, logpdf
using ADTypes: ADTypes
# The problem types and `solve` are SciMLBase's. Every package that can actually run that `solve`
# (OptimizationOptimJL, OptimizationMOI, …) brings OptimizationBase with it, so depending on it
# here bought nothing but its load time, and that of SparseConnectivityTracer and
# SparseMatrixColorings behind it: ~100 ms on every `using`.
using SciMLBase: SciMLBase, OptimizationFunction, OptimizationProblem, solve
using StatsAPI: StatsAPI, fit, coef, params

# Links
export IntervalLink

# Bases
export AbstractBasis, ConstantBasis, PolynomialBasis, FourierBasis, IndicatorBasis
export BSplineBasis, CyclicBSplineBasis
export CombinedBasis, TrigBasis
export nbasis, eval_basis, eval_basis!, design_matrix

# Parametrization
export ParamSpec, paramnames, ncoef, coefs, parameters_at, initial_coefficients
export blockindex, blockrange, apply_link, invert_link

# Families and objectives
export TimeVaryingFamily, DistConstructor, PeriodicDesign, group_by_time, distributions, linear_predictors
export PeriodicNLL, PointwiseNLL, PointwiseLoss

# Fitting
export TimeVaryingFit, solve_objective, curve
export fit, coef, params

include("links.jl")
include("bases.jl")
include("spec.jl")
include("family.jl")
include("fit.jl")

end # module
