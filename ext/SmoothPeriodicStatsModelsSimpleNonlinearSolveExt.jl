"""
Coefficient fits by nonlinear least squares for SmoothPeriodicStatsModels.

This extension loads with `using SimpleNonlinearSolve`. It adds [`fit_θ`](@ref),
[`fit_θ!`](@ref), [`fit_θᴬ!`](@ref), [`fit_θᴮ!`](@ref), [`solve_nlls`](@ref) and
[`default_nlls_solver`](@ref).
"""
module SmoothPeriodicStatsModelsSimpleNonlinearSolveExt

using SmoothPeriodicStatsModels
const SPSM = SmoothPeriodicStatsModels

# Declared in the parent (src/fit_stubs.jl); `import` because the real methods are added here.
import SmoothPeriodicStatsModels: default_nlls_solver, solve_nlls, fit_θ, fit_θ!, fit_θᴬ!, fit_θᴮ!

using SmoothPeriodicStatsModels: ARPeriodicHMM, sARPeriodicHMM, TrigBasis, design_matrix,
    _trig_order, _transitions!, _emissions!, transition_residual, bernoulli_residual

using SimpleNonlinearSolve: SimpleGaussNewton
using SciMLBase: solve, successful_retcode, NonlinearFunction, NonlinearLeastSquaresProblem
using Distributions: succprob

include("../src/fit/nlls.jl")
include("../src/fit/trig_conversion.jl")

end # module
