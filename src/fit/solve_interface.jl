# # One convention for every fit in the package
#
# Each fitting entry point takes the same four keywords, so that the solver is chosen at the
# call site rather than baked into the model:
#
# - `solver`       the optimization algorithm to hand to `solve`, required: there is no default,
#                  so that the solver package is the user's choice and not a dependency
# - `adtype`       the automatic differentiation backend used to build the `OptimizationFunction`
# - `ensemble`     how a batch of independent problems is run (serial, threaded, distributed)
# - `solve_kwargs` a `NamedTuple` forwarded verbatim to `solve`
#
# `solve_kwargs` is a `NamedTuple` rather than a `kwargs...` slurp on purpose: `fit_mle` already
# has `maxiter` (the number of EM iterations) sitting next to the solver's own `maxiters`, and a
# slurp would silently swallow that typo.
#
# The nonlinear least squares solver of the coefficient fits is in `nlls.jl`, which the
# SimpleNonlinearSolve extension includes.

"""
    default_adtype()

Automatic differentiation backend used to build every `OptimizationFunction`: `AutoForwardDiff()`.
"""
default_adtype() = AutoForwardDiff()

"""
    default_ensemble()

How a batch of independent problems — one per hidden state, per site, … — is run.
`EnsembleSerial` by default; pass `EnsembleThreads()` or `EnsembleDistributed()` to parallelize.
"""
default_ensemble() = EnsembleSerial()

# Optimization 5 takes no derivative as `SciMLBase.NoAD()`; `NoAutoDiff()` of ADTypes means the
# same and is translated.
_optimization_adtype(::NoAutoDiff) = NoAD()
_optimization_adtype(adtype) = adtype

"""
    solve_unconstrained(objective, u0, p; solver, adtype, solve_kwargs, return_sol) -> minimizer

Minimize `objective(u, p)` from the starting point `u0` and return the minimizer, or the whole
solution with `return_sol = true`. An unsuccessful solver return code warns instead of
throwing. `adtype = NoAutoDiff()` uses no derivative, for a derivative-free `solver`.
"""
function solve_unconstrained(objective, u0, p;
    solver=_required_solver(), adtype=default_adtype(), solve_kwargs=(;), return_sol=false)
    prob = OptimizationProblem(OptimizationFunction(objective, _optimization_adtype(adtype)), u0, p)
    sol = solve(prob, solver; solve_kwargs...)
    successful_retcode(sol.retcode) || @warn "solver returned $(sol.retcode)"
    return return_sol ? sol : sol.u
end
