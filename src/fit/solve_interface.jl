# # One convention for every fit in the package
#
# Each fitting entry point takes the same four keywords, so that the solver is chosen at the
# call site rather than baked into the model:
#
# - `solver`       the optimization (or nonlinear least squares) algorithm to hand to `solve`
# - `adtype`       the automatic differentiation backend used to build the `OptimizationFunction`
# - `ensemble`     how a batch of independent problems is run (serial, threaded, distributed)
# - `solve_kwargs` a `NamedTuple` forwarded verbatim to `solve`
#
# `solve_kwargs` is a `NamedTuple` rather than a `kwargs...` slurp on purpose: `fit_mle` already
# has `maxiter` (the number of EM iterations) sitting next to the solver's own `maxiters`, and a
# slurp would silently swallow that typo.

"""
    default_solver()

Algorithm used when a fitting function is called without a `solver`. `LBFGS` is a reasonable
default here because every problem in the package is smooth and unconstrained.
"""
default_solver() = LBFGS()

"""
    default_nlls_solver()

Algorithm used by the nonlinear least squares fits ([`fit_θᴬ!`](@ref), [`fit_θᴮ!`](@ref)).
"""
default_nlls_solver() = SimpleGaussNewton()

"""
    default_adtype()

Automatic differentiation backend used to build every `OptimizationFunction`.

Note that `AutoForwardDiff` needs `ForwardDiff` to be loaded: it is a dependency of this package
precisely so that choosing it here cannot fail at run time.
"""
default_adtype() = AutoForwardDiff()

"""
    default_ensemble()

How a batch of independent problems — one per hidden state, per site, … — is run.
`EnsembleSerial` by default; pass `EnsembleThreads()` or `EnsembleDistributed()` to parallelize.
"""
default_ensemble() = EnsembleSerial()

"""
    solve_unconstrained(objective, u0, p; solver, adtype, solve_kwargs) -> minimizer

Minimize `objective(u, p)` from the starting point `u0`, and return the minimizer. Warns, rather
than throwing, when the solver reports an unsuccessful return code — the caller usually has a
usable iterate anyway and an EM loop should not die on one bad M-step.
"""
function solve_unconstrained(objective, u0, p;
    solver=default_solver(), adtype=default_adtype(), solve_kwargs=(;))
    prob = OptimizationProblem(OptimizationFunction(objective, adtype), u0, p)
    sol = solve(prob, solver; solve_kwargs...)
    successful_retcode(sol.retcode) || @warn "solver returned $(sol.retcode)"
    return sol.u
end

"""
    solve_nlls(residual, u0, p, n_resid; solver, solve_kwargs) -> minimizer

Minimize `sum(abs2, residual(u, p))` as a `NonlinearLeastSquaresProblem` with `n_resid`
residuals, and return the minimizer. Warns on an unsuccessful return code.
"""
function solve_nlls(residual, u0, p, n_resid::Integer;
    solver=default_nlls_solver(), solve_kwargs=(;))
    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(residual; resid_prototype=zeros(n_resid)), u0, p)
    sol = solve(prob, solver; solve_kwargs...)
    successful_retcode(sol.retcode) || @warn "nonlinear least squares returned $(sol.retcode)"
    return sol.u
end
