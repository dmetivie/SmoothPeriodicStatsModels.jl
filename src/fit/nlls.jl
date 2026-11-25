# # Nonlinear least squares
#
# The solver of the coefficient fits `fit_θ*`, which turn evaluated arrays back into
# coefficients. Included by the SimpleNonlinearSolve extension.

"""
    default_nlls_solver()

Algorithm used by the nonlinear least squares fits ([`fit_θᴬ!`](@ref), [`fit_θᴮ!`](@ref)).
"""
default_nlls_solver() = SimpleGaussNewton()

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
