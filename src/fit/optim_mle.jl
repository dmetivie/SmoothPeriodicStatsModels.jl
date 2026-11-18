"""
    fit_mle(opt::OptimMLE, y; solver, adtype, solve_kwargs) -> SciMLBase solution

Minimize `opt.ℓ(θ, y)` from `opt.θ₀`. Returns the solution object, so that `sol.u` are the fitted
parameters and `sol.retcode` reports whether the solver converged.
"""
function Distributions.fit_mle(Opt::OptimMLE, y::AbstractArray;
    solver=Opt.solver, adtype=default_adtype(), solve_kwargs=(;))
    OptFunc = OptimizationFunction(Opt.ℓ, adtype)
    prob = OptimizationProblem(OptFunc, Opt.θ₀, y)

    return solve(prob, solver; solve_kwargs...)
end

"""
    fit_loss_optim(ℓ, y, θ0; solver, adtype, solve_kwargs) -> SciMLBase solution

`fit_mle(OptimMLE(ℓ, solver, vec(θ0)), y; kwargs...)`: minimize the loss `ℓ(θ, y)` from `θ0`.
"""
fit_loss_optim(ℓ, y, θ0; solver=_required_solver(), kwargs...) =
    fit_mle(OptimMLE(ℓ, solver, vec(θ0)), y; kwargs...)
