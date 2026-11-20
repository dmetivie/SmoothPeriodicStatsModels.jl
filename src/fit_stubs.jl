# # Fitting lives in an extension
#
# Everything needed to *define* a model, simulate from it and evaluate its likelihood is in this
# package. Everything needed to *fit* one is in
# `ext/SmoothPeriodicStatsModelsOptimizationExt.jl`, which loads once the solver stack is
# available. These are the names that extension provides: declared here so that they exist, are
# exported, and fail with an instruction rather than a bare `UndefVarError`.

const _SOLVER_STACK = "Optimization, SimpleNonlinearSolve"

@noinline function _needs_solver_stack(name)
    throw(ArgumentError("""
    `$name` needs the solver stack, which is a weak dependency of SmoothPeriodicStatsModels.
    Load it, and the package of the solver you pass to the fits, to enable every fitting routine:

        using $_SOLVER_STACK
        using OptimizationOptimJL   # for example, for `solver = LBFGS()`

    (Only simulation and likelihood evaluation work without it.)"""))
end

# Every fit takes its optimization algorithm explicitly: this is the value of `solver` when the
# call gives none, and it says so.
@noinline _required_solver() = throw(ArgumentError("""
    no `solver` given. Every fit takes its optimization algorithm explicitly, any algorithm of
    Optimization.jl, for example

        using OptimizationOptimJL
        fit_mle(m, data...; n2t, solver = LBFGS())"""))

for f in (:default_nlls_solver, :default_adtype, :default_ensemble,
    :solve_unconstrained, :solve_nlls,
    :fit_θ, :fit_θ!, :fit_θᴬ!, :fit_θᴮ!,
    # `fit_mle` and `fit` are owned by Distributions and StatsAPI respectively and already
    # imported, so the extension adds methods to those directly.
    :fit_mle_RO, :fit_loss_optim,
    :update_B!, :update_α!, :update_dist, :update_R!)
    # A `Vararg` fallback is less specific than every method the extension defines, so the real
    # implementation always wins once the extension has loaded.
    @eval begin
        function $f end
        $f(args...; kwargs...) = _needs_solver_stack($(QuoteNode(f)))
    end
end
