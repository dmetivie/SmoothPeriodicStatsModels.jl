# # Fitting lives in two extensions
#
# Everything needed to *define* a model, simulate from it and evaluate its likelihood is in this
# package. Everything needed to *fit* one is in two extensions:
#
# - `ext/SmoothPeriodicStatsModelsOptimizationExt.jl`, loaded by Optimization.jl: every fit;
# - `ext/SmoothPeriodicStatsModelsSimpleNonlinearSolveExt.jl`, loaded by SimpleNonlinearSolve.jl:
#   the nonlinear least squares fits of coefficients to evaluated arrays, `fit_θ*`.
#
# These are the names those extensions provide: declared here so that they exist, are exported,
# and fail with an instruction rather than a bare `UndefVarError` or `MethodError`.

const _OPTIMIZATION_EXT = :SmoothPeriodicStatsModelsOptimizationExt
const _NLLS_EXT = :SmoothPeriodicStatsModelsSimpleNonlinearSolveExt

# A call that no method of `f` takes. When the extension that defines `f` is loaded, the call has
# the wrong arguments, a `MethodError`; otherwise the extension is what is missing.
@noinline function _unavailable(f, args, ext::Symbol)
    Base.get_extension(@__MODULE__, ext) === nothing || throw(MethodError(f, args))
    throw(ArgumentError(ext === _NLLS_EXT ? """
    `$(nameof(f))` solves a nonlinear least squares problem, in an extension of
    SmoothPeriodicStatsModels that loads with SimpleNonlinearSolve. Load it:

        using SimpleNonlinearSolve""" : """
    `$(nameof(f))` is a fitting routine, in an extension of SmoothPeriodicStatsModels that loads
    with Optimization. Load it, and the package of the solver you pass as `solver`:

        using Optimization, OptimizationOptimJL   # for `solver = LBFGS()`, for example

    (Simulation and likelihood evaluation work without it.)"""))
end

# Every fit takes its optimization algorithm explicitly: this is the value of `solver` when the
# call gives none, and it says so.
@noinline _required_solver() = throw(ArgumentError("""
    no `solver` given. Every fit takes its optimization algorithm explicitly, any algorithm of
    Optimization.jl, for example

        using OptimizationOptimJL
        fit_mle(m, data...; n2t, solver = LBFGS())"""))

for (f, ext) in ((:default_adtype, _OPTIMIZATION_EXT), (:default_ensemble, _OPTIMIZATION_EXT),
    (:solve_unconstrained, _OPTIMIZATION_EXT), (:fit_mle_RO, _OPTIMIZATION_EXT),
    (:fit_loss_optim, _OPTIMIZATION_EXT), (:update_B!, _OPTIMIZATION_EXT),
    (:update_α!, _OPTIMIZATION_EXT), (:update_dist, _OPTIMIZATION_EXT), (:update_R!, _OPTIMIZATION_EXT),
    (:default_nlls_solver, _NLLS_EXT), (:solve_nlls, _NLLS_EXT),
    (:fit_θ, _NLLS_EXT), (:fit_θ!, _NLLS_EXT), (:fit_θᴬ!, _NLLS_EXT), (:fit_θᴮ!, _NLLS_EXT))
    # A `Vararg` fallback is less specific than every method the extension defines, so the real
    # implementation always wins once the extension has loaded.
    @eval begin
        function $f end
        $f(args...; kwargs...) = _unavailable($f, args, $(QuoteNode(ext)))
    end
end
