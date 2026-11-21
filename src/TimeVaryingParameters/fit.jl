# # Fitting
#
# One entry point per objective, all sharing the same solver keywords. The solver itself is
# injected: any `Optimization.jl` algorithm, passed as `solver`. (In the package,
# `OptimizationOptimJL` sets a default through an extension; this vendored copy has none, see
# the first lines of `TimeVaryingParameters.jl`.)

const DEFAULT_OPTIMIZER = Ref{Any}(nothing)

function default_optimizer()
    opt = DEFAULT_OPTIMIZER[]
    opt === nothing && throw(ArgumentError(
        "no `solver` given: pass any algorithm of Optimization.jl, for example " *
        "`solver = BFGS()` after `using OptimizationOptimJL`"))
    return opt
end

"""
    solve_objective(objective, θ0; solver, adtype, solve_kwargs) -> SciMLBase solution

Minimise `objective(θ, nothing)` from `θ0`. `objective` is any callable of that signature.
"""
function solve_objective(objective, θ0::AbstractVector;
        solver = default_optimizer(),
        adtype = ADTypes.AutoForwardDiff(),
        solve_kwargs = (;),
        warn_unsuccessful = true,
    )
    prob = OptimizationProblem(OptimizationFunction(objective, adtype), collect(θ0), nothing)
    sol = solve(prob, solver; solve_kwargs...)
    if warn_unsuccessful && !SciMLBase.successful_retcode(sol.retcode)
        @warn "optimisation returned $(sol.retcode)"
    end
    return sol
end

"""
    TimeVaryingFit

A fitted time-varying model: the family, the flat coefficient vector, and the solver
solution it came from.

Accessors: [`coef`](@ref) for the coefficients (whole vector, or one parameter's block),
[`curve`](@ref) for a parameter over a period, and `fitted(t)` for the distribution at `t`.
"""
struct TimeVaryingFit{F, C <: AbstractVector, R}
    family::F
    coefficients::C
    solution::R
end

@inline (fit::TimeVaryingFit)(t) = fit.family(t, fit.coefficients)

"""
    coef(fit) -> Vector
    coef(fit, name::Symbol) -> SubArray

The fitted coefficients: all of them, or the block belonging to one parameter.
"""
StatsAPI.coef(fit::TimeVaryingFit) = fit.coefficients
StatsAPI.coef(fit::TimeVaryingFit, name::Symbol) =
    coefs(fit.family.spec, fit.coefficients, name)

"""
    params(fit, t) -> Tuple

The parameter values at time `t`, in the order the spec names them. `fit(t)` gives the
distribution they build.
"""
StatsAPI.params(fit::TimeVaryingFit, t) = parameters_at(fit.family.spec, fit.coefficients, t)

"""
    curve(spec::ParamSpec, θ, name::Symbol, times) -> Vector
    curve(fit::TimeVaryingFit, name::Symbol, times) -> Vector

One parameter evaluated over `times`, the seasonal curve a periodic model is usually
plotted as. One matrix-vector product and one link, rather than
[`parameters_at`](@ref) once per time.
"""
function curve(spec::ParamSpec, θ::AbstractVector, name::Symbol, times::AbstractVector)
    j = blockindex(spec, name)
    η = design_matrix(spec.bases[j], times) * coefs(spec, θ, j)
    return apply_link.(Ref(spec.links[j]), η)
end

"""
    curve(spec::ParamSpec, θ::AbstractMatrix, name::Symbol, times) -> Matrix

The same curve once per column of `θ`, stacked one row per column, so the result is
`size(θ, 2) × length(times)`. This is the shape a model fitted independently per group --
per hidden state, per site -- is usually kept in, and it costs one matrix product rather
than one per column.
"""
function curve(spec::ParamSpec, θ::AbstractMatrix, name::Symbol, times::AbstractVector)
    j = blockindex(spec, name)
    η = design_matrix(spec.bases[j], times) * view(θ, blockrange(spec, j), :)
    return apply_link.(Ref(spec.links[j]), permutedims(η))
end

curve(fit::TimeVaryingFit, name::Symbol, times::AbstractVector) =
    curve(fit.family.spec, fit.coefficients, name, times)

paramnames(fit::TimeVaryingFit) = paramnames(fit.family)

function Base.show(io::IO, ::MIME"text/plain", fit::TimeVaryingFit)
    println(io, "TimeVaryingFit:")
    show(io, MIME"text/plain"(), fit.family.spec)
    print(io, "  retcode: ", fit.solution.retcode)
    return
end

"""
    fit(fam::TimeVaryingFamily, y, ts; init, weights = nothing, n2t = nothing, nT = nothing, kwargs...)

Maximum likelihood for a time-varying family.

`y` are the observations and `ts` the times they were seen at. `weights`, if given, is one
non-negative weight per observation -- which is what makes this usable as the M-step of an
EM algorithm.

When the problem is periodic, pass `n2t` (the period index of every observation) and `nT`
(the period). The design is then built once at `1:nT` and the likelihood is accumulated per
period index, which costs `nT` distribution constructions per iteration instead of `N`.
Without them the generic per-observation path is used.

`init` names a starting point for every parameter: either a coefficient vector, or a single
value placed on the intercept through the link. Remaining keywords (`solver`, `adtype`,
`solve_kwargs`) go to [`solve_objective`](@ref).
"""
function StatsAPI.fit(fam::TimeVaryingFamily, y::AbstractVector, ts = nothing;
        init,
        weights = nothing,
        n2t = nothing,
        nT = nothing,
        kwargs...,
    )
    θ0 = initial_coefficients(fam.spec; init...)
    check_weights(weights, length(y))
    objective = if n2t === nothing
        @argcheck ts !== nothing "give either `ts`, or `n2t` together with `nT`"
        @argcheck length(ts) == length(y)
        PointwiseNLL(fam, y, ts, weights)
    else
        @argcheck nT !== nothing "`n2t` needs `nT`, the number of distinct period indices"
        @argcheck length(n2t) == length(y)
        design = PeriodicDesign(fam.spec, 1:nT)
        PeriodicNLL(fam, design, y, group_by_time(n2t, nT), weights)
    end
    sol = solve_objective(objective, θ0; kwargs...)
    return TimeVaryingFit(fam, sol.u, sol)
end

"""
    fit(spec::ParamSpec, loss, y, ts; init, weights = nothing, kwargs...)

Minimise `Σₙ wₙ · loss(θ(tsₙ), yₙ)` for an arbitrary `loss(params::Tuple, y) -> Number`.

The M-estimation escape hatch, for objectives that are not a loglikelihood. Returns the
solver solution rather than a `TimeVaryingFit`, since there is no family to carry.
"""
function StatsAPI.fit(spec::ParamSpec, loss, y::AbstractVector, ts;
        init, weights = nothing, kwargs...)
    @argcheck length(ts) == length(y)
    check_weights(weights, length(y))
    θ0 = initial_coefficients(spec; init...)
    return solve_objective(PointwiseLoss(spec, loss, y, ts, weights), θ0; kwargs...)
end

check_weights(::Nothing, _n) = nothing
function check_weights(w, n)
    @argcheck length(w) == n "one weight per observation: got $(length(w)) for $(n)"
    @argcheck all(>=(0), w) "weights must be non-negative"
    return nothing
end
