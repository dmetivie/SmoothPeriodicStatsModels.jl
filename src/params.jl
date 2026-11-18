# # Explicit parameters, and what a fit reports
#
# Every model here is a thin pair: a *structure* — the bases and links of its `ParamSpec`, the
# period, the number of states — and a *parameter container* `ps`, held apart the way `Lux.jl`
# holds them apart from a layer. `ps` is a `ComponentArray`, which is both named
# (`ps.ρ`, `ps.A`) and a plain `AbstractVector`, so the same object reads as `θᴬ` to a person and
# goes into an `OptimizationProblem` or through `ForwardDiff` without being flattened first.
#
# That is what replaces passing `θᴬ`, `θᴮ`, `θᴿ`, `θ_α`, `θ_Y` loose beside the model, and why
# every fit here returns two things rather than six.

"""
    parameters(model) -> ComponentArray

The model's coefficients, as one named container. `parameters(m).σ` is the block that generates
`σ(t)`; `parameters(m)` as a whole is an `AbstractVector` of every free coefficient.
"""
function parameters end

"""
    nparams(model) -> Int

The number of free coefficients, `length(parameters(model))`.
"""
nparams(model) = length(parameters(model))

"""
    seasonal_params(spec::ParamSpec, dims::Integer...) -> ComponentArray
    seasonal_params(spec::ParamSpec, θ::AbstractVector) -> ComponentArray
    seasonal_params(spec::ParamSpec, θ::AbstractVector, K::Integer) -> ComponentArray
    seasonal_params(spec::ParamSpec, θ::AbstractMatrix) -> ComponentArray

A coefficient container laid out by `spec`: one block per named parameter, in the spec's order,
so that `ps.σ` and `coefs(spec, ps, :σ)` are the same memory.

- `seasonal_params(spec)` is all zeros. Trailing `dims` widen every block:
  `seasonal_params(spec, K)` gives one column of coefficients per hidden state.
- `seasonal_params(spec, θ)` copies a flat coefficient vector into the layout.
- `seasonal_params(spec, θ, K)` repeats `θ` in each of `K` columns.
- `seasonal_params(spec, θ::AbstractMatrix)` takes one column of `θ` per state.

```julia
spec = ParamSpec(μ = TrigBasis(365, 1) => IdentityLink(), σ = TrigBasis(365, 2) => LogLink())
ps = seasonal_params(spec, collect(1.0:8.0))
ps.μ == [1, 2, 3]
ps.σ == [4, 5, 6, 7, 8]
```
"""
function seasonal_params(spec::ParamSpec, dims::Integer...)
    names = paramnames(spec)
    blocks = ntuple(j -> zeros(length(blockrange(spec, j)), dims...), length(names))
    return ComponentArray(NamedTuple{names}(blocks))
end

function seasonal_params(spec::ParamSpec, θ::AbstractVector)
    @argcheck length(θ) == ncoef(spec) "`θ` has $(length(θ)) coefficients, the spec has $(ncoef(spec))"
    ps = seasonal_params(spec)
    ps .= θ
    return ps
end

# One column of coefficients per group — per hidden state, per site. The flat vector is shared
# by every column, which is the cold start the ensemble fits use.
seasonal_params(spec::ParamSpec, θ::AbstractVector, K::Integer) =
    seasonal_params(spec, repeat(collect(float.(θ)), 1, K))

seasonal_params(spec::ParamSpec, θ::AbstractMatrix) =
    ComponentArray(NamedTuple{paramnames(spec)}(
        ntuple(j -> collect(view(θ, blockrange(spec, j), :)), length(paramnames(spec)))))

# `seasonal_params(spec, θ)`, in a copy of `ps`'s container when that already has the layout it
# would build -- one vector block per parameter, named and placed as `spec` says -- so that the
# `ComponentArray`'s type is not computed from values at run time again. Any other `ps` (a plain
# vector, the blocks in another order, one column per state) goes through `seasonal_params`.
_params_like(spec::ParamSpec, ps::AbstractVector, θ::AbstractVector) =
    _has_spec_layout(spec, ps) ? copyto!(similar(ps, Float64), θ) : seasonal_params(spec, θ)

_has_spec_layout(::ParamSpec, ::AbstractVector) = false
function _has_spec_layout(spec::ParamSpec, ps::ComponentArray)
    (propertynames(ps) == paramnames(spec) && length(ps) == ncoef(spec)) || return false
    # `ntuple` over `Val`, so that each name is a constant and `getproperty` is inferred
    names = paramnames(spec)
    return all(ntuple(j -> _block_at(getproperty(ps, names[j]), blockrange(spec, j)), Val(length(names))))
end
_block_at(block, range) = block isa SubArray && parentindices(block) == (range,)

"""
    ncolumns(spec, ps) -> Int

How many columns of coefficients `ps` holds: 1 for a single model, `K` when every block carries
one column per hidden state.
"""
function ncolumns(spec::ParamSpec, ps::AbstractVector)
    n, nc = length(ps), ncoef(spec)
    @argcheck n % nc == 0 "`ps` has $n coefficients, not a multiple of the spec's $nc"
    return n ÷ nc
end

"""
    column(spec, ps, k) -> Vector

Hidden state `k`'s coefficients, as one flat vector in the spec's order, from a `ps` that holds
one column per state.
"""
column(spec::ParamSpec, ps::AbstractVector, k::Integer) =
    reduce(vcat, (collect(float.(view(getproperty(ps, nm), :, k))) for nm in paramnames(spec)))

"""
    seasonal_curve(spec, ps, name::Symbol, times) -> Vector | Matrix

One parameter of `ps` over `times`, link applied: a vector, or a `K × length(times)` matrix
when `ps` holds one column of coefficients per hidden state.
"""
function seasonal_curve(spec::ParamSpec, ps::AbstractVector, name::Symbol, times)
    j = blockindex(spec, name)
    return _link_curve(spec.links[j], design_matrix(spec.bases[j], times), getproperty(ps, name))
end

# `B * β` is `length(times)` long for one coefficient vector and `length(times) × K` for a block
# of them; the transpose puts the groups in rows, which is the layout `AR1` and `MixtureModel`
# both index as `[k, t]`.
_link_curve(link, B::AbstractMatrix, β::AbstractVector) = apply_link.(Ref(link), B * β)
_link_curve(link, B::AbstractMatrix, β::AbstractMatrix) = apply_link.(Ref(link), permutedims(B * β))

# ## What a fit reports

"""
    FitHistory(; converged, iterations, logtots, solution, ps_iters)

How a fit went: the second value every `fit_mle` in this package returns.

- `converged`   whether the procedure reported success
- `iterations`  how many steps it took
- `logtots`     the loglikelihood after each of them; a one-element vector for a single solve
- `solution`    what the solver returned: a `SciMLBase` solution, a vector of them when the fit
                solved several independent problems, or `nothing` for the EM fits, which solve
                inside every iteration and keep none of those solves
- `ps_iters`    the coefficients before the first iteration and after each one, as `NamedTuple`s
                of arrays, when asked for; otherwise `nothing`

When there is a `solution`, `history.retcode`, `history.objective` and `history.stats` read the
solver's report, as a vector when there were several problems. `history.best_iteration` is the
iterate with the highest loglikelihood, and `loglikelihood(history)` the last one.
"""
struct FitHistory{S,P}
    converged::Bool
    iterations::Int
    logtots::Vector{Float64}
    solution::S
    ps_iters::P
end

FitHistory(; converged, iterations, logtots=Float64[], solution=nothing, ps_iters=nothing) =
    FitHistory(converged, Int(iterations), collect(Float64, logtots), solution, ps_iters)

"""
    loglikelihood(history::FitHistory) -> Float64

The loglikelihood after the last iteration of the fit, `-Inf` when none was recorded.
"""
StatsAPI.loglikelihood(h::FitHistory) = isempty(h.logtots) ? -Inf : last(h.logtots)

@inline function Base.getproperty(h::FitHistory, name::Symbol)
    if name === :retcode || name === :objective || name === :stats
        return _solution_property(getfield(h, :solution), name)
    elseif name === :best_iteration
        # Which iterate attained the highest loglikelihood. For every deterministic fit in this
        # package that is simply the last one, since EM is monotone; it is a real question only
        # for the spatial HMM, whose emission is a randomized quasi-Monte-Carlo estimate and
        # whose `fit_mle` therefore returns the best iterate rather than the final one.
        logtots = getfield(h, :logtots)
        return isempty(logtots) ? 0 : argmax(logtots)
    end
    return getfield(h, name)
end

Base.propertynames(h::FitHistory) = (:converged, :iterations, :logtots, _solver_properties(h)...,
    :ps_iters, :best_iteration)

_solver_properties(::FitHistory) = (:solution, :retcode, :objective, :stats)
_solver_properties(::FitHistory{Nothing}) = ()

# A `SciMLBase` solution is itself an `AbstractVector` -- it indexes into the minimizer -- so the
# many-solutions method has to name its element type rather than take any vector.
_solution_property(sol::AbstractSciMLSolution, name::Symbol) = getproperty(sol, name)
_solution_property(sols::AbstractVector{<:AbstractSciMLSolution}, name::Symbol) =
    [getproperty(sol, name) for sol in sols]

@noinline _solution_property(::Nothing, name::Symbol) = throw(ArgumentError(
    "`$name` is a solver property, and this fit keeps no solver solution: an EM fit solves inside every iteration"))
@noinline _solution_property(solution, name::Symbol) = throw(ArgumentError(
    "`$name` is a solver property, and this fit reports a $(typeof(solution)); see `.solution`"))

# A plain copy of a `ComponentArray`, for `ps_iters`: its blocks as `Array`s. A `ComponentArray`
# carries its layout in its type, which JLD cannot write, and its blocks are views, which JLD
# writes but cannot read back.
_plain(ps::ComponentArray) = map(copy, NamedTuple(ps))

"""
    converged(history::FitHistory) -> Bool
    converged(f::SeasonalFit) -> Bool

Whether the fit reported convergence.
"""
converged(h::FitHistory) = h.converged

function Base.show(io::IO, ::MIME"text/plain", h::FitHistory)
    print(io, "FitHistory: ", h.converged ? "converged" : "did NOT converge",
        " in ", h.iterations, " iteration", h.iterations == 1 ? "" : "s")
    isempty(h.logtots) || print(io, ", loglikelihood ", round(last(h.logtots); digits=4))
    h.ps_iters === nothing || print(io, ", ", length(h.ps_iters), " parameter iterates kept")
    return
end
