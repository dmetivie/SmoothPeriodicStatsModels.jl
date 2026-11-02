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

The number of free coefficients — `length(parameters(model))`. The name and the meaning are
`HMMBase`'s. Note `TimeVaryingParameters.nparams(spec)` counts a spec's *named parameters*
instead, which is why that one is not exported.
"""
nparams(model) = length(parameters(model))

"""
    seasonal_params(spec::ParamSpec, dims::Integer...) -> ComponentArray
    seasonal_params(spec::ParamSpec, θ::AbstractVector) -> ComponentArray

A coefficient container laid out by `spec`: one block per named parameter, in the spec's order,
so that `ps.σ` and `coefs(spec, ps, :σ)` are the same memory.

Zeros by default. Trailing `dims` widen every block — `seasonal_params(spec, K)` gives one
*column* of coefficients per hidden state. Passing a flat `θ` copies it into the layout, which
is how a v3 coefficient vector is carried over.
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

"""
    ncolumns(spec, ps) -> Int

How many independent fits `ps` holds: 1 for a single model, `K` when every block carries one
column per hidden state.

A `ComponentArray` is an `AbstractVector` whatever the shape of its blocks — `ps.μ` may be a
matrix while `ps` is still flat — so the number of columns has to be counted against the spec
rather than read off `typeof(ps)`.
"""
function ncolumns(spec::ParamSpec, ps::AbstractVector)
    n, nc = length(ps), ncoef(spec)
    @argcheck n % nc == 0 "`ps` has $n coefficients, not a multiple of the spec's $nc"
    return n ÷ nc
end

"""
    column(spec, ps, k) -> Vector

Hidden state `k`'s coefficients, flat and in the spec's order.

Not a reshape: a `ComponentArray` lays its blocks out one after another, so the flat order is
`[vec(μ); vec(ρ); vec(σ)]` and state `k` is a stride through each block, not a contiguous slice.
"""
column(spec::ParamSpec, ps::AbstractVector, k::Integer) =
    reduce(vcat, (collect(float.(view(getproperty(ps, nm), :, k))) for nm in paramnames(spec)))

"""
    seasonal_curve(spec, ps, name::Symbol, times) -> Vector | Matrix

One parameter of `ps` over `times`, links applied: a vector, or a `K × length(times)` matrix
when `ps` holds one column of coefficients per hidden state.

`TimeVaryingParameters.curve` cannot be used directly here. It addresses a flat vector through
the spec's offsets, which describe *one* column, whereas a `ComponentArray` with `K` columns is
still flat but laid out `[vec(μ); vec(ρ); …]` — so `coefs(spec, ps, :ρ)` would read across the
wrong block. The named access `ps.ρ` is what knows the real shape.
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

How a fit went — the second half of every `fit_mle` return in this package.

- `converged`   whether the procedure reported success
- `iterations`  how many steps it took
- `logtots`     the loglikelihood after each of them; a one-element vector for a single solve
- `solution`    what the solver returned: a `SciMLBase` solution, a vector of them when the fit
                solved several independent problems, or `nothing`
- `ps_iters`    the parameter trajectory when it was asked for, otherwise `nothing`

`history.retcode`, `history.objective` and `history.stats` reach into `solution`, collecting over
the problems when there were several — which `history.solution.retcode` could not do, since
`solution` is then a vector.

This is `HMMBase.EMHistory`'s shape, widened to cover the one-shot solves too, so that a fit in
this package reports the same way whether it ran EM or a single optimisation.
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

Base.propertynames(::FitHistory) =
    (fieldnames(FitHistory)..., :retcode, :objective, :stats, :best_iteration)

# A `SciMLBase` solution is itself an `AbstractVector` -- it indexes into the minimizer -- so the
# many-solutions method has to name its element type rather than take any vector.
_solution_property(sol::AbstractSciMLSolution, name::Symbol) = getproperty(sol, name)
_solution_property(sols::AbstractVector{<:AbstractSciMLSolution}, name::Symbol) =
    [getproperty(sol, name) for sol in sols]

@noinline _solution_property(solution, name::Symbol) = throw(ArgumentError(
    "`$name` is a solver property, and this fit reports a $(typeof(solution)); see `.solution`"))

converged(h::FitHistory) = h.converged

function Base.show(io::IO, ::MIME"text/plain", h::FitHistory)
    print(io, "FitHistory: ", h.converged ? "converged" : "did NOT converge",
        " in ", h.iterations, " iteration", h.iterations == 1 ? "" : "s")
    isempty(h.logtots) || print(io, ", loglikelihood ", round(last(h.logtots); digits=4))
    h.ps_iters === nothing || print(io, ", ", length(h.ps_iters), " parameter iterates kept")
    return
end
