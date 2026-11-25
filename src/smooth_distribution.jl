# # A seasonal distribution, one per hidden state
#
# Any `Distributions.jl` law with a `seasonal_family` declaration, its parameters smooth
# functions of the period index, with one column of coefficients per hidden state. It is the
# marginal a weather generator fits given the regimes, such as the Gamma of the temperature
# range given the hidden states.

"""
    SmoothDistribution(D::Type{<:Distribution}, T; K = 1, season = 1, init = nothing)
    SmoothDistribution(family::TimeVaryingFamily, T; K = 1, init = nothing)

A distribution `D` whose parameters are smooth functions of the period index `t ∈ 1:T`, with
`K` hidden states, each with its own coefficients.

- `D` is any law with a [`seasonal_family`](@ref) declaration, which gives the parameters and
  their links.
- `season` gives the basis of every parameter: an integer `d` for `TrigBasis(T, d)`, any basis,
  or a `NamedTuple` with one per parameter, such as `(α = 0, θ = 2)`.
- `init` is the starting point of the fit: a `NamedTuple` of parameter values, such as
  `(α = 2.0, θ = 1.0)`, giving constant curves, or a coefficient vector; one start for every
  state, or a vector of `K` starts. Without it every coefficient is zero.

```julia
m = SmoothDistribution(Gamma, 366; K = 4, season = 2)
m = fit_mle(m, y, OneHot(z, 4); n2t, solver = LBFGS())   # the fitted model
curve(m, :θ)          # 4 × 366, the scale of each state over the year
m(10; state = 2)      # the Gamma of state 2 at period index 10
rand(m, n2t; z)       # one draw per observation
```

`parameters(m)` holds one block per parameter, `ncoef × K`.
"""
struct SmoothDistribution{F<:TimeVaryingFamily,P<:AbstractVector}
    family::F
    nT::Int
    ps::P
end

function SmoothDistribution(D::Type{<:Distribution}, T::Integer; K::Integer=1, season=1, init=nothing)
    fam0 = seasonal_family(D, ConstantBasis())
    names = paramnames(fam0.spec)
    bases = _season_bases(T, season, names)
    spec = ParamSpec(; (nm => (bases[nm] => link) for (nm, link) in zip(names, fam0.spec.links))...)
    return SmoothDistribution(TimeVaryingFamily(fam0.dist, spec), T; K, init)
end

function SmoothDistribution(family::TimeVaryingFamily, T::Integer; K::Integer=1, init=nothing)
    @argcheck T >= 1 "the period `T` is at least 1, not $T"
    @argcheck K >= 1 "the number of states `K` is at least 1, not $K"
    return SmoothDistribution(family, Int(T), _start_params(family.spec, K, init))
end

parameters(m::SmoothDistribution) = m.ps

_nstates(m::SmoothDistribution) = ncolumns(m.family.spec, m.ps)

"""
    curve(m::SmoothDistribution, name::Symbol, times = 1:m.nT; state = :)

One parameter over `times`, link applied: a `K × length(times)` matrix, or the vector of one
state with `state = k`.
"""
function TimeVaryingParameters.curve(m::SmoothDistribution, name::Symbol, times=1:m.nT; state=:)
    c = seasonal_curve(m.family.spec, m.ps, name, times)
    state === (:) && return c
    _check_state(state, _nstates(m))
    return c[state, :]
end

"""
    (m::SmoothDistribution)(t; state) -> Distribution

The law of state `state` at period index `t`; `state` may be left out when `K = 1`.
"""
function (m::SmoothDistribution)(t; state=nothing)
    K = _nstates(m)
    if state === nothing
        K == 1 || throw(ArgumentError("this model has $K states: pass `state`"))
        state = 1
    end
    _check_state(state, K)
    return m.family(t, column(m.family.spec, m.ps, state))
end

# The `T` laws of each state.
function _state_distributions(m::SmoothDistribution)
    design = PeriodicDesign(m.family.spec, 1:m.nT)
    return [distributions(m.family, design, column(m.family.spec, m.ps, k)) for k = 1:_nstates(m)]
end

"""
    loglikelihoods(m::SmoothDistribution, y; n2t, past = nothing) -> Matrix

The `N × K` loglikelihoods: entry `[n, k]` is the logdensity of `y[n]` under state `k` at period
index `n2t[n]`. `past` is accepted and unused: the observations are independent given the period
index and the state.
"""
function loglikelihoods(m::SmoothDistribution, y::AbstractVector{<:Real};
    n2t::AbstractVector{<:Integer}, past=nothing)
    _check_n2t(n2t, length(y), m.nT)
    dists = _state_distributions(m)
    return [logpdf(dists[k][n2t[n]], y[n]) for n in eachindex(y), k in eachindex(dists)]
end

"""
    loglikelihood(m::SmoothDistribution, y; n2t) -> Float64
    loglikelihood(m::SmoothDistribution, y, w; n2t) -> Float64

The loglikelihood of `y`: for a model of one state, or weighted by the state weights `w`
(`OneHot(z, K)` or an `N × K` matrix), `Σ w[n, k] ⋅ logpdf` over the positive weights.
"""
function StatsAPI.loglikelihood(m::SmoothDistribution, y::AbstractVector{<:Real}; n2t::AbstractVector{<:Integer})
    K = _nstates(m)
    K == 1 || throw(ArgumentError("this model has $K states: pass the weights, OneHot(z, K) or an N × K matrix"))
    return sum(loglikelihoods(m, y; n2t))
end

function StatsAPI.loglikelihood(m::SmoothDistribution, y::AbstractVector{<:Real}, w; n2t::AbstractVector{<:Integer})
    W = _state_weights(w, length(y), _nstates(m))
    LL = loglikelihoods(m, y; n2t)
    return sum(W[n, k] * LL[n, k] for n in axes(LL, 1), k in axes(LL, 2) if W[n, k] > 0; init=0.0)
end

"""
    rand([rng, ]m::SmoothDistribution, n2t; z = nothing, past = nothing) -> Vector

One draw per period index `n2t[n]`, from state `z[n]`. `z` is needed when `K > 1`. `past` is
accepted and unused.
"""
function rand(rng::AbstractRNG, m::SmoothDistribution, n2t::AbstractVector{<:Integer}; z=nothing, past=nothing)
    N, K = length(n2t), _nstates(m)
    _check_n2t(n2t, N, m.nT)
    if z === nothing
        K == 1 || throw(ArgumentError("this model has $K states: pass the state of every draw as `z`"))
        z = fill(1, N)
    end
    _check_state_path(z, N, K)
    dists = _state_distributions(m)
    return [rand(rng, dists[z[n]][n2t[n]]) for n in eachindex(n2t)]
end

rand(m::SmoothDistribution, n2t::AbstractVector{<:Integer}; kwargs...) = rand(GLOBAL_RNG, m, n2t; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", m::SmoothDistribution)
    K = _nstates(m)
    println(io, "SmoothDistribution of ", m.family.dist, " over a period of ", m.nT,
        K > 1 ? ", $K states" : "", ", ", nparams(m), " coefficients")
    show(io, MIME"text/plain"(), m.family.spec)
    return
end
