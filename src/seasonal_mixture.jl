# # A seasonal mixture, expressed for ExpectationMaximization.jl
#
# The EM loop for a seasonal mixture is the ordinary one: the only thing that is periodic is the
# *parametrization* of the weights and of the components. So rather than carry a second EM driver,
# this package states the parametrization as three types and hands the loop to
# `ExpectationMaximization.fit_mle!`.
#
# That works because its driver never indexes `α` itself — `α` only travels into `E_step!` and
# `M_step!`, both of which dispatch, and its built-in `M_step!` leaves `α` untyped. A method typed
# on the wrappers below is therefore strictly more specific and wins, with no change to that
# package. The algorithm really is `ClassicEM`; only the parameters are seasonal.
#
# `loglikelihoods!` lives here because evaluating a likelihood needs no solver. The matching
# `M_step!` lives in the fitting extension, next to the `update_*` functions it calls.

"""
    SeasonalSample(y, n2t, n_in_t)

The sample of a seasonal fit: the observations together with the period index of each one.

`n2t[n] ∈ 1:T` is the period index of observation `n`, and `n_in_t[t]` the indices of the
observations sharing period index `t`. `y` is stored as given, not copied.
"""
struct SeasonalSample{T,V<:AbstractVector{T}} <: AbstractVector{T}
    y::V
    n2t::Vector{Int}
    n_in_t::Vector{Vector{Int}}
end

# `n2t` and `n_in_t` are deliberately not parametric: they are period indices, built here rather
# than taken from the caller, so there is nothing for a type parameter to vary over.
SeasonalSample(y::AbstractVector, n2t::AbstractVector, nT::Integer) =
    SeasonalSample{eltype(y),typeof(y)}(y, collect(Int, n2t), [findall(==(t), n2t) for t = 1:nT])

Base.size(s::SeasonalSample) = size(s.y)
Base.getindex(s::SeasonalSample, n) = s.y[n]

"""
    SeasonalWeights(α, θ, basis, cfg)

The seasonal mixing weights of a mixture, as the weight vector the EM algorithm of
`ExpectationMaximization` works on.

`α` is `K × T`: `α[k, t]` is the weight of component `k` at period index `t`. `θ` holds the
`(K-1) × (2d+1)` multinomial-logit coefficients that generate it, with component `K` as the
reference, and `basis` is the `T × (2d+1)` design matrix from [`trig_basis`](@ref). `cfg` holds
the M-step settings.

Indexing it returns the time-averaged weight of a component.
"""
mutable struct SeasonalWeights{T,C} <: AbstractVector{T}
    α::Matrix{T}
    θ::Matrix{T}
    basis::Matrix{T}
    cfg::C
end

# The three matrices come from different places — the initial mixtures, the caller's coefficients
# and `trig_basis` — so they are promoted to a common element type rather than assumed to share one.
function SeasonalWeights(α::AbstractMatrix, θ::AbstractMatrix, basis::AbstractMatrix, cfg)
    T = promote_type(eltype(α), eltype(θ), eltype(basis))
    return SeasonalWeights{T,typeof(cfg)}(Matrix{T}(α), Matrix{T}(θ), Matrix{T}(basis), cfg)
end

Base.size(a::SeasonalWeights) = (size(a.α, 1),)
Base.getindex(a::SeasonalWeights, k::Integer) = sum(view(a.α, k, :)) / size(a.α, 2)

"""
    SeasonalFamily(dists, θ, fam, design)

One component of a seasonal mixture during an EM fit. It holds the `T` distributions it takes
over a period, the coefficients `θ` that generate them, its [`seasonal_family`](@ref) `fam`, and
`design`, that family's bases evaluated over the period.

It does not implement the `Distribution` interface itself: `logpdf`, `rand` and `params` apply to
each `dists[t]`.
"""
struct SeasonalFamily{D<:Distribution,T<:Real,F,G} <: Distribution{Univariate,Continuous}
    dists::Vector{D}
    θ::Vector{T}
    fam::F
    design::G
end

"""
    loglikelihoods!(LL, dists, α::SeasonalWeights, y::SeasonalSample)

E-step hook: `LL[n, k] = log αₖ(tₙ) + log fₖ,ₜₙ(yₙ)`, the weight and the component both taken at
the period index of observation `n`.
"""
function loglikelihoods!(LL::AbstractMatrix, dists, α::SeasonalWeights, y::SeasonalSample)
    for k in eachindex(dists)
        dₖ = dists[k].dists
        @inbounds for (n, t) in enumerate(y.n2t)
            LL[n, k] = log(α.α[k, t]) + logpdf(dₖ[t], y.y[n])
        end
    end
    return LL
end

# # The seasonal mixture as a model: structure apart from coefficients
#
# Everything above states the parametrization for `ExpectationMaximization`'s driver. What
# follows is the model a user holds: a component family, a weight parametrization, a period and
# one `ComponentArray` of coefficients — the same `(structure, ps)` pair as `SmoothAR1`, fitted
# by the same `fit_mle(model, data) -> (model, history)` call.

"""
    seasonal_family(D::Type, basis) -> TimeVaryingFamily

How the parameters of the distribution `D` vary with the period index: one basis and one link
per parameter, in the order `D`'s constructor takes them.

`Exponential`, `Rayleigh`, `Gamma`, `Weibull`, `Normal`, `LogNormal`, `Bernoulli` and `Poisson`
are declared. Any other family becomes usable as a mixture component or an HMM emission with one
method:

```julia
SmoothPeriodicStatsModels.seasonal_family(::Type{Pareto}, basis) =
    TimeVaryingFamily(Pareto, ParamSpec(α = basis => LogLink(), θ = basis => LogLink()))
```

The method is declared on the type without parameters (`Rayleigh`, not `Rayleigh{Float64}`).
"""
function seasonal_family end

# `Vector{Rayleigh{Float64}}` is what a fit actually holds, so the lookup is made on the
# unparametrized type and the declarations below stay one line each.
function seasonal_family(D::Type{<:Distribution}, basis)
    W = Base.typename(D).wrapper
    W === D && throw(ArgumentError("""
    no `seasonal_family` method for $D, so this package does not know how its parameters vary
    with the period index. Declare it:

        SmoothPeriodicStatsModels.seasonal_family(::Type{$D}, basis) =
            TimeVaryingFamily($D, ParamSpec(name = basis => LogLink(), ...))

    one entry per constructor argument, in order. Alternatively write an `update_dist` method
    for $D, which replaces the generic M-step entirely."""))
    return seasonal_family(W, basis)
end

seasonal_family(::Type{Exponential}, basis) =
    TimeVaryingFamily(Exponential, ParamSpec(θ=basis => LogLink()))
seasonal_family(::Type{Rayleigh}, basis) =
    TimeVaryingFamily(Rayleigh, ParamSpec(σ=basis => LogLink()))
seasonal_family(::Type{Gamma}, basis) =
    TimeVaryingFamily(Gamma, ParamSpec(α=basis => LogLink(), θ=basis => LogLink()))
seasonal_family(::Type{Weibull}, basis) =
    TimeVaryingFamily(Weibull, ParamSpec(α=basis => LogLink(), θ=basis => LogLink()))
seasonal_family(::Type{Normal}, basis) =
    TimeVaryingFamily(Normal, ParamSpec(μ=basis => IdentityLink(), σ=basis => LogLink()))
seasonal_family(::Type{LogNormal}, basis) =
    TimeVaryingFamily(LogNormal, ParamSpec(μ=basis => IdentityLink(), σ=basis => LogLink()))
seasonal_family(::Type{Bernoulli}, basis) =
    TimeVaryingFamily(Bernoulli, ParamSpec(p=basis => LogitLink()))
seasonal_family(::Type{Poisson}, basis) =
    TimeVaryingFamily(Poisson, ParamSpec(λ=basis => LogLink()))

"""
    mixture_weight_spec(T, d) -> ParamSpec

The parametrization of the mixing weights over a period of `T`: one trigonometric polynomial of
degree `d` per component but the last. The polynomials become weights together, through a
softmax with the last component as the reference ([`αₜ`](@ref) for `K = 2`).
"""
mixture_weight_spec(T::Integer, d::Integer) = ParamSpec(α=TrigBasis(T, d) => IdentityLink())

"""
    SmoothSeasonalMixture(D::Type, K, d, T; init, dα = d)
    SmoothSeasonalMixture(pairs, T; init, dα = 1)
    SmoothSeasonalMixture(families, spec_α, nT; init)
    SmoothSeasonalMixture(families, spec_α, nT, ps)

A mixture whose weights and component parameters are smooth functions of the period index.
It holds one [`seasonal_family`](@ref) per component, the weights' parametrization `spec_α`, the
period `nT`, and the coefficients `ps`.

- `SmoothSeasonalMixture(D, K, d, T; init, dα = d)`: `K` components of family `D`, each parameter a
  trigonometric polynomial of degree `d`, the weights of degree `dα`.
- `SmoothSeasonalMixture(pairs, T; init, dα = 1)`: one `distribution => degree` pair per
  component, so components may differ in family and degree.
- `SmoothSeasonalMixture(families, spec_α, nT; init)` and `(families, spec_α, nT, ps)`: any
  `TimeVaryingFamily` per component and any weight parametrization.

`init` gives each component its starting parameter: a number, a coefficient vector, or a
`NamedTuple` for a family with several parameters. The `K` starting components must differ.

```julia
m  = SmoothSeasonalMixture(Exponential, 2, 1, 366; init = [1.0, 5.0])
ps = parameters(m)                     # ComponentArray(α = …, Y = …)
ps.α                                   # (2dα+1) × (K-1) weight coefficients
component_coefs(m, 2)                  # component 2's coefficients, a writable view

m, history = fit_mle(m, y; n2t, solver)
m(196)                                 # the `MixtureModel` at period index 196
curve(m, :θ)                           # K × T, each component's scale over the period
mixture_weights(m)                     # K × T, the mixing weights

m2 = SmoothSeasonalMixture([Exponential => 0, Gamma => 2], 366;
                           init = [2.0, (α = 3.0, θ = 1.0)])
```

`ps.Y` holds the components' coefficients end to end; [`component_coefs`](@ref) returns one
component's block.
"""
struct SmoothSeasonalMixture{F<:Tuple,A<:ParamSpec,P<:AbstractVector}
    components::F
    spec_α::A
    nT::Int
    ps::P
end

function SmoothSeasonalMixture(families, spec_α::ParamSpec, nT::Integer; init)
    fams = Tuple(families)
    K = length(fams)
    @argcheck K > 1 "a mixture needs at least two components"
    @argcheck length(init) == K "`init` gives one starting parameter per component, so $K of them"
    starts = [_component_start(f.spec, v) for (f, v) in zip(fams, init)]
    # Two components are the same starting model only if the *family* and the coefficients both
    # match: an `Exponential` and a `Rayleigh` that happen to start from the same coefficients
    # are two different laws, and perfectly identifiable.
    @argcheck allunique(zip(fams, starts)) "the starting components must differ: identical ones give identical posteriors, and EM cannot move off that point"
    ps = ComponentArray(α=zeros(ncoef(spec_α), K - 1), Y=reduce(vcat, starts))
    return SmoothSeasonalMixture(fams, spec_α, Int(nT), ps)
end

SmoothSeasonalMixture(component::TimeVaryingFamily, spec_α::ParamSpec, nT::Integer, K::Integer; init) =
    SmoothSeasonalMixture(ntuple(_ -> component, K), spec_α, nT; init)

SmoothSeasonalMixture(D::Type, K::Integer, d::Integer, T::Integer; dα::Integer=d, init) =
    SmoothSeasonalMixture(seasonal_family(D, TrigBasis(T, d)), mixture_weight_spec(T, dα), T, K; init)

# `[Exponential => 0, Gamma => 2]`: the compact spelling of a mixture whose components differ.
SmoothSeasonalMixture(pairs::AbstractVector{<:Pair}, T::Integer; dα::Integer=1, init) =
    SmoothSeasonalMixture([seasonal_family(D, TrigBasis(T, d)) for (D, d) in pairs],
        mixture_weight_spec(T, dα), T; init)

# A scalar start names the only parameter, a `NamedTuple` names them all, a vector *is* the
# coefficient block. All three go through `initial_coefficients`, which inverts each link and
# leaves the harmonics at zero — the flat start every guess in this package uses.
_component_start(spec::ParamSpec, v::NamedTuple) = initial_coefficients(spec; v...)
_component_start(spec::ParamSpec, v) =
    initial_coefficients(spec; NamedTuple{paramnames(spec)}((v,))...)

parameters(m::SmoothSeasonalMixture) = m.ps

"""
    ncomponents(m::SmoothSeasonalMixture) -> Int

How many components the mixture has.
"""
Distributions.ncomponents(m::SmoothSeasonalMixture) = length(m.components)

"""
    component_coefs(m::SmoothSeasonalMixture, k) -> SubArray

Component `k`'s coefficients: a view into `parameters(m).Y`, so writing through it sets them.
When all components share family and degree, `reshape(parameters(m).Y, :, ncomponents(m))`
gives them as columns.
"""
component_coefs(m::SmoothSeasonalMixture, k::Integer) = view(m.ps.Y, component_range(m, k))

"""
    component_range(m::SmoothSeasonalMixture, k) -> UnitRange

Where component `k`'s coefficients sit inside `parameters(m).Y`.
"""
function component_range(m::SmoothSeasonalMixture, k::Integer)
    @argcheck 1 <= k <= ncomponents(m)
    counts = map(ncoef, m.components)
    off = sum(counts[j] for j = 1:(k-1); init=0)
    return (off+1):(off+counts[k])
end

"""
    mixture_weights(m::SmoothSeasonalMixture, times = 1:m.nT) -> Matrix

The mixing weights, `K × length(times)`: `α[k, t]` is the weight of component `k` at `times[t]`.
Each column sums to one.
"""
function mixture_weights(m::SmoothSeasonalMixture, times=1:m.nT)
    K = ncomponents(m)
    P = design_matrix(m.spec_α.bases[1], times) * m.ps.α   # length(times) × (K-1)
    α = Matrix{Float64}(undef, K, size(P, 1))
    for t in axes(P, 1)
        shift = max(zero(eltype(P)), maximum(view(P, t, :)))
        α[K, t] = exp(-shift)
        total = α[K, t]
        for k = 1:K-1
            α[k, t] = exp(P[t, k] - shift)
            total += α[k, t]
        end
        for k = 1:K
            α[k, t] /= total
        end
    end
    return α
end

"""
    curve(m::SmoothSeasonalMixture, name::Symbol, times = 1:m.nT) -> Matrix

One parameter of the components over `times`, `K × length(times)` with one row per component:
`:θ` for `Exponential`, `:μ` and `:σ` for `Normal`, links applied. Every component must have
that parameter. The mixing weights are [`mixture_weights`](@ref).
"""
function TimeVaryingParameters.curve(m::SmoothSeasonalMixture, name::Symbol, times=1:m.nT)
    missing_in = [k for (k, f) in enumerate(m.components) if !(name in paramnames(f.spec))]
    isempty(missing_in) || throw(ArgumentError(
        "`$name` is not a parameter of component$(length(missing_in) == 1 ? "" : "s") " *
        "$(join(missing_in, ", ")), which take " *
        "$(join(unique(paramnames(m.components[k].spec) for k in missing_in), " and "))" *
        (name === :α ? "; the mixing weights are `mixture_weights(m)`" : "")))
    # One row per component: the bases may differ between them, so each is evaluated against
    # its own and only the number of columns is shared.
    rows = map(enumerate(m.components)) do (k, f)
        j = blockindex(f.spec, name)
        B = design_matrix(f.spec.bases[j], times)
        apply_link.(Ref(f.spec.links[j]), B * view(component_coefs(m, k), blockrange(f.spec, j)))
    end
    return permutedims(reduce(hcat, rows))
end

"""
    components(m::SmoothSeasonalMixture) -> Vector{Vector{<:Distribution}}

Component `k`'s distribution at every period index, as `components(m)[k][t]`.
"""
function Distributions.components(m::SmoothSeasonalMixture)
    return [distributions(f, PeriodicDesign(f.spec, 1:m.nT), component_coefs(m, k))
            for (k, f) in enumerate(m.components)]
end

"""
    mixtures(m::SmoothSeasonalMixture) -> Vector{MixtureModel}

The `MixtureModel` at each of the `nT` period indices.
"""
function mixtures(m::SmoothSeasonalMixture)
    K, dists, α = ncomponents(m), components(m), mixture_weights(m)
    return [MixtureModel([dists[k][t] for k = 1:K], α[:, t]) for t = 1:m.nT]
end

"""
    (m::SmoothSeasonalMixture)(t) -> MixtureModel

The mixture at one period index.
"""
(m::SmoothSeasonalMixture)(t) = MixtureModel(
    [f(t, component_coefs(m, k)) for (k, f) in enumerate(m.components)],
    vec(mixture_weights(m, t:t)))

"""
    loglikelihood(m::SmoothSeasonalMixture, y; n2t) -> Float64

The loglikelihood of `y` under `m`, each observation scored against the mixture at its own
period index `n2t[n]`.
"""
function StatsAPI.loglikelihood(m::SmoothSeasonalMixture, y::AbstractVector;
    n2t=n_to_t(length(y), m.nT)::AbstractVector{<:Integer})
    mix = mixtures(m)
    return sum(logpdf(mix[t], y[n]) for (n, t) in enumerate(n2t))
end

"""
    rand([rng, ]m::SmoothSeasonalMixture, n2t::AbstractVector{<:Integer}) -> Vector
    rand([rng, ]m::SmoothSeasonalMixture, N::Integer) -> Vector

Independent draws, the `n`-th from the mixture at period index `n2t[n]`. With `N`, the period
indices are `n_to_t(N, m.nT)`.
"""
rand(rng::AbstractRNG, m::SmoothSeasonalMixture, n2t::AbstractVector{<:Integer}) =
    (mix = mixtures(m); [rand(rng, mix[t]) for t in n2t])
rand(rng::AbstractRNG, m::SmoothSeasonalMixture, N::Integer) = rand(rng, m, n_to_t(N, m.nT))
rand(m::SmoothSeasonalMixture, n2t::AbstractVector{<:Integer}) = rand(GLOBAL_RNG, m, n2t)
rand(m::SmoothSeasonalMixture, N::Integer) = rand(GLOBAL_RNG, m, N)

function Base.show(io::IO, ::MIME"text/plain", m::SmoothSeasonalMixture)
    laws = unique(string(f.dist) for f in m.components)
    println(io, "SmoothSeasonalMixture of ", ncomponents(m), " ", join(laws, " + "),
        " over a period of ", m.nT, ", ", nparams(m), " coefficients")
    println(io, "  weights, softmax over the first ", ncomponents(m) - 1, ":")
    show(io, MIME"text/plain"(), m.spec_α)
    for (k, f) in enumerate(m.components)
        println(io, "  component ", k, ", ", f.dist, ", coefficients ", component_range(m, k), ":")
        show(io, MIME"text/plain"(), f.spec)
    end
    return
end
