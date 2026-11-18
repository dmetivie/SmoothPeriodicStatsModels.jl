# # The periodic HMM with any emission, with its coefficients attached
#
# The transitions are those of the Bernoulli HMMs: a multinomial logit of a basis expansion over
# the period index, state `K` the reference. The emission of each block of rows is a seasonal family,
# the same `TimeVaryingFamily` a seasonal mixture component is, with one coefficient vector per
# state. Fitting it is the mixture's weighted M-step with the posterior state probabilities as
# weights.

"""
    SmoothPeriodicHMM(families, K, T; init, θᴬ, a, basis_A)
    SmoothPeriodicHMM(D::Type, K, d, T; init, dA = d)
    SmoothPeriodicHMM(families, a, θᴬ, θᴮ, T; basis_A)

A periodic HMM without memory whose transition probabilities and emission parameters are smooth
functions of the period index. The observations are `D × N`, split into blocks of rows,
one per family; given the state, the blocks are independent.

- `families`: one per block, a `distribution => degree` pair (a [`seasonal_family`](@ref) with a
  trigonometric polynomial of that degree per parameter) or any `TimeVaryingFamily`. A
  univariate law covers one row; a multivariate law, built from scalar parameters, covers as
  many rows as its length.
- `init[k]`: the starting parameters of state `k`, one per block (just the one when there is a
  single block): a number, a `NamedTuple` for a family with several parameters, or a coefficient
  vector. The states must start apart.
- `θᴬ`, `K × (K-1) × (2d+1)`: the transition coefficients, as for [`SmoothARPeriodicHMM`](@ref),
  zero (uniform transitions) by default; `a` the initial distribution, uniform by default.
- `θᴮ[k, b]`: the coefficient vector of block `b` in state `k`.

```julia
m0 = SmoothPeriodicHMM([Gamma => 1, Normal => 2], 2, 365;
                       init = [[(α = 2.0, θ = 1.0), (μ = 0.0, σ = 1.0)],
                               [(α = 1.0, θ = 5.0), (μ = 3.0, σ = 2.0)]])
m, history = fit_mle(m0, Y; n2t, solver)       # Y is 2 × N
parameters(m).A                        # the transition coefficients
emission_coefs(m, 2, 1)                # state 2's Gamma coefficients, a writable view
curve(m, :μ)                           # K × T, the Normal mean of each state
PeriodicHMMDists(m)                    # the evaluated model
```

`rand`, `viterbi`, `loglikelihood` and `loglikelihoods` act as on `PeriodicHMMDists(m)`. It is a
`HiddenMarkovModels.AbstractHMM` whose controls are [`SeasonalControls(n2t)`](@ref SeasonalControls).
"""
struct SmoothPeriodicHMM{P<:AbstractVector,F<:Tuple,S<:PeriodicHMMDists,BA,C} <: HMMs.AbstractHMM
    ps::P          # ComponentArray(A = θᴬ, B = the emission coefficients end to end)
    families::F
    stored::S
    nT::Int
    basis_A::BA
    cfg::C
end

function SmoothPeriodicHMM(families, a::AbstractVector, θᴬ::AbstractArray{<:Real,3},
    θᴮ::AbstractMatrix{<:AbstractVector{<:Real}}, T::Integer;
    basis_A=TrigBasis(T, _trig_order(size(θᴬ, 3), "θᴬ")), cfg=nothing)
    fams = _families(families, T)
    K = size(θᴬ, 1)
    @argcheck size(θᴬ, 2) == K - 1 "θᴬ is $(size(θᴬ)): K × (K-1) × coefficients"
    @argcheck size(θᴮ) == (K, length(fams)) "θᴮ is $(size(θᴮ)), and $K states with $(length(fams)) blocks need K × blocks"
    for k = 1:K, (b, f) in enumerate(fams)
        @argcheck length(θᴮ[k, b]) == ncoef(f) "θᴮ[$k, $b] has $(length(θᴮ[k, b])) coefficients and block $b takes $(ncoef(f))"
    end
    _check_basis(basis_A, size(θᴬ, 3), "θᴬ")
    ps = ComponentArray(A=collect(float.(θᴬ)), B=reduce(vcat, [float.(θᴮ[k, b]) for k = 1:K for b in eachindex(fams)]))
    X = design_matrix(basis_A, 1:T)
    A = _transitions!(zeros(K, K, T), ps.A, X)
    B = map(enumerate(fams)) do (b, f)
        design = PeriodicDesign(f.spec, 1:T)
        dists = [distributions(f, design, view(ps.B, _emission_range(fams, K, k, b))) for k = 1:K]
        [dists[k][t] for k = 1:K, t = 1:T]
    end
    stored = PeriodicHMMDists(collect(float.(a)), A, Tuple(B))
    return SmoothPeriodicHMM(ps, fams, stored, Int(T), basis_A, cfg)
end

function SmoothPeriodicHMM(families, K::Integer, T::Integer; init,
    θᴬ=zeros(K, K - 1, 3), a=fill(1 / K, K), basis_A=TrigBasis(T, _trig_order(size(θᴬ, 3), "θᴬ")))
    fams = _families(families, T)
    @argcheck length(init) == K "`init` gives the starting parameters of each of the $K states"
    starts = [_state_starts(fams, v, k) for (k, v) in enumerate(init)]
    @argcheck allunique(starts) "the states must start apart: identical starting states get identical posteriors, and EM cannot move off that point"
    θᴮ = [starts[k][b] for k = 1:K, b in eachindex(fams)]
    return SmoothPeriodicHMM(fams, a, θᴬ, θᴮ, T; basis_A)
end

SmoothPeriodicHMM(D::Type{<:Distribution}, K::Integer, d::Integer, T::Integer; dA::Integer=d, init) =
    SmoothPeriodicHMM([D => d], K, T; init, θᴬ=zeros(K, K - 1, 2dA + 1))

# `Gamma => 1` or a `TimeVaryingFamily`, one per block
_family(f::TimeVaryingFamily, _) = f
_family((D, d)::Pair, T) = seasonal_family(D, TrigBasis(T, d))
_families(f::Union{TimeVaryingFamily,Pair}, T) = (_family(f, T),)
_families(families, T) = Tuple(_family(f, T) for f in families)

# One state's starting coefficients, one vector per block. With one block the start may be given
# directly; a coefficient vector is a start of its own, so only then is a vector unwrapped.
function _state_starts(fams::Tuple, v, k)
    length(fams) == 1 && !_is_block_list(v) && return [_component_start(fams[1].spec, v)]
    @argcheck length(v) == length(fams) "`init[$k]` gives $(length(v)) starts for $(length(fams)) blocks"
    return [_component_start(f.spec, vb) for (f, vb) in zip(fams, v)]
end
_is_block_list(v) = v isa Union{Tuple,AbstractVector} && !(v isa AbstractVector{<:Real})

parameters(m::SmoothPeriodicHMM) = m.ps

"""
    PeriodicHMMDists(m::SmoothPeriodicHMM) -> PeriodicHMMDists

The evaluated model: the transition matrices and the emission laws the coefficients of `m`
generate at every period index.
"""
PeriodicHMMDists(m::SmoothPeriodicHMM) = m.stored

size(m::SmoothPeriodicHMM, dim=:) = size(m.stored, dim)
_period(m::SmoothPeriodicHMM) = m.nT

"""
    emission_range(m::SmoothPeriodicHMM, k, b) -> UnitRange

Where the coefficients of block `b` in state `k` sit inside `parameters(m).B`: state by state,
block by block within a state.
"""
emission_range(m::SmoothPeriodicHMM, k::Integer, b::Integer) = _emission_range(m.families, size(m.stored.A, 1), k, b)

function _emission_range(fams::Tuple, K, k, b)
    @argcheck 1 <= k <= K && 1 <= b <= length(fams)
    counts = map(ncoef, fams)
    off = (k - 1) * sum(counts) + sum(counts[i] for i = 1:(b-1); init=0)
    return (off+1):(off+counts[b])
end

"""
    emission_coefs(m::SmoothPeriodicHMM, k, b) -> SubArray

The coefficients of block `b` in state `k`: a view into `parameters(m).B`, so writing through it
sets them (call `fit_mle` or rebuild the model to re-evaluate).
"""
emission_coefs(m::SmoothPeriodicHMM, k::Integer, b::Integer) = view(m.ps.B, emission_range(m, k, b))

"""
    curve(m::SmoothPeriodicHMM, name::Symbol, times = 1:m.nT; block) -> Matrix

One emission parameter over `times`, `K × length(times)` with one row per state, link applied.
`block` is needed only when several blocks have a parameter of that name.
"""
function TimeVaryingParameters.curve(m::SmoothPeriodicHMM, name::Symbol, times=1:m.nT; block=nothing)
    having = [b for (b, f) in enumerate(m.families) if name in paramnames(f.spec)]
    if block === nothing
        isempty(having) && throw(ArgumentError("no block has a parameter `$name`; the blocks take " *
                                               join((join(paramnames(f.spec), ", ") for f in m.families), "; ")))
        length(having) == 1 || throw(ArgumentError("blocks $(join(having, ", ")) all have a parameter `$name`; choose one with `block`"))
        block = only(having)
    end
    @argcheck block in having "block $block has no parameter `$name`"
    f = m.families[block]
    j = blockindex(f.spec, name)
    X = design_matrix(f.spec.bases[j], times)
    rows = [apply_link.(Ref(f.spec.links[j]), X * view(emission_coefs(m, k, block), blockrange(f.spec, j)))
            for k = 1:size(m, 1)]
    return permutedims(reduce(hcat, rows))
end

# The accessors forward to the evaluated twin, which `_refresh!` keeps in step with `ps`.
HMMs.initialization(m::SmoothPeriodicHMM) = HMMs.initialization(m.stored)
HMMs.transition_matrix(m::SmoothPeriodicHMM, c::Tuple) = HMMs.transition_matrix(m.stored, c)
HMMs.obs_distributions(m::SmoothPeriodicHMM, c::Tuple) = HMMs.obs_distributions(m.stored, c)
HMMs.obs_logdensities!(logb::AbstractVector, m::SmoothPeriodicHMM, obs, c::Tuple; kwargs...) =
    HMMs.obs_logdensities!(logb, m.stored, obs, c; kwargs...)

# Rewrite the evaluated arrays from the coefficients, in place.
function _refresh!(m::SmoothPeriodicHMM)
    K, T = size(m.stored.A, 1), m.nT
    _transitions!(m.stored.A, m.ps.A, design_matrix(m.basis_A, 1:T))
    for (b, f) in enumerate(m.families)
        design = PeriodicDesign(f.spec, 1:T)
        Bb = m.stored.B[b]
        for k = 1:K
            Bb[k, :] = distributions(f, design, emission_coefs(m, k, b))
        end
    end
    return m
end

rand(rng::AbstractRNG, m::SmoothPeriodicHMM, N::Integer; kwargs...) = rand(rng, m.stored, N; kwargs...)
rand(rng::AbstractRNG, m::SmoothPeriodicHMM, n2t::_IndexVector; kwargs...) = rand(rng, m.stored, n2t; kwargs...)
rand(rng::AbstractRNG, m::SmoothPeriodicHMM, z::_IndexVector, n2t::_IndexVector) = rand(rng, m.stored, z, n2t)
rand(m::SmoothPeriodicHMM, N::Integer; kwargs...) = rand(GLOBAL_RNG, m, N; kwargs...)
rand(m::SmoothPeriodicHMM, n2t::_IndexVector; kwargs...) = rand(GLOBAL_RNG, m, n2t; kwargs...)
rand(m::SmoothPeriodicHMM, z::_IndexVector, n2t::_IndexVector) = rand(GLOBAL_RNG, m, z, n2t)

loglikelihoods(m::SmoothPeriodicHMM, Y::AbstractVecOrMat{<:Real}; n2t=n_to_t(_ncols(Y), m.nT), kwargs...) =
    loglikelihoods(m.stored, Y; n2t, kwargs...)
viterbi(m::SmoothPeriodicHMM, Y::AbstractVecOrMat{<:Real}; n2t=n_to_t(_ncols(Y), m.nT), kwargs...) =
    viterbi(m.stored, Y; n2t, kwargs...)
viterbi(m::SmoothPeriodicHMM, Ys::AbstractVector{<:AbstractVecOrMat{<:Real}};
    n2t=[n_to_t(_ncols(Y), m.nT) for Y in Ys], kwargs...) = viterbi(m.stored, Ys; n2t, kwargs...)

function Base.show(io::IO, ::MIME"text/plain", m::SmoothPeriodicHMM)
    K, D, T = size(m)
    println(io, "SmoothPeriodicHMM: ", K, " states, ", _rows(D), ", period ", T)
    print(io, "  transitions  ", m.basis_A, ", ", length(m.ps.A), " coefficients")
    for (f, r) in zip(m.families, m.stored.blocks)
        println(io)
        print(io, "  ", rpad(_rows(r), 11), "  ", _law_name(f), ", ", ncoef(f), " coefficients per state")
    end
    return
end

_rows(D::Integer) = D == 1 ? "1 row" : "$D rows"
_rows(r::UnitRange) = length(r) == 1 ? "row $(first(r))" : "rows $r"
_law_name(f::TimeVaryingFamily{<:DistConstructor}) = string(f.dist)
_law_name(f::TimeVaryingFamily) = "law of (" * join(paramnames(f.spec), ", ") * ")"
