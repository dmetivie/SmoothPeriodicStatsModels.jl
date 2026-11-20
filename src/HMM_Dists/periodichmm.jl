# # A periodic HMM with any emission laws, evaluated
#
# The Bernoulli HMMs emit one binary value per site, with memory. Here the emission is any
# `Distributions.jl` law and there is no memory: the rows of the `D × N` observations are split
# into blocks, each block has one law per state and period index, and the blocks are independent
# given the state. A univariate law covers one row, a multivariate law as many rows as its
# length.

"""
    PeriodicHMMDists([a, ]A, B)

A periodic hidden Markov model with `K` states, period `T` and any emission laws, without memory.

- `a`: the initial state distribution, length `K`; uniform if omitted.
- `A`: the `K × K × T` transition matrices; `A[:, :, t]` is the step leaving period index `t`.
- `B`: one `K × T` matrix of laws per block of rows, as a tuple (a single matrix for one block).
  `B[b][k, t]` is the law of the rows of block `b` in state `k` at period index `t`. A univariate law
  covers one row, a multivariate law as many rows as its `length`; the blocks follow one another
  down the rows.

Given the state, the blocks are independent: the emission is the product of their laws. It is
the evaluated twin of [`SmoothPeriodicHMM`](@ref), and a `HiddenMarkovModels.AbstractHMM` whose
controls are [`SeasonalControls(n2t)`](@ref SeasonalControls).

```julia
hmm = PeriodicHMMDists(a, A, ([Normal(k, 1) for k in 1:2, t in 1:T],))
z, Y = rand(hmm, n2t; seq = true)    # Y is 1 × N
viterbi(hmm, Y; n2t)
loglikelihood(hmm, Y; n2t)
```
"""
struct PeriodicHMMDists{T,BB<:Tuple,R<:Tuple} <: HMMs.AbstractHMM
    a::Vector{T}
    A::Array{T,3}
    B::BB          # one `K × T` matrix per block, each with a concrete element type
    blocks::R      # the rows of each block
    function PeriodicHMMDists(a::AbstractVector{T}, A::AbstractArray{T,3}, B::Tuple) where {T}
        B = map(Bb -> convert(Matrix, Bb), B)
        blocks = _check_dists_hmm(a, A, B)
        return new{T,typeof(B),typeof(blocks)}(a, A, B, blocks)
    end
end

PeriodicHMMDists(a::AbstractVector, A::AbstractArray{<:Real,3}, B::AbstractMatrix{<:Distribution}) =
    PeriodicHMMDists(a, A, (B,))
PeriodicHMMDists(a::AbstractVector, A::AbstractArray{<:Real,3}, B::AbstractVector{<:AbstractMatrix}) =
    PeriodicHMMDists(a, A, Tuple(B))
PeriodicHMMDists(A::AbstractArray{T,3}, B) where {T} = PeriodicHMMDists(fill(one(T) / size(A, 1), size(A, 1)), A, B)

# Check the shapes and return the rows of each block.
function _check_dists_hmm(a, A, B::Tuple)
    @argcheck !isempty(B) "no emission block"
    @argcheck isprobvec(a) "the initial distribution must sum to one"
    @argcheck all(t -> istransmat(view(A, :, :, t)), OneTo(size(A, 3))) "every A[:, :, t] must be a transition matrix"
    K, T = size(A, 1), size(A, 3)
    @argcheck length(a) == K "$(length(a)) initial probabilities for $K states"
    lengths = map(B) do Bb
        @argcheck size(Bb) == (K, T) "each block holds a K × T = $((K, T)) matrix of laws, not $(size(Bb))"
        d = length(first(Bb))
        @argcheck all(dist -> length(dist) == d, Bb) "the laws of one block must all cover the same number of rows"
        d
    end
    stops = cumsum(collect(lengths))
    return ntuple(b -> (stops[b]-lengths[b]+1):stops[b], length(B))
end

"""
    size(hmm::PeriodicHMMDists, [dim]) -> Tuple | Int

`(K, D, T)`: the number of states, of observation rows and the period.
"""
size(hmm::PeriodicHMMDists, dim=:) = (size(hmm.A, 1), last(last(hmm.blocks)), size(hmm.A, 3))[dim]

copy(hmm::PeriodicHMMDists) = PeriodicHMMDists(copy(hmm.a), copy(hmm.A), map(copy, hmm.B))

_period(hmm::PeriodicHMMDists) = size(hmm.A, 3)

function Base.show(io::IO, ::MIME"text/plain", hmm::PeriodicHMMDists)
    K, D, T = size(hmm)
    println(io, "PeriodicHMMDists: ", K, " states, ", _rows(D), ", period ", T)
    for (b, (Bb, r)) in enumerate(zip(hmm.B, hmm.blocks))
        print(io, "  ", rpad(_rows(r), 11), "  ", Base.typename(eltype(Bb)).wrapper)
        b < length(hmm.B) && println(io)
    end
    return
end

# ## Reading one block of an observation
#
# A univariate law reads one entry, a multivariate law a view of its rows. The element type of a
# block's matrix decides which, so the choice is made at compile time.
_block_obs(::AbstractMatrix{<:UnivariateDistribution}, x::AbstractVector, r) = x[first(r)]
_block_obs(::AbstractMatrix, x::AbstractVector, r) = view(x, r)

# Loops over the blocks: a tuple recursion, so each block's call is static even when the laws
# differ from block to block.
@inline _foreach_block(f, ::Tuple{}, ::Tuple{}) = nothing
@inline function _foreach_block(f, B::Tuple, R::Tuple)
    f(first(B), first(R))
    return _foreach_block(f, Base.tail(B), Base.tail(R))
end

# ## The product over blocks, as one law
#
# Univariate blocks only: `Distributions`' own `product_distribution`. One multivariate block
# covering every row: that law itself. Anything else needs a product that mixes univariate and
# multivariate laws, which `Distributions` does not provide.

"""
    BlockProduct(dists, blocks)

The product of independent laws over consecutive blocks of a vector: `dists[b]` is the law of
`x[blocks[b]]`, univariate laws covering one entry. The emission of a [`PeriodicHMMDists`](@ref)
that mixes univariate and multivariate laws.
"""
struct BlockProduct{D<:Tuple,R<:Tuple,S<:ValueSupport,T} <: Distribution{Multivariate,S}
    dists::D
    blocks::R
end

function BlockProduct(dists::Tuple, blocks::Tuple)
    S = mapreduce(d -> Distributions.value_support(typeof(d)), promote_type, dists)
    T = mapreduce(eltype, promote_type, dists)
    return BlockProduct{typeof(dists),typeof(blocks),S,T}(dists, blocks)
end

Base.length(d::BlockProduct) = last(last(d.blocks))
Base.eltype(::Type{<:BlockProduct{D,R,S,T}}) where {D,R,S,T} = T

_part(::UnivariateDistribution, x, r) = x[first(r)]
_part(::Distribution, x, r) = view(x, r)

Distributions._logpdf(d::BlockProduct, x::AbstractVector{<:Real}) =
    sum(map((dist, r) -> logpdf(dist, _part(dist, x, r)), d.dists, d.blocks))

function Distributions._rand!(rng::AbstractRNG, d::BlockProduct, x::AbstractVector{<:Real})
    _foreach_block(d.dists, d.blocks) do dist, r
        _rand_part!(rng, dist, x, r)
    end
    return x
end
_rand_part!(rng, dist::UnivariateDistribution, x, r) = (x[first(r)] = rand(rng, dist))
_rand_part!(rng, dist::Distribution, x, r) = rand!(rng, dist, view(x, r))

_product(dists::Tuple{Vararg{UnivariateDistribution}}, _) = product_distribution(dists...)
_product(dists::Tuple{MultivariateDistribution}, _) = only(dists)
_product(dists::Tuple, blocks) = BlockProduct(dists, blocks)

"""
    emission(hmm::PeriodicHMMDists, k, t) -> Distribution

The law of a whole observation in state `k` at period index `t`: the product of the blocks' laws.
"""
emission(hmm::PeriodicHMMDists, k::Integer, t::Integer) = _product(map(Bb -> Bb[k, t], hmm.B), hmm.blocks)

# ## Simulation

# The element type of the observations: `Bool` for Bernoulli laws only, `Float64` once a
# continuous law is in.
_obs_eltype(hmm::PeriodicHMMDists) = mapreduce(Bb -> eltype(eltype(Bb)), promote_type, hmm.B)

"""
    rand([rng, ]hmm::PeriodicHMMDists, n2t; z_ini, seq = false)
    rand([rng, ]hmm::PeriodicHMMDists, N::Integer; kwargs...)
    rand([rng, ]hmm::PeriodicHMMDists, z, n2t) -> Matrix

Draw a state path and the `D × N` observations along it: `seq = true` returns `(z, Y)`,
otherwise just `Y`. `n2t[n]` is the period index of step `n`; a length `N` gives
`n2t = n_to_t(N, T)`. Given a state path `z`, only the observations are drawn. `z_ini` is the
first state, drawn from `hmm.a` by default. The step from `n-1` to `n` uses `A[:, :, n2t[n-1]]`.
"""
function rand(rng::AbstractRNG, hmm::PeriodicHMMDists, z::AbstractVector{<:Integer},
    n2t::AbstractVector{<:Integer})
    @argcheck length(z) == length(n2t) "$(length(z)) states and $(length(n2t)) period indices"
    Y = Matrix{_obs_eltype(hmm)}(undef, size(hmm, 2), length(z))
    _foreach_block(hmm.B, hmm.blocks) do Bb, r
        _rand_block!(Y, rng, Bb, r, z, n2t)
    end
    return Y
end

function _rand_block!(Y, rng, Bb::AbstractMatrix{<:UnivariateDistribution}, r, z, n2t)
    j = first(r)
    for n in eachindex(z)
        Y[j, n] = rand(rng, Bb[z[n], n2t[n]])
    end
    return Y
end

function _rand_block!(Y, rng, Bb::AbstractMatrix, r, z, n2t)
    for n in eachindex(z)
        rand!(rng, Bb[z[n], n2t[n]], view(Y, r, n))
    end
    return Y
end

function rand(rng::AbstractRNG, hmm::PeriodicHMMDists, n2t::AbstractVector{<:Integer};
    z_ini=rand(rng, Categorical(hmm.a))::Integer, seq=false)
    N = length(n2t)
    z = zeros(Int, N)
    (N >= 1) && (z[1] = z_ini)
    for n = 2:N
        z[n] = rand(rng, Categorical(hmm.A[z[n-1], :, n2t[n-1]]))
    end
    Y = rand(rng, hmm, z, n2t)
    return seq ? (z, Y) : Y
end

rand(rng::AbstractRNG, hmm::PeriodicHMMDists, N::Integer; kwargs...) =
    rand(rng, hmm, n_to_t(N, _period(hmm)); kwargs...)
rand(hmm::PeriodicHMMDists, n2t::AbstractVector{<:Integer}; kwargs...) = rand(GLOBAL_RNG, hmm, n2t; kwargs...)
rand(hmm::PeriodicHMMDists, N::Integer; kwargs...) = rand(GLOBAL_RNG, hmm, N; kwargs...)
rand(hmm::PeriodicHMMDists, z::AbstractVector{<:Integer}, n2t::AbstractVector{<:Integer}) =
    rand(GLOBAL_RNG, hmm, z, n2t)

# ## Likelihoods and decoding

# Observations come as a `D × N` matrix, or as a vector when there is one row.
_as_rows(y::AbstractVector{<:Real}) = reshape(y, 1, :)
_as_rows(Y::AbstractMatrix{<:Real}) = Y
_ncols(y::AbstractVector{<:Real}) = length(y)
_ncols(Y::AbstractMatrix) = size(Y, 2)

function _check_rows(hmm::PeriodicHMMDists, Y::AbstractMatrix)
    @argcheck size(Y, 1) == size(hmm, 2) "the observations have $(size(Y, 1)) rows and the emission blocks cover $(size(hmm, 2))"
    return Y
end

"""
    loglikelihoods(hmm::PeriodicHMMDists, Y; n2t) -> Matrix

The `N × K` emission loglikelihoods: entry `[n, k]` is the logdensity of observation `n` in state
`k` at period index `n2t[n]`. `Y` is `D × N`, or a vector when `D = 1`. `robust = true` gives an
observation that is impossible in every state the same finite value in all of them.
"""
function loglikelihoods(hmm::PeriodicHMMDists, Y::AbstractVecOrMat{<:Real};
    n2t=n_to_t(_ncols(Y), _period(hmm))::AbstractVector{<:Integer}, robust=false)
    Y = _check_rows(hmm, _as_rows(Y))
    @argcheck size(Y, 2) == length(n2t) "`Y` has $(size(Y, 2)) observations and `n2t` $(length(n2t)) entries"
    LL = zeros(size(Y, 2), size(hmm, 1))
    _foreach_block(hmm.B, hmm.blocks) do Bb, r
        _add_block_loglikelihoods!(LL, Bb, r, Y, n2t)
    end
    robust && _clamp_inf!(LL)
    return LL
end

function _add_block_loglikelihoods!(LL, Bb, r, Y, n2t)
    for n in axes(Y, 2)
        x = _block_obs(Bb, view(Y, :, n), r)
        for k in axes(LL, 2)
            LL[n, k] += logpdf(Bb[k, n2t[n]], x)
        end
    end
    return LL
end

"""
    viterbi(hmm::PeriodicHMMDists, Y; n2t, robust = false) -> Vector{Int}
    viterbi(hmm::PeriodicHMMDists, Ys; n2t, robust = false) -> Vector{Vector{Int}}

The most likely hidden state path of `Y` (`D × N`, or a vector when `D = 1`), or of each of
several independent sequences `Ys[i]` with period indices `n2t[i]`.
"""
viterbi(hmm::PeriodicHMMDists, Y::AbstractVecOrMat{<:Real};
    n2t=n_to_t(_ncols(Y), _period(hmm))::AbstractVector{<:Integer}, robust=false) =
    viterbi(hmm.a, hmm.A, loglikelihoods(hmm, Y; n2t, robust); n2t)

function viterbi(hmm::PeriodicHMMDists, Ys::AbstractVector{<:AbstractVecOrMat{<:Real}};
    n2t=[n_to_t(_ncols(Y), _period(hmm)) for Y in Ys], robust=false)
    @argcheck length(Ys) == length(n2t) "$(length(Ys)) sequences and $(length(n2t)) period-index vectors"
    return [viterbi(hmm, Y; n2t=n2tᵢ, robust) for (Y, n2tᵢ) in zip(Ys, n2t)]
end
