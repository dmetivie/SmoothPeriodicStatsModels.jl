# struct sARPeriodicHMM <: AbstractPeriodicHMM
#     a::Vector
#     A::Array
#     B::Array
#     # sARPeriodicHMM{F,T}(a, A, B) where {F,T} = assert_hmm(a, A, B) && new(a, A, B)
# end

"""
    sARPeriodicHMM([a, ]A, B) -> sARPeriodicHMM

A periodic hidden Markov model with autoregressive emissions whose memory order differs from site
to site.

- `a`: the initial state distribution, length `K`; uniform if omitted.
- `A`: the `K × K × T` transition matrices; `A[:, :, t]` is the step leaving period index `t`.
- `B`: a `K × T × D` array of vectors: `B[k, t, j][h]` is the law of site `j` in state `k` at
  period index `t` given its memory category `h`, so `length(B[k, t, j]) == 2^order_j`.
"""
struct sARPeriodicHMM{F,T,BB<:Array{<:AbstractVector{<:Distribution{F}},3}} <: HMMs.AbstractHMM
    a::Vector{T}
    A::Array{T,3}
    # a parameter for the same reason as `ARPeriodicHMM.B`: an abstractly typed field read
    # once per observation is a dynamic dispatch per observation
    B::BB
    function sARPeriodicHMM{F,T}(a, A, B) where {F,T}
        B = convert(Array, B)
        return assert_hmm(a, A, B) && new{F,T,typeof(B)}(a, A, B)
    end
end

sARPeriodicHMM(a::AbstractVector{T}, A::AbstractArray{T,3}, B::AbstractArray{<:AbstractVector{<:Distribution{F}},3}) where {F,T} =
    sARPeriodicHMM{F,T}(a, A, B)

sARPeriodicHMM(A, B) =
    sARPeriodicHMM(ones(size(A, 1)) ./ size(A, 1), A, B)

function assert_hmm(hmm::sARPeriodicHMM)
    assert_hmm(hmm.a, hmm.A, hmm.B)
end

"""
    assert_hmm(a, A, B)

Throw an `ArgumentError` if the initial state distribution and the transition matrix rows does not sum to 1,
and if the observation distributions do not have the same dimensions.
"""
function assert_hmm(a::AbstractVector, A::AbstractArray{T,3} where {T}, B::AbstractArray)
    @argcheck isprobvec(a)
    @argcheck all(t -> istransmat(A[:, :, t]), OneTo(size(A, 3))) ArgumentError("All transition matrice A(t) for all t must be transition matrix")
    @argcheck length(a) == size(A, 1) == size(B, 1) ArgumentError("Number of transition rates must match length of chain")
    @argcheck size(A, 3) == size(B, 2) ArgumentError("Period length must be the same for transition matrix and distribution")
    return true
end

function rand(
    rng::AbstractRNG,
    hmm::sARPeriodicHMM,
    z::AbstractVector{<:Integer},
    n2t::AbstractVector{<:Integer};
    Y_past=[zeros(Bool, Int(log2(s))) for s in size(hmm, 4)]
)
    D = size(hmm, 2)
    y = Matrix{Bool}(undef, D, length(z))
    orders = Int.(log2.(size(hmm, 4)))
    _check_Y_past(hmm, Y_past)
    for j in 1:D
        order = orders[j]
        for n in eachindex(z)
            h = 1 # memory category of site j
            for m = 1:order
                h += _previous(y, Y_past[j], j, n - m) * 2^(order - m)
            end
            y[j, n] = rand(rng, hmm.B[z[n], n2t[n], j][h])
        end
    end
    return y
end

rand(hmm::sARPeriodicHMM, z::AbstractVector{<:Integer}, n2t::AbstractVector{<:Integer}; kwargs...) = rand(GLOBAL_RNG, hmm, z, n2t; kwargs...)

"""
    rand([rng, ]hmm::sARPeriodicHMM, n2t; z_ini, Y_past, seq = false)
    rand([rng, ]hmm::sARPeriodicHMM, N::Integer; kwargs...)
    rand([rng, ]hmm::sARPeriodicHMM, z, n2t; Y_past) -> Matrix{Bool}

Draw a state path and the `D × N` observations along it; `seq = true` returns `(z, y)`.
`Y_past[j]` holds the `order_j` observations of site `j` before `y`, in time order, zeros by
default. Otherwise as for [`ARPeriodicHMM`](@ref).
"""
function rand(rng::AbstractRNG, hmm::sARPeriodicHMM, n2t::AbstractVector{<:Integer};
    z_ini=rand(rng, Categorical(hmm.a))::Integer, seq=false, kwargs...)
    N = length(n2t)
    z = zeros(Int, N)
    (N >= 1) && (z[1] = z_ini)
    for n = 2:N
        z[n] = rand(rng, Categorical(hmm.A[z[n-1], :, n2t[n-1]]))
    end
    y = rand(rng, hmm, z, n2t; kwargs...)
    return seq ? (z, y) : y
end

rand(hmm::sARPeriodicHMM, n2t::AbstractVector{<:Integer}; kwargs...) =
    rand(GLOBAL_RNG, hmm, n2t; kwargs...)
rand(rng::AbstractRNG, hmm::sARPeriodicHMM, N::Integer; kwargs...) =
    rand(rng, hmm, n_to_t(N, size(hmm, 3)); kwargs...)
rand(hmm::sARPeriodicHMM, N::Integer; kwargs...) = rand(GLOBAL_RNG, hmm, N; kwargs...)

"""
    size(hmm::sARPeriodicHMM, [dim]) -> Tuple | Int

`(K, D, T, sizes)`: the number of states, of sites, the period, and the number of memory
categories of each site.
"""
size(hmm::sARPeriodicHMM, dim=:) = (size(hmm.B, 1), size(hmm.B, 3), size(hmm.B, 2), length.(hmm.B[1, 1, :]))[dim]

copy(hmm::sARPeriodicHMM) = sARPeriodicHMM(copy(hmm.a), copy(hmm.A), deepcopy(hmm.B))

function sort_wrt_ref!(hmm::sARPeriodicHMM, ref_station)
    K, T = size(hmm.B, 1), size(hmm.B, 2)
    sorting = [[succprob(hmm.B[k, t, ref_station][1]) for k = 1:K] for t = 1:T] # 1 is by convention the driest category i.e. Y|d....d
    new_order = sortperm.(sorting, rev=true)
    for t = 1:T
        hmm.B[:, t, :] = hmm.B[new_order[t], t, :]
        hmm.A[:, :, t] = hmm.A[new_order[t], new_order[t], t]
    end
end