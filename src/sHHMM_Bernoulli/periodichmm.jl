"""
    sARPeriodicHMM([a, ]A, B) -> sARPeriodicHMM

Build an Auto Regressive Periodic Hidden Markov Chain `sARPeriodicHMM` with transition matrix `A(t)` and observation distributions `B(t)`.  
If the initial state distribution `a` is not specified, a uniform distribution is assumed. 

Observations distributions can be of different types (for example `Normal` and `Exponential`),  
but they must be of the same dimension.

Alternatively, `B(t)` can be an emission matrix where `B[i,j,t]` is the probability of observing symbol `j` in state `i`.

**Arguments**
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractArray{T,3}`: transition matrix.
- `B::AbstractMatrix{<:Distribution{F}}`: Y distributions.
- or `B::AbstractMatrix`: emission matrix.
"""
# struct sARPeriodicHMM <: AbstractPeriodicHMM
#     a::Vector
#     A::Array
#     B::Array
#     # sARPeriodicHMM{F,T}(a, A, B) where {F,T} = assert_hmm(a, A, B) && new(a, A, B)
# end

struct sARPeriodicHMM{F,T} <: AbstractPeriodicHMM{F}
    a::Vector{T}
    A::Array{T,3}
    B::Array{<:AbstractVector{<:Distribution{F}},3}
    sARPeriodicHMM{F,T}(a, A, B) where {F,T} = assert_hmm(a, A, B) && new(a, A, B)
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
    y_ini=[rand(Bernoulli(), Int(log2(size(hmm, 4)[j]))) for j in 1:size(hmm, 2)]
)
    D = size(hmm, 2)
    y = Matrix{Bool}(undef, length(z), D)
    orders = Int.(log2.(size(hmm, 4)))

    # @argcheck length(n2t) == length(z)
    # Check the initial conditions
    @argcheck length(y_ini) == D "Initial condition size is not correct: You give $(length(y_ini)) instead of $(D)"
    @argcheck length.(y_ini) == orders "Initial condition order is not correct: You give $(length.(y_ini)) instead of $(orders)"
    previous_day_category = zeros(Int, D)
    for j in 1:D
        order = orders[j]
        if order > 0
            # One could do some specialized for each value of order e.g. for order = 1, we have simply previous_day_category = y[n-1,:].+1
            y[1:order, j] .= y_ini[j]
            for n in eachindex(z)[order+1:end]
                t = n2t[n] # periodic t
                @views previous_day_category = bin2digit([y[n-m, j] for m = 1:order])
                y[n, j] = rand(rng, hmm.B[z[n], t, j][previous_day_category])
            end
        else
            for n in eachindex(z)
                t = n2t[n] # periodic t
                y[n, j] = rand(rng, hmm.B[z[n], t, j][1])
            end
        end
    end
    return y
end

"""
    size(hmm, [dim]) -> Int | Tuple
Return the number of states in `hmm`, the dimension of the Y and the length of the chain.
"""
size(hmm::sARPeriodicHMM, dim=:) = (size(hmm.B, 1), size(hmm.B, 3), size(hmm.B, 2), length.(hmm.B[1, 1, :]))[dim]
# K                # D             # T          # number of states
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