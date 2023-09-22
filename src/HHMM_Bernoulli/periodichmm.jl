"""
    HierarchicalPeriodicHMM([a, ]A, B) -> HierarchicalPeriodicHMM

Build an HierarchicalPeriodicHMM with transition matrix `A(t)` and observation distributions `B(t)`.  
If the initial state distribution `a` is not specified, a uniform distribution is assumed. 

Observations distributions can be of different types (for example `Normal` and `Exponential`),  
but they must be of the same dimension.

Alternatively, `B(t)` can be an emission matrix where `B[i,j,t]` is the probability of observing symbol `j` in state `i`.

**Arguments**
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractArray{T,3}`: transition matrix.
- `B::AbstractMatrix{<:Distribution{F}}`: 𝐘 distributions.
- or `B::AbstractMatrix`: emission matrix.

**Example**
```julia
using Distributions, HierarchicalPeriodicHMM
# from distributions
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
# from an emission matrix
hmm = HMM([0.9 0.1; 0.1 0.9], [0. 0.5 0.5; 0.25 0.25 0.5])
```
"""
struct HierarchicalPeriodicHMM{F,T} <: AbstractPeriodicHMM{F}
    a::Vector{T}
    A::Array{T,3}
    B::Array{<:Distribution{F},4}
    HierarchicalPeriodicHMM{F,T}(a, A, B) where {F,T} = assert_hmm(a, A, B) && new(a, A, B)
end

HierarchicalPeriodicHMM(a::AbstractVector{T}, A::AbstractArray{T,3}, B::AbstractArray{<:Distribution{F},4}) where {F,T} =
    HierarchicalPeriodicHMM{F,T}(a, A, B)

HierarchicalPeriodicHMM(A::AbstractArray{T,3}, B::AbstractArray{<:Distribution{F},4}) where {F,T} =
    HierarchicalPeriodicHMM{F,T}(ones(size(A, 1)) ./ size(A, 1), A, B)

function assert_hmm(hmm::HierarchicalPeriodicHMM)
    assert_hmm(hmm.a, hmm.A, hmm.B)
end

"""
    assert_hmm(a, A, B)

Throw an `ArgumentError` if the initial state distribution and the transition matrix rows does not sum to 1,
and if the observation distributions do not have the same dimensions.
"""
function assert_hmm(a::AbstractVector, A::AbstractArray{T,3} where {T}, B::AbstractArray{<:Distribution,4})
    @argcheck isprobvec(a)
    @argcheck all(t -> istransmat(A[:, :, t]), OneTo(size(A, 3))) ArgumentError("All transition matrice A(t) for all t must be transition matrix")
    @argcheck length(a) == size(A, 1) == size(B, 1) ArgumentError("Number of transition rates must match length of chain")
    @argcheck size(A, 3) == size(B, 2) ArgumentError("Period length must be the same for transition matrix and distribution")
    return true
end

#TODO? Function could/should be optimzed (for example depending on order value, or succprob is taken once or )
#TODO I wanted to re-use the rand() in PeriodicHiddenMarkovModels which works for AbstractPeriodicHMM, but I encountered unsupported keyword errors
# rand(hmm_fit, n2t) works 
# rand(hmm_fit, n2t; z_init = 1) does not work 
# ERROR: MethodError: no method matching rand(::HierarchicalPeriodicHMM{Univariate, Float64}, ::Vector{Int64}; z_init=1)
# Closest candidates are:
#   rand(::HMMBase.AbstractHMM, ::AbstractVector{<:Integer}) at C:\Users\metivier\.julia\packages\HMMBase\ZYC4P\src\hmm.jl:183 got unsupported keyword argument "z_init"
#   rand(::AR1{<:AbstractMatrix}, ::AbstractVector{<:Integer}, ::AbstractVector{<:Integer}; y₁) at C:\Users\metivier\.julia\dev\PeriodicAutoRegressive\src\AR1.jl:41 got unsupported keyword argument "z_init"
#   rand(::AR1{<:AbstractVector}, ::AbstractVector{<:Integer}; y₁) at C:\Users\metivier\.julia\dev\PeriodicAutoRegressive\src\AR1.jl:31 got unsupported keyword argument "z_init"

function rand(
    rng::AbstractRNG,
    hmm::HierarchicalPeriodicHMM,
    z::AbstractVector{<:Integer},
    n2t::AbstractVector{<:Integer};
    y_ini=rand(Bernoulli(), Int(log2(size(hmm, 4))), size(hmm, 2))
)
    D = size(hmm, 2)
    y = Matrix{Bool}(undef, length(z), D)
    order = Int(log2(size(hmm, 4)))

    # @argcheck length(n2t) == length(z)
    # Check the initial conditions
    @argcheck size(y_ini) == (order, D) "Initial condition is not correct: You give $(size(y_ini)) instead of $((order, D))" 

    p = zeros(D)
    if order > 0
        # One could do some specialized for each value of order e.g. for order = 1, we have simply previous_day_category = y[n-1,:].+1
        y[1:order, :] = y_ini
        previous_day_category = zeros(Int, D)
        for n in eachindex(z)[order+1:end]
            t = n2t[n] # periodic t
            previous_day_category[:] = bin2digit.(eachcol([y[n-m, j] for m = 1:order, j = 1:D]))
            p[:] = succprob.(hmm.B[CartesianIndex.(z[n], t, 1:D, previous_day_category)])
            y[n, :] = rand(rng, Product(Bernoulli.(p)))
        end
    else
        for n in eachindex(z)
            t = n2t[n] # periodic t
            p[:] = succprob.(hmm.B[CartesianIndex.(z[n], t, 1:D, 1)])
            y[n, :] = rand(rng, Product(Bernoulli.(p)))
        end
    end
    return y
end

"""
    size(hmm, [dim]) -> Int | Tuple
Return the number of states in `hmm`, the dimension of the 𝐘 and the length of the chain.
"""
size(hmm::HierarchicalPeriodicHMM, dim=:) = (size(hmm.B, 1), size(hmm.B, 3), size(hmm.B, 2), size(hmm.B, 4))[dim]
                                            # K                # D             # T          # number of states
copy(hmm::HierarchicalPeriodicHMM) = HierarchicalPeriodicHMM(copy(hmm.a), copy(hmm.A), copy(hmm.B))

function sort_wrt_ref!(hmm::HierarchicalPeriodicHMM, ref_station)
    K, T = size(hmm.B, 1), size(hmm.B, 2)
    sorting = [[succprob(hmm.B[k, t, ref_station, 1]) for k = 1:K] for t = 1:T] # 1 is by convention the driest category i.e. Y|d....d
    new_order = sortperm.(sorting, rev = true)
    for t = 1:T
        hmm.B[:, t, :, :] = hmm.B[new_order[t], t, :, :]
        hmm.A[:, :, t] = hmm.A[new_order[t], new_order[t], t]
    end
end

function sort_wrt_ref!(α::AbstractVector, B::AbstractArray{F,3}, ref_station) where {F<:Bernoulli}
    K = size(α, 1)
    sorting = [succprob(B[k, ref_station, 1]) for k = 1:K] # 1 is by my convention the driest category i.e. Y|d....d
    B[:, :, :] = B[sortperm(sorting, rev=true), :, :]
    α[:] = α[sortperm(sorting, rev=true)]
end