# `ARPeriodicHMM` subtypes `HMMs.AbstractHMM` rather than
# `PeriodicHiddenMarkovModels.AbstractPeriodicHMM`: Julia has single inheritance, and subtyping
# `HiddenMarkovModels`' abstraction is what makes its inference and Baum-Welch apply here (see
# `src/hmm_interface.jl`). The old supertype bought nothing that is lost: the `forwardlog!`,
# `backwardlog!` and `viterbi` of that package are called in their loose-parameter form, on
# `a`, `A` and `LL`, never on the model itself.

"""
    ARPeriodicHMM([a, ]A, B) -> ARPeriodicHMM

Build an Auto Regressive Periodic Hidden Markov Chain `ARPeriodicHMM` with transition matrix `A(t)` and observation distributions `B(t)`.  
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
struct ARPeriodicHMM{F,T} <: HMMs.AbstractHMM
    a::Vector{T}
    A::Array{T,3}
    B::Array{<:Distribution{F},4}
    ARPeriodicHMM{F,T}(a, A, B) where {F,T} = assert_hmm(a, A, B) && new(a, A, B)
end

ARPeriodicHMM(a::AbstractVector{T}, A::AbstractArray{T,3}, B::AbstractArray{<:Distribution{F},4}) where {F,T} =
    ARPeriodicHMM{F,T}(a, A, B)

ARPeriodicHMM(A::AbstractArray{T,3}, B::AbstractArray{<:Distribution{F},4}) where {F,T} =
    ARPeriodicHMM{F,T}(ones(size(A, 1)) ./ size(A, 1), A, B)

function assert_hmm(hmm::ARPeriodicHMM)
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
# ERROR: MethodError: no method matching rand(::ARPeriodicHMM{Univariate, Float64}, ::Vector{Int64}; z_init=1)
# Closest candidates are:
#   rand(::HMMBase.AbstractHMM, ::AbstractVector{<:Integer}) at C:\Users\metivier\.julia\packages\HMMBase\ZYC4P\src\hmm.jl:183 got unsupported keyword argument "z_init"
#   rand(::AR1{<:AbstractMatrix}, ::AbstractVector{<:Integer}, ::AbstractVector{<:Integer}; y₁) at C:\Users\metivier\.julia\dev\PeriodicAutoRegressive\src\AR1.jl:41 got unsupported keyword argument "z_init"
#   rand(::AR1{<:AbstractVector}, ::AbstractVector{<:Integer}; y₁) at C:\Users\metivier\.julia\dev\PeriodicAutoRegressive\src\AR1.jl:31 got unsupported keyword argument "z_init"

function rand(
    rng::AbstractRNG,
    hmm::ARPeriodicHMM,
    z::AbstractVector{<:Integer},
    n2t::AbstractVector{<:Integer};
    y_ini=rand(rng, Bernoulli(), size(hmm, 2), Int(log2(size(hmm, 4))))
)
    D = size(hmm, 2)
    y = Matrix{Bool}(undef, D, length(z))
    order = Int(log2(size(hmm, 4)))

    # @argcheck length(n2t) == length(z)
    # Check the initial conditions
    @argcheck size(y_ini) == (D, order) "Initial condition is not correct: You give $(size(y_ini)) instead of $((D, order))"

    y[:, 1:order] = y_ini   # empty slice when order == 0
    rand_emissions!(y, rng, hmm.B, z, n2t, order)
    return y
end

rand(hmm::ARPeriodicHMM, z::AbstractVector{<:Integer}, n2t::AbstractVector{<:Integer}; kwargs...) = rand(GLOBAL_RNG, hmm, z, n2t; kwargs...)

"""
    rand([rng, ]hmm::ARPeriodicHMM, n2t; z_ini, y_ini, seq = false)
    rand([rng, ]hmm::ARPeriodicHMM, N::Integer; kwargs...)

Draw a state path and the observations along it: `seq = true` returns `(z, y)`, otherwise just
`y`. `n2t[n]` is the period index of step `n`, or give a length `N` and get `n_to_t(N, T)`.

The state path uses `A[z[n-1], :, n2t[n-1]]` — the transition matrix of the period the step
*leaves*, this package's convention — and each observation is drawn from the emission selected
by the memory its predecessors encode, which is why simulation cannot go through
`HiddenMarkovModels`' own `rand`: that one fixes the whole control sequence first, and here the
controls depend on the draws. See [`SeasonalControls`](@ref).
"""
function rand(rng::AbstractRNG, hmm::ARPeriodicHMM, n2t::AbstractVector{<:Integer};
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

rand(hmm::ARPeriodicHMM, n2t::AbstractVector{<:Integer}; kwargs...) =
    rand(GLOBAL_RNG, hmm, n2t; kwargs...)
rand(rng::AbstractRNG, hmm::ARPeriodicHMM, N::Integer; kwargs...) =
    rand(rng, hmm, n_to_t(N, size(hmm, 3)); kwargs...)
rand(hmm::ARPeriodicHMM, N::Integer; kwargs...) = rand(GLOBAL_RNG, hmm, N; kwargs...)

"""
    rand_emissions!(y, rng, B, z, n2t, order)

Fill `y[:, order+1:end]` from the emission distributions `B` along the state path `z`: site `j`
at time `n` is drawn from `B[z[n], n2t[n], j, h]`, where `h` encodes the `order` previous values
of site `j` the way `bin2digit` does. The first `order` columns of `y` are the initial condition.
"""
function rand_emissions!(y::AbstractMatrix{Bool}, rng::AbstractRNG, B::AbstractArray{F,4} where {F<:UnivariateDistribution}, z::AbstractVector{<:Integer}, n2t::AbstractVector{<:Integer}, order::Integer)
    D = size(y, 1)
    for n in eachindex(z)[order+1:end]
        t = n2t[n] # periodic t
        for j in OneTo(D)
            h = 1 # memory category of site j
            for m = 1:order
                h += y[j, n-m] * 2^(order - m)
            end
            y[j, n] = rand(rng, B[z[n], t, j, h])
        end
    end
end

"""
    size(hmm, [dim]) -> Int | Tuple
Return the number of states in `hmm`, the dimension of the Y and the length of the chain.
"""
size(hmm::ARPeriodicHMM, dim=:) = (size(hmm.B, 1), size(hmm.B, 3), size(hmm.B, 2), size(hmm.B, 4))[dim]
                                            # K                # D             # T          # number of states
copy(hmm::ARPeriodicHMM) = ARPeriodicHMM(copy(hmm.a), copy(hmm.A), copy(hmm.B))

function sort_wrt_ref!(hmm::ARPeriodicHMM, ref_station)
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