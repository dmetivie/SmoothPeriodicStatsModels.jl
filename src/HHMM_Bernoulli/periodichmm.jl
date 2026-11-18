# `ARPeriodicHMM` subtypes `HMMs.AbstractHMM` rather than
# `PeriodicHiddenMarkovModels.AbstractPeriodicHMM`: Julia has single inheritance, and subtyping
# `HiddenMarkovModels`' abstraction is what makes its inference and Baum-Welch apply here (see
# `src/hmm_interface.jl`). The old supertype bought nothing that is lost: the only inference
# this package still writes is `viterbi`, in its loose-parameter form on `a`, `A` and `LL`
# (`src/viterbi.jl`), never on the model itself.

"""
    ARPeriodicHMM([a, ]A, B) -> ARPeriodicHMM

A periodic hidden Markov model with `K` states, `D` sites, period `T` and autoregressive
emissions of memory `order`.

- `a`: the initial state distribution, length `K`; uniform if omitted.
- `A`: the `K × K × T` transition matrices; `A[:, :, t]` is the step leaving period index `t`.
- `B`: the `K × T × D × 2^order` emission distributions: `B[k, t, j, h]` is the law of site `j`
  in state `k` at period index `t`, given the memory category `h` encoding the `order` previous
  values of that site.

`ARPeriodicHMM(a, θᴬ, θᴮ, T; basis_A, basis_B)` builds one from coefficients.
"""
struct ARPeriodicHMM{F,T,BB<:Array{<:Distribution{F},4}} <: HMMs.AbstractHMM
    a::Vector{T}
    A::Array{T,3}
    # `B`'s concrete type is a parameter. Declared `Array{<:Distribution{F},4}`, the field is
    # abstract, and `HiddenMarkovModels` reads it once per observation, so every read was a
    # dynamic dispatch: 1.1 KB allocated per observation and 3x the forward-pass time.
    B::BB
    function ARPeriodicHMM{F,T}(a, A, B) where {F,T}
        B = convert(Array, B)
        return assert_hmm(a, A, B) && new{F,T,typeof(B)}(a, A, B)
    end
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
    Y_past=zeros(Bool, size(hmm, 2), Int(log2(size(hmm, 4))))
)
    D, order = size(hmm, 2), Int(log2(size(hmm, 4)))
    @argcheck size(Y_past) == (D, order) "`Y_past` is $(size(Y_past)) instead of $((D, order)): $D sites, memory $order"
    y = Matrix{Bool}(undef, D, length(z))
    rand_emissions!(y, rng, hmm.B, z, n2t, Y_past)
    return y
end

rand(hmm::ARPeriodicHMM, z::AbstractVector{<:Integer}, n2t::AbstractVector{<:Integer}; kwargs...) = rand(GLOBAL_RNG, hmm, z, n2t; kwargs...)

"""
    rand([rng, ]hmm::ARPeriodicHMM, n2t; z_ini, Y_past, seq = false)
    rand([rng, ]hmm::ARPeriodicHMM, N::Integer; kwargs...)
    rand([rng, ]hmm::ARPeriodicHMM, z, n2t; Y_past) -> Matrix{Bool}

Draw a state path and the `D × N` observations along it: `seq = true` returns `(z, y)`,
otherwise just `y`. `n2t[n]` is the period index of step `n`; a length `N` gives
`n2t = n_to_t(N, T)`. Given a state path `z`, only the observations are drawn.

`z_ini` is the state of the first observation, drawn from `hmm.a` by default. `Y_past` is the
`D × order` initial condition, all zeros by default: the `order` observations before `y`, in
time order, so that `[Y_past y]` is the simulated series. It is the `Y_past` that `fit_mle`,
`loglikelihood` and `viterbi` take with `y`. The step from `n-1` to `n` uses `A[:, :, n2t[n-1]]`.
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
    rand_emissions!(y, rng, B, z, n2t, Y_past)

Fill `y` with emissions drawn along the state path `z`. Site `j` at time `n` is drawn from `B[z[n], n2t[n], j, h]`, where `h` encodes the `order` previous values of site `j` in
`[Y_past y]` the way `bin2digit` does.
"""
function rand_emissions!(y::AbstractMatrix{Bool}, rng::AbstractRNG, B::AbstractArray{F,4} where {F<:UnivariateDistribution}, z::AbstractVector{<:Integer}, n2t::AbstractVector{<:Integer}, Y_past::AbstractMatrix)
    D, order = size(y, 1), size(Y_past, 2)
    for n in eachindex(z)
        t = n2t[n] # periodic t
        for j in OneTo(D)
            h = 1 # memory category of site j
            for m = 1:order
                h += _previous(y, Y_past, j, n - m) * 2^(order - m)
            end
            y[j, n] = rand(rng, B[z[n], t, j, h])
        end
    end
    return y
end

# The value of site `j` at step `n` of the series `[Y_past y]`; steps `n ≤ 0` are in `Y_past`,
# whose last column is step 0. For site-dependent memory, `Y_past` is that site's vector.
_previous(y::AbstractMatrix, Y_past::AbstractMatrix, j, n) = n ≥ 1 ? y[j, n] : Y_past[j, end+n]
_previous(y::AbstractMatrix, y_past::AbstractVector, j, n) = n ≥ 1 ? y[j, n] : y_past[end+n]

"""
    size(hmm::ARPeriodicHMM, [dim]) -> Tuple | Int

`(K, D, T, 2^order)`: the number of states, of sites, the period and the number of memory
categories.
"""
size(hmm::ARPeriodicHMM, dim=:) = (size(hmm.B, 1), size(hmm.B, 3), size(hmm.B, 2), size(hmm.B, 4))[dim]

copy(hmm::ARPeriodicHMM) = ARPeriodicHMM(copy(hmm.a), copy(hmm.A), copy(hmm.B))

"""
    sort_wrt_ref!(hmm::ARPeriodicHMM, ref_station)
    sort_wrt_ref!(hmm::sARPeriodicHMM, ref_station)
    sort_wrt_ref!(α, B, ref_station)

Relabel the hidden states, at each period index, by decreasing success probability of site
`ref_station` in memory category 1, so that state 1 is the one where that site is most likely a
success. The third form sorts the weights `α` and the `K × D × 2^order` Bernoulli array `B` of a
mixture the same way.
"""
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