
#-----------------equivalent to periodichmm.jl  ---------------------------#

"""
    ARPeriodicHMMSpatial(a, A, R, B, h[, QMC_m = 30]) -> ARPeriodicHMMSpatial

A periodic hidden Markov model whose emission in each state is a [`SpatialBernoulli`](@ref): `D`
correlated binary sites, with autoregressive marginals.

- `a`: the initial state distribution, length `K`.
- `A`: the `K × K × T` transition matrices; `A[:, :, t]` is the step leaving period index `t`.
- `R`: the `K × T` spatial ranges.
- `B`: the `K × T × D × 2^order` marginal success probabilities, given each site's memory
  category.
- `h`: the `D × D` distance matrix between sites.
- `QMC_m`: the quasi-Monte-Carlo budget of the emission likelihood, `D * QMC_m` points of
  `MvNormalCDF.mvnormcdf`; higher is more accurate and slower.
"""
struct ARPeriodicHMMSpatial{T,AM} <: HMMs.AbstractHMM
    a::Vector{T}
    A::Array{T,3}
    R::Array{T,2}
    B::Array{T,4}
    h::AM
    QMC_m::Int
end

ARPeriodicHMMSpatial(a::AbstractVector, A::AbstractArray, R::AbstractArray, B::AbstractArray,
    h::AbstractMatrix) = ARPeriodicHMMSpatial(a, A, R, B, h, 30)

"""
    size(hmm::ARPeriodicHMMSpatial, [dim]) -> Tuple | Int

`(K, D, T, 2^order)`: the number of states, of sites, the period and the number of memory
categories.
"""
size(hmm::ARPeriodicHMMSpatial, dim=:) = (size(hmm.B, 1), size(hmm.B, 3), size(hmm.B, 2), size(hmm.B, 4))[dim]

# simulate with given z sequence.
function rand(rng::AbstractRNG,
    hmm::ARPeriodicHMMSpatial,
    z::AbstractVector{<:Integer},
    n2t::AbstractVector{<:Integer};
    Y_past=zeros(Bool, size(hmm, 2), Int(log2(size(hmm, 4))))
)
    N = length(n2t)
    D = size(hmm, 2)
    order = Int(log2(size(hmm, 4)))
    y = Matrix{Bool}(undef, D, N)

    _check_Y_past(hmm, Y_past)

    for n in 1:N
        t = n2t[n]
        # memory category of each site, encoded as `bin2digit` does: the most recent value
        # carries the highest bit
        lambdas = map(1:D) do d
            lc = 1
            for m in 1:order
                lc += _previous(y, Y_past, d, n - m) * 2^(order - m)
            end
            hmm.B[z[n], t, d, lc]
        end
        y[:, n] = rand(rng, SpatialBernoulli(hmm.R[z[n], t], lambdas, hmm.h))
    end
    return y
end

rand(hmm::ARPeriodicHMMSpatial, z::AbstractVector{<:Integer}, n2t::AbstractVector{<:Integer}; kwargs...) = rand(GLOBAL_RNG, hmm, z, n2t; kwargs...)

"""
    rand([rng, ]hmm::ARPeriodicHMMSpatial, n2t; z = nothing, past = nothing) -> (; z, Y)
    rand([rng, ]hmm::ARPeriodicHMMSpatial, n2t; z_ini, Y_past, seq = false)
    rand([rng, ]hmm::ARPeriodicHMMSpatial, N::Integer; kwargs...)
    rand([rng, ]hmm::ARPeriodicHMMSpatial, z, n2t; Y_past) -> Matrix{Bool}

Draw a state path and the `D × N` observations along it; `seq = true` returns `(z, y)`. Given
`z`, only the observations are drawn. Otherwise as for [`ARPeriodicHMM`](@ref).
"""
function rand(rng::AbstractRNG, hmm::ARPeriodicHMMSpatial, n2t::AbstractVector{<:Integer}; kwargs...)
    _keyword_rand(kwargs) && return _rand_keyword(rng, hmm, n2t; kwargs...)
    return _rand_sequence(rng, hmm, n2t; kwargs...)
end

rand(hmm::ARPeriodicHMMSpatial, n2t::AbstractVector{<:Integer}; kwargs...) =
    rand(GLOBAL_RNG, hmm, n2t; kwargs...)
rand(rng::AbstractRNG, hmm::ARPeriodicHMMSpatial, N::Integer; kwargs...) =
    rand(rng, hmm, n_to_t(N, size(hmm, 3)); kwargs...)
rand(hmm::ARPeriodicHMMSpatial, N::Integer; kwargs...) = rand(GLOBAL_RNG, hmm, N; kwargs...)

# ## peut être pas necessaire car déjà dans rand(rng, hmm, z, n2t, y_ini) pour les AbsractHMM
# function rand(rng::AbstractRNG,
#     hmm::ARPeriodicHMMSpatial,
#     n2t::AbstractVector{<:Integer};
#     z_ini=rand(Categorical(hmm.a))::Integer, y_ini=fill(0, size(hmm, 2)),
#     seq=false
# )
#     N = length(n2t)
#     z = zeros(Int, N)
#     (N >= 1) && (z[1] = z_ini)
#     for n = 2:N
#         tₙ₋₁ = n2t[n-1] # periodic t-1
#         z[n] = rand(rng, Categorical(hmm.A[z[n-1], :, tₙ₋₁]))
#     end
#     y = my_rand(hmm, z, n2t, y_ini)
#     return seq ? (z, y) : y
# end
