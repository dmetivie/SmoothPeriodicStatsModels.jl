
#-----------------equivalent to periodichmm.jl  ---------------------------#

"""
    ARPeriodicHMMSpatial([a, ]A, R, B,h) -> ARPeriodicHMMSpatial

Build an Auto Regressive Periodic Hidden Markov Chain with Spatial Bernoulli emission `ARPeriodicHMMSpatial`.  
If the initial state distribution `a` is not specified, it does not work. Please give initial state distribution.


**Arguments**
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractArray{T,3}`: transition matrix.
- `B`: rain probabilities
- `R` : range parameter
- `h` distance matrix.
- `QMC_m`: quasi-Monte Carlo budget of the emission likelihood, as `m = D * QMC_m` points of
  `MvNormalCDF.mvnormcdf`. It is a field rather than a keyword because the emission is read
  through `HiddenMarkovModels`' `obs_distributions`/`obs_logdensities!`, whose signatures have
  nowhere to put one. Raise it for accuracy, lower it for speed.
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
    size(hmm::ARPeriodicHMMSpatial, dim=:)
(K, D, T, number of memory)
"""
size(hmm::ARPeriodicHMMSpatial, dim=:) = (size(hmm.B, 1), size(hmm.B, 3), size(hmm.B, 2), size(hmm.B, 4))[dim]

# simulate with given z sequence.
function rand(rng::AbstractRNG,
    hmm::ARPeriodicHMMSpatial,
    z::AbstractVector{<:Integer},
    n2t::AbstractVector{<:Integer};
    y_ini=rand(rng, Bool, size(hmm, 2), Int(log2(size(hmm, 4))))
)
    N = length(n2t)
    D = size(hmm, 2)
    size_order = size(hmm, 4)
    order = Int(log2(size_order))
    y = Matrix{Bool}(undef, D, N)

    @argcheck size(y_ini) == (D, order) "Initial condition is not correct: You give $(size(y_ini)) instead of $((D, order))"

    if order == 0
        for n in 1:N
            t = n2t[n]
            y[:, n] = rand(rng, SpatialBernoulli(hmm.R[z[n], t], hmm.B[z[n], t, :, 1], hmm.h))
        end
    else
        # lag_cat encoding (matches bin2digit): lc = 1 + sum(y_{n-m} * 2^(order-m) for m in 1:order)
        # most recent lag (m=1) carries the highest bit weight 2^(order-1)
        # y_ini has shape (D, order): y_ini[d, m] = observation at lag m (column 1 = most recent)
        y[:, 1:order] = y_ini
        for n in order+1:N
            t = n2t[n]
            lambdas = map(1:D) do d
                lc = 1
                for m in 1:order
                    lc += Int(y[d, n-m]) * 2^(order - m)
                end
                hmm.B[z[n], t, d, lc]
            end
            y[:, n] = rand(rng, SpatialBernoulli(hmm.R[z[n], t], lambdas, hmm.h))
        end
    end
    return y
end

rand(hmm::ARPeriodicHMMSpatial, z::AbstractVector{<:Integer}, n2t::AbstractVector{<:Integer}; kwargs...) = rand(GLOBAL_RNG, hmm, z, n2t; kwargs...)

"""
    rand([rng, ]hmm::ARPeriodicHMMSpatial, n2t; z_ini, y_ini, seq = false)
    rand([rng, ]hmm::ARPeriodicHMMSpatial, N::Integer; kwargs...)

Draw a state path and the observations along it; `seq = true` returns `(z, y)`.

Ported here rather than inherited: this model used to subtype
`PeriodicHiddenMarkovModels.AbstractPeriodicHMM`, which supplied these three methods, and it now
subtypes `HiddenMarkovModels.AbstractHMM` instead (see `src/hmm_interface.jl`). The path uses
`A[z[n-1], :, n2t[n-1]]` — the transition matrix of the period each step *leaves*, this
package's convention.
"""
function rand(rng::AbstractRNG, hmm::ARPeriodicHMMSpatial, n2t::AbstractVector{<:Integer};
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
