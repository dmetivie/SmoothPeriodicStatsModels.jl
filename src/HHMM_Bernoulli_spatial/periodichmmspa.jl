
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
"""
struct ARPeriodicHMMSpatial{T, AM} <: AbstractPeriodicHMM{Multivariate}
    a::Vector{T}
    A::Array{T,3}
    R::Array{T,2}
    B::Array{T,4}
    h::AM
end

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
    y_ini=rand(Bool, Int(log2(size(hmm, 4))), size(hmm, 2))
)
    N = length(n2t)
    D = size(hmm, 2)
    size_order = size(hmm, 4)
    order = Int(log2(size_order))
    y = Matrix{Bool}(undef, N, D)

    @argcheck size(y_ini) == (order, D) "Initial condition is not correct: You give $(size(y_ini)) instead of $((order, D))"

    if order == 0
        for n in 1:N
            t = n2t[n]
            y[n, :] = rand(rng, SpatialBernoulli(hmm.R[z[n], t], hmm.B[z[n], t, :, 1], hmm.h))
        end
    else
        # lag_cat encoding (matches bin2digit): lc = 1 + sum(y_{n-m} * 2^(order-m) for m in 1:order)
        # most recent lag (m=1) carries the highest bit weight 2^(order-1)
        # y_ini has shape (order, D): y_ini[m, d] = observation at lag m (row 1 = most recent)
        y[1:order, :] = y_ini
        for n in order+1:N
            t = n2t[n]
            lambdas = map(1:D) do d
                lc = 1
                for m in 1:order
                    lc += Int(y[n-m, d]) * 2^(order - m)
                end
                hmm.B[z[n], t, d, lc]
            end
            y[n, :] = rand(rng, SpatialBernoulli(hmm.R[z[n], t], lambdas, hmm.h))
        end
    end
    return y
end

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
