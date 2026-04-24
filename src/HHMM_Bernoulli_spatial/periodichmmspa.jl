
#-----------------equivalent to periodichmm.jl  ---------------------------#

"""
    PeriodicHMMSpaMemory([a, ]A, R, B,h) -> PeriodicHMMSpaMemory

Build an Auto Regressive Periodic Hidden Markov Chain with Spatial Bernoulli emission `PeriodicHMMSpaMemory`.  
If the initial state distribution `a` is not specified, it does not work. Please give initial state distribution.


**Arguments**
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractArray{T,3}`: transition matrix.
- `B`: rain probabilities
- `R` : range parameter
-  `h` distance matrix.
"""
struct PeriodicHMMSpaMemory{T, AM} <: AbstractPeriodicHMM{Multivariate}
    a::Vector{T}
    A::Array{T,3}
    R::Array{T,2}
    B::Array{T,4}
    h::AM
end

"""
    size(hmm::PeriodicHMMSpaMemory, dim=:)
(K, D, T, number of memory)
"""
size(hmm::PeriodicHMMSpaMemory, dim=:) = (size(hmm.B, 1), size(hmm.B, 3), size(hmm.B, 2), size(hmm.B, 4))[dim]

# simulate with given z sequence.
function rand(rng::AbstractRNG,
    hmm::PeriodicHMMSpaMemory,
    z::AbstractVector{<:Integer},
    n2t::AbstractVector{<:Integer}, y_ini=fill(0, size(hmm, 2))
)
    N = length(n2t)
    y = Matrix{eltype(eltype(hmm.B))}(undef, size(hmm, 2), length(z))
    if size(hmm, 4) == 2 # with memory of order 1
        y[:, 1] = y_ini
        for n in 2:N
            t = n2t[n] # periodic t
            y_previous = y[:, n-1]
            lambdas = (1 .- y_previous) .* hmm.B[z[n], t, :, 1] .+ y_previous .* hmm.B[z[n], t, :, 2]
            y[:, n] = rand(rng, SpatialBernoulli(hmm.R[z[n], t], lambdas, hmm.h))
        end
        return y'
    elseif size(hmm, 4) == 1 # no memory
        for n in 1:N
            t = n2t[n] # periodic t
            lambdas = hmm.B[z[n], t, :, 1]
            y[:, n] = rand(rng, SpatialBernoulli(hmm.R[z[n], t], lambdas, hmm.h))
        end
        return y'
    else
        error("Unsupported memory order: size(hmm, 4) = $(size(hmm, 4))")
    end
end

# ## peut être pas necessaire car déjà dans rand(rng, hmm, z, n2t, y_ini) pour les AbsractHMM
# function rand(rng::AbstractRNG,
#     hmm::PeriodicHMMSpaMemory,
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