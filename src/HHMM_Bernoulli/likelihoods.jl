#! TODO fix convention Y size(Y) = D, N not the opposite. (Here it does not change)
likelihoods!(L::AbstractMatrix, hmm::ARPeriodicHMM, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(L, 1), size(hmm, 3))::AbstractVector{<:Integer}) = likelihoods!(L, hmm.B, Y, lag_cat; n2t=n2t::AbstractVector{<:Integer})

"""
    likelihoods!(L, B, Y, lag_cat; n2t)

Fill `L[n, k]` with the likelihood of observation `n` in state `k`, i.e. the product over the
sites `j` of `pdf(B[k, n2t[n], j, lag_cat[n, j]], Y[n, j])`.

Equivalent to `pdf(product_distribution(B[k, n2t[n], :, lag_cat[n, :]]), Y[n, :])`, without
building the index vector, the distribution and the observation slice for every `(n, k)`.
`B` is an argument so this specializes on its concrete element type; the `hmm.B` field is
abstractly typed.
"""
function likelihoods!(L::AbstractMatrix, B::AbstractArray{F,4} where {F<:UnivariateDistribution}, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(L, 1), size(B, 2))::AbstractVector{<:Integer})
    N, K, D = size(Y, 1), size(B, 1), size(Y, 2)
    @argcheck size(L) == (N, K)

    for i in OneTo(K)
        for n in OneTo(N)
            t = n2t[n] # periodic t
            p = one(eltype(L))
            for j in OneTo(D)
                p *= pdf(B[i, t, j, lag_cat[n, j]], Y[n, j])
            end
            L[n, i] = p
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,2} where {F<:MultivariateDistribution}, Y::AbstractMatrix; n2t=n_to_t(size(Y, 1), size(B, 2))::AbstractVector{<:Integer})
    N, K = size(Y, 1), size(B, 1)
    @argcheck size(LL) == (N, K)

    for n in OneTo(N)
        t = n2t[n]
        for k in OneTo(K)
            LL[n, k] = logpdf(B[k, t], Y[n, :])
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,2} where {F<:UnivariateDistribution}, Y::AbstractVector; n2t=n_to_t(size(Y, 1), size(B, 2))::AbstractVector{<:Integer})
    N, K = size(Y, 1), size(B, 1)
    @argcheck size(LL) == (N, K)

    for k in OneTo(K)
        for n in OneTo(N)
            t = n2t[n]
            LL[n, k] = logpdf(B[k, t], Y[n])
        end
    end
end

loglikelihoods!(LL::AbstractMatrix, hmm::ARPeriodicHMM, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer}) = loglikelihoods!(LL, hmm.B, Y, lag_cat; n2t=n2t::AbstractVector{<:Integer})

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,4} where {F<:UnivariateDistribution}, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(Y, 1), size(B, 2))::AbstractVector{<:Integer})
    N, K, D = size(Y, 1), size(B, 1), size(Y, 2)
    @argcheck size(LL) == (N, K)

    for k in OneTo(K)
        for n in OneTo(N)
            t = n2t[n]
            # Equivalent to `logpdf(product_distribution(B[k, t, :, lag_cat[n, :]]), Y[n, :])`
            ll = zero(eltype(LL))
            for j in OneTo(D)
                ll += logpdf(B[k, t, j, lag_cat[n, j]], Y[n, j])
            end
            LL[n, k] = ll
        end
    end
end


function loglikelihoods(hmm::ARPeriodicHMM, Y::AbstractMatrix{<:Bool}, Y_past::AbstractMatrix{<:Bool}; robust = false, n2t=n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer})
    N, K = size(Y, 1), size(hmm, 1)
    LL = Matrix{Float64}(undef, N, K)

    lag_cat = conditional_to(Y, Y_past)

    loglikelihoods!(LL, hmm, Y, lag_cat; n2t=n2t)
    if robust
        replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    end
    return LL
end

# * Bayesian Criterion * #

function complete_loglikelihood(hmm::ARPeriodicHMM, y::AbstractMatrix, y_past::AbstractMatrix, z::AbstractVector; n2t=n_to_t(size(y, 1), size(hmm.B, 2))::AbstractVector{<:Integer})
    lag_cat = conditional_to(y, y_past)
    N = size(y, 1)

    return sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n = 1:N-1) +
           emission_loglikelihood(hmm.B, y, z, lag_cat, n2t)
end

"""
    emission_loglikelihood(B, y, z, lag_cat, n2t)

Loglikelihood of the observations `y` along the state path `z`, i.e. the sum over `n` and over
the sites `j` of `logpdf(B[z[n], n2t[n], j, lag_cat[n, j]], y[n, j])`. This is the emission part
of [`complete_loglikelihood`](@ref), the other one being the transition term.

Equivalent to `sum(logpdf(product_distribution(B[z[n], n2t[n], :, lag_cat[n, :]]), y[n, :]))`.
`B` is an argument so this specializes on its concrete element type; the `hmm.B` field is
abstractly typed.
"""
function emission_loglikelihood(B::AbstractArray{F,4} where {F<:UnivariateDistribution}, y::AbstractMatrix, z::AbstractVector, lag_cat::AbstractMatrix{<:Integer}, n2t::AbstractVector{<:Integer})
    N, D = size(y)
    ll = 0.0
    for n = 1:N
        t = n2t[n]
        for j = 1:D
            ll += logpdf(B[z[n], t, j, lag_cat[n, j]], y[n, j])
        end
    end
    return ll
end

function complete_loglikelihood(hmm::PeriodicHMM, y::AbstractMatrix, z::AbstractVector; n2t=n_to_t(size(y, 1), size(hmm.B, 2))::AbstractVector{<:Integer})
    N, D = size(y, 1), size(y, 2)

    return sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n = 1:N-1) + sum(logpdf(product_distribution(hmm.B[CartesianIndex.(z[n], n2t[n], 1:D)]), y[n, :]) for n = 1:N)
end

nb_param(K, memory::Integer, d, D) = (2d + 1) * (K * 2^memory * D + K * (K - 1))
bic_sel(LL, K, memory::Integer, d, D, penality) = LL - penality * nb_param(K, memory, d, D) / 2

# function complete_loglikelihood(hmm::HMM, y::AbstractMatrix, z::AbstractVector)
#     N, D = size(y, 1), size(y, 2)

#     return sum(log(hmm.A[z[n], z[n+1]]) for n = 1:N-1) + sum(logpdf(product_distribution(hmm.B[CartesianIndex.(z[n], 1:D)]), y[n, :]) for n = 1:N)
# end
