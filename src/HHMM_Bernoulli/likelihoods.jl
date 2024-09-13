#! TODO fix convention Y size(Y) = D, N not the opposite. (Here it does not change)
function likelihoods!(L::AbstractMatrix, hmm::HierarchicalPeriodicHMM, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(L, 1), size(hmm, 3))::AbstractVector{<:Integer})
    N, K, D = size(Y, 1), size(hmm, 1), size(hmm, 2)
    @argcheck size(L) == (N, K)

    for i in OneTo(K)
        for n in OneTo(N)
            t = n2t[n] # periodic t
            L[n, i] = pdf(product_distribution(hmm.B[CartesianIndex.(i, t, 1:D, lag_cat[n, :])]), Y[n, :])
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::HierarchicalPeriodicHMM, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer})
    N, K, D = size(Y, 1), size(hmm, 1), size(hmm, 2)
    @argcheck size(LL) == (N, K)

    for i in OneTo(K)
        for n in OneTo(N)
            t = n2t[n]
            LL[n, i] = logpdf(product_distribution(hmm.B[CartesianIndex.(i, t, 1:D, lag_cat[n, :])]), Y[n, :])
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

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,4} where {F<:UnivariateDistribution}, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(Y, 1), size(B, 2))::AbstractVector{<:Integer})
    N, K, D = size(Y, 1), size(B, 1), size(Y, 2)
    @argcheck size(LL) == (N, K)

    for k in OneTo(K)
        for n in OneTo(N)
            t = n2t[n]
            LL[n, k] = logpdf(product_distribution(B[CartesianIndex.(k, t, 1:D, lag_cat[n, :])]), Y[n, :])
        end
    end
end


function loglikelihoods(hmm::HierarchicalPeriodicHMM, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}; robust = false, n2t=n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer})
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

function complete_loglikelihood(hmm::HierarchicalPeriodicHMM, y::AbstractArray, y_past::AbstractArray, z::AbstractVector; n2t=n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer})
    N, D = size(y)
    lag_cat = conditional_to(y, y_past)

    return sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n = 1:N-1) + sum(logpdf(product_distribution(hmm.B[CartesianIndex.(z[n], n2t[n], 1:D, lag_cat[n, :])]), y[n, :]) for n = 1:N)
end

function complete_loglikelihood(hmm::PeriodicHMM, y::AbstractArray, z::AbstractVector; n2t=n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer})
    N, D = size(y, 1), size(y, 2)

    return sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n = 1:N-1) + sum(logpdf(product_distribution(hmm.B[CartesianIndex.(z[n], n2t[n], 1:D)]), y[n, :]) for n = 1:N)
end

nb_param(K, memory, d, D) = (2d + 1) * (K * 2^memory * D + K * (K - 1))
bic_sel(LL, K, memory, d, D, penality) = LL - penality * nb_param(K, memory, d, D) / 2

# function complete_loglikelihood(hmm::HMM, y::AbstractArray, z::AbstractVector)
#     N, D = size(y, 1), size(y, 2)

#     return sum(log(hmm.A[z[n], z[n+1]]) for n = 1:N-1) + sum(logpdf(product_distribution(hmm.B[CartesianIndex.(z[n], 1:D)]), y[n, :]) for n = 1:N)
# end
