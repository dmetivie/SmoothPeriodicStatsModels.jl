#! TODO fix convention Y size(Y) = D, N not the opposite. (Here it does not change)
function likelihoods!(L::AbstractMatrix, hmm::sARPeriodicHMM, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(L, 1), size(hmm, 3))::AbstractVector{<:Integer})
    N, K, D = size(Y, 1), size(hmm, 1), size(hmm, 2)
    @argcheck size(L) == (N, K)

    for i in OneTo(K)
        for n in OneTo(N)
            t = n2t[n] # periodic t
            L[n, i] = pdf(product_distribution([hmm.B[k, t, j][lag_cat[n, j]] for j in 1:D]), Y[n, :])
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,3} where {F<:AbstractVector}, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(Y, 1), size(B, 2))::AbstractVector{<:Integer})
    N, K, D = size(Y, 1), size(B, 1), size(Y, 2)
    @argcheck size(LL) == (N, K)

    for k in OneTo(K)
        for n in OneTo(N)
            t = n2t[n]
            @views LL[n, k] = logpdf(product_distribution([B[k, t, j][lag_cat[n, j]] for j in 1:D]), Y[n, :])
        end
    end
end

loglikelihoods!(LL::AbstractMatrix, hmm::sARPeriodicHMM, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer}) = loglikelihoods!(LL, hmm.B, Y, lag_cat; n2t=n2t::AbstractVector{<:Integer})

function loglikelihoods(hmm::sARPeriodicHMM, Y::AbstractMatrix{<:Bool}, Y_past::AbstractVector; robust = false, n2t=n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer})
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

function complete_loglikelihood(hmm::sARPeriodicHMM, y::AbstractMatrix, y_past::AbstractVector, z::AbstractVector; n2t=n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer})
    N, D = size(y)
    lag_cat = conditional_to(y, y_past)

    return sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n = 1:N-1) + sum(logpdf(product_distribution([hmm.B[z[n], n2t[n], j][lag_cat[n, j]] for j in 1:D]), y[n, :]) for n = 1:N)
end

nb_param(K, memory::AbstractVector, d) = (2d + 1) * (K * sum(2 .^ memory) + K * (K - 1))

bic_sel(LL, K, memory, d, penality) = LL - penality * nb_param(K, memory, d) / 2