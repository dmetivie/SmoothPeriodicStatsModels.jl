function update_B!(B::AbstractArray{T,4} where {T}, γ::AbstractMatrix, 𝐘::AbstractMatrix{<:Bool}, estimator, idx_tj::Matrix{Vector{Vector{Int}}})
    @argcheck size(γ, 1) == size(𝐘, 1)
    @argcheck size(γ, 2) == size(B, 1)
    N = size(γ, 1)
    K = size(B, 1)
    T = size(B, 2)
    D = size(B, 3)
    size_order = size(B, 4)
    ## For periodicHMM only the n 𝐘 corresponding to B(t) are used to update B(t)

    @inbounds for t in OneTo(T)
        for i in OneTo(K)
            for j = 1:D
                for m = 1:size_order
                    if sum(γ[idx_tj[t, j][m], i]) > 0
                        B[i, t, j, m] = estimator(Bernoulli, 𝐘[idx_tj[t, j][m], j], γ[idx_tj[t, j][m], i])
                    else
                        B[i, t, j, m] = Bernoulli(eps())
                    end
                end
            end
        end
    end
end

function fit_mle!(
    hmm::HierarchicalPeriodicHMM,
    𝐘::AbstractArray{<:Bool},
    𝐘_past::AbstractArray{<:Bool}
    ;
    n2t=n_to_t(size(𝐘, 1), size(hmm, 3))::AbstractVector{<:Integer},
    display=:none,
    maxiter=100,
    tol=1e-3,
    robust=false,
    estimator=fit_mle)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, T, size_order = size(𝐘, 1), size(hmm, 1), size(hmm, 3), size(hmm, 4)
    @argcheck T == size(hmm.B, 2)
    history = EMHistory(false, 0, [])

    # Allocate order for in-place updates
    c = zeros(N)
    α = zeros(N, K)
    β = zeros(N, K)
    γ = zeros(N, K)
    ξ = zeros(N, K, K)
    LL = zeros(N, K)

    # assign category for observation depending in the 𝐘_past 𝐘
    lag_cat = conditional_to(𝐘, 𝐘_past)
    idx_tj = idx_observation_of_past_cat(lag_cat, n2t, T, size_order)

    loglikelihoods!(LL, hmm, 𝐘, lag_cat; n2t=n2t)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL; n2t=n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL; n2t=n2t)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, ξ, α, β, LL; n2t=n2t)
        update_B!(hmm.B, γ, 𝐘, estimator, idx_tj)
    
        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely 𝐘.
        robust && (hmm.A .+= eps())
    
        @check isprobvec(hmm.a)
        @check istransmats(hmm.A)
    
        loglikelihoods!(LL, hmm, 𝐘, lag_cat; n2t=n2t)
    
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    
        forwardlog!(α, c, hmm.a, hmm.A, LL; n2t=n2t)
        backwardlog!(β, c, hmm.a, hmm.A, LL; n2t=n2t)
        posteriors!(γ, α, β)
    
        logtotp = sum(c)
        (display == :iter) && println("Iteration $it: logtot = $logtotp")
    
        push!(history.logtots, logtotp)
        history.iterations += 1
    
        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history.converged = true
            break
        end
    
        logtot = logtotp
    end

    if !history.converged
        if display in [:iter, :final]
            println("EM has not converged after $(history.iterations) iterations, logtot = $logtot")
        end
    end

    history
end