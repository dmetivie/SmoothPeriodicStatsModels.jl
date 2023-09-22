function fit_mle_all_slices(hmm::HierarchicalPeriodicHMM, 𝐘::AbstractArray{<:Bool}, 𝐘_past::AbstractArray{<:Bool};
    n2t=n_to_t(size(𝐘, 1), size(hmm, 3))::AbstractVector{<:Integer},
    𝐘ₜ_extanted = [0],
    robust=false,
    history=false,
    kwargs...)

    hmm = copy(hmm)

    N, K, T = size(𝐘, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(𝐘, 1) == size(n2t, 1)

    # assign category for observation depending in the past 𝐘
    lag_cat = conditional_to(𝐘, 𝐘_past)

    n_in_t = [findall(n2t .== t) for t = 1:T] #

    hist = Vector{EMHistory}(undef, T)

    # Initial condition
    α = hcat([vec(sum(hmm.A[:, :, t], dims=1) / K) for t = 1:T]...)

    for t = 1:T
        n_in_t_extanded = sort(vcat([n_in_t[tt] for tt in cycle.(t .+ 𝐘ₜ_extanted, T)]...)) # extend dataset
        # n_in_t_extanded = n_in_t[t]#sort(vcat([n_in_t[tt] for tt in cycle[[t - 12, t - 7, t, t + 6, t + 13]]]...)) # extend dataset
        hist[t] = fit_mle_B_slice!(@view(α[:, t]), @view(hmm.B[:, t, :, :]), 𝐘[n_in_t_extanded, :], lag_cat[n_in_t_extanded, :]; kwargs...)
    end

    LL = zeros(N, K)

    loglikelihoods!(LL, hmm.B, 𝐘, lag_cat; n2t=n2t)
    for k = 1:K, n = 1:N
        LL[n, k] += log(α[k, n2t[n]])
    end

    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    fit_mle_A_from_slice!(hmm.a, hmm.A, LL, n2t, robust=robust)

    return history ? (hmm, hist) : hmm
end

function fit_mle_A_from_slice!(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL, n2t; robust=false)

    N, K = size(LL)
    T = size(A, 3)
    c = zeros(N)
    γ = zeros(N, K)

    # get posterior of each category
    c[:] = logsumexp(LL, dims=2)
    γ[:, :] = exp.(LL .- c)
    a[:] = γ[1, :] # initial date
    max_aposteriori = argmaxrow(γ)
    max_aposteriori_next = lead(max_aposteriori)

    Q = zeros(K, K, T)
    for n = 1:N
        t = n2t[n]
        if max_aposteriori_next[n] === missing
            continue
        else
            Q[max_aposteriori[n], max_aposteriori_next[n], t] += 1
        end
    end
    robust && (Q .+= eps())
    for t = 1:T
        A[:, :, t] = Q[:, :, t] ./ sum(Q[:, :, t], dims=2) 
    end
end

function fit_mle_B_slice!(α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli},
    𝐘::AbstractMatrix{<:Bool}, lag_cat::AbstractMatrix{<:Integer};
    rand_ini=true,
    n_random_ini=10, display_random=false,
    Dirichlet_α=0.8,
    ref_station=1, kwargs...)

    size_order = size(B, 3)
    Idx = idx_observation_of_past_cat(lag_cat, size_order)

    if rand_ini == true
        α[:], B[:], h = fit_em_multiD_rand(α, B, 𝐘, lag_cat, Idx; n_random_ini=n_random_ini, Dirichlet_α=Dirichlet_α, display_random=display_random, kwargs...)
    else
        h = fit_em_multiD!(α, B, 𝐘, lag_cat, Idx; kwargs...)
    end
    sort_wrt_ref!(α, B, ref_station)
    return h
end

function fit_em_multiD_rand(α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli},
    𝐘::AbstractMatrix{<:Integer}, lag_cat::AbstractMatrix{<:Integer}, idx_j::AbstractVector{Vector{Vector{Int}}};
    n_random_ini=10, Dirichlet_α=0.8, display_random=false, kwargs...)

    D = size(𝐘, 2)
    K = size(α, 1)
    size_order = size(B, 3)

    h = fit_em_multiD!(α, B, 𝐘, lag_cat, idx_j; kwargs...)
    log_max = h.logtots[end]
    α_max, B_max = copy(α), copy(B)
    h_max = h
    (display_random == :iter) && println("random IC 1: logtot = $(h.logtots[end])")
    for i = 1:(n_random_ini-1)
        B[:, :, :] = random_product_Bernoulli(D, K, size_order)
        α[:] = rand(Dirichlet(K, Dirichlet_α))
        h = fit_em_multiD!(α, B, 𝐘, lag_cat, idx_j; kwargs...)
        (display_random == :iter) && println("random IC $(i+1): logtot = $(h.logtots[end])")
        if h.logtots[end] > log_max
            log_max = h.logtots[end]
            h_max = h
            α_max[:], B_max[:] = copy(α), copy(B)
        end
    end
    return α_max, B_max, h_max
end

function fit_em_multiD!(α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli},
    𝐘::AbstractMatrix{<:Bool}, lag_cat::AbstractMatrix{<:Integer}, idx_j::AbstractVector{Vector{Vector{Int}}};
    display=:none, maxiter=100, tol=1e-3, robust=false)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, D, size_order = size(𝐘, 1), size(B, 1), size(B, 2), size(B, 3)
    history = EMHistory(false, 0, [])

    # Allocate order for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # Initial parameters already in α, B

    # E-step
    # evaluate likelihood for each type k
    for k in OneTo(K), n in OneTo(N)
        LL[n, k] = logpdf(product_distribution(B[CartesianIndex.(k, 1:D, lag_cat[n, :])]), 𝐘[n, :])
    end
    for k = 1:K
        LL[:, k] .+= log(α[k])
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    c[:] = logsumexp(LL, dims=2)
    γ[:, :] = exp.(LL .- c)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter

        # M-step
        # with γ in hand, maximize (update) the parameters
        α[:] = mean(γ, dims=1)
        for k in OneTo(K)
            for j = 1:D
                for m = 1:size_order
                    if sum(γ[idx_j[j][m], k]) > 0
                        B[k, j, m] = fit_mle(Bernoulli, 𝐘[idx_j[j][m], j], γ[idx_j[j][m], k])
                    else
                        B[k, j, m] = Bernoulli(1 / 2)
                    end
                end
            end
        end

        # E-step
        # evaluate likelihood for each type k
        @inbounds for k in OneTo(K), n in OneTo(N)
            LL[n, k] = logpdf(product_distribution(B[CartesianIndex.(k, 1:D, lag_cat[n, :])]), 𝐘[n, :])
        end
        [LL[:, k] .+= log(α[k]) for k = 1:K]
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
        # get posterior of each category
        c[:] = logsumexp(LL, dims=2)
        γ[:, :] = exp.(LL .- c)

        # Loglikelihood
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

#TODO PeriodicHMM version
# function fit_mle_all_slices(hmm::PeriodicHMM, 𝐘;
#     n2t=n_to_t(size(𝐘, 1), size(hmm, 3))::AbstractVector{<:Integer},
#     robust=false,
#     smooth=true, window=-15:15, kernel=:step,
#     history=false,
#     kwargs...)

#     hmm = copy(hmm)
#     N = size(𝐘, 1)
#     @argcheck size(𝐘, 1) == size(n2t, 1)

#     K, T = size(hmm, 1), size(hmm, 3)
#     α = hcat([vec(sum(hmm.A[:, :, t], dims=1) / K) for t = 1:T]...)
#     n_in_t = [findall(n2t .== t) for t = 1:T] #
#     hist = Vector{HMMBase.EMHistory}(undef, T)
#     cycle = CyclicArray(1:T, "1D")
#     for t = 1:T
#         n_in_t_extanded = sort(vcat([n_in_t[tt] for tt in cycle[[t - 12, t - 6, t, t + 6, t + 12]]]...)) # extend dataset
#         hist[t] = fit_mle_B_slice!(@view(α[:, t]), @view(hmm.B[:, t]), 𝐘[n_in_t_extanded, :]; kwargs...)
#         # hist[t] = fit_mle_B_slice!(@view(α[:,t]), @view(hmm.B[:,t]), 𝐘[n_in_t[t],:]; kwargs...)
#     end

#     LL = zeros(N, K)

#     if smooth == true
#         smooth_B = lagou(hmm.B, dims=2, window=window, kernel=kernel)
#         smooth_α = lagou(α, dims=2, window=window, kernel=kernel)

#         # evaluate likelihood for each type k
#         loglikelihoods!(LL, smooth_B, 𝐘, n2t)
#         [LL[n, k] += log(smooth_α[k, n2t[n]]) for k = 1:K, n = 1:N]
#     else
#         # evaluate likelihood for each type k
#         loglikelihoods!(LL, hmm.B, 𝐘, n2t)
#         [LL[n, k] += log(α[k, n2t[n]]) for k = 1:K, n = 1:N]
#     end
#     robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

#     fit_mle_A_from_slice!(hmm.a, hmm.A, LL, n2t, robust=robust)

#     return history ? (hmm, hist) : hmm
# end

#TODO PeriodicHMM version
# function fit_mle_B_slice!(α::AbstractVector, B::AbstractVector{F} where {F<:Distribution}, 𝐘;
#     rand_ini=true,
#     n_random_ini=10, display_random=false,
#     Dirichlet_α=0.8, Dirichlet_categories=0.85,
#     ref_station=1, kwargs...)
#     if rand_ini == true
#         α[:], B[:], h = fit_em_multiD_rand(α, B, 𝐘; n_random_ini=n_random_ini, Dirichlet_α=Dirichlet_α, Dirichlet_categories=Dirichlet_categories, display_random=display_random, kwargs...)
#     else
#         h = fit_em_multiD!(α, B, 𝐘; kwargs...)
#     end
#     sort_wrt_ref!(α, B, ref_station)
#     h
# end