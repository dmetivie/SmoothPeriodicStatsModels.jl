function fit_mle_all_slices(hmm::sARPeriodicHMM, Y::AbstractMatrix{<:Bool}, Y_past::AbstractVector;
    n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer},
    Yₜ_extanted = [0],
    robust=false,
    history=false,
    kwargs...)

    hmm = copy(hmm)

    N, K, T = size(Y, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(Y, 1) == size(n2t, 1)

    # assign category for observation depending in the past Y
    lag_cat = conditional_to(Y, Y_past)

    n_in_t = [findall(n2t .== t) for t = 1:T] #

    hist = Vector{Dict}(undef, T)

    # Initial condition
    α = hcat([vec(sum(hmm.A[:, :, t], dims=1) / K) for t = 1:T]...)

    for t = 1:T
        n_in_t_extanded = sort(vcat([n_in_t[tt] for tt in cycle.(t .+ Yₜ_extanted, T)]...)) # extend dataset
        hist[t] = fit_mle_B_slice!(@view(α[:, t]), hmm.B[:, t, :], Y[n_in_t_extanded, :], lag_cat[n_in_t_extanded, :]; kwargs...)
    end

    LL = zeros(N, K)

    loglikelihoods!(LL, hmm.B, Y, lag_cat; n2t=n2t)
    for k = 1:K, n = 1:N
        LL[n, k] += log(α[k, n2t[n]])
    end

    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    fit_mle_A_from_slice!(hmm.a, hmm.A, LL, n2t, robust=robust)

    return history ? (hmm, hist) : hmm
end

function fit_mle_B_slice!(α::AbstractVector, B::AbstractMatrix,
    Y::AbstractMatrix{<:Bool}, lag_cat::AbstractMatrix{<:Integer};
    rand_ini=true,
    n_random_ini=10, display_random=false,
    Dirichlet_α=0.8,
    ref_station=1, kwargs...)

    size_order = [length(B[1,j]) for j in axes(B,2)]
    Idx = idx_observation_of_past_cat(lag_cat, size_order)

    if rand_ini == true
        α[:], B[:], h = fit_em_multiD_rand(α, B, Y, lag_cat, Idx; n_random_ini=n_random_ini, Dirichlet_α=Dirichlet_α, display_random=display_random, kwargs...)
    else
        h = fit_em_multiD!(α, B, Y, lag_cat, Idx; kwargs...)
    end
    sort_wrt_ref!(α, B, ref_station)
    return h
end

function fit_em_multiD_rand(α::AbstractVector, B::AbstractMatrix,
    Y::AbstractMatrix{<:Integer}, lag_cat::AbstractMatrix{<:Integer}, idx_j::AbstractVector{Vector{Vector{Int}}};
    n_random_ini=10, Dirichlet_α=0.8, display_random=false, kwargs...)

    D = size(Y, 2)
    K = size(α, 1)
    size_order = [length(B[1,j]) for j in 1:D]

    h = fit_em_multiD!(α, B, Y, lag_cat, idx_j; kwargs...)
    log_max = h["logtots"][end]
    α_max, B_max = copy(α), copy(B)
    h_max = h
    (display_random == :iter) && println("random IC 1: logtot = $(h["logtots"][end])")
    for i = 1:(n_random_ini-1)
        B = [[Bernoulli(rand()) for h in 1:size_order[j]] for k in 1:K, j in 1:D]
        α[:] = rand(Dirichlet(K, Dirichlet_α))
        h = fit_em_multiD!(α, B, Y, lag_cat, idx_j; kwargs...)
        (display_random == :iter) && println("random IC $(i+1): logtot = $(h["logtots"][end])")
        if h["logtots"][end] > log_max
            log_max = h["logtots"][end]
            h_max = h
            α_max[:], B_max[:] = copy(α), copy(B)
        end
    end
    return α_max, B_max, h_max
end

function fit_em_multiD!(α::AbstractVector, B::AbstractMatrix,
    Y::AbstractMatrix{<:Bool}, lag_cat::AbstractMatrix{<:Integer}, idx_j::AbstractVector{Vector{Vector{Int}}};
    display=:none, maxiter=100, tol=1e-3, robust=false)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, D = size(Y, 1), size(B, 1), size(B, 2)
    size_order = [length(B[1,j]) for j in 1:D]

    history = Dict("converged" => false, "iterations" => 0, "logtots" => Float64[])

    # Allocate order for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # Initial parameters already in α, B

    # E-step
    # evaluate likelihood for each type k
    for k in OneTo(K), n in OneTo(N)
        LL[n, k] = logpdf(product_distribution([B[k, j][lag_cat[n, j]] for j in 1:D]), Y[n, :])
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
                for m = 1:size_order[j]
                    if sum(γ[idx_j[j][m], k]) > 0
                        B[k, j][m] = fit_mle(Bernoulli, Y[idx_j[j][m], j], γ[idx_j[j][m], k])
                    else
                        B[k, j][m] = Bernoulli(1 / 2)
                    end
                end
            end
        end

        # E-step
        # evaluate likelihood for each type k
        @inbounds for k in OneTo(K), n in OneTo(N)
            LL[n, k] = logpdf(product_distribution([B[k, j][lag_cat[n, j]] for j in 1:D]), Y[n, :])
        end
        for k = 1:K
            LL[:, k] .+= log(α[k])
        end
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
        # get posterior of each category
        c[:] = logsumexp(LL, dims=2)
        γ[:, :] = exp.(LL .- c)

        # Loglikelihood
        logtotp = sum(c)
        (display == :iter) && println("Iteration $it: logtot = $logtotp")

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history["converged"] = true
            break
        end

        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println("EM has not converged after $(history["iterations"]) iterations, logtot = $logtot")
        end
    end

    history
end