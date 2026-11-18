"""
    fit_mle_all_slices([rng, ]hmm, Y, Y_past; n2t, Yₜ_extanted = [0], robust = false, history = false, kwargs...)

A starting point for fitting a periodic HMM, estimated one period index at a time and ignoring
the dynamics. At each `t`, the weights and Bernoulli emissions of a `K`-component mixture are
fitted by EM to the observations at period indices `t .+ Yₜ_extanted` (wrapped into `1:T`) with
[`fit_mle_B_slice!`](@ref), which receives the other keywords. The transition matrices are then
the transition frequencies of the most likely state of each observation
([`fit_mle_A_from_slice!`](@ref)).

Returns a copy of `hmm` holding the estimates, and with `history = true` also each slice's EM
history. Defined for `ARPeriodicHMM` and `sARPeriodicHMM`.
"""
fit_mle_all_slices(hmm::ARPeriodicHMM, Y::AbstractMatrix{<:Bool}, Y_past::AbstractMatrix{<:Bool}; kwargs...) = fit_mle_all_slices(GLOBAL_RNG, hmm, Y, Y_past; kwargs...)

function fit_mle_all_slices(rng::AbstractRNG, hmm::ARPeriodicHMM, Y::AbstractMatrix{<:Bool}, Y_past::AbstractMatrix{<:Bool};
    n2t=n_to_t(size(Y, 2), size(hmm, 3))::AbstractVector{<:Integer},
    Yₜ_extanted = [0],
    robust=false,
    history=false,
    kwargs...)

    hmm = copy(hmm)

    N, K, T = size(Y, 2), size(hmm, 1), size(hmm, 3)
    @argcheck size(Y, 2) == size(n2t, 1)

    # assign category for observation depending in the past Y
    lag_cat = conditional_to(Y, Y_past)

    n_in_t = [findall(n2t .== t) for t = 1:T] #

    hist = Vector{Dict}(undef, T)

    # Initial condition
    α = hcat([vec(sum(hmm.A[:, :, t], dims=1) / K) for t = 1:T]...)

    for t = 1:T
        n_in_t_extanded = sort(vcat([n_in_t[tt] for tt in cycle.(t .+ Yₜ_extanted, T)]...)) # extend dataset
        hist[t] = fit_mle_B_slice!(rng, @view(α[:, t]), @view(hmm.B[:, t, :, :]), Y[:, n_in_t_extanded], lag_cat[:, n_in_t_extanded]; kwargs...)
    end

    LL = zeros(N, K)

    loglikelihoods!(LL, hmm.B, Y, lag_cat; n2t=n2t)
    for k = 1:K, n = 1:N
        LL[n, k] += log(α[k, n2t[n]])
    end

    robust && _clamp_inf!(LL)

    fit_mle_A_from_slice!(hmm.a, hmm.A, LL, n2t, robust=robust)

    return history ? (hmm, hist) : hmm
end

"""
    fit_mle_A_from_slice!(a, A, LL, n2t; robust = false)

The initial distribution and the transition matrices from the most likely state of each
observation. From the `N × K` matrix `LL` of joint log-weights (log prior weight plus emission
loglikelihood), it sets `a` to the posterior of the first observation and each `A[:, :, t]` to
the transition frequencies, from period index `t`, of the most likely states.
`robust = true` adds `eps()` to every count, so that no row is empty.
"""
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

    Q = zeros(K, K, T)
    for n = 1:N-1
        Q[max_aposteriori[n], max_aposteriori[n+1], n2t[n]] += 1
    end
    robust && (Q .+= eps())
    for t = 1:T
        A[:, :, t] = Q[:, :, t] ./ sum(Q[:, :, t], dims=2) 
    end
end

"""
    fit_mle_B_slice!([rng, ]α, B, Y, lag_cat; rand_ini = true, n_random_ini = 10, Dirichlet_α = 0.8, display_random = false, ref_station = 1, kwargs...)

Fit in place a mixture of product-Bernoulli laws to the observations of one slice. The weights
`α` and the Bernoulli emissions `B` (`K × D × 2^order`, or `K × D` vectors for site-dependent
memory) are fitted to `Y`, site `j` of observation `n` using memory category `lag_cat[j, n]`. With `rand_ini = true` EM
also restarts from `n_random_ini - 1` random points (weights from `Dirichlet(K, Dirichlet_α)`)
and keeps the best ([`fit_em_multiD_rand`](@ref)); otherwise it runs once from the current
values ([`fit_em_multiD!`](@ref)), which receives the other keywords. States are then sorted with
[`sort_wrt_ref!`](@ref). Returns the EM history.
"""
function fit_mle_B_slice!(rng::AbstractRNG, α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli},
    Y::AbstractMatrix{<:Bool}, lag_cat::AbstractMatrix{<:Integer};
    rand_ini=true,
    n_random_ini=10, display_random=false,
    Dirichlet_α=0.8,
    ref_station=1, kwargs...)

    size_order = size(B, 3)
    Idx = idx_observation_of_past_cat(lag_cat, size_order)

    if rand_ini == true
        α[:], B[:], h = fit_em_multiD_rand(rng, α, B, Y, lag_cat, Idx; n_random_ini=n_random_ini, Dirichlet_α=Dirichlet_α, display_random=display_random, kwargs...)
    else
        h = fit_em_multiD!(α, B, Y, lag_cat, Idx; kwargs...)
    end
    sort_wrt_ref!(α, B, ref_station)
    return h
end

fit_mle_B_slice!(α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli}, Y::AbstractMatrix{<:Bool}, lag_cat::AbstractMatrix{<:Integer}; kwargs...) = fit_mle_B_slice!(GLOBAL_RNG, α, B, Y, lag_cat; kwargs...)

"""
    fit_em_multiD_rand([rng, ]α, B, Y, lag_cat, idx_j; n_random_ini = 10, Dirichlet_α = 0.8, display_random = false, kwargs...) -> (α, B, history)

Run [`fit_em_multiD!`](@ref) from the given `(α, B)` and from `n_random_ini - 1` random starts,
and return the estimates and history with the highest loglikelihood. `display_random = :iter`
prints each run's result.
"""
function fit_em_multiD_rand(rng::AbstractRNG, α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli},
    Y::AbstractMatrix{<:Integer}, lag_cat::AbstractMatrix{<:Integer}, idx_j::AbstractVector{Vector{Vector{Int}}};
    n_random_ini=10, Dirichlet_α=0.8, display_random=false, kwargs...)

    D = size(Y, 1)
    K = size(α, 1)
    size_order = size(B, 3)

    h = fit_em_multiD!(α, B, Y, lag_cat, idx_j; kwargs...)
    log_max = h["logtots"][end]
    α_max, B_max = copy(α), copy(B)
    h_max = h
    (display_random == :iter) && println("random IC 1: logtot = $(h["logtots"][end])")
    for i = 1:(n_random_ini-1)
        B[:, :, :] = random_product_Bernoulli(rng, K, D, size_order)
        α[:] = rand(rng, Dirichlet(K, Dirichlet_α))
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

fit_em_multiD_rand(α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli}, Y::AbstractMatrix{<:Integer}, lag_cat::AbstractMatrix{<:Integer}, idx_j::AbstractVector{Vector{Vector{Int}}}; kwargs...) = fit_em_multiD_rand(GLOBAL_RNG, α, B, Y, lag_cat, idx_j; kwargs...)

"""
    fit_em_multiD!(α, B, Y, lag_cat, idx_j; display = :none, maxiter = 100, tol = 1e-3, robust = false) -> Dict

EM for a mixture of product-Bernoulli laws with memory, updating the weights `α` and emissions
`B` in place. `idx_j[j][h]` lists the observations where site `j` is in memory category `h`
([`idx_observation_of_past_cat`](@ref)). Stops when the loglikelihood changes by less than `tol`
or after `maxiter` iterations; `display` is `:none`, `:iter` or `:final`. Returns a `Dict` with
keys `"converged"`, `"iterations"` and `"logtots"`.
"""
function fit_em_multiD!(α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli},
    Y::AbstractMatrix{<:Bool}, lag_cat::AbstractMatrix{<:Integer}, idx_j::AbstractVector{Vector{Vector{Int}}};
    display=:none, maxiter=100, tol=1e-3, robust=false)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, D, size_order = size(Y, 2), size(B, 1), size(B, 2), size(B, 3)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => Float64[])

    # Allocate order for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # Initial parameters already in α, B

    # E-step
    # evaluate likelihood for each type k
    for k in OneTo(K), n in OneTo(N)
        LL[n, k] = logpdf(product_distribution(B[CartesianIndex.(k, 1:D, view(lag_cat, :, n))]), view(Y, :, n))
    end
    for k = 1:K
        LL[:, k] .+= log(α[k])
    end
    robust && _clamp_inf!(LL)
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
                        B[k, j, m] = fit_mle(Bernoulli, Y[j, idx_j[j][m]], γ[idx_j[j][m], k])
                    else
                        B[k, j, m] = Bernoulli(1 / 2)
                    end
                end
            end
        end

        # E-step
        # evaluate likelihood for each type k
        @inbounds for k in OneTo(K), n in OneTo(N)
            LL[n, k] = logpdf(product_distribution(B[CartesianIndex.(k, 1:D, view(lag_cat, :, n))]), view(Y, :, n))
        end
        for k = 1:K
            LL[:, k] .+= log(α[k])
        end
        robust && _clamp_inf!(LL)
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

#TODO PeriodicHMM version
# function fit_mle_all_slices(hmm::PeriodicHMM, Y;
#     n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer},
#     robust=false,
#     smooth=true, window=-15:15, kernel=:step,
#     history=false,
#     kwargs...)

#     hmm = copy(hmm)
#     N = size(Y, 1)
#     @argcheck size(Y, 1) == size(n2t, 1)

#     K, T = size(hmm, 1), size(hmm, 3)
#     α = hcat([vec(sum(hmm.A[:, :, t], dims=1) / K) for t = 1:T]...)
#     n_in_t = [findall(n2t .== t) for t = 1:T] #
#     hist = Vector{HMMBase.EMHistory}(undef, T)
#     cycle = CyclicArray(1:T, "1D")
#     for t = 1:T
#         n_in_t_extanded = sort(vcat([n_in_t[tt] for tt in cycle[[t - 12, t - 6, t, t + 6, t + 12]]]...)) # extend dataset
#         hist[t] = fit_mle_B_slice!(@view(α[:, t]), @view(hmm.B[:, t]), Y[n_in_t_extanded, :]; kwargs...)
#         # hist[t] = fit_mle_B_slice!(@view(α[:,t]), @view(hmm.B[:,t]), Y[n_in_t[t],:]; kwargs...)
#     end

#     LL = zeros(N, K)

#     if smooth == true
#         smooth_B = lagou(hmm.B, dims=2, window=window, kernel=kernel)
#         smooth_α = lagou(α, dims=2, window=window, kernel=kernel)

#         # evaluate likelihood for each type k
#         loglikelihoods!(LL, smooth_B, Y, n2t)
#         [LL[n, k] += log(smooth_α[k, n2t[n]]) for k = 1:K, n = 1:N]
#     else
#         # evaluate likelihood for each type k
#         loglikelihoods!(LL, hmm.B, Y, n2t)
#         [LL[n, k] += log(α[k, n2t[n]]) for k = 1:K, n = 1:N]
#     end
#     robust && _clamp_inf!(LL)

#     fit_mle_A_from_slice!(hmm.a, hmm.A, LL, n2t, robust=robust)

#     return history ? (hmm, hist) : hmm
# end

#TODO PeriodicHMM version
# function fit_mle_B_slice!(α::AbstractVector, B::AbstractVector{F} where {F<:Distribution}, Y;
#     rand_ini=true,
#     n_random_ini=10, display_random=false,
#     Dirichlet_α=0.8, Dirichlet_categories=0.85,
#     ref_station=1, kwargs...)
#     if rand_ini == true
#         α[:], B[:], h = fit_em_multiD_rand(α, B, Y; n_random_ini=n_random_ini, Dirichlet_α=Dirichlet_α, Dirichlet_categories=Dirichlet_categories, display_random=display_random, kwargs...)
#     else
#         h = fit_em_multiD!(α, B, Y; kwargs...)
#     end
#     sort_wrt_ref!(α, B, ref_station)
#     h
# end