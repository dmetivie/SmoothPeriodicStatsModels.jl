"""
    conditional_to(Y, Y_past::AbstractVector) -> Matrix{Int}

The memory category of every site, for sites that each carry their own memory order. `Y` is
`D × N` and `Y_past[j]` is the initial condition of site `j`, its `order_j` values before `Y[j, :]` in
time order.
"""
function conditional_to(Y::AbstractMatrix{<:Bool}, Y_past::AbstractVector)
    D, N = size(Y)
    lag_cat = ones(Int, D, N)
    for (j, y_past) in enumerate(Y_past)
        order = length(y_past)
        order == 0 && continue
        for m = 1:order
            w = 2^(order - m)
            for n = 1:N
                lag_cat[j, n] += w * (n > m ? Y[j, n-m] : y_past[order+n-m])
            end
        end
    end
    return lag_cat
end

n_per_category(h, t, y, n_in_t, n_occurence_history::AbstractMatrix) = (n_in_t[t] ∩ n_occurence_history[h, y])

function randARPeriodicHMM(rng::AbstractRNG, K, T, orders::AbstractVector; ref_station=1, ξ=ones(K) / K)
    D = length(orders)
    size_order = 2 .^ orders
    B_rand = [[Bernoulli(rand(rng)) for h in 1:size_order[j]] for k in 1:K, t in 1:T, j in 1:D] # completly random -> bad
    Q_rand = zeros(K, K, T)
    for t in 1:T
        Q_rand[:, :, t] = randtransmat(rng, K) # completly random -> bad
    end
    hmm_random = sARPeriodicHMM(ξ, Q_rand, B_rand)
    sort_wrt_ref!(hmm_random, ref_station)
    return hmm_random
end

randARPeriodicHMM(K, T, order; ref_station=1, ξ=ones(K) / K) = randARPeriodicHMM(GLOBAL_RNG, K, T, order; ref_station=ref_station, ξ=ξ)

function idx_observation_of_past_cat(lag_cat, size_order::AbstractVector)
    D = size(lag_cat, 1)
    idx_j = Vector{Vector{Vector{Int}}}(undef, D)
    for j = 1:D
        idx_j[j] = [findall(view(lag_cat, j, :) .== m) for m = 1:size_order[j]]
    end
    return idx_j
end

function sort_wrt_ref!(α::AbstractVector, B::AbstractMatrix, ref_station)
    K = size(α, 1)
    sorting = [succprob(B[k, ref_station][1]) for k = 1:K] # 1 is by my convention the driest category i.e. Y|d....d
    B[:, :] = B[sortperm(sorting, rev=true), :]
    α[:] = α[sortperm(sorting, rev=true)]
end