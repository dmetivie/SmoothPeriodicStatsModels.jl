"""
    conditional_to(Y, Y_past::AbstractVector) -> Matrix{Int}

Memory category `lag_cat[j, n]` for sites that each carry their own memory order: `Y` is `D × N`
and `Y_past[j]` is the initial condition of site `j`, most recent value first.
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
                lag_cat[j, n] += w * (n > m ? Y[j, n-m] : y_past[m-n+1])
            end
        end
    end
    return lag_cat
end

n_per_category(h, t, y, n_in_t, n_occurence_history::AbstractMatrix) = (n_in_t[t] ∩ n_occurence_history[h, y])

function randARPeriodicHMM(rng::AbstractRNG, K, T, orders::AbstractVector; ref_station=1, ξ=ones(K) / K)
    D = length(orders)
    size_order = 2 .^ orders
    B_rand = [[Bernoulli(rand()) for h in 1:size_order[j]] for k in 1:K, t in 1:T, j in 1:D] # completly random -> bad
    Q_rand = zeros(K, K, T)
    for t in 1:T
        Q_rand[:, :, t] = PeriodicHiddenMarkovModels.randtransmat(rng, K) # completly random -> bad
    end
    hmm_random = sARPeriodicHMM(ξ, Q_rand, B_rand)
    sort_wrt_ref!(hmm_random, ref_station)
    return hmm_random
end

randARPeriodicHMM(K, T, order; ref_station=1, ξ=ones(K) / K) = randARPeriodicHMM(GLOBAL_RNG, K, T, order; ref_station=ref_station, ξ=ξ)

function idx_observation_of_past_cat(lag_cat, size_order::AbstractVector)
    # Matrix(T,D) of vector that give the index of data of same past.
    # ie. size_order = 1 (no order) -> every data is in category 1
    # ie size_order = 2 (order on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
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