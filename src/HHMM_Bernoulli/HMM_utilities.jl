argmaxrow(A::AbstractMatrix{<:Real}) = [argmax(A[i, :]) for i = axes(A, 1)]

function cycle(t, T)
    if 1 ≤ t ≤ T
        return t
    elseif T < t ≤ 2T
        return t - T
    elseif -T < t ≤ 0
        return t + T
    else
        println("Not implemented")
    end
end

n_per_category(s, h, t, y, n_in_t, n_occurence_history) = (n_in_t[t] ∩ n_occurence_history[s, h, y])

bin2digit(x) = sum(x[length(x)-i+1] * 2^(i - 1) for i = 1:length(x)) + 1
bin2digit(x::Tuple) = bin2digit([x...])

function dayx(lag_obs::AbstractArray)
    order = length(lag_obs)
    t = tuple.([lag_obs[m] for m = 1:order]...)
    bin2digit.(t)
end

function conditional_to(Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool})
    order = size(Y_past, 1)
    if order == 0
        return ones(Int, size(Y))
    else
        lag_obs = [copy(lag(Y, m)) for m = 1:order]  # transform dry=0 into 1 and wet=1 into 2 for array indexing
        for m = 1:order
            lag_obs[m][1:m, :] .= reverse(Y_past[1:m, :], dims=1) # avoid the missing first row
        end
        return dayx(lag_obs)
    end
end

function idx_observation_of_past_cat(lag_cat, n2t, T, size_order)
    # Matrix(T,D) of vector that give the index of data of same Y_past.
    # ie. size_order = 1 (no order) -> every data is in category 1
    # ie size_order = 2 (order on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
    D = size(lag_cat, 2)
    idx_tj = Matrix{Vector{Vector{Int}}}(undef, T, D)
    n_in_t = [findall(n2t .== t) for t = 1:T] # could probably be speeded up e.g. recusivly suppressing already classified label with like sortperm
    for t in OneTo(T)
        n_t = n_in_t[t]
        for j = 1:D
            n_tm = [findall(lag_cat[n_t, j] .== m) for m = 1:size_order]
            idx_tj[t, j] = [n_t[n_tm[m]] for m = 1:size_order]
            ##
        end
    end
    return idx_tj
end

function idx_observation_of_past_cat(lag_cat, size_order)
    # Matrix(T,D) of vector that give the index of data of same past.
    # ie. size_order = 1 (no order) -> every data is in category 1
    # ie size_order = 2 (order on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
    D = size(lag_cat, 2)
    idx_j = Vector{Vector{Vector{Int}}}(undef, D)
    for j = 1:D
        idx_j[j] = [findall(lag_cat[:, j] .== m) for m = 1:size_order]
    end
    return idx_j
end

random_product_Bernoulli(rng::AbstractRNG, args...) = Bernoulli.(rand(rng, args...))

#TODO add to rand(HierarchicalPeriodicHMM, blabla)
function randhierarchicalPeriodicHMM(rng::AbstractRNG, K, T, D, order; ref_station=1, ξ=ones(K) / K)
    size_order = 2^order
    B_rand = random_product_Bernoulli(rng, K, T, D, size_order)  # completly random -> bad
    Q_rand = zeros(K, K, T)
    for t in 1:T
        Q_rand[:, :, t] = PeriodicHiddenMarkovModels.randtransmat(rng, K) # completly random -> bad
    end
    hmm_random = HierarchicalPeriodicHMM(ξ, Q_rand, B_rand)
    sort_wrt_ref!(hmm_random, ref_station)
    return hmm_random
end

randhierarchicalPeriodicHMM(K, T, D, order; ref_station=1, ξ=ones(K) / K) = randhierarchicalPeriodicHMM(GLOBAL_RNG, K, T, D, order; ref_station=ref_station, ξ=ξ)

# TODO: site dependent order #
# function conditional_to(Y::AbstractArray, order::AbstractVector;
#     Y_past=[0 1 0 1 1 0 1 0 0 0
#         1 1 0 1 1 1 1 1 1 1
#         1 1 0 1 1 1 0 1 1 1
#         1 1 0 1 1 0 0 0 1 0
#         1 1 0 1 1 0 0 1 0 1]
# )
#     D = size(Y, 2)
#     lag_cat = zeros(Int, size(Y))
#     for j = 1:D
#         lag_cat[:, j] = conditional_to(Y[:, j], order[j], Y_past=Y_past[:, j])
#     end
#     return lag_cat
# end
