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

n_per_category(s, h, t, y, n_in_t, n_occurence_history::AbstractArray{<:AbstractVector, 3}) = (n_in_t[t] ∩ n_occurence_history[s, h, y])

"""
    bin2digit(x) -> Int

Index in `1:2^length(x)` of the binary pattern `x`, read with `x[1]` as the most significant bit
and counted from 1:

```julia
bin2digit([0, 0]) == 1    bin2digit([0, 1]) == 2
bin2digit([1, 0]) == 3    bin2digit([1, 1]) == 4
```

Used for the last index `h` of an emission array `B[k, t, j, h]`, where `x[m]` is the value of
the site `m` steps earlier, so `h = 1` is the all-dry history.

Any `x` supporting `length` and `getindex` works, `Tuple` included.
"""
bin2digit(x) = sum(x[length(x)-i+1] * 2^(i - 1) for i = 1:length(x)) + 1

"""
    dayx(lag_obs) -> Matrix{Int}

Memory category of every (site, observation), from `lag_obs[m][j, n]`, the value of site `j` at
`m` steps before `n`. Entry `[j, n]` is `bin2digit((lag_obs[1][j, n], …, lag_obs[order][j, n]))`,
that is `1 + sum(lag_obs[m][j, n] * 2^(order - m) for m = 1:order)`.
"""
function dayx(lag_obs::AbstractArray)
    # Accumulated in place rather than as `bin2digit.(tuple.(lag_obs...))`: splatting a `Vector`
    # whose length is only known at run time makes that broadcast uninferrable.
    order = length(lag_obs)
    lag_cat = fill(1, size(lag_obs[1]))
    for m = 1:order
        w = 2^(order - m)
        lag_obs_m = lag_obs[m]
        for i in eachindex(lag_cat, lag_obs_m)
            lag_cat[i] += w * lag_obs_m[i]
        end
    end
    return lag_cat
end

"""
    conditional_to(Y, Y_past) -> Matrix{Int}

Memory category `lag_cat[j, n]` of every site `j` at every observation `n`, from the `D × N`
observations `Y` and the `D × order` initial condition `Y_past`, whose column `m` is the value
`m` steps before the first observation.
"""
function conditional_to(Y::AbstractMatrix{<:Bool}, Y_past::AbstractMatrix{<:Bool})
    D, N = size(Y)
    order = size(Y_past, 2)
    if order == 0
        return ones(Int, D, N)
    else
        # lag_obs[m][j, n] is the value of site j, m steps before n; the first m observations
        # take it from the initial condition instead
        lag_obs = [[n > m ? Y[j, n-m] : Y_past[j, m-n+1] for j = 1:D, n = 1:N] for m = 1:order]
        return dayx(lag_obs)
    end
end

function idx_observation_of_past_cat(lag_cat, n2t, T, size_order)
    # Matrix(T,D) of vector that give the index of data of same Y_past.
    # ie. size_order = 1 (no order) -> every data is in category 1
    # ie size_order = 2 (order on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
    D = size(lag_cat, 1)
    idx_tj = Matrix{Vector{Vector{Int}}}(undef, T, D)
    n_in_t = [findall(n2t .== t) for t = 1:T] # could probably be speeded up e.g. recusivly suppressing already classified label with like sortperm
    for t in OneTo(T)
        n_t = n_in_t[t]
        for j = 1:D
            n_tm = [findall(view(lag_cat, j, n_t) .== m) for m = 1:size_order]
            idx_tj[t, j] = [n_t[n_tm[m]] for m = 1:size_order]
            ##
        end
    end
    return idx_tj
end

function idx_observation_of_past_cat(lag_cat, size_order::Integer)
    # Matrix(T,D) of vector that give the index of data of same past.
    # ie. size_order = 1 (no order) -> every data is in category 1
    # ie size_order = 2 (order on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
    D = size(lag_cat, 1)
    idx_j = Vector{Vector{Vector{Int}}}(undef, D)
    for j = 1:D
        idx_j[j] = [findall(view(lag_cat, j, :) .== m) for m = 1:size_order]
    end
    return idx_j
end

random_product_Bernoulli(rng::AbstractRNG, args...) = Bernoulli.(rand(rng, args...))

#TODO add to rand(ARPeriodicHMM, blabla)
function randARPeriodicHMM(rng::AbstractRNG, K::Integer, T::Integer, D::Integer, order::Integer; ref_station=1, ξ=ones(K) / K)
    size_order = 2^order
    B_rand = random_product_Bernoulli(rng, K, T, D, size_order)  # completly random -> bad
    Q_rand = zeros(K, K, T)
    for t in 1:T
        Q_rand[:, :, t] = PeriodicHiddenMarkovModels.randtransmat(rng, K) # completly random -> bad
    end
    hmm_random = ARPeriodicHMM(ξ, Q_rand, B_rand)
    sort_wrt_ref!(hmm_random, ref_station)
    return hmm_random
end

randARPeriodicHMM(K, T, D, order; ref_station=1, ξ=ones(K) / K) = randARPeriodicHMM(GLOBAL_RNG, K, T, D, order; ref_station=ref_station, ξ=ξ)

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
