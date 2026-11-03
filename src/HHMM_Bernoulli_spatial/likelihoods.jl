function likelihoods!(L::AbstractMatrix, hmm::ARPeriodicHMMSpatial, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(L, 1), size(hmm, 3))::AbstractVector{<:Integer}, QMC_m=30)
    N, K, D = size(Y, 2), size(hmm, 1), size(hmm, 2)
    @argcheck size(L) == (N, K)

    for i in 1:K
        # @show i  # debug
        for n in 1:N
            t = n2t[n] # periodic t
            modelit = SpatialBernoulli(hmm.R[i, t], hmm.B[CartesianIndex.(i, t, 1:D, view(lag_cat, :, n))], hmm.h)

            L[n, i] = pdf(modelit, view(Y, :, n); m=D * QMC_m)
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::ARPeriodicHMMSpatial, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer}, QMC_m=30)
    N, K, D, T = size(Y, 2), size(hmm, 1), size(hmm, 2), size(hmm, 3)
    @argcheck size(LL) == (N, K)

    Sigmat = zeros(D, D, T, K)
    @threads for t in 1:T
        for k in 1:K
            Sigmat[:, :, t, k] = expkernel.(hmm.h; range=hmm.R[k, t])
        end
    end

    @threads for n in 1:N
        # Preallocate vectors outside loops to avoid repeated allocations
        a = fill(-Inf, D)
        b = fill(Inf, D)
        finite_bounds = zeros(D)
        zerosvec = zeros(D)
        for i in 1:K
            # @show n
            t = n2t[n] # periodic t
            finite_bounds .= quantile.(Normal(), hmm.B[CartesianIndex.(i, t, 1:D, view(lag_cat, :, n))])
            a .= ifelse.(view(Y, :, n) .== 0, finite_bounds, -Inf)
            b .= ifelse.(view(Y, :, n) .== 1, finite_bounds, Inf)
            hy = mvnormcdf(zerosvec, Sigmat[:, :, t, i], a, b; m=D * QMC_m)
            LL[n, i] = log(hy[1])
        end
    end
end

function loglikelihoods(hmm::ARPeriodicHMMSpatial, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}; robust=false, n2t=n_to_t(size(Y, 2), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m=30)
    N, K = size(Y, 2), size(hmm, 1)
    LL = Matrix{Float64}(undef, N, K)

    lag_cat = conditional_to(Y, Y_past)

    loglikelihoods!(LL, hmm, Y, lag_cat; n2t=n2t, QMC_m=QMC_m)
    if robust
        replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    end
    return LL
end

# function complete_loglikelihood(hmm::ARPeriodicHMMSpatial, y::AbstractArray, y_past::AbstractArray, z::AbstractVector; n2t = n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m = 30)
# 	N, D = size(y)
# 	lag_cat = conditional_to(y, y_past)
# 	return sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n ∈ 1:(N-1)) + sum(logpdf(SpatialBernoulli(hmm.R[z[n], n2t[n]], hmm.B[CartesianIndex.(z[n], n2t[n], 1:D, lag_cat[n, :])], hmm.h), y[n, :]; m = D * QMC_m) for n ∈ 1:N)
# end

nb_param_HMMSpa(K, memory, d, D) = (2d + 1) * (K * 2^memory * D + K * (K - 1) + K)

"""
    complete_loglikelihood(hmm::ARPeriodicHMMSpatial, y, y_past, z; n2t, QMC_m)

Loglikelihood of the observations *and* the state path `z`: the transition terms, indexed by
the period each step leaves, plus one `D`-dimensional orthant probability per observation.

Three things were wrong with this before and are worth recording, because none of them could
have been caught by running it — it could not run at all. `D, N = size(y)` was written the
other way round, so `D` and `N` were swapped relative to the `D × N` layout the rest of the
file uses; `logpdf` was not defined for `SpatialBernoulli` at all (only `pdf` was), so the call
was a `MethodError`; and the sums accumulated into `Threads.threadid()`-indexed slots, which
has been unsound since Julia 1.8 because a task may migrate between threads mid-loop. The
reductions below are per-chunk and need no thread identity.
"""
function complete_loglikelihood(hmm::ARPeriodicHMMSpatial, y::AbstractArray, y_past::AbstractArray, z::AbstractVector; n2t=n_to_t(size(y, 2), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m=hmm.QMC_m)
    D, N = size(y)
    lag_cat = conditional_to(y, y_past)

    log_A_sum = sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n = 1:(N-1); init=0.0)

    # The latent-field correlation depends only on `(state, period)`, so there are `K·T` of
    # them however long the series is. Building one per observation instead is what the
    # three-argument `SpatialBernoulli` constructor does, and what this function used to do
    # through it.
    #
    # This is an *allocation* win, not a speed one: measured over three interleaved rounds at
    # `D = 10..20`, `N = 300..3000`, the time is indistinguishable (0.93-1.09x) while the
    # allocation falls about 8% (120.7 -> 111.3 MiB at `D = 20, N = 3000`). Timing the rebuild
    # on its own suggests it is 11% of the runtime, and that is misleading: in here it happens
    # inside the spawned tasks, where it overlaps with the `mvnormcdf` that actually dominates.
    ΣU = [expkernel.(hmm.h; range=hmm.R[k, t]) for k in axes(hmm.R, 1), t in axes(hmm.R, 2)]

    # One partial sum per chunk of observations, combined at the end: no `threadid()`, so a
    # task that migrates cannot corrupt a neighbour's accumulator. `mvnormcdf` draws from
    # `Random.default_rng()`, which is task-local, so the tasks do not share a stream either.
    chunks = Iterators.partition(1:N, max(1, cld(N, Threads.nthreads())))
    partials = map(collect(chunks)) do ns
        Threads.@spawn begin
            acc = 0.0
            for n in ns
                t = n2t[n]
                d = SpatialBernoulli(hmm.R[z[n], t],
                    hmm.B[CartesianIndex.(z[n], t, 1:D, view(lag_cat, :, n))],
                    hmm.h, ΣU[z[n], t])
                acc += logpdf(d, view(y, :, n); m=D * QMC_m)
            end
            acc
        end
    end
    return log_A_sum + sum(fetch, partials)
end

# for pairwise likelihood : 

function pairwise_loglikelihood(R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 2), size(R, 1))::AbstractVector{<:Integer}, eps=1e-10, pairwise_indices2=Tuple.(findall(wp .> 0))
)
    D, N = size(Y)
    T = size(R, 1)

    Iij = fill(convert(eltype(R), NaN), 4, D, D, T)
    @inbounds for t in 1:T
        for (i, j) in pairwise_indices2
            # @show (i,j)
            B_ij = @view B[t, [i, j]]
            h_ij = @view h[[i, j], [i, j]]
            if i == j
                Iij[1, i, j, t] = B_ij[1]
                Iij[4, i, j, t] = 1 - B_ij[1]
            else
                Iij[1, i, j, t] = ifelse(!isnan(Iij[1, j, i, t]), Iij[1, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1]), quantile(Normal(), B_ij[2]), exp(-h_ij[1, 2] / R[t])))
                Iij[2, i, j, t] = ifelse(!isnan(Iij[3, j, i, t]), Iij[3, j, i, t], B_ij[1] - Iij[1, i, j, t])
                Iij[3, i, j, t] = ifelse(!isnan(Iij[2, j, i, t]), Iij[2, j, i, t], B_ij[2] - Iij[1, i, j, t])
                Iij[4, i, j, t] = ifelse(i == j, 1.0 - Iij[1, i, j, t], 1.0 - Iij[1, i, j, t] - Iij[2, i, j, t] - Iij[3, i, j, t])
            end
        end
    end

    Iij .= max.(Iij, eps)  # Replace elements < eps with eps

    pairwise_sum = 0.0
    @inbounds for (i, j) in pairwise_indices2
        for t in 1:T
            if i != j
                pairwise_sum += wp[i, j] * n_pair[1, i, j, t] * log(Iij[1, i, j, t]) +
                                wp[i, j] * n_pair[2, i, j, t] * log(Iij[2, i, j, t]) +
                                wp[i, j] * n_pair[3, i, j, t] * log(Iij[3, i, j, t]) +
                                wp[i, j] * n_pair[4, i, j, t] * log(Iij[4, i, j, t])
            else
                pairwise_sum += wp[i, j] * n_pair[1, i, j, t] * log(Iij[1, i, j, t]) +
                                wp[i, j] * n_pair[4, i, j, t] * log(Iij[4, i, j, t])
            end
        end
    end

    return pairwise_sum
end

function pairwise_loglikelihood_memory(R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 2), size(R, 1))::AbstractVector{<:Integer}, eps=1e-10, pairwise_indices2=Tuple.(findall(wp .> 0)))
    D, N = size(Y)
    T = size(R, 1)
    size_order = size(B, 3)  # B is (T, D, size_order)
    n_situations = 4 * size_order^2

    Iij = fill(convert(eltype(R), NaN), n_situations, D, D, T)
    @inbounds for t in 1:T
        for (i, j) in pairwise_indices2
            h_ij = @view h[[i, j], [i, j]]
            rho = exp(-h_ij[1, 2] / R[t])

            for lag_cat_i in 1:size_order
                for lag_cat_j in 1:size_order
                    # block index: ordered so (size_order, size_order) → block 0, ..., (1, 1) → last block
                    block_ij = (size_order - lag_cat_i) * size_order * 4 + (size_order - lag_cat_j) * 4
                    block_ji = (size_order - lag_cat_j) * size_order * 4 + (size_order - lag_cat_i) * 4
                    B_i = B[t, i, lag_cat_i]
                    B_j = B[t, j, lag_cat_j]

                    if i == j
                        if lag_cat_i == lag_cat_j
                            Iij[block_ij+1, i, j, t] = B_i          # P(1,1) = P(1) since i == j
                            Iij[block_ij+4, i, j, t] = 1 - B_i      # P(0,0) = P(0)
                        end
                        # cross lag-cat blocks at i == j are never observed; leave as NaN
                    else
                        # P(1,1): reuse from symmetric (j,i) block (lag_cat_j, lag_cat_i) if already computed
                        p11 = ifelse(!isnan(Iij[block_ji+1, j, i, t]),
                            Iij[block_ji+1, j, i, t],
                            norm_cdf_2d_vfast(quantile(Normal(), B_i), quantile(Normal(), B_j), rho))
                        Iij[block_ij+1, i, j, t] = p11
                        # P(1,0): reuse as P(0,1) from (j,i)
                        Iij[block_ij+2, i, j, t] = ifelse(!isnan(Iij[block_ji+3, j, i, t]),
                            Iij[block_ji+3, j, i, t], B_i - p11)
                        # P(0,1): reuse as P(1,0) from (j,i)
                        Iij[block_ij+3, i, j, t] = ifelse(!isnan(Iij[block_ji+2, j, i, t]),
                            Iij[block_ji+2, j, i, t], B_j - p11)
                        # P(0,0)
                        Iij[block_ij+4, i, j, t] = 1.0 - Iij[block_ij+1, i, j, t] - Iij[block_ij+2, i, j, t] - Iij[block_ij+3, i, j, t]
                    end
                end
            end
        end
    end
    Iij .= max.(Iij, eps)  # Replace elements < eps with eps

    pairwise_sum = 0.0
    @inbounds for (i, j) in pairwise_indices2
        for t in 1:T
            if i != j
                pairwise_sum += wp[i, j] * sum(n_pair[k, i, j, t] * log(Iij[k, i, j, t]) for k in 1:n_situations)
            else
                # only diagonal lag-cat blocks contribute (cross-lag pairs never observed for i == j)
                for lag_cat in 1:size_order
                    block = (size_order - lag_cat) * size_order * 4 + (size_order - lag_cat) * 4
                    pairwise_sum += wp[i, j] * (n_pair[block+1, i, j, t] * log(Iij[block+1, i, j, t]) +
                                                n_pair[block+4, i, j, t] * log(Iij[block+4, i, j, t]))
                end
            end
        end
    end
    return pairwise_sum
end


