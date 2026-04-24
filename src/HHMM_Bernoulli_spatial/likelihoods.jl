

function likelihoods!(L::AbstractMatrix, hmm::PeriodicHMMSpaMemory, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t = n_to_t(size(L, 1), size(hmm, 3))::AbstractVector{<:Integer}, QMC_m = 30)
	N, K, D = size(Y, 1), size(hmm, 1), size(hmm, 2)
	@argcheck size(L) == (N, K)

	for i in 1:K
		# @show i  # debug
		for n in 1:N
			t = n2t[n] # periodic t
			modelit = SpatialBernoulli(hmm.R[i, t], hmm.B[CartesianIndex.(i, t, 1:D, lag_cat[n, :])], hmm.h)

			L[n, i] = pdf(modelit, Y[n, :]; m = D * QMC_m)
		end
	end
end



function loglikelihoods!(LL::AbstractMatrix, hmm::PeriodicHMMSpaMemory, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t = n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer}, QMC_m = 30)
	N, K, D, T = size(Y, 1), size(hmm, 1), size(hmm, 2), size(hmm, 3)
	@argcheck size(LL) == (N, K)

	Sigmat = zeros(D, D, T, K)
	@threads for t in 1:T
		for k in 1:K
			Sigmat[:, :, t, k] = expkernel.(hmm.h; range = hmm.R[k, t])
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
			finite_bounds .= quantile.(Normal(), hmm.B[CartesianIndex.(i, t, 1:D, lag_cat[n, :])])
			a .= ifelse.(Y[n, :] .== 0, finite_bounds, -Inf)
			b .= ifelse.(Y[n, :] .== 1, finite_bounds, Inf)
			hy = mvnormcdf(zerosvec, Sigmat[:, :, t, i], a, b; m = D * QMC_m)
			LL[n, i] = log(hy[1])
		end
	end
end



function loglikelihoods(hmm::PeriodicHMMSpaMemory, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}; robust = false, n2t = n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m = 30)
	N, K = size(Y, 1), size(hmm, 1)
	LL = Matrix{Float64}(undef, N, K)

	lag_cat = conditional_to(Y, Y_past)

	loglikelihoods!(LL, hmm, Y, lag_cat; n2t = n2t, QMC_m = QMC_m)
	if robust
		replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
	end
	return LL
end




# function complete_loglikelihood(hmm::PeriodicHMMSpaMemory, y::AbstractArray, y_past::AbstractArray, z::AbstractVector; n2t = n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m = 30)
# 	N, D = size(y)
# 	lag_cat = conditional_to(y, y_past)
# 	return sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n ∈ 1:(N-1)) + sum(logpdf(SpatialBernoulli(hmm.R[z[n], n2t[n]], hmm.B[CartesianIndex.(z[n], n2t[n], 1:D, lag_cat[n, :])], hmm.h), y[n, :]; m = D * QMC_m) for n ∈ 1:N)
# end

nb_param_HMMSpa(K, memory, d, D) = (2d + 1) * (K * 2^memory * D + K * (K - 1) + K)




function complete_loglikelihood(hmm::PeriodicHMMSpaMemory, y::AbstractArray, y_past::AbstractArray, z::AbstractVector; n2t = n_to_t(size(y, 1), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m = 30)
	N, D = size(y)
	lag_cat = conditional_to(y, y_past)

	# Parallelizing the two summations
	log_A_sum_threads = zeros(Float64, Threads.nthreads())
	@threads for n ∈ 1:(N-1)
		tid = Threads.threadid()
		log_A_sum_threads[tid] += log(hmm.A[z[n], z[n+1], n2t[n]])
	end
	log_A_sum = sum(log_A_sum_threads)


	logpdf_sum_threads = zeros(Float64, Threads.nthreads())
	@threads for n ∈ 1:N
		tid = Threads.threadid()
		logpdf_sum_threads[tid] += logpdf(
			SpatialBernoulli(hmm.R[z[n], n2t[n]],
				hmm.B[CartesianIndex.(z[n], n2t[n], 1:D, lag_cat[n, :])],
				hmm.h),
			y[n, :]; m = D * QMC_m,
		)
	end
	logpdf_sum = sum(logpdf_sum_threads)


	return log_A_sum + logpdf_sum
end

# for pairwise likelihood : 


function my_loglikelihood(R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t = n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, eps = 1e-10, pairwise_indices2 = Tuple.(findall(wp .> 0))
)
	N, D = size(Y)
	T = size(R, 1)
	# println("T = size(R,1)",T)
	# @show R

	Iij = ones(eltype(R), 4, D, D, T)
	@inbounds for t in 1:T



		for (i, j) in pairwise_indices2
			# @show (i,j)
			B_ij = @view B[t, [i, j]]
			h_ij = @view h[[i, j], [i, j]]
			if i == j
				Iij[1, i, j, t] = B_ij[1]
				Iij[4, i, j, t] = 1 - B_ij[1]
			else
				Iij[1, i, j, t] = ifelse(Iij[1, j, i, t] != 1.0, Iij[1, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1]), quantile(Normal(), B_ij[2]), exp(-h_ij[1, 2] / R[t])))
				Iij[2, i, j, t] = ifelse(Iij[3, j, i, t] != 1.0, Iij[3, j, i, t], B_ij[1] - Iij[1, i, j, t])
				Iij[3, i, j, t] = ifelse(Iij[2, j, i, t] != 1.0, Iij[2, j, i, t], B_ij[2] - Iij[1, i, j, t])
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

	return (pairwise_sum)
end

function my_loglikelihood_memory1(R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t = n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, eps = 1e-10, pairwise_indices2 = Tuple.(findall(wp .> 0)))
	N, D = size(Y)
	T = size(R, 1)
	# println("T = size(R,1)",T)
	# @show R


	Iij = ones(eltype(R), 16, D, D, T)
	@inbounds for t in 1:T
		for (i, j) in pairwise_indices2
			# @show (i,j)
			h_ij = @view h[[i, j], [i, j]]

			B_ij = @view B[t, [i, j], :]

			if i == j
				Iij[1, i, j, t] = B_ij[1, 2]
				Iij[4, i, j, t] = 1 - B_ij[1, 2]
				Iij[13, i, j, t] = B_ij[1, 1]
				Iij[16, i, j, t] = 1 - B_ij[1, 1]
			end
			if i != j
				Iij[1, i, j, t] = ifelse(Iij[1, j, i, t] != 1.0, Iij[1, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1, 2]), quantile(Normal(), B_ij[2, 2]), exp(-h_ij[1, 2] / R[t])))
				Iij[2, i, j, t] = ifelse(Iij[3, j, i, t] != 1.0, Iij[3, j, i, t], B_ij[1, 2] - Iij[1, i, j, t])
				Iij[3, i, j, t] = ifelse(Iij[2, j, i, t] != 1.0, Iij[2, j, i, t], B_ij[2, 2] - Iij[1, i, j, t])
				Iij[4, i, j, t] = ifelse(i == j, 1.0 - Iij[1, i, j, t], 1.0 - Iij[1, i, j, t] - Iij[2, i, j, t] - Iij[3, i, j, t])

				Iij[5, i, j, t] = ifelse(Iij[9, j, i, t] != 1.0, Iij[9, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1, 2]), quantile(Normal(), B_ij[2, 1]), exp(-h_ij[1, 2] / R[t])))
				Iij[6, i, j, t] = ifelse(Iij[11, j, i, t] != 1.0, Iij[11, j, i, t], B_ij[1, 2] - Iij[5, i, j, t])
				Iij[7, i, j, t] = ifelse(Iij[10, j, i, t] != 1.0, Iij[10, j, i, t], B_ij[2, 1] - Iij[5, i, j, t])
				Iij[8, i, j, t] = 1.0 - Iij[5, i, j, t] - Iij[6, i, j, t] - Iij[7, i, j, t]


				Iij[9, i, j, t] = ifelse(Iij[5, j, i, t] != 1.0, Iij[5, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1, 1]), quantile(Normal(), B_ij[2, 2]), exp(-h_ij[1, 2] / R[t])))
				Iij[10, i, j, t] = ifelse(Iij[7, j, i, t] != 1.0, Iij[7, j, i, t], B_ij[1, 1] - Iij[9, i, j, t])
				Iij[11, i, j, t] = ifelse(Iij[6, j, i, t] != 1.0, Iij[6, j, i, t], B_ij[2, 2] - Iij[9, i, j, t])
				Iij[12, i, j, t] = 1.0 - Iij[9, i, j, t] - Iij[10, i, j, t] - Iij[11, i, j, t]

				Iij[13, i, j, t] = ifelse(Iij[13, j, i, t] != 1.0, Iij[13, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1, 1]), quantile(Normal(), B_ij[2, 1]), exp(-h_ij[1, 2] / R[t])))
				Iij[14, i, j, t] = ifelse(Iij[15, j, i, t] != 1.0, Iij[15, j, i, t], B_ij[1, 1] - Iij[13, i, j, t])
				Iij[15, i, j, t] = ifelse(Iij[14, j, i, t] != 1.0, Iij[14, j, i, t], B_ij[2, 1] - Iij[13, i, j, t])
				Iij[16, i, j, t] = ifelse(i == j, 1.0 - Iij[13, i, j, t], 1.0 - Iij[13, i, j, t] - Iij[14, i, j, t] - Iij[15, i, j, t])
			end
		end
	end
	Iij .= max.(Iij, eps)  # Replace elements < eps with eps

	# bad_indices = findall(Iij .< 0)
	# println(bad_indices)




	pairwise_sum = 0.0
	@inbounds for (i, j) in pairwise_indices2
		for t in 1:T
			if i != j
				pairwise_sum += wp[i, j] * sum(n_pair[k, i, j, t] * log(Iij[k, i, j, t]) for k in 1:16)
			else
				pairwise_sum += wp[i, j] * n_pair[1, i, j, t] * log(Iij[1, i, j, t]) +
								wp[i, j] * n_pair[4, i, j, t] * log(Iij[4, i, j, t]) +
								wp[i, j] * n_pair[13, i, j, t] * log(Iij[13, i, j, t]) +
								wp[i, j] * n_pair[16, i, j, t] * log(Iij[16, i, j, t])
			end
		end
	end
	return (pairwise_sum)
end


