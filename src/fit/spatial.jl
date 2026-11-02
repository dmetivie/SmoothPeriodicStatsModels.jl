# A and B are updated using the code from HHMM - update_A_B_jump.jl.

# The 164-line Baum-Welch loop that used to follow is now `HiddenMarkovModels.forward_backward!`
# driven from `src/fit/smooth_hmm.jl`, where this package writes only the M-step as
# `StatsAPI.fit!`. The two were checked against each other the only way a *stochastic*
# likelihood allows: given the same posteriors, the retired `update_a!`/`update_A!`/`update_B!`/
# `update_R!` sequence and the new `fit!` produce the same coefficients to machine precision
# (θᴬ 2.8e-16, θᴮ 6.1e-16, θᴿ 5.3e-16). Comparing end-to-end optima would have been meaningless:
# `mvnormcdf` is randomized, so two runs never agree exactly.
#
# `update_R!` and `fit_mle_one_R!` below stay — they are the spatial M-step, and the only one
# this model does not share with its siblings.


function fit_mle_one_R!(theta_R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 2), size(B, 1))::AbstractVector{<:Integer}, solver, adtype=default_adtype(), return_sol=false, solve_kwargs=(;))
    T = size(B, 1)
    pairwise_indices2 = Tuple.(findall(wp .> 0))
    size_order = size(B, 3)  # 1 for no memory, 2^order for memory >= 1

    function optimfunction2(u, p)
        Rt = similar(u, T)
        for t in 1:T
            Rt[t] = exp(polynomial_trigo(t / T, u))
        end
        if size_order == 1
            return -pairwise_loglikelihood(Rt, B[:, :, 1], p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2)
        else
            return -pairwise_loglikelihood_memory(Rt, B, p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2)
        end
    end
    optf2 = OptimizationFunction((u, p) -> optimfunction2(u, p), adtype)
    u0 = collect(theta_R)

    prob = OptimizationProblem(optf2, u0, [Y, n_pair, h, wp])

    sol = solve(prob, solver; solve_kwargs...)

    if !successful_retcode(sol.retcode)
        @warn "sol.retcode = $(sol.retcode)"
    end

    theta_R[:] .= view(sol.u, :)
end

function update_R!(hmm::ARPeriodicHMMSpatial,
    Range_θ::AbstractArray{N,2} where {N},
    γ::AbstractMatrix, wp, Y, SituationIdx::AbstractArray{<:Integer,3}; n2t=n_to_t(size(Y, 2), size(hmm, 3))::AbstractVector{<:Integer}, solver, adtype=default_adtype(), maxiters=10, solve_kwargs=(;))
    @argcheck size(γ, 1) == size(Y, 2)
    N_obs = size(γ, 1)
    R = hmm.R
    D = size(hmm, 2)
    K = size(R, 1)
    T = size(R, 2)
    size_order = size(hmm, 4)
    n_situations = 4 * size_order^2
    pairwise_indices2 = Tuple.(findall(wp .> 0))

    @threads for k in 1:K
        B = hmm.B[k, :, :, :]  # B[t, d, lag_cat]
        h = hmm.h
        w = γ[:, k]
        n_pair = zeros(eltype(R), n_situations, D, D, T)

        @inbounds for tk in 1:N_obs
            t = n2t[tk]
            for (i, j) in pairwise_indices2
                n_pair[SituationIdx[tk, i, j], i, j, t] += w[tk]
            end
        end

        fit_mle_one_R!(view(Range_θ, k, :), B, h, Y, wp, n_pair; n2t=n2t, solver, adtype,
            solve_kwargs=merge((; maxiters), solve_kwargs))
    end

    for k in 1:K
        R[k, :] .= [exp(polynomial_trigo(t / T, view(Range_θ, k, :))) for t = 1:T]
    end
end
