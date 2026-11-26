# The spatial range M-step. The transitions and marginal probabilities are refitted by the
# M-steps shared with the other HMMs, in `src/fit/smooth_hmm.jl`.

"""
    fit_mle_one_R!(θᴿ, B, h, Y, wp, n_pair; n2t, basis, solver, adtype, solve_kwargs, additional_loss, index) -> θᴿ

Refit in place the spatial range coefficients `θᴿ` of one hidden state. The range at period
index `t` is `exp(basis[t, :] ⋅ θᴿ)`, and the fit maximizes the weighted pairwise loglikelihood
([`pairwise_loglikelihood`](@ref), or [`pairwise_loglikelihood_memory`](@ref) when `B`, the
`T × D × 2^order` marginal probabilities, has memory). `n_pair` holds the posterior-weighted
counts of joint outcomes and `wp` the pair weights. `basis`, the `T × length(θᴿ)` design
matrix, defaults to the trigonometric one. An
[`additional_loss`](@ref SmoothPeriodicStatsModels.additional_loss) is called with `part = :R` and `index`.
"""
function fit_mle_one_R!(theta_R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 2), size(B, 1))::AbstractVector{<:Integer}, basis=trig_basis(size(B, 1), (length(theta_R) - 1) ÷ 2), solver, adtype=default_adtype(), return_sol=false, solve_kwargs=(;), additional_loss=nothing, index=nothing)
    T = size(B, 1)
    loss = _resolve_loss(additional_loss)
    pairwise_indices2 = Tuple.(findall(wp .> 0))
    size_order = size(B, 3)  # 1 for no memory, 2^order for memory >= 1

    function optimfunction2(u, p)
        Rt = similar(u, T)
        for t in 1:T
            Rt[t] = exp(_linear_predictor(basis, t, u))
        end
        if size_order == 1
            return -pairwise_loglikelihood(Rt, B[:, :, 1], p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2)
        else
            return -pairwise_loglikelihood_memory(Rt, B, p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2)
        end
    end
    phi = _RangePhi(basis)
    u0 = collect(theta_R)
    _check_loss(loss, phi, u0, :R, index, T)
    objective = _penalized((u, p) -> optimfunction2(u, p), loss, phi, :R, T)
    optf2 = OptimizationFunction(objective, adtype)

    prob = OptimizationProblem(optf2, u0, _penalized_data(loss, [Y, n_pair, h, wp], index))

    sol = solve(prob, solver; solve_kwargs...)

    if !successful_retcode(sol.retcode)
        @warn "sol.retcode = $(sol.retcode)"
    end

    theta_R[:] .= view(sol.u, :)
end

"""
    update_R!(hmm::ARPeriodicHMMSpatial, θᴿ, γ, wp, Y, SituationIdx; n2t, basis, solver, adtype, maxiters = 10, solve_kwargs, additional_loss)

M-step for the spatial ranges of the hidden states. For each state `k`, it weights the joint
outcomes of each pair of sites by the posteriors `γ[:, k]`, refits `θᴿ[k, :]` with
[`fit_mle_one_R!`](@ref), and writes the resulting ranges into `hmm.R`. `SituationIdx` gives the joint-outcome category of every
pair and observation; `basis` is the design matrix of the range, trigonometric by default;
`maxiters` caps the solver iterations. The states are refitted in parallel threads, in each of
which an [`additional_loss`](@ref SmoothPeriodicStatsModels.additional_loss) is called with
`part = :R` and `index = k`.
"""
function update_R!(hmm::ARPeriodicHMMSpatial,
    Range_θ::AbstractArray{N,2} where {N},
    γ::AbstractMatrix, wp, Y, SituationIdx::AbstractArray{<:Integer,3}; n2t=n_to_t(size(Y, 2), size(hmm, 3))::AbstractVector{<:Integer}, basis=trig_basis(size(hmm, 3), (size(Range_θ, 2) - 1) ÷ 2), solver, adtype=default_adtype(), maxiters=10, solve_kwargs=(;), additional_loss=nothing)
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

        fit_mle_one_R!(view(Range_θ, k, :), B, h, Y, wp, n_pair; n2t=n2t, basis, solver, adtype,
            solve_kwargs=merge((; maxiters), solve_kwargs), additional_loss, index=k)
    end

    for k in 1:K
        R[k, :] .= [exp(_linear_predictor(basis, t, view(Range_θ, k, :))) for t = 1:T]
    end
end
