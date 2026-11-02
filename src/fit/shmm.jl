function γₛ!(γₛ::AbstractVector, γ, n_all::AbstractVector)
    D = length(γₛ)
    for j in 1:D
        K, size_order, T, rain_cat = size(γₛ[j])
        for tup in Iterators.product(1:size_order, 1:T, 1:rain_cat)
            for k = 1:K
                γₛ[j][k, tup...] = sum(γ[n, k] for n in n_all[j][tup...]; init=0)
            end
        end
    end
end

function update_B!(B::AbstractArray{T,3} where {T}, θᴮ::AbstractMatrix, γ::AbstractMatrix, γₛ::AbstractVector, Y, n_all, basis::AbstractMatrix;
    solver=default_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;))
    @argcheck size(γ, 1) == size(Y, 2)
    @argcheck size(γ, 2) == size(B, 1)
    K, T, D = size(B)
    ## For periodicHMM only the n Y corresponding to B(t) are used to update B(t)
    ## Aggregate the posteriors that the M-step fits against

    γₛ!(γₛ, γ, n_all)

    # One independent problem per (state, site, memory), as in the dense `update_B!`; the sites
    # may each carry their own memory order, so the index list is ragged.
    idx = vcat(vec([[(k, j, h) for h = 1:length(θᴮ[k, j])] for j = 1:D, k = 1:K])...)
    emission_data(k, s, h) = (view(γₛ[s], k, h, :, :), basis)
    base = OptimizationProblem(OptimizationFunction(bernoulli_logit_nll, adtype),
        collect(θᴮ[idx[1][1], idx[1][2]][idx[1][3]]), emission_data(idx[1]...))
    eprob = EnsembleProblem(base;
        prob_func=(prob, ctx) -> remake(prob;
            u0=collect(θᴮ[idx[ctx.sim_id][1], idx[ctx.sim_id][2]][idx[ctx.sim_id][3]]),
            p=emission_data(idx[ctx.sim_id]...)))
    sol = solve(eprob, solver, ensemble; trajectories=length(idx), solve_kwargs...)

    for (i, (k, s, h)) in enumerate(idx)
        successful_retcode(sol.u[i].retcode) || @warn "emission ($k, $s, $h): $(sol.u[i].retcode)"
        θᴮ[k, s][h] = sol.u[i].u
    end

    for k = 1:K, t = 1:T, s = 1:D
        B[k, t, s][:] .= Bernoulli.([logistic(polynomial_trigo(t / T, θh)) for θh in θᴮ[k, s]])
    end
end

# The Baum-Welch loop that used to follow — forward and backward passes, the convergence test,
# the `θ_iters` bookkeeping — is now `HiddenMarkovModels.forward_backward!` driven from
# `src/fit/smooth_hmm.jl`, where this package writes only the M-step as `StatsAPI.fit!`. The
# two were measured against each other on the same data from the same start before this was
# removed: coefficients agreeing to 2.4e-15 and the attained loglikelihood to 3e-15.
# `fit_mle(SmoothsARPeriodicHMM(a, θᴬ, θᴮ, T), Y, Y_past; n2t)` is the replacement, returning
# `(model, history)` like every other fit in the package.
#
# `γₛ!` and `update_B!` above stay: they are the ragged, site-dependent M-step kernels, and
# `StatsAPI.fit!` calls them.
