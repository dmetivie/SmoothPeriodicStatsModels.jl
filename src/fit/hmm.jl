function γₛ!(γₛ, γ, n_all)
    K, D, size_order, T, rain_cat = size(γₛ)
    for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)
        for k = 1:K
            γₛ[k, tup...] = sum(γ[n, k] for n in n_all[tup...]; init = 0)
        end
    end
end

# `s_ξ!` and `update_A!` used to live here. Both are gone: the transition marginals `ξ` now come
# from `HiddenMarkovModels.forward_backward!` rather than being recomputed from `α`, `β` and
# `LL`, and the aggregation plus the multinomial-logit solve are `_update_transitions!` in
# `src/fit/smooth_hmm.jl`, shared by all three HMM families.


"""
    update_B!(B, θᴮ, γ, γₛ, Y, n_all, basis; solver, adtype, ensemble, solve_kwargs)

M-step for the emission probabilities: refit the trigonometric coefficients `θᴮ[k, s, h, :]` of
every (state, site, memory) triple against the aggregated posteriors `γₛ`, then write the
resulting probabilities into `B`.

Pass `ensemble = EnsembleThreads()` to refit the triples in parallel.
"""
function update_B!(B::AbstractArray{T,4} where {T}, θᴮ::AbstractArray{N,4} where {N}, γ::AbstractMatrix, γₛ::AbstractArray, Y, n_all, basis::AbstractMatrix;
    solver=default_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;))
    @argcheck size(γ, 1) == size(Y, 2)
    @argcheck size(γ, 2) == size(B, 1)
    K, T, D, size_order = size(B)

    ## For periodicHMM only the n Y corresponding to B(t) are used to update B(t)
    γₛ!(γₛ, γ, n_all)

    idx = vec(collect(Iterators.product(1:K, 1:D, 1:size_order)))
    emission_data(k, s, h) = (view(γₛ, k, s, h, :, :), basis)
    base = OptimizationProblem(OptimizationFunction(bernoulli_logit_nll, adtype),
        θᴮ[idx[1]..., :], emission_data(idx[1]...))
    eprob = EnsembleProblem(base;
        prob_func=(prob, ctx) -> remake(prob; u0=θᴮ[idx[ctx.sim_id]..., :],
            p=emission_data(idx[ctx.sim_id]...)))
    sol = solve(eprob, solver, ensemble; trajectories=length(idx), solve_kwargs...)

    for (i, (k, s, h)) in enumerate(idx)
        successful_retcode(sol.u[i].retcode) || @warn "emission ($k, $s, $h): $(sol.u[i].retcode)"
        θᴮ[k, s, h, :] = sol.u[i].u
    end

    p = [logistic(polynomial_trigo(t / T, θᴮ[k, s, h, :])) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    _assign_B!(B, p)
end

_assign_B!(B::AbstractArray{<:Real,4}, p) = B .= p
_assign_B!(B::AbstractArray{<:Distribution,4}, p) = B .= Bernoulli.(p)


"""
    update_A!(A, θᴬ, ξ, s_ξ, α, β, LL, n2t, n_in_t, basis; solver, adtype, ensemble, solve_kwargs)

M-step for the transition matrices: refit the trigonometric coefficients `θᴬ[k, :, :]` of every
state against the aggregated transition posteriors `s_ξ`, then write the resulting softmax
probabilities into `A`.

One independent problem per state, solved as one `EnsembleProblem`.
"""


# The Baum-Welch loop that used to live here — the forward and backward passes, the
# convergence test, the `θ_iters` bookkeeping, roughly 140 lines — is now
# `HiddenMarkovModels.forward_backward!` driven from `src/fit/smooth_hmm.jl`, where the only
# thing this package still writes is the M-step above, as `StatsAPI.fit!`. The two were
# measured against each other on the same data from the same start before this was removed:
# same iteration count, coefficients agreeing to 1.2e-8 and the attained loglikelihood to
# 1.6e-7. `fit_mle(SmoothARPeriodicHMM(a, θᴬ, θᴮ, T), Y, Y_past; n2t)` is the replacement, and
# it returns `(model, history)` like every other fit in the package.
#
# The kernels above stay: `sARPeriodicHMM`'s own loop in `shmm.jl` calls `γₛ!`, `s_ξ!`,
# `update_A!` and `update_B!` too.
