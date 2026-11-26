"""
    γₛ!(γₛ, γ, n_all)

Aggregate the state posteriors `γ` (`N × K`) by site, memory category, period index and observed
value: `γₛ[k, j, h, t, y] = Σ γ[n, k]` over the observations `n ∈ n_all[j, h, t, y]`.
"""
function γₛ!(γₛ, γ, n_all)
    K, D, size_order, T, rain_cat = size(γₛ)
    for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)
        for k = 1:K
            γₛ[k, tup...] = sum(γ[n, k] for n in n_all[tup...]; init = 0)
        end
    end
end


"""
    update_B!(B, θᴮ, γ, γₛ, Y, n_all, basis; solver, adtype, ensemble, solve_kwargs, additional_loss)

M-step for the emission probabilities of the Bernoulli HMMs. It refits the coefficients
`θᴮ[k, s, h, :]` of every (state, site, memory category) against the posteriors `γ` aggregated
into `γₛ`, then writes the resulting probabilities into `B`. `basis` is the `T × nB` design matrix of the coefficients. Also defined
for the site-dependent layout of `sARPeriodicHMM`. `ensemble = EnsembleThreads()` refits them in
parallel. An [`additional_loss`](@ref SmoothPeriodicStatsModels.additional_loss) is called with `part = :B` and `index = (k, s, h)`.
"""
function update_B!(B::AbstractArray{T,4} where {T}, θᴮ::AbstractArray{N,4} where {N}, γ::AbstractMatrix, γₛ::AbstractArray, Y, n_all, basis::AbstractMatrix;
    solver=_required_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;), additional_loss=nothing)
    @argcheck size(γ, 1) == size(Y, 2)
    @argcheck size(γ, 2) == size(B, 1)
    K, T, D, size_order = size(B)
    loss = _resolve_loss(additional_loss)

    ## For periodicHMM only the n Y corresponding to B(t) are used to update B(t)
    γₛ!(γₛ, γ, n_all)

    idx = vec(collect(Iterators.product(1:K, 1:D, 1:size_order)))
    emission_data(k, s, h) = _penalized_data(loss, (view(γₛ, k, s, h, :, :), basis), (k, s, h))
    phi = _BernoulliPhi(basis)
    for (k, s, h) in idx
        _check_loss(loss, phi, collect(θᴮ[k, s, h, :]), :B, (k, s, h), T)
    end
    base = OptimizationProblem(OptimizationFunction(_penalized(bernoulli_logit_nll, loss, phi, :B, T), adtype),
        θᴮ[idx[1]..., :], emission_data(idx[1]...))
    eprob = EnsembleProblem(base;
        prob_func=(prob, ctx) -> remake(prob; u0=θᴮ[idx[ctx.sim_id]..., :],
            p=emission_data(idx[ctx.sim_id]...)))
    sol = solve(eprob, solver, ensemble; trajectories=length(idx), solve_kwargs...)

    for (i, (k, s, h)) in enumerate(idx)
        successful_retcode(sol.u[i].retcode) || @warn "emission ($k, $s, $h): $(sol.u[i].retcode)"
        θᴮ[k, s, h, :] = sol.u[i].u
    end

    p = [logistic(_linear_predictor(basis, t, view(θᴮ, k, s, h, :))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    _assign_B!(B, p)
end

_assign_B!(B::AbstractArray{<:Real,4}, p) = B .= p
_assign_B!(B::AbstractArray{<:Distribution,4}, p) = B .= Bernoulli.(p)
