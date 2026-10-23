"""
    fit_mle_AR1(y, d, T; n2t, θ0, solver, adtype, solve_kwargs) -> (θμ, θρ, θσ)

Maximum likelihood fit of a periodic AR(1)

    y[n] = μ(t) + ρ(t) * y[n-1] + σ(t) * ε[n],   with t = n2t[n]


whose three seasonal parameters are trigonometric polynomials of degree `d` over a period of `T`
steps. Returns their coefficient vectors, each of length `2d+1`, in the convention of
[`polynomial_trigo`](@ref): recover the curves with `μₜ(t/T, θμ)`, `ρₜ(t/T, θρ)`, `σₜ(t/T, θσ)`.
"""
function fit_mle_AR1(y::AbstractVector, d::Integer, T::Integer;
    n2t=n_to_t(length(y), T)::AbstractVector{<:Integer},
    θ0=zeros(3 * (2d + 1)),
    solver=default_solver(), adtype=default_adtype(), solve_kwargs=(;))
    nc = 2d + 1
    basis = trig_basis(T, d)
    stats = ar1_sufficient_statistics(y, n2t, T)
    u = solve_unconstrained(ar1_nll, collect(θ0), (stats, basis); solver, adtype, solve_kwargs)
    return u[1:nc], u[nc+1:2nc], u[2nc+1:3nc]
end

"""
    fit_mle_AR1(y, z, d, T; n2t, K, θ0, solver, adtype, ensemble, solve_kwargs) -> (θμ, θρ, θσ)

Maximum likelihood fit of one periodic AR(1) per hidden state, given the state sequence `z`. Each
returned array is `(2d+1) × K`, column `k` holding the coefficients of state `k`.

The states share no parameters, so the `K` fits are independent and are run as one
`EnsembleProblem`. A state that `z` never visits has no data and keeps its starting value.
"""
function fit_mle_AR1(y::AbstractVector, z::AbstractVector{<:Integer}, d::Integer, T::Integer;
    n2t=n_to_t(length(y), T)::AbstractVector{<:Integer},
    K=maximum(z),
    θ0=zeros(3 * (2d + 1)),
    solver=default_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;))
    nc = 2d + 1
    basis = trig_basis(T, d)
    stats = [ar1_sufficient_statistics(y, n2t, T, [n for n = 2:length(y) if z[n] == k]) for k = 1:K]

    base = OptimizationProblem(OptimizationFunction(ar1_nll, adtype), collect(θ0), (stats[1], basis))
    eprob = EnsembleProblem(base;
        prob_func=(prob, ctx) -> remake(prob; u0=collect(θ0), p=(stats[ctx.sim_id], basis)))
    sol = solve(eprob, solver, ensemble; trajectories=K, solve_kwargs...)

    θμ, θρ, θσ = (zeros(nc, K) for _ = 1:3)
    for k = 1:K
        successful_retcode(sol.u[k].retcode) || @warn "AR(1) state $k: $(sol.u[k].retcode)"
        u = sol.u[k].u
        θμ[:, k] = u[1:nc]
        θρ[:, k] = u[nc+1:2nc]
        θσ[:, k] = u[2nc+1:3nc]
    end
    return θμ, θρ, θσ
end

# Legacy spellings. The name said "model" but these always built *and* solved; they are now
# aliases of `fit_mle_AR1`, which returns the same `(θμ, θρ, θσ)` triple.
function model_for_loglikelihood_AR1_full(y::AbstractVector, d::Integer, T::Integer,
    n2t=n_to_t(length(y), T); silence=nothing, kwargs...)
    _warn_legacy_solver_kwargs(silence, nothing)
    return fit_mle_AR1(y, d, T; n2t, kwargs...)
end

function model_for_loglikelihood_AR1_full(y::AbstractVector, z::AbstractVector{<:Integer},
    d::Integer, T::Integer, n2t=n_to_t(length(y), T); silence=nothing, kwargs...)
    _warn_legacy_solver_kwargs(silence, nothing)
    return fit_mle_AR1(y, z, d, T; n2t, kwargs...)
end
