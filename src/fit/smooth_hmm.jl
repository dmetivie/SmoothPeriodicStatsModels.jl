# # Fitting a smooth periodic HMM, with HiddenMarkovModels.jl's inference
#
# The E-step is `HiddenMarkovModels.forward_backward!`: the forward and backward passes, the
# state and transition marginals, and the multi-sequence bookkeeping come from there. The M-step,
# `StatsAPI.fit!`, is this package's, because what is specific to a *smooth* model is how the
# posteriors turn back into coefficients. The loop around them, `_baum_welch!` below, is also
# this package's rather than `HiddenMarkovModels.baum_welch!`: it records the loglikelihood after
# each M-step instead of before, and `baum_welch` would `deepcopy` the guess, duplicating the
# observation index sets its fitting context holds.

"""
    fit_mle(m, Ys, Y_pasts; n2t, solver, kwargs...) -> (m, history)

Fit a smooth periodic HMM to several independent sequences. The model is a
[`SmoothARPeriodicHMM`](@ref), [`SmoothsARPeriodicHMM`](@ref) or
[`SmoothARPeriodicHMMSpatial`](@ref); `Ys[i]` is the `D × Nᵢ` sequence `i`, `Y_pasts[i]` its initial observations and `n2t[i]` its period indices, by default
`n_to_t(Nᵢ, T)`. Every sequence starts from the initial distribution, which is fitted to the
first step of all of them. Same keywords and return as the single-sequence fit.
"""
function fit_mle(m::_SmoothHMM, Ys::AbstractVector{<:AbstractMatrix{<:Bool}},
    Y_pasts::AbstractVector; n2t=[n_to_t(size(Y, 2), m.nT) for Y in Ys], kwargs...)
    return _fit_mle(m, _sequences(m, Ys, Y_pasts, n2t)...; kwargs...)
end

# The Baum-Welch loop of the two Bernoulli models; the spatial one has its own, below.
_snapshot(m::SmoothARPeriodicHMM) = _plain(m.ps)
_snapshot(m::SmoothsARPeriodicHMM) = (A=copy(m.θᴬ), B=deepcopy(m.θᴮ))

# The fitted model without its fitting context, which holds the index sets of the data it was
# fitted to and is no part of the model.
_detach(m::SmoothARPeriodicHMM) =
    SmoothARPeriodicHMM(m.ps, m.stored, m.nT, m.basis_A, m.basis_B, nothing, m.stored_from)
_detach(m::SmoothsARPeriodicHMM) =
    SmoothsARPeriodicHMM(m.θᴬ, m.θᴮ, m.stored, m.nT, m.basis_A, m.basis_B, nothing, m.stored_from)
_detach(m::SmoothARPeriodicHMMSpatial) =
    SmoothARPeriodicHMMSpatial(m.ps, m.stored, m.nT, m.basis_A, m.basis_B, m.basis_R, nothing, m.stored_from)

# The Baum-Welch loop, shared by every model whose emission is evaluated one control at a time:
# the two Bernoulli models and `SmoothPeriodicHMM`. `obs` holds the columns of the observations.
_columns(Y::AbstractMatrix) = [view(Y, :, n) for n in axes(Y, 2)]

function _baum_welch!(fitted, obs, controls, seq_ends; maxiter, tol, display, robust, ps_iters)
    # With `robust`, the E-step reads clamped emission loglikelihoods. The wrapper holds `fitted`
    # itself, whose arrays the M-step rewrites in place, so it stays current.
    estep_model = robust ? RobustEmissions(fitted) : fitted

    storage = HMMs.initialize_forward_backward(estep_model, obs, controls; seq_ends)
    logtots = Float64[]
    trajectory = ps_iters ? [_snapshot(fitted)] : nothing

    # The E-step is run once up front and then once after every M-step, so that `logtots[k]` is
    # the loglikelihood the parameters *after* `k` M-steps attain — which makes
    # `logtots[end] == loglikelihood(fitted, …)`, the invariant the seasonal mixture's history
    # also has. `HiddenMarkovModels`' own loop pushes before fitting instead, so its last entry
    # describes the previous parameters; the number of forward passes is the same either way.
    HMMs.forward_backward!(storage, estep_model, obs, controls; seq_ends)
    logtot = sum(storage.logL)
    (display == :iter) && println("Iteration 0: logtot = $logtot")
    for it = 1:maxiter
        StatsAPI.fit!(fitted, storage, obs, controls; seq_ends)
        robust && (fitted.stored.A .+= eps())
        ps_iters && push!(trajectory, _snapshot(fitted))

        HMMs.forward_backward!(storage, estep_model, obs, controls; seq_ends)
        push!(logtots, sum(storage.logL))
        (display == :iter) && println("Iteration $it: logtot = $(round(logtots[end], digits=6))")

        if abs(logtots[end] - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $(logtots[end])")
            return _detach(fitted), FitHistory(; converged=true, iterations=it, logtots,
                ps_iters=trajectory, loglikelihood=logtots[end])
        end
        logtot = logtots[end]
    end
    # `logtot` is the loglikelihood of the returned model, the starting one when `maxiter = 0`
    (display in [:iter, :final]) &&
        println("EM has not converged after $maxiter iterations, logtot = $logtot")
    return _detach(fitted), FitHistory(; converged=false, iterations=maxiter, logtots,
        ps_iters=trajectory, loglikelihood=logtot)
end

# The initial distribution: each sequence contributes its first state marginal.
function _update_initialization!(m, γ, seq_ends)
    init = m.stored.a
    fill!(init, 0)
    for k in eachindex(seq_ends)
        t1, _ = HMMs.seq_limits(seq_ends, k)
        init .+= view(γ, :, t1)
    end
    init ./= sum(init)
    return m
end

# `ξ[n]` is the marginal of the step `n -> n+1`, which `forward_backward!` builds with
# `transition_matrix(hmm, control_seq[n + 1])`. That control carries `n2t[n]`, the period the
# step leaves, and the accessor reads `A(n2t[n])`, so the grouping is by `n2t[n]` — the grouping
# this package has always used, whatever the next period index is. (A model indexed by the
# period a step *arrives* at would group by `n2t[n + 1]` instead; that mismatch is the
# off-by-one that was fixed in PeriodicHiddenMarkovModels 0.3.2.) The last step of each
# sequence has no successor and its `ξ` is zero.
function _transition_counts(ξ, n_in_t, K, T)
    s_ξ = zeros(T, K, K)
    for t = 1:T, n in n_in_t[t]
        @views s_ξ[t, :, :] .+= ξ[n]
    end
    return s_ξ
end

"""
    SmoothHMMContext(n_in_t, n_all, basis_A, basis_B, solver, adtype, ensemble, solve_kwargs)

What the M-step of a smooth periodic HMM needs besides the posteriors: the observation index
sets of the data, the design matrices of the bases, and the solver settings. [`fit_mle`](@ref)
builds it and stores it in the model's `cfg` field.
"""
struct SmoothHMMContext{I,A,XB,AD,E,S}
    n_in_t::Vector{Vector{Int}}
    n_all::I
    basis_A::A       # `nothing` when K == 1: a one-state chain has no transition to fit
    basis_B::XB
    solver::S
    adtype::AD
    ensemble::E
    solve_kwargs::NamedTuple
end

function SmoothHMMContext(m::SmoothARPeriodicHMM, Y::AbstractMatrix{<:Bool},
    lag_cat::AbstractMatrix{<:Integer}, n2t::AbstractVector{<:Integer};
    solver, adtype, ensemble, solve_kwargs)
    K, D, T, size_order = size(m.stored)
    rain_cat = 2 # dry or wet

    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(view(Y, j, :) .== y, view(lag_cat, j, :) .== h))
                           for j = 1:D, h = 1:size_order, y = 0:1]
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history)
             for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]

    return SmoothHMMContext(n_in_t, n_all, K ≥ 2 ? design_matrix(m.basis_A, 1:T) : nothing,
        design_matrix(m.basis_B, 1:T), solver, adtype, ensemble, solve_kwargs)
end

"""
    fit!(m::SmoothARPeriodicHMM, fb_storage, obs_seq, control_seq; seq_ends)

One M-step of a smooth periodic HMM. It refits the initial distribution, the transition
coefficients and the emission coefficients from the posteriors in `fb_storage`, then re-evaluates
the model's `A` and `B`.
"""
function StatsAPI.fit!(m::SmoothARPeriodicHMM, fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector, control_seq::AbstractVector; seq_ends)
    cfg = m.cfg
    cfg === nothing && throw(ArgumentError(
        "this model carries no fitting context; call `fit_mle(m, Y, Y_past; n2t)` rather than `baum_welch` directly"))
    (; γ, ξ) = fb_storage
    K, T = size(m.stored.A, 1), m.nT
    D, size_order = size(m.ps.B, 2), size(m.ps.B, 3)

    _update_initialization!(m, γ, seq_ends)
    cfg.basis_A !== nothing && _update_transitions!(m, _transition_counts(ξ, cfg.n_in_t, K, T), cfg)

    # `γₛ!` wants `γ[n, k]`; the storage is `γ[k, n]`. A transpose is a lazy view, so this is
    # a relabelling rather than a copy.
    γₛ = zeros(K, D, size_order, T, 2)
    γₛ!(γₛ, transpose(γ), cfg.n_all)
    _update_emissions!(m, γₛ, cfg)

    return _refresh!(m)
end

# The transition M-step: one independent multinomial-logit problem per state. Shared by the
# four models — the transition parametrization is the same whatever the emission; only where
# `θᴬ` lives differs, which `_theta_A` answers.
function _update_transitions!(m::Union{_SmoothHMM,SmoothPeriodicHMM}, s_ξ, cfg)
    K = size(m.stored.A, 1)
    θᴬ = _theta_A(m)
    transition_data(k) = (view(s_ξ, :, k, :), cfg.basis_A, K)
    base = OptimizationProblem(OptimizationFunction(multinomial_logit_nll, cfg.adtype),
        collect(vec(θᴬ[1, :, :])), transition_data(1))
    eprob = EnsembleProblem(base;
        prob_func=(prob, ctx) -> remake(prob; u0=collect(vec(θᴬ[ctx.sim_id, :, :])),
            p=transition_data(ctx.sim_id)))
    sol = solve(eprob, cfg.solver, cfg.ensemble; trajectories=K, cfg.solve_kwargs...)
    for k = 1:K
        successful_retcode(sol.u[k].retcode) || @warn "transition $k: $(sol.u[k].retcode)"
        θᴬ[k, :, :] = reshape(sol.u[k].u, K - 1, :)
    end
    return m
end

# The emission M-step: one independent Bernoulli-logit problem per (state, site, memory).
function _update_emissions!(m::Union{SmoothARPeriodicHMM,SmoothARPeriodicHMMSpatial}, γₛ, cfg)
    K, D, size_order = size(m.ps.B, 1), size(m.ps.B, 2), size(m.ps.B, 3)
    θᴮ = m.ps.B
    idx = vec(collect(Iterators.product(1:K, 1:D, 1:size_order)))
    emission_data(k, s, h) = (view(γₛ, k, s, h, :, :), cfg.basis_B)
    base = OptimizationProblem(OptimizationFunction(bernoulli_logit_nll, cfg.adtype),
        collect(θᴮ[idx[1]..., :]), emission_data(idx[1]...))
    eprob = EnsembleProblem(base;
        prob_func=(prob, ctx) -> remake(prob; u0=collect(θᴮ[idx[ctx.sim_id]..., :]),
            p=emission_data(idx[ctx.sim_id]...)))
    sol = solve(eprob, cfg.solver, cfg.ensemble; trajectories=length(idx), cfg.solve_kwargs...)
    for (i, (k, s, h)) in enumerate(idx)
        successful_retcode(sol.u[i].retcode) || @warn "emission ($k, $s, $h): $(sol.u[i].retcode)"
        θᴮ[k, s, h, :] = sol.u[i].u
    end
    return m
end

"""
    fit_mle(m::SmoothARPeriodicHMM, Y, Y_past; n2t, solver, kwargs...) -> (m, history)

Fit a smooth periodic autoregressive HMM by Baum-Welch, starting from `m`.

`Y` is `D × N` and `Y_past` the `D × order` initial observations, those before `Y` in time
order, so that `[Y_past Y]` is the series; `rand(m, n2t; Y_past)` draws a `Y` that follows it.
Returns the fitted model and a [`FitHistory`](@ref). For several independent sequences, pass
vectors of them.

Keywords: `maxiter`, `tol` (absolute change of the loglikelihood), the solver settings `solver`,
`adtype`, `ensemble` and `solve_kwargs` of the M-step (`ensemble = EnsembleThreads()` solves
the emission problems in parallel), and `ps_iters = true` to keep the coefficients of every
iteration in `history.ps_iters`. `robust = true` clamps infinite emission loglikelihoods in the
E-step and keeps the transition probabilities off zero.
"""
function fit_mle(m::SmoothARPeriodicHMM, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool};
    n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer}, kwargs...)
    return _fit_mle(m, _sequences(m, Y, Y_past, n2t)...; kwargs...)
end

function _fit_mle(m::SmoothARPeriodicHMM, Y, lag_cat, n2t, seq_ends;
    maxiter=100, tol=1e-3, display=:none, robust=false, ps_iters=false,
    solver=_required_solver(), adtype=default_adtype(), ensemble=default_ensemble(),
    solve_kwargs=(;))
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0
    cfg = SmoothHMMContext(m, Y, lag_cat, n2t; solver, adtype, ensemble, solve_kwargs)
    fitted = SmoothARPeriodicHMM(copy(m.ps), copy(m.stored), m.nT, m.basis_A, m.basis_B, cfg)
    _refresh!(fitted)   # the coefficients may have been edited in place since `m` was built
    return _baum_welch!(fitted, _columns(Y), SeasonalControls(n2t, lag_cat), seq_ends;
        maxiter, tol, display, robust, ps_iters)
end

# # The site-dependent variant
#
# The same shape as above: a context holding what the M-step needs beyond the parameters, one
# `StatsAPI.fit!`, and a `fit_mle` wrapper driving `HiddenMarkovModels`' forward-backward. Only
# the emission bookkeeping differs, because each site has its own memory order and its own
# number of coefficient blocks.

"""
    SmoothsHMMContext

What the M-step of a `SmoothsARPeriodicHMM` needs besides the posteriors, as
[`SmoothHMMContext`](@ref) for the dense model.
"""
struct SmoothsHMMContext{I,A,XB,AD,E,S}
    n_in_t::Vector{Vector{Int}}
    n_all::I
    basis_A::A
    basis_B::XB
    γₛ::Vector{Array{Float64,4}}
    solver::S
    adtype::AD
    ensemble::E
    solve_kwargs::NamedTuple
end

function SmoothsHMMContext(m::SmoothsARPeriodicHMM, Y::AbstractMatrix{<:Bool},
    lag_cat::AbstractMatrix{<:Integer}, n2t::AbstractVector{<:Integer};
    solver, adtype, ensemble, solve_kwargs)
    K, D, T, size_orders = size(m.stored)
    rain_cat = 2

    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [[findall(.&(view(Y, j, :) .== y, view(lag_cat, j, :) .== h))
                            for h = 1:size_orders[j], y = 0:1] for j = 1:D]
    n_all = [[n_per_category(tup..., n_in_t, n_occurence_history[j])
              for tup in Iterators.product(1:size_orders[j], 1:T, 1:rain_cat)] for j = 1:D]

    return SmoothsHMMContext(n_in_t, n_all, K ≥ 2 ? design_matrix(m.basis_A, 1:T) : nothing,
        design_matrix(m.basis_B, 1:T), [zeros(K, size_orders[j], T, rain_cat) for j = 1:D],
        solver, adtype, ensemble, solve_kwargs)
end

"""
    fit!(m::SmoothsARPeriodicHMM, fb_storage, obs_seq, control_seq; seq_ends)

One M-step of the site-dependent model, as for `SmoothARPeriodicHMM`.
"""
function StatsAPI.fit!(m::SmoothsARPeriodicHMM, fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector, control_seq::AbstractVector; seq_ends)
    cfg = m.cfg
    cfg === nothing && throw(ArgumentError(
        "this model carries no fitting context; call `fit_mle(m, Y, Y_past; n2t)` rather than `baum_welch` directly"))
    (; γ, ξ) = fb_storage
    K, T = size(m.stored.A, 1), m.nT

    _update_initialization!(m, γ, seq_ends)
    cfg.basis_A !== nothing && _update_transitions!(m, _transition_counts(ξ, cfg.n_in_t, K, T), cfg)

    γₛ!(cfg.γₛ, transpose(γ), cfg.n_all)
    _update_emissions_sites!(m, cfg)

    return _refresh!(m)
end

_theta_A(m::SmoothARPeriodicHMM) = m.ps.A
_theta_A(m::SmoothsARPeriodicHMM) = m.θᴬ

function _update_emissions_sites!(m::SmoothsARPeriodicHMM, cfg)
    K, D = size(m.stored.B, 1), size(m.stored.B, 3)
    θᴮ = m.θᴮ
    idx = vcat(vec([[(k, j, h) for h = 1:length(θᴮ[k, j])] for j = 1:D, k = 1:K])...)
    emission_data(k, s, h) = (view(cfg.γₛ[s], k, h, :, :), cfg.basis_B)
    base = OptimizationProblem(OptimizationFunction(bernoulli_logit_nll, cfg.adtype),
        collect(θᴮ[idx[1][1], idx[1][2]][idx[1][3]]), emission_data(idx[1]...))
    eprob = EnsembleProblem(base;
        prob_func=(prob, ctx) -> remake(prob;
            u0=collect(θᴮ[idx[ctx.sim_id][1], idx[ctx.sim_id][2]][idx[ctx.sim_id][3]]),
            p=emission_data(idx[ctx.sim_id]...)))
    sol = solve(eprob, cfg.solver, cfg.ensemble; trajectories=length(idx), cfg.solve_kwargs...)
    for (i, (k, s, h)) in enumerate(idx)
        successful_retcode(sol.u[i].retcode) || @warn "emission ($k, $s, $h): $(sol.u[i].retcode)"
        θᴮ[k, s][h] = sol.u[i].u
    end
    return m
end

"""
    fit_mle(m::SmoothsARPeriodicHMM, Y, Y_past; n2t, solver, kwargs...) -> (m, history)

Fit a site-dependent periodic autoregressive HMM by Baum-Welch. `Y_past[j]` is site `j`'s own
initial condition, its values before `Y[j, :]` in time order, as many as its order. Same
keywords and same `(model, history)`
return as the dense [`fit_mle`](@ref).
"""
function fit_mle(m::SmoothsARPeriodicHMM, Y::AbstractMatrix{<:Bool}, Y_past::AbstractVector;
    n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer}, kwargs...)
    return _fit_mle(m, _sequences(m, Y, Y_past, n2t)...; kwargs...)
end

function _fit_mle(m::SmoothsARPeriodicHMM, Y, lag_cat, n2t, seq_ends;
    maxiter=100, tol=1e-3, display=:none, robust=false, ps_iters=false,
    solver=_required_solver(), adtype=default_adtype(), ensemble=default_ensemble(),
    solve_kwargs=(;))
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0
    cfg = SmoothsHMMContext(m, Y, lag_cat, n2t; solver, adtype, ensemble, solve_kwargs)
    fitted = SmoothsARPeriodicHMM(copy(m.θᴬ), deepcopy(m.θᴮ), copy(m.stored), m.nT,
        m.basis_A, m.basis_B, cfg)
    _refresh!(fitted)   # the coefficients may have been edited in place since `m` was built
    return _baum_welch!(fitted, _columns(Y), SeasonalControls(n2t, lag_cat), seq_ends;
        maxiter, tol, display, robust, ps_iters)
end

# # The spatial variant
#
# Two things make this one different from its siblings, and both are about the emission.
#
# It is **expensive and threaded**: one `D`-dimensional orthant probability per (observation,
# state), estimated by quasi-Monte Carlo, and `loglikelihoods!` parallelises over observations.
# `HiddenMarkovModels`' forward pass is necessarily sequential, so evaluating the emission one
# control at a time costs 8-10x (measured at `D = 10..20`, `N = 300`, 28 threads; caching the
# `D x D` kernel recovers none of it, because `mvnormcdf` dominates). `CachedEmissions`, in
# `src/hmm_interface.jl`, is how both are had at once.
#
# It is also **stochastic**: `mvnormcdf` is randomized, so two E-steps at the same parameters
# disagree and the loglikelihood path is noisy. That is why `fit_mle` returns the best iterate
# rather than the last.

"""
    SpatialHMMContext

What the M-step of a `SmoothARPeriodicHMMSpatial` needs besides the posteriors. That is the
content of [`SmoothHMMContext`](@ref), plus the pairs of sites, their weights and the joint-outcome category
of every pair and observation used by the range M-step.
"""
struct SpatialHMMContext{I,A,XB,XR,AD,E,S,W,X,YY}
    n_in_t::Vector{Vector{Int}}
    n_all::I
    Y::YY                   # the D x N observations; `update_R!` needs the matrix, not the views
    basis_A::A
    basis_B::XB
    basis_R::XR
    situation_idx::X        # N x D x D, the pair-situation of every observation
    wp::W                   # D x D pair weights of the composite likelihood
    QMC_E::Int
    maxiters_R::Int
    solver::S
    adtype::AD
    ensemble::E
    solve_kwargs::NamedTuple
end

function SpatialHMMContext(m::SmoothARPeriodicHMMSpatial, Y::AbstractMatrix{<:Bool},
    lag_cat::AbstractMatrix{<:Integer}, n2t::AbstractVector{<:Integer};
    wp, QMC_E, maxiters_R, solver, adtype, ensemble, solve_kwargs)
    K, D, T, size_order = size(m.stored)
    rain_cat = 2

    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(view(Y, j, :) .== y, view(lag_cat, j, :) .== h))
                           for j = 1:D, h = 1:size_order, y = 0:1]
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history)
             for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]

    # Which of the four (or 4*size_order^2) joint cells each pair of sites falls in, at every
    # observation. It depends only on the data, so it is built once and reused every M-step.
    N = size(Y, 2)
    situation_idx = Array{Int,3}(undef, N, D, D)
    for n = 1:N, i = 1:D, j = 1:D
        cur = 2 * (1 - Int(Y[i, n])) + (1 - Int(Y[j, n])) + 1
        situation_idx[n, i, j] =
            (size_order - lag_cat[i, n]) * size_order * 4 + (size_order - lag_cat[j, n]) * 4 + cur
    end

    return SpatialHMMContext(n_in_t, n_all, Y, K >= 2 ? design_matrix(m.basis_A, 1:T) : nothing,
        design_matrix(m.basis_B, 1:T), design_matrix(m.basis_R, 1:T), situation_idx, wp,
        Int(QMC_E), Int(maxiters_R), solver, adtype, ensemble, solve_kwargs)
end

"""
    fit!(m::SmoothARPeriodicHMMSpatial, fb_storage, obs_seq, control_seq; seq_ends)

One M-step of the spatial model: the transition and emission refits, then the spatial range
through [`update_R!`](@ref), which maximizes a pairwise composite likelihood.
"""
function StatsAPI.fit!(m::SmoothARPeriodicHMMSpatial, fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector, control_seq::AbstractVector; seq_ends)
    cfg = m.cfg
    cfg === nothing && throw(ArgumentError(
        "this model carries no fitting context; call `fit_mle(m, Y, Y_past; n2t)` rather than `baum_welch` directly"))
    (; γ, ξ) = fb_storage
    K, T = size(m.stored.A, 1), m.nT
    D, size_order = size(m.ps.B, 2), size(m.ps.B, 3)

    _update_initialization!(m, γ, seq_ends)
    cfg.basis_A !== nothing && _update_transitions!(m, _transition_counts(ξ, cfg.n_in_t, K, T), cfg)

    γₛ = zeros(K, D, size_order, T, 2)
    γₛ!(γₛ, transpose(γ), cfg.n_all)
    _update_emissions!(m, γₛ, cfg)

    # `update_R!` reads `B` and writes `R`, so the emission coefficients have to be refreshed
    # into the stored arrays before it runs.
    _refresh!(m)
    update_R!(m.stored, m.ps.R, permutedims(γ), cfg.wp, cfg.Y, cfg.situation_idx;
        n2t=_controls_n2t(control_seq), basis=cfg.basis_R, solver=cfg.solver, adtype=cfg.adtype,
        maxiters=cfg.maxiters_R, solve_kwargs=cfg.solve_kwargs)
    m.stored_from[] = deepcopy(m.ps)   # `update_R!` writes the range coefficients and their evaluation
    return m
end

_theta_A(m::SmoothARPeriodicHMMSpatial) = m.ps.A
_controls_n2t(c::SeasonalControls) = c.n2t

"""
    fit_mle(m::SmoothARPeriodicHMMSpatial, Y, Y_past; n2t, solver, kwargs...) -> (m, history)

Fit a spatial periodic HMM by Baum-Welch, starting from `m`.

Returns the iterate with the highest loglikelihood, not the last one: the emission likelihood is
a quasi-Monte-Carlo estimate, so the loglikelihood path is noisy. `history.best_iteration` says
which iterate was returned, `0` for the starting model, and `loglikelihood(history)` its
loglikelihood; `history.logtots` holds the whole path, and `history.ps_iters` the coefficients
of every iteration with `ps_iters = true`.

Keywords beyond those of the other HMM fits: `QMC_E` multiplies the model's `QMC_m` for the
E-step, `maxiters_R` caps the solver iterations of the range M-step, and `tdist`/`wp` choose the
pairs of sites in its composite likelihood.
"""
function fit_mle(m::SmoothARPeriodicHMMSpatial, Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool};
    n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer}, kwargs...)
    return _fit_mle(m, _sequences(m, Y, Y_past, n2t)...; kwargs...)
end

function _fit_mle(m::SmoothARPeriodicHMMSpatial, Y, lag_cat, n2t, seq_ends;
    maxiter=100, tol=1e-3, display=:none, robust=false, ps_iters=false,
    tdist=1, wp=1.0 .* (m.stored.h .< maximum(m.stored.h) * tdist),
    QMC_E=1, maxiters_R=10,
    solver=_required_solver(), adtype=default_adtype(), ensemble=default_ensemble(),
    solve_kwargs=(;))
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N = size(Y, 2)
    controls = SeasonalControls(n2t, lag_cat)
    obs = [view(Y, :, n) for n = 1:N]

    cfg = SpatialHMMContext(m, Y, lag_cat, n2t; wp, QMC_E, maxiters_R,
        solver, adtype, ensemble, solve_kwargs)
    fitted = SmoothARPeriodicHMMSpatial(copy(m.ps), deepcopy(m.stored), m.nT,
        m.basis_A, m.basis_B, m.basis_R, cfg)
    _refresh!(fitted)   # the coefficients may have been edited in place since `m` was built

    K = size(fitted.stored.A, 1)
    cached = CachedEmissions(fitted, zeros(N, K))
    storage = HMMs.initialize_forward_backward(cached, obs, controls; seq_ends)
    logtots = Float64[]
    trajectory = ps_iters ? [_plain(fitted.ps)] : nothing

    # The threaded batch evaluation, which is the whole reason `CachedEmissions` exists.
    function estep!()
        loglikelihoods!(cached.LL, fitted.stored, Y, lag_cat; n2t=n2t,
            QMC_m=fitted.stored.QMC_m * cfg.QMC_E)
        robust && _clamp_inf!(cached.LL)
        HMMs.forward_backward!(storage, cached, obs, controls; seq_ends)
        return sum(storage.logL)
    end

    logtot = estep!()
    (display == :iter) && println("Iteration 0: logtot = $logtot")
    # The best iterate so far, the start included. `a` is no coefficient, so it is kept apart.
    best_it, best_logtot, best_ps, best_a = 0, logtot, copy(fitted.ps), copy(fitted.stored.a)
    converged = false
    it_done = 0

    for it = 1:maxiter
        ps_before = copy(fitted.ps)
        StatsAPI.fit!(fitted, storage, obs, controls; seq_ends)
        robust && (fitted.stored.A .+= eps())
        ps_iters && push!(trajectory, _plain(fitted.ps))

        logtotp = estep!()
        push!(logtots, logtotp)
        it_done = it
        (display == :iter) && println("Iteration $it: logtot = $(round(logtotp, digits=6))")

        if logtotp > best_logtot
            best_it, best_logtot, best_ps, best_a = it, logtotp, copy(fitted.ps), copy(fitted.stored.a)
        end

        # The original relative-change test on the coefficients, and the loglikelihood test.
        Δ = maximum(abs, (fitted.ps .- ps_before) ./ ifelse.(iszero.(ps_before), eps(), ps_before))
        if Δ < tol || abs(logtotp - logtot) < tol
            converged = true
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            break
        end
        if abs((logtotp - logtot) / logtotp) > tol && logtotp < logtot
            (display in [:iter, :final]) &&
                println("stop: the loglikelihood has decreased dramatically")
            break
        end
        logtot = logtotp
    end

    # Restore the best iterate. With a noisy likelihood the last one is not reliably the best,
    # and this is what the spatial tutorial used to do by hand with `argmax(history["logtots"])`.
    fitted.ps .= best_ps
    fitted.stored.a .= best_a
    _refresh!(fitted)
    (display in [:iter, :final]) && !converged &&
        println("EM has not converged after $it_done iterations, logtot = $best_logtot")
    return _detach(fitted), FitHistory(; converged, iterations=it_done, logtots,
        ps_iters=trajectory, best_iteration=best_it, loglikelihood=best_logtot)
end

# # Model selection

"""
    fit(SeasonalFit, m, Y, Y_past; n2t, solver, kwargs...) -> SeasonalFit
    fit(SeasonalFit, m, Ys, Y_pasts; n2t, solver, kwargs...) -> SeasonalFit

Fit a smooth periodic HMM with [`fit_mle`](@ref), same keywords, and return a
[`SeasonalFit`](@ref), whose `bic` compares numbers of states, degrees or memory orders. `nobs`
is the number of observed steps, summed over the sequences.
"""
function StatsAPI.fit(::Type{SeasonalFit}, m::_SmoothHMM, Y, Y_past; kwargs...)
    fitted, history = fit_mle(m, Y, Y_past; kwargs...)
    return SeasonalFit(fitted, history, _nsteps(Y), loglikelihood(history))
end

_nsteps(Y::AbstractArray{<:Real}) = _ncols(Y)
_nsteps(Ys::AbstractVector{<:AbstractArray}) = sum(_ncols, Ys)
