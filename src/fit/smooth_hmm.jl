# # Fitting a smooth periodic HMM, with HiddenMarkovModels.jl driving
#
# Baum-Welch is `HiddenMarkovModels.baum_welch!`. The only thing this package writes is the
# M-step, `StatsAPI.fit!`, because the only thing specific to a *smooth* model is how the
# posteriors turn back into coefficients. The forward pass, the backward pass, the transition
# marginals, the convergence test and the multi-sequence bookkeeping all come from there.

"""
    SmoothHMMContext(n_in_t, n_all, basis_A, basis_B, solver, adtype, ensemble, solve_kwargs)

Everything the M-step needs that is neither a parameter nor supplied by the Baum-Welch loop:
the observation index sets, which depend on the data, and the solver settings.

Built once by [`fit_mle`](@ref) and carried in the model's `cfg` field, because
`StatsAPI.fit!`'s signature is fixed by `HiddenMarkovModels` and has nowhere else to put them.
"""
struct SmoothHMMContext{I,A,AD,E,S}
    n_in_t::Vector{Vector{Int}}
    n_all::I
    basis_A::A       # `nothing` when K == 1: a one-state chain has no transition to fit
    basis_B::Matrix{Float64}
    solver::S
    adtype::AD
    ensemble::E
    solve_kwargs::NamedTuple
end

function SmoothHMMContext(m::SmoothARPeriodicHMM, Y::AbstractMatrix{<:Bool},
    lag_cat::AbstractMatrix{<:Integer}, n2t::AbstractVector{<:Integer};
    solver, adtype, ensemble, solve_kwargs)
    K, D, T, size_order = size(m.stored)
    dA, dB = degrees(m)
    rain_cat = 2 # dry or wet

    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(view(Y, j, :) .== y, view(lag_cat, j, :) .== h))
                           for j = 1:D, h = 1:size_order, y = 0:1]
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history)
             for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]

    return SmoothHMMContext(n_in_t, n_all, K ≥ 2 ? trig_basis(T, dA) : nothing,
        trig_basis(T, dB), solver, adtype, ensemble, solve_kwargs)
end

"""
    fit!(m::SmoothARPeriodicHMM, fb_storage, obs_seq, control_seq; seq_ends)

One M-step: refit the initial distribution, the transition coefficients and the emission
coefficients from the posteriors `HiddenMarkovModels` has just computed, then re-evaluate the
model's `A` and `B` from them.

The posteriors arrive in that library's layout — `γ` is `states × time` and `ξ[n]` is the
`K × K` marginal of the step `n -> n+1` — so the two aggregations below transpose and regroup
before handing them to the kernels, which are unchanged from the hand-rolled loop this
replaces.
"""
function StatsAPI.fit!(m::SmoothARPeriodicHMM, fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector, control_seq::AbstractVector; seq_ends)
    cfg = m.cfg
    cfg === nothing && throw(ArgumentError(
        "this model carries no fitting context; call `fit_mle(m, Y, Y_past; n2t)` rather than `baum_welch` directly"))
    (; γ, ξ) = fb_storage
    K, T = size(m.stored.A, 1), m.nT
    D, size_order = size(m.ps.B, 2), size(m.ps.B, 3)

    # ## The initial distribution
    #
    # One sequence contributes its first state marginal; several contribute one each.
    init = m.stored.a
    fill!(init, 0)
    for k in eachindex(seq_ends)
        t1, _ = HMMs.seq_limits(seq_ends, k)
        init .+= view(γ, :, t1)
    end
    init ./= sum(init)

    # ## The transitions
    #
    # `ξ[n]` is the marginal of the step `n -> n+1`, which `forward_backward!` builds with
    # `transition_matrix(hmm, control_seq[n + 1])`. Our accessor reads that as `A(n2t[n])`, so
    # `ξ[n]` belongs to the period the step *leaves* and the grouping is by `n2t[n]` — the
    # grouping this package has always used. (A model that did not shift its accessor would
    # have to group by `n2t[n + 1]` instead; that mismatch is the off-by-one that was fixed in
    # PeriodicHiddenMarkovModels 0.3.2.)
    if cfg.basis_A !== nothing
        s_ξ = zeros(T, K, K)
        for t = 1:T, n in cfg.n_in_t[t]
            n == length(ξ) && continue    # the last step has no successor; `ξ[end]` is zero
            @views s_ξ[t, :, :] .+= ξ[n]
        end
        _update_transitions!(m, s_ξ, cfg)
    end

    # ## The emissions
    #
    # `γₛ!` wants `γ[n, k]`; the storage is `γ[k, n]`. A transpose is a lazy view, so this is
    # a relabelling rather than a copy.
    γₛ = zeros(K, D, size_order, T, 2)
    γₛ!(γₛ, transpose(γ), cfg.n_all)
    _update_emissions!(m, γₛ, cfg)

    return _refresh!(m)
end

# The transition M-step: one independent multinomial-logit problem per state. Shared by both
# Bernoulli models — the transition parametrization is the same whether or not the sites differ
# in memory order; only where `θᴬ` lives differs, which `_theta_A` answers.
function _update_transitions!(m::Union{SmoothARPeriodicHMM,SmoothsARPeriodicHMM,SmoothARPeriodicHMMSpatial}, s_ξ, cfg)
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
    fit_mle(m::SmoothARPeriodicHMM, Y, Y_past; n2t, kwargs...) -> (m, history)

Fit a smooth periodic autoregressive HMM by Baum-Welch, starting from `m`.

`Y` is `D × N` (sites first, the `Distributions.jl` convention) and `Y_past` the `D × order`
initial condition. Returns a model of the same shape and a [`FitHistory`](@ref);
`ARPeriodicHMM(m)` is the evaluated twin to simulate from, `parameters(m).B` the emission
coefficients.

The loop is `HiddenMarkovModels.baum_welch!`; this package supplies only the M-step. Its
keywords are accepted here: `maxiter` (`max_iterations`), `tol` (`atol`), and the solver
settings every fit in this package takes — pass `ensemble = EnsembleThreads()` to run the
independent emission problems in parallel. `ps_iters = true` keeps the coefficient trajectory
in `history.ps_iters`, which is what `θ_iters` used to return beside the model.
"""
function fit_mle(m::SmoothARPeriodicHMM, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool};
    n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer},
    maxiter=100, tol=1e-3, display=:none, robust=false, ps_iters=false,
    solver=default_solver(), adtype=default_adtype(), ensemble=default_ensemble(),
    solve_kwargs=(;))
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N = size(Y, 2)
    lag_cat = conditional_to(Y, Y_past)
    controls = SeasonalControls(n2t, lag_cat)
    obs = [view(Y, :, n) for n = 1:N]

    cfg = SmoothHMMContext(m, Y, lag_cat, n2t; solver, adtype, ensemble, solve_kwargs)
    fitted = SmoothARPeriodicHMM(copy(m.ps), copy(m.stored), m.nT, cfg)

    # `baum_welch!` rather than `baum_welch`: the latter `deepcopy`s the guess, which would
    # duplicate the observation index sets the context holds.
    storage = HMMs.initialize_forward_backward(fitted, obs, controls; seq_ends=(N,))
    logtots = Float64[]
    trajectory = ps_iters ? [copy(fitted.ps)] : nothing

    # The E-step is run once up front and then once after every M-step, so that `logtots[k]` is
    # the loglikelihood the parameters *after* `k` M-steps attain — which makes
    # `logtots[end] == loglikelihood(fitted, …)`, the invariant the seasonal mixture's history
    # also has. `HiddenMarkovModels`' own loop pushes before fitting instead, so its last entry
    # describes the previous parameters; the number of forward passes is the same either way.
    HMMs.forward_backward!(storage, fitted, obs, controls; seq_ends=(N,))
    logtot = sum(storage.logL)
    (display == :iter) && println("Iteration 0: logtot = $logtot")
    for it = 1:maxiter
        StatsAPI.fit!(fitted, storage, obs, controls; seq_ends=(N,))
        robust && (fitted.stored.A .+= eps())
        ps_iters && push!(trajectory, copy(fitted.ps))

        HMMs.forward_backward!(storage, fitted, obs, controls; seq_ends=(N,))
        push!(logtots, sum(storage.logL))
        (display == :iter) && println("Iteration $it: logtot = $(round(logtots[end], digits=6))")

        if abs(logtots[end] - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $(logtots[end])")
            return fitted, FitHistory(; converged=true, iterations=it, logtots, ps_iters=trajectory)
        end
        logtot = logtots[end]
    end
    (display in [:iter, :final]) &&
        println("EM has not converged after $maxiter iterations, logtot = $(isempty(logtots) ? NaN : logtots[end])")
    return fitted, FitHistory(; converged=false, iterations=maxiter, logtots, ps_iters=trajectory)
end

"""
    loglikelihood(m::SmoothARPeriodicHMM, Y, Y_past; n2t) -> Float64

The marginal loglikelihood of `Y` under `m`, from `HiddenMarkovModels`' forward pass.
"""
function StatsAPI.loglikelihood(m::SmoothARPeriodicHMM, Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool}; n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer})
    N = size(Y, 2)
    controls = SeasonalControls(n2t, conditional_to(Y, Y_past))
    return HMMs.logdensityof(m, [view(Y, :, n) for n = 1:N], controls; seq_ends=(N,))
end

# # The site-dependent variant
#
# The same shape as above: a context holding what the M-step needs beyond the parameters, one
# `StatsAPI.fit!`, and a `fit_mle` wrapper driving `HiddenMarkovModels`' forward-backward. Only
# the emission bookkeeping differs, because each site has its own memory order and its own
# number of coefficient blocks.

struct SmoothsHMMContext{I,A,AD,E,S}
    n_in_t::Vector{Vector{Int}}
    n_all::I
    basis_A::A
    basis_B::Matrix{Float64}
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
    dA = (size(m.θᴬ, 3) - 1) ÷ 2
    dB = (length(m.θᴮ[1, 1][1]) - 1) ÷ 2
    rain_cat = 2

    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [[findall(.&(view(Y, j, :) .== y, view(lag_cat, j, :) .== h))
                            for h = 1:size_orders[j], y = 0:1] for j = 1:D]
    n_all = [[n_per_category(tup..., n_in_t, n_occurence_history[j])
              for tup in Iterators.product(1:size_orders[j], 1:T, 1:rain_cat)] for j = 1:D]

    return SmoothsHMMContext(n_in_t, n_all, K ≥ 2 ? trig_basis(T, dA) : nothing,
        trig_basis(T, dB), [zeros(K, size_orders[j], T, rain_cat) for j = 1:D],
        solver, adtype, ensemble, solve_kwargs)
end

"""
    fit!(m::SmoothsARPeriodicHMM, fb_storage, obs_seq, control_seq; seq_ends)

One M-step of the site-dependent model. Identical in structure to the dense one: the
transition aggregation groups `ξ[n]` by the period the step *leaves*, and the emission
aggregation transposes `γ` into the `(observation, state)` layout the kernels read.
"""
function StatsAPI.fit!(m::SmoothsARPeriodicHMM, fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector, control_seq::AbstractVector; seq_ends)
    cfg = m.cfg
    cfg === nothing && throw(ArgumentError(
        "this model carries no fitting context; call `fit_mle(m, Y, Y_past; n2t)` rather than `baum_welch` directly"))
    (; γ, ξ) = fb_storage
    K, T = size(m.stored.A, 1), m.nT

    init = m.stored.a
    fill!(init, 0)
    for k in eachindex(seq_ends)
        t1, _ = HMMs.seq_limits(seq_ends, k)
        init .+= view(γ, :, t1)
    end
    init ./= sum(init)

    if cfg.basis_A !== nothing
        s_ξ = zeros(T, K, K)
        for t = 1:T, n in cfg.n_in_t[t]
            n == length(ξ) && continue
            @views s_ξ[t, :, :] .+= ξ[n]
        end
        _update_transitions!(m, s_ξ, cfg)
    end

    γₛ!(cfg.γₛ, transpose(γ), cfg.n_all)
    _update_emissions_sites!(m, cfg)

    return _refresh!(m)
end

# `_update_transitions!` is shared with the dense model through the `θᴬ` accessor below, since
# the transition parametrization is the same in both.
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
    fit_mle(m::SmoothsARPeriodicHMM, Y, Y_past; n2t, kwargs...) -> (m, history)

Fit a site-dependent periodic autoregressive HMM by Baum-Welch. `Y_past[j]` is site `j`'s own
initial condition, of length equal to its order. Same keywords and same `(model, history)`
return as the dense [`fit_mle`](@ref).
"""
function fit_mle(m::SmoothsARPeriodicHMM, Y::AbstractMatrix{<:Bool}, Y_past::AbstractVector;
    n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer},
    maxiter=100, tol=1e-3, display=:none, robust=false, ps_iters=false,
    solver=default_solver(), adtype=default_adtype(), ensemble=default_ensemble(),
    solve_kwargs=(;))
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0
    @argcheck 2 .^ length.(Y_past) == size(m.stored, 4) "the initial condition's orders $(length.(Y_past)) do not match the model's $(Int.(log2.(size(m.stored, 4))))"

    N = size(Y, 2)
    lag_cat = conditional_to(Y, Y_past)
    controls = SeasonalControls(n2t, lag_cat)
    obs = [view(Y, :, n) for n = 1:N]

    cfg = SmoothsHMMContext(m, Y, lag_cat, n2t; solver, adtype, ensemble, solve_kwargs)
    fitted = SmoothsARPeriodicHMM(copy(m.θᴬ), deepcopy(m.θᴮ), copy(m.stored), m.nT, cfg)

    storage = HMMs.initialize_forward_backward(fitted, obs, controls; seq_ends=(N,))
    logtots = Float64[]
    trajectory = ps_iters ? [(A=copy(fitted.θᴬ), B=deepcopy(fitted.θᴮ))] : nothing

    HMMs.forward_backward!(storage, fitted, obs, controls; seq_ends=(N,))
    logtot = sum(storage.logL)
    (display == :iter) && println("Iteration 0: logtot = $logtot")
    for it = 1:maxiter
        StatsAPI.fit!(fitted, storage, obs, controls; seq_ends=(N,))
        robust && (fitted.stored.A .+= eps())
        ps_iters && push!(trajectory, (A=copy(fitted.θᴬ), B=deepcopy(fitted.θᴮ)))

        HMMs.forward_backward!(storage, fitted, obs, controls; seq_ends=(N,))
        push!(logtots, sum(storage.logL))
        (display == :iter) && println("Iteration $it: logtot = $(round(logtots[end], digits=6))")

        if abs(logtots[end] - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $(logtots[end])")
            return fitted, FitHistory(; converged=true, iterations=it, logtots, ps_iters=trajectory)
        end
        logtot = logtots[end]
    end
    (display in [:iter, :final]) &&
        println("EM has not converged after $maxiter iterations, logtot = $(isempty(logtots) ? NaN : logtots[end])")
    return fitted, FitHistory(; converged=false, iterations=maxiter, logtots, ps_iters=trajectory)
end

"""
    loglikelihood(m::SmoothsARPeriodicHMM, Y, Y_past; n2t) -> Float64

The marginal loglikelihood of `Y` under `m`, from `HiddenMarkovModels`' forward pass.
"""
function StatsAPI.loglikelihood(m::SmoothsARPeriodicHMM, Y::AbstractMatrix{<:Bool},
    Y_past::AbstractVector; n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer})
    N = size(Y, 2)
    controls = SeasonalControls(n2t, conditional_to(Y, Y_past))
    return HMMs.logdensityof(m, [view(Y, :, n) for n = 1:N], controls; seq_ends=(N,))
end

# # The spatial variant
#
# Two things make this one different from its siblings, and both are about the emission.
#
# It is **expensive and threaded**: one `D`-dimensional orthant probability per (observation,
# state), estimated by quasi-Monte Carlo, and `loglikelihoods!` parallelises over observations.
# `HiddenMarkovModels`' forward pass is necessarily sequential, so evaluating the emission one
# control at a time costs 8-10x (measured at `D = 10..20`, `N = 300`, 28 threads; caching the
# `D x D` kernel recovers none of it, because `mvnormcdf` dominates). `CachedEmissions` below
# is how both are had at once.
#
# It is also **stochastic**: `mvnormcdf` is randomized, so two E-steps at the same parameters
# disagree and the loglikelihood path is noisy. That is why `fit_mle` returns the best iterate
# rather than the last.

"""
    CachedEmissions(model, LL)

An `AbstractHMM` that answers `initialization` and `transition_matrix` from `model` but reads
its emission loglikelihoods out of `LL`, an `N x K` matrix filled beforehand.

Internal to the spatial fitter. It exists so the expensive emission can be evaluated by the
threaded [`loglikelihoods!`](@ref) in one batch while the forward-backward recursions and the
transition marginals still come from `HiddenMarkovModels`. The cache cannot go stale: it is
created, refilled and used inside a single `fit_mle` call and never escapes it — which is also
why [`SpatialControls`](@ref) carries the observation index, the only way `obs_logdensities!`
can find its row.
"""
struct CachedEmissions{M} <: HMMs.AbstractHMM
    model::M
    LL::Matrix{Float64}
end

HMMs.initialization(c::CachedEmissions) = HMMs.initialization(c.model)
HMMs.transition_matrix(c::CachedEmissions, ctl::Tuple) = HMMs.transition_matrix(c.model, ctl)
HMMs.obs_distributions(c::CachedEmissions, ctl::Tuple) = HMMs.obs_distributions(c.model, ctl)

function HMMs.obs_logdensities!(logb::AbstractVector, c::CachedEmissions, obs, ctl::Tuple;
    error_if_not_finite::Bool=true)
    logb .= view(c.LL, ctl[3], :)
    return nothing
end

struct SpatialHMMContext{I,A,AD,E,S,W,X,YY}
    n_in_t::Vector{Vector{Int}}
    n_all::I
    Y::YY                   # the D x N observations; `update_R!` needs the matrix, not the views
    basis_A::A
    basis_B::Matrix{Float64}
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
    dA, dB, _ = degrees(m)
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

    return SpatialHMMContext(n_in_t, n_all, Y, K >= 2 ? trig_basis(T, dA) : nothing,
        trig_basis(T, dB), situation_idx, wp, Int(QMC_E), Int(maxiters_R),
        solver, adtype, ensemble, solve_kwargs)
end

"""
    fit!(m::SmoothARPeriodicHMMSpatial, fb_storage, obs_seq, control_seq; seq_ends)

One M-step: the shared transition and emission refits, then the spatial range through
[`update_R!`](@ref), which maximises a *composite* (pairwise) likelihood rather than the full
one — deterministic and `ForwardDiff`-differentiable, where the full emission is neither.
"""
function StatsAPI.fit!(m::SmoothARPeriodicHMMSpatial, fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector, control_seq::AbstractVector; seq_ends)
    cfg = m.cfg
    cfg === nothing && throw(ArgumentError(
        "this model carries no fitting context; call `fit_mle(m, Y, Y_past; n2t)` rather than `baum_welch` directly"))
    (; γ, ξ) = fb_storage
    K, T = size(m.stored.A, 1), m.nT
    D, size_order = size(m.ps.B, 2), size(m.ps.B, 3)

    init = m.stored.a
    fill!(init, 0)
    for k in eachindex(seq_ends)
        t1, _ = HMMs.seq_limits(seq_ends, k)
        init .+= view(γ, :, t1)
    end
    init ./= sum(init)

    if cfg.basis_A !== nothing
        s_ξ = zeros(T, K, K)
        for t = 1:T, n in cfg.n_in_t[t]
            n == length(ξ) && continue
            @views s_ξ[t, :, :] .+= ξ[n]
        end
        _update_transitions!(m, s_ξ, cfg)
    end

    γₛ = zeros(K, D, size_order, T, 2)
    γₛ!(γₛ, transpose(γ), cfg.n_all)
    _update_emissions!(m, γₛ, cfg)

    # `update_R!` reads `B` and writes `R`, so the emission coefficients have to be refreshed
    # into the stored arrays before it runs.
    _refresh!(m)
    update_R!(m.stored, m.ps.R, permutedims(γ), cfg.wp, cfg.Y, cfg.situation_idx;
        n2t=_controls_n2t(control_seq), solver=cfg.solver, adtype=cfg.adtype,
        maxiters=cfg.maxiters_R, solve_kwargs=cfg.solve_kwargs)
    return m
end

_theta_A(m::SmoothARPeriodicHMMSpatial) = m.ps.A
_controls_n2t(c::SpatialControls) = c.n2t
_controls_n2t(c::SeasonalControls) = c.n2t

"""
    fit_mle(m::SmoothARPeriodicHMMSpatial, Y, Y_past; n2t, kwargs...) -> (m, history)

Fit a spatial periodic HMM by Baum-Welch, starting from `m`.

**Returns the best iterate, not the last.** The emission likelihood is a randomized
quasi-Monte-Carlo estimate, so the loglikelihood path is noisy and the final step is not
reliably the best one; `history.best_iteration` says which was taken, and `history.logtots` and
`history.ps_iters` hold the whole path. Every other `fit_mle` in this package returns its last
iterate, because every other likelihood here is deterministic.

Keywords beyond the usual solver ones: `QMC_E` multiplies the model's `QMC_m` for the E-step
only, `maxiters_R` caps the inner solve of the range M-step, and `tdist`/`wp` choose which pairs
of sites enter the composite likelihood that M-step maximises.
"""
function fit_mle(m::SmoothARPeriodicHMMSpatial, Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool};
    n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer},
    maxiter=100, tol=1e-3, display=:none, robust=false, ps_iters=true,
    tdist=1, wp=1.0 .* (m.stored.h .< maximum(m.stored.h) * tdist),
    QMC_E=1, maxiters_R=10,
    solver=default_solver(), adtype=default_adtype(), ensemble=default_ensemble(),
    solve_kwargs=(;))
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N = size(Y, 2)
    lag_cat = conditional_to(Y, Y_past)
    controls = SpatialControls(n2t, lag_cat)
    obs = [view(Y, :, n) for n = 1:N]

    cfg = SpatialHMMContext(m, Y, lag_cat, n2t; wp, QMC_E, maxiters_R,
        solver, adtype, ensemble, solve_kwargs)
    fitted = SmoothARPeriodicHMMSpatial(copy(m.ps), deepcopy(m.stored), m.nT, cfg)

    K = size(fitted.stored.A, 1)
    cached = CachedEmissions(fitted, zeros(N, K))
    storage = HMMs.initialize_forward_backward(cached, obs, controls; seq_ends=(N,))
    logtots = Float64[]
    trajectory = ps_iters ? [copy(fitted.ps)] : nothing

    # The threaded batch evaluation, which is the whole reason `CachedEmissions` exists.
    function estep!()
        loglikelihoods!(cached.LL, fitted.stored, Y, lag_cat; n2t=n2t,
            QMC_m=fitted.stored.QMC_m * cfg.QMC_E)
        robust && replace!(cached.LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
        HMMs.forward_backward!(storage, cached, obs, controls; seq_ends=(N,))
        return sum(storage.logL)
    end

    logtot = estep!()
    (display == :iter) && println("Iteration 0: logtot = $logtot")
    best_logtot, best_ps = logtot, copy(fitted.ps)
    converged = false
    it_done = 0

    for it = 1:maxiter
        ps_before = copy(fitted.ps)
        StatsAPI.fit!(fitted, storage, obs, controls; seq_ends=(N,))
        robust && (fitted.stored.A .+= eps())
        ps_iters && push!(trajectory, copy(fitted.ps))

        logtotp = estep!()
        push!(logtots, logtotp)
        it_done = it
        (display == :iter) && println("Iteration $it: logtot = $(round(logtotp, digits=6))")

        if logtotp > best_logtot
            best_logtot, best_ps = logtotp, copy(fitted.ps)
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
    _refresh!(fitted)
    (display in [:iter, :final]) && !converged &&
        println("EM has not converged after $it_done iterations, logtot = $best_logtot")
    return fitted, FitHistory(; converged, iterations=it_done, logtots, ps_iters=trajectory)
end

"""
    loglikelihood(m::SmoothARPeriodicHMMSpatial, Y, Y_past; n2t) -> Float64

The marginal loglikelihood of `Y` under `m`. Stochastic: the emission is a quasi-Monte-Carlo
estimate, so repeated calls differ. Raise the model's `QMC_m` to tighten it.
"""
function StatsAPI.loglikelihood(m::SmoothARPeriodicHMMSpatial, Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool}; n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer})
    N, K = size(Y, 2), size(m.stored.A, 1)
    lag_cat = conditional_to(Y, Y_past)
    controls = SpatialControls(n2t, lag_cat)
    obs = [view(Y, :, n) for n = 1:N]
    cached = CachedEmissions(m, zeros(N, K))
    loglikelihoods!(cached.LL, m.stored, Y, lag_cat; n2t=n2t)
    return HMMs.logdensityof(cached, obs, controls; seq_ends=(N,))
end
