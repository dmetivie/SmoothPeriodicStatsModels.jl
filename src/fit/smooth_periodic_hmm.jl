# # Fitting the periodic HMM with any emission
#
# Baum-Welch with the loop the Bernoulli models use. The transition and initialization M-steps
# are theirs too. The emission M-step is the seasonal mixture's: for each state and block, a
# weighted maximum likelihood fit of the block's family, the weights being the posterior
# probabilities of that state, through `update_dist`.

"""
    DistHMMContext(n_in_t, basis_A, designs, ys, solver, adtype, ensemble, solve_kwargs)

What the M-step of a [`SmoothPeriodicHMM`](@ref) needs besides the posteriors. It holds the
observations of each period index, the transition design matrix, the design of each emission block, the
observations of each block, and the solver settings. [`fit_mle`](@ref) builds it and stores it in
the model's `cfg` field.
"""
struct DistHMMContext{A,DS<:Tuple,YS<:Tuple,AD,E,S}
    n_in_t::Vector{Vector{Int}}
    basis_A::A       # `nothing` when K == 1: a one-state chain has no transition to fit
    designs::DS
    ys::YS
    solver::S
    adtype::AD
    ensemble::E
    solve_kwargs::NamedTuple
end

function DistHMMContext(m::SmoothPeriodicHMM, Y::AbstractMatrix, n2t::AbstractVector{<:Integer};
    solver, adtype, ensemble, solve_kwargs)
    K, T = size(m.stored.A, 1), m.nT
    designs = map(f -> PeriodicDesign(f.spec, 1:T), m.families)
    ys = map((Bb, r) -> _block_observations(Bb, Y, r), m.stored.B, m.stored.blocks)
    return DistHMMContext(group_by_time(n2t, T), K ≥ 2 ? design_matrix(m.basis_A, 1:T) : nothing,
        designs, ys, solver, adtype, ensemble, solve_kwargs)
end

# A univariate block is one row of `Y`, a multivariate block one view of its rows per observation.
_block_observations(::AbstractMatrix{<:UnivariateDistribution}, Y, r) = view(Y, first(r), :)
_block_observations(::AbstractMatrix, Y, r) = [view(Y, r, n) for n in axes(Y, 2)]

"""
    fit!(m::SmoothPeriodicHMM, fb_storage, obs_seq, control_seq; seq_ends)

One M-step: refit the initial distribution, the transition coefficients and, for every state and
block, the emission coefficients, then re-evaluate the model.
"""
function StatsAPI.fit!(m::SmoothPeriodicHMM, fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector, control_seq::AbstractVector; seq_ends)
    cfg = m.cfg
    cfg === nothing && throw(ArgumentError(
        "this model carries no fitting context; call `fit_mle(m, Y; n2t)` rather than `baum_welch` directly"))
    (; γ, ξ) = fb_storage
    K, T = size(m.stored.A, 1), m.nT

    _update_initialization!(m, γ, seq_ends)
    cfg.basis_A !== nothing && _update_transitions!(m, _transition_counts(ξ, cfg.n_in_t, K, T), cfg)
    _update_emission_blocks!(m, γ, cfg)
    return _refresh!(m)
end

_theta_A(m::SmoothPeriodicHMM) = m.ps.A

# Each state's fit leaves out the observations whose posterior weight in that state is at most
# this. They add nothing to its estimate, at most `1e-10 × |logdensity|` each to its objective,
# but their density can still break the solve: an observation far in the tail of a state's law
# can have density zero there in floating point. A copula law, for one, evaluates its marginal
# cdfs, which round to 0 or 1 in the far tail. The objective is then infinite right next to the
# current coefficients, the line search cannot take a step, and that state stops moving while
# the loglikelihood of the others settles. Measured on a Gaussian copula with Weibull or
# LogNormal marginals, that is a fit stuck at its start, or ending 100–300 nats below the true
# parameters; with the cutoff, 20 nats above them.
const _NEGLIGIBLE_WEIGHT = 1e-10

# One weighted fit per (state, block). They write to disjoint coefficients, so with
# `EnsembleThreads()` they run in parallel.
function _update_emission_blocks!(m::SmoothPeriodicHMM, γ, cfg)
    K = size(m.stored.A, 1)
    groups = [[filter(n -> γ[k, n] > _NEGLIGIBLE_WEIGHT, g) for g in cfg.n_in_t] for k = 1:K]
    jobs = vec(collect(Iterators.product(1:K, eachindex(m.families))))
    function solve_job(i)
        k, b = jobs[i]
        θ = emission_coefs(m, k, b)
        _, θ̂ = update_dist(m.families[b], cfg.designs[b], θ, view(γ, k, :), cfg.ys[b], groups[k];
            cfg.solver, cfg.adtype, cfg.solve_kwargs)
        θ .= θ̂
        return nothing
    end
    if cfg.ensemble isa EnsembleThreads
        Threads.@threads for i in eachindex(jobs)
            solve_job(i)
        end
    else
        foreach(solve_job, eachindex(jobs))
    end
    return m
end

_snapshot(m::SmoothPeriodicHMM) = _plain(m.ps)
_detach(m::SmoothPeriodicHMM) = SmoothPeriodicHMM(m.ps, m.families, m.stored, m.nT, m.basis_A, nothing, m.stored_from)

"""
    fit_mle(m::SmoothPeriodicHMM, Y; n2t, solver, kwargs...) -> (m, history)
    fit_mle(m::SmoothPeriodicHMM, Ys; n2t, solver, kwargs...) -> (m, history)

Fit a periodic HMM with any emission by Baum-Welch, starting from `m`. `Y` is `D × N`, or a
vector when `D = 1`; there are no initial observations, the model having no memory. For several
independent sequences, pass a vector of them and a vector of period-index vectors. Returns the
fitted model and a [`FitHistory`](@ref).

Keywords: `maxiter`, `tol` (absolute change of the loglikelihood), `display`,
`ps_iters = true` to keep the coefficients of every iteration, and the solver settings `solver`,
`adtype`, `ensemble` and `solve_kwargs` of the M-step (`ensemble = EnsembleThreads()` fits the
emission blocks in parallel). `robust = true` clamps infinite emission loglikelihoods in the
E-step, so that an observation impossible in every state does not stop the fit.

The emission fit of each state leaves out the observations whose posterior probability in that
state is at most `1e-10`.

With `seq_ends`, `Y` is several independent sequences end to end. `past` is checked and unused,
the model having no memory; a `past.z` is refused, every sequence starting from the fitted
initial distribution.
"""
function fit_mle(m::SmoothPeriodicHMM, Y::AbstractVecOrMat{<:Real};
    n2t=n_to_t(_ncols(Y), m.nT)::AbstractVector{<:Integer}, past=nothing, seq_ends=nothing, kwargs...)
    Yc, n2tc, ends = _sequences(Y, n2t)
    seq_ends === nothing || (ends = _check_seq_ends(seq_ends, size(Yc, 2)))
    _memoryless_pasts(m, past, seq_ends === nothing ? nothing : length(ends))
    return _fit_mle(m, Yc, n2tc, ends; kwargs...)
end

function fit_mle(m::SmoothPeriodicHMM, Ys::AbstractVector{<:AbstractVecOrMat{<:Real}};
    n2t=[n_to_t(_ncols(Y), m.nT) for Y in Ys], past=nothing, kwargs...)
    _memoryless_pasts(m, past, length(Ys))
    return _fit_mle(m, _sequences(Ys, n2t)...; kwargs...)
end

function _fit_mle(m::SmoothPeriodicHMM, Y, n2t, seq_ends;
    maxiter=100, tol=1e-3, display=:none, robust=false, ps_iters=false,
    solver=_required_solver(), adtype=default_adtype(), ensemble=default_ensemble(),
    solve_kwargs=(;))
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0
    @argcheck size(Y, 1) == size(m, 2) "the observations have $(size(Y, 1)) rows and the emission blocks cover $(size(m, 2))"
    cfg = DistHMMContext(m, Y, n2t; solver, adtype, ensemble, solve_kwargs)
    fitted = SmoothPeriodicHMM(copy(m.ps), m.families, copy(m.stored), m.nT, m.basis_A, cfg)
    _refresh!(fitted)   # the coefficients may have been edited in place since `m` was built
    return _baum_welch!(fitted, _columns(Y), SeasonalControls(n2t), seq_ends;
        maxiter, tol, display, robust, ps_iters)
end

"""
    fit(SeasonalFit, m::SmoothPeriodicHMM, Y; n2t, solver, kwargs...) -> SeasonalFit

Fit with [`fit_mle`](@ref), same keywords, and return a [`SeasonalFit`](@ref) for model
selection. `nobs` is the number of observed steps, summed over the sequences.
"""
function StatsAPI.fit(::Type{SeasonalFit}, m::SmoothPeriodicHMM, Y; kwargs...)
    fitted, history = fit_mle(m, Y; kwargs...)
    return SeasonalFit(fitted, history, _nsteps(Y), loglikelihood(history))
end
