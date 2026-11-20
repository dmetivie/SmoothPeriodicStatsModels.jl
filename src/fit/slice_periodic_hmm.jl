# # A starting point for `SmoothPeriodicHMM`, one period index at a time
#
# The slice method of the Bernoulli HMMs, for any emission. At each period index, a mixture of
# the `K` states' emission laws with constant parameters is fitted by EM to the observations of a
# window of neighbouring indices, ignoring the dynamics. The states are put in the same order at
# every index, the transitions are fitted to the steps between the most likely states, and the
# per-index estimates are smoothed into the coefficients of the model.
#
# Each law is fitted through its family with every basis made constant: its coefficients are then
# the linear predictors themselves, one per parameter, which is what the smoothing step needs.
# This works for a law given as a function of its parameters, where the parameters cannot be
# read back from the law.

_constant_family(f::TimeVaryingFamily) = TimeVaryingFamily(f.dist, ParamSpec(;
    (name => (ConstantBasis() => link) for (name, link) in zip(paramnames(f.spec), f.spec.links))...))

"""
    fit_mle_all_slices([rng, ]m::SmoothPeriodicHMM, Y; n2t, Yₜ_extanted = [0], kwargs...) -> SmoothPeriodicHMM

A starting point for fitting `m`, estimated one period index at a time and ignoring the dynamics.
At each `t`, a mixture of the `K` states' emission laws, with constant parameters, is fitted by
EM to the observations at period indices `t .+ Yₜ_extanted`: once from the laws of `m` at `t`
and `n_random_ini - 1` times from random groupings of the observations, the best run being kept.
The states are ordered by decreasing `sortby`. The emission estimates are then smoothed into
the bases of `m` by least squares, and the transition coefficients fitted to the steps between
the most likely states of consecutive observations. `Y` may be a vector of sequences, with
`n2t` a vector of period-index vectors.

Keywords:
- `Yₜ_extanted = [0]`: the window, as offsets from `t`; `-2:2` pools five period indices.
- `periodic = true`: whether the window wraps around the cycle; `false` for an index that is not
  periodic.
- `n_random_ini = 5`: the number of EM runs at each period index.
- `sortby`: a function of the tuple of a state's laws, one per block, returning the number the
  states are ordered by; by default the mean of the first row, `laws -> first(mean(first(laws)))`.
- `slices = true` also returns the estimates at each period index, as a `PeriodicHMMDists`.
- `robust`, `maxiter` and `tol` of each EM run, `solver`, `adtype` and `solve_kwargs` of its
  M-step; `ensemble = EnsembleThreads()` fits the period indices in parallel.
"""
fit_mle_all_slices(m::SmoothPeriodicHMM, Y::AbstractVecOrMat{<:Real}; kwargs...) =
    fit_mle_all_slices(GLOBAL_RNG, m, Y; kwargs...)

function fit_mle_all_slices(rng::AbstractRNG, m::SmoothPeriodicHMM, Y::AbstractVecOrMat{<:Real};
    n2t=n_to_t(_ncols(Y), m.nT)::AbstractVector{<:Integer}, kwargs...)
    return _fit_slices(rng, m, _sequences(Y, n2t)...; kwargs...)
end

function fit_mle_all_slices(rng::AbstractRNG, m::SmoothPeriodicHMM, Ys::AbstractVector{<:AbstractVecOrMat{<:Real}};
    n2t=[n_to_t(_ncols(Y), m.nT) for Y in Ys], kwargs...)
    return _fit_slices(rng, m, _sequences(Ys, n2t)...; kwargs...)
end

fit_mle_all_slices(m::SmoothPeriodicHMM, Ys::AbstractVector{<:AbstractVecOrMat{<:Real}}; kwargs...) =
    fit_mle_all_slices(GLOBAL_RNG, m, Ys; kwargs...)

_default_sortby(laws) = first(mean(first(laws)))

function _fit_slices(rng::AbstractRNG, m::SmoothPeriodicHMM, Y, n2t, seq_ends;
    Yₜ_extanted=[0], periodic=true, n_random_ini=5, sortby=_default_sortby, slices=false,
    robust=false, maxiter=100, tol=1e-3,
    solver=_required_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;))
    @argcheck size(Y, 1) == size(m, 2) "the observations have $(size(Y, 1)) rows and the emission blocks cover $(size(m, 2))"
    @argcheck n_random_ini >= 1
    K, T = size(m.stored.A, 1), m.nT
    fams, blocks = m.families, m.stored.blocks
    cfg = (; solver, adtype, solve_kwargs, maxiter, tol, robust, n_random_ini, sortby)

    designs = map(f -> PeriodicDesign(f.spec, 1:T), fams)
    fams0 = map(_constant_family, fams)
    designs0 = map(f -> PeriodicDesign(f.spec, 1:1), fams0)
    univariate = map(Bb -> eltype(Bb) <: UnivariateDistribution, m.stored.B)
    # the linear predictors of `m`: where each slice's first EM run, and every M-step, starts
    η_model = [linear_predictors(designs[b], emission_coefs(m, k, b)) for k = 1:K, b in eachindex(fams)]

    n_in_t = group_by_time(n2t, T)
    windows = map(1:T) do t
        ts = periodic ? unique(mod1.(t .+ Yₜ_extanted, T)) : filter(in(1:T), unique(t .+ Yₜ_extanted))
        sort!(reduce(vcat, (n_in_t[s] for s in ts); init=Int[]))
    end

    # one seed per period index, so that the result does not depend on the threading
    seeds = rand(rng, UInt, T)
    fits = Vector{Any}(undef, T)
    fit_index(t) = (fits[t] = _fit_slice(Xoshiro(seeds[t]), t, windows[t], Y, blocks, univariate,
        fams0, designs0, η_model, cfg))
    if ensemble isa EnsembleThreads
        Threads.@threads for t = 1:T
            fit_index(t)
        end
    else
        foreach(fit_index, 1:T)
    end
    fitted = findall(!isnothing, fits)
    isempty(fitted) && throw(ArgumentError("no period index could be fitted; check the data and the starting model"))
    length(fitted) < T && @warn "$(T - length(fitted)) of the $T period indices had no usable fit and are left out of the smoothing"

    # the estimates at every period index; where there is none, the laws of `m`
    α = fill(1 / K, K, T)
    foreach(t -> α[:, t] = fits[t].α, fitted)
    laws_m = PeriodicHMMDists(m).B
    B = map(eachindex(fams)) do b
        [fits[t] === nothing ? laws_m[b][k, t] : fits[t].laws[k, b] for k = 1:K, t = 1:T]
    end

    # transitions: the frequencies of the most likely state of each observation
    hmm = PeriodicHMMDists(fill(1 / K, K), fill(1 / K, K, K, T), Tuple(B))
    LL = loglikelihoods(hmm, Y; n2t, robust)
    for n in axes(LL, 1), k = 1:K
        LL[n, k] += log(α[k, n2t[n]])
    end
    a, counts = _slice_transitions(LL, n2t, seq_ends, K, T)

    # the emission estimates, smoothed into the bases of `m`
    θᴮ = map(Iterators.product(1:K, eachindex(fams))) do (k, b)
        spec = fams[b].spec
        reduce(vcat, [_smooth_parameter(designs[b].designs[j][fitted, :],
                [fits[t].η[k, b][j] for t in fitted], spec.links[j]; solver, adtype, solve_kwargs)
                      for j in eachindex(paramnames(spec))])
    end
    XA = design_matrix(m.basis_A, 1:T)
    start = SmoothPeriodicHMM(fams, a, zeros(K, K - 1, size(XA, 2)), θᴮ, T; basis_A=m.basis_A)
    # the transition coefficients: the M-step of the fit, on the counted steps
    if K > 1
        _update_transitions!(start, counts, (; basis_A=XA, solver, adtype, ensemble, solve_kwargs))
        _refresh!(start)
    end
    slices || return start
    # with the estimates at each period index, whose transitions are the frequencies of the steps
    frequencies = permutedims(counts .+ eps(), (2, 3, 1))
    return start, PeriodicHMMDists(a, frequencies ./ sum(frequencies; dims=2), Tuple(B))
end

# The mixture at one period index: `nothing` when the window is too small or every EM run fails.
function _fit_slice(rng, t, idx, Y, blocks, univariate, fams0, designs0, η_model, cfg)
    K, nb = size(η_model)
    n = length(idx)
    n >= K || return nothing
    # the observations of each block in the window: numbers for a univariate law, vectors otherwise
    ys = map((u, r) -> u ? Y[first(r), idx] : [Y[r, i] for i in idx], univariate, blocks)
    Z = _standardized(view(Y, :, idx))
    LL, γ = zeros(n, K), zeros(n, K)
    best = nothing
    for run = 1:cfg.n_random_ini
        α = fill(1 / K, K)
        η = [Float64[ηj[t] for ηj in η_model[k, b]] for k = 1:K, b = 1:nb]
        laws = Any[only(distributions(fams0[b], designs0[b], η[k, b])) for k = 1:K, b = 1:nb]
        # a law can refuse a parameter value the solver reaches (a correlation of 1, say): that
        # run is dropped, and the others stand
        logtot = try
            run > 1 && _random_grouping!(γ, rng, Z)
            _slice_em!(α, η, laws, LL, γ, ys, fams0, designs0, cfg; from_grouping=run > 1)
        catch err
            err isa InterruptException && rethrow()
            -Inf
        end
        if isfinite(logtot) && (best === nothing || logtot > best.logtot)
            best = (; α, η, laws, logtot)
        end
    end
    best === nothing && return nothing
    order = sortperm([cfg.sortby(Tuple(best.laws[k, :])) for k = 1:K]; rev=true)
    return (; α=best.α[order], η=best.η[order, :], laws=best.laws[order, :], best.logtot)
end

function _standardized(Y)
    Z = Matrix{Float64}(Y)
    for row in eachrow(Z)
        μ, σ = mean(row), std(row)
        row .= (row .- μ) ./ (σ > 0 ? σ : one(σ))
    end
    return Z
end

# Random soft groups: `K` observations are drawn as centres, and each observation is shared
# between the groups according to its distance to them. Every group gets some weight, so no
# state starts empty.
function _random_grouping!(γ, rng, Z)
    n, K = size(γ)
    centres = randperm(rng, n)[1:K]
    for i = 1:n
        for (k, c) in enumerate(centres)
            γ[i, k] = exp(-sum(abs2, view(Z, :, i) .- view(Z, :, c)) / 2)
        end
        s = sum(view(γ, i, :))
        s > 0 ? (γ[i, :] ./= s) : (γ[i, :] .= 1 / K)
    end
    return γ
end

# EM for the mixture of one window. `from_grouping` starts with an M-step on the given `γ`.
function _slice_em!(α, η, laws, LL, γ, ys, fams0, designs0, cfg; from_grouping)
    logtot = from_grouping ? -Inf : _slice_estep!(LL, γ, α, laws, ys, cfg.robust)
    for _ = 1:cfg.maxiter
        _slice_mstep!(α, η, laws, γ, ys, fams0, designs0, cfg)
        new = _slice_estep!(LL, γ, α, laws, ys, cfg.robust)
        done = abs(new - logtot) < cfg.tol
        logtot = new
        (done || !isfinite(new)) && break
    end
    return logtot
end

function _slice_estep!(LL, γ, α, laws, ys, robust)
    for k in axes(LL, 2)
        col = view(LL, :, k)
        fill!(col, log(α[k]))
        for b in eachindex(ys)
            _add_logpdf!(col, laws[k, b], ys[b])
        end
    end
    robust && _clamp_inf!(LL)
    logtot = 0.0
    for n in axes(LL, 1)
        c = logsumexp(view(LL, n, :))
        logtot += c
        for k in axes(LL, 2)
            γ[n, k] = exp(LL[n, k] - c)
        end
    end
    return logtot
end

# a function barrier: the laws of a window are held untyped
function _add_logpdf!(col, law, y)
    @inbounds for n in eachindex(col, y)
        col[n] += logpdf(law, y[n])
    end
    return col
end

function _slice_mstep!(α, η, laws, γ, ys, fams0, designs0, cfg)
    n = size(γ, 1)
    for k in eachindex(α)
        γₖ = view(γ, :, k)
        α[k] = sum(γₖ) / n
        group = [findall(>(_NEGLIGIBLE_WEIGHT), γₖ)]
        for b in eachindex(ys)
            dists, θ = update_dist(fams0[b], designs0[b], η[k, b], γₖ, ys[b], group;
                cfg.solver, cfg.adtype, cfg.solve_kwargs)
            η[k, b], laws[k, b] = θ, only(dists)
        end
    end
    return nothing
end

# The initial distribution, and the number of steps from state `k` to `l` leaving period index
# `t`, `counts[t, k, l]`, between the most likely states of consecutive observations of a
# sequence.
function _slice_transitions(LL, n2t, seq_ends, K, T)
    z = [argmax(view(LL, n, :)) for n in axes(LL, 1)]
    counts = zeros(T, K, K)
    a = zeros(K)
    t1 = 1
    for t2 in seq_ends
        first_row = view(LL, t1, :)
        a .+= exp.(first_row .- logsumexp(first_row))
        for n = t1:(t2-1)
            counts[n2t[n], z[n], z[n+1]] += 1
        end
        t1 = t2 + 1
    end
    a = all(isfinite, a) && sum(a) > 0 ? a ./ sum(a) : fill(1 / K, K)
    return a, counts
end

# The coefficients of one parameter from its linear predictors `η` at the period indices of the
# design `X`: least squares on the parameter itself, started from least squares on `η`. Fitting
# the parameter rather than `η` keeps a saturated estimate (a probability of zero, whose logit
# the solver left at some large negative number) from dragging the curve.
function _smooth_parameter(X, η, link; solver, adtype, solve_kwargs)
    keep = findall(isfinite, η)
    X, η = X[keep, :], η[keep]
    c = X \ η
    link isa IdentityLink && return c
    θ = apply_link.(Ref(link), η)
    return solve_unconstrained(_parameter_residual, c, (X, θ, link); solver, adtype, solve_kwargs)
end

_parameter_residual(c, (X, θ, link)) = sum(abs2, apply_link.(Ref(link), X * c) .- θ) / length(θ)
