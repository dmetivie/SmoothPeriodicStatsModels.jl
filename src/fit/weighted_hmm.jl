# # The complete-data M-step: a smooth HMM fitted given its hidden states
#
# One M-step of Baum-Welch with the posteriors replaced by given state weights: a known path,
# `OneHot(z, K)`, or soft weights. It is how sites are added to an HMM whose states are already
# estimated, and it reuses `update_B!`, the emission M-step of `fit_mle_RO`, so that from a model
# of zero coefficients it gives the coefficients `fit_mle_RO` gives. Its objective is that sum,
# not an average.

"""
    fit_mle(m::SmoothARPeriodicHMM, Y, W; n2t, past, fixed = (:A,), solver, adtype, ensemble, solve_kwargs) -> SmoothARPeriodicHMM

The emission coefficients, and the transitions unless they are `fixed`, of the HMM `m` fitted to
the `D × N` observations `Y` given the state weights `W`: `OneHot(z, K)` for a known state path,
or an `N × K` matrix of floats. Starts from `parameters(m)` and returns the fitted model alone.

- `past = (; Y = Y_past)` holds the observations before `Y`, at least as many as the memory.
- `fixed` lists the parts kept as in `m`: `:A`, the transitions and the initial distribution,
  and `:B`, the emissions. Fitting `:A` needs `W = OneHot(z, K)`: the transitions are then fitted
  to the steps of `z`, and the initial distribution is the state `z[1]`.

From `SmoothARPeriodicHMM(K, D, T; memory, season = d)`, the emission coefficients are those of
`fit_mle_RO(Y, Y_past, z, n2t, d, K, T)`. A dense `Bool` matrix third argument is `Y_past`, the
positional form of Baum-Welch: pass weights as `OneHot` or floats.
"""
function fit_mle(m::SmoothARPeriodicHMM, Y::AbstractMatrix{<:Bool},
    W::Union{OneHot,AbstractMatrix{<:AbstractFloat}};
    n2t=n_to_t(size(Y, 2), m.nT)::AbstractVector{<:Integer}, past=nothing, fixed=(:A,),
    solver=_required_solver(), adtype=default_adtype(), ensemble=default_ensemble(),
    solve_kwargs=(;))
    K, D, T, size_order = size(m)
    N = size(Y, 2)
    @argcheck size(Y, 1) == D "`Y` has $(size(Y, 1)) rows for $D sites"
    _check_n2t(n2t, N, T)
    W = _state_weights(W, N, K)
    fixed = _fixed_parts(fixed)
    Y_past, z0 = _hmm_past(m, past; required=true)
    z0 === nothing || _no_past_z()
    lag_cat = _lag_cat(m, Y, Y_past)

    fitted = SmoothARPeriodicHMM(copy(m.ps), copy(m.stored), m.nT, m.basis_A, m.basis_B, nothing)
    _refresh!(fitted)   # the coefficients may have been edited in place since `m` was built
    if !(:B in fixed)
        cfg = SmoothHMMContext(fitted, Y, lag_cat, n2t; solver, adtype, ensemble, solve_kwargs)
        update_B!(fitted.stored.B, fitted.ps.B, W, zeros(K, D, size_order, T, 2), Y, cfg.n_all,
            cfg.basis_B; solver, adtype, ensemble, solve_kwargs)
    end
    if !(:A in fixed) && K >= 2
        W isa OneHot || throw(ArgumentError(
            "fitting the transitions needs consecutive states: pass OneHot(z, K), or fixed = (:A,)"))
        z = W.z
        counts = zeros(T, K, K)
        for n = 1:N-1
            counts[n2t[n], z[n], z[n+1]] += 1
        end
        _update_transitions!(fitted, counts,
            (; basis_A=design_matrix(m.basis_A, 1:T), solver, adtype, ensemble, solve_kwargs))
        N >= 1 && (fill!(fitted.stored.a, 0); fitted.stored.a[z[1]] = 1)
    end
    return _refresh!(fitted)
end

# `fixed`, checked: some of `:A` and `:B`.
function _fixed_parts(fixed)
    parts = fixed isa Symbol ? (fixed,) : Tuple(fixed)
    for p in parts
        p in (:A, :B) || throw(ArgumentError("`fixed` lists `:A`, `:B` or both, not `$(repr(p))`"))
    end
    return parts
end

# The spatial model has no fit given state weights in 4.2; without this, the positional
# Baum-Welch would read a `OneHot` as `Y_past`.
fit_mle(::SmoothARPeriodicHMMSpatial, ::AbstractMatrix{<:Bool}, ::OneHot; kwargs...) =
    throw(ArgumentError("the fit given state weights is defined for `SmoothARPeriodicHMM`; for the spatial model, call Baum-Welch with `past`"))
