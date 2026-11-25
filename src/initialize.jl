# # Starting points of the smooth HMMs
#
# Baum-Welch needs states that start apart. `initialize` estimates them from the data by the
# slice method: a mixture fitted at every period index, ignoring the dynamics, then smoothed into
# the bases of the model.

"""
    SliceEM(window = [0], n_random = 10, dirichlet = 0.8)
    SliceEM(; window, n_random, dirichlet)

The slice method of [`initialize`](@ref). At each period index `t`, a mixture of the states'
emissions is fitted by EM to the observations at the period indices `t .+ window`, from
`n_random` starts whose weights are drawn from `Dirichlet(K, dirichlet)`, and the best run is
kept; the transitions are then those between the most likely states of consecutive
observations.
"""
struct SliceEM{W<:AbstractVector{<:Integer}}
    window::W
    n_random::Int
    dirichlet::Float64
    function SliceEM(window::AbstractVector{<:Integer}, n_random::Integer, dirichlet::Real)
        isempty(window) && throw(ArgumentError("the window of `SliceEM` holds at least one offset"))
        n_random >= 1 || throw(ArgumentError("`n_random` is the number of EM runs, at least 1, not $n_random"))
        isfinite(dirichlet) && dirichlet > 0 ||
            throw(ArgumentError("`dirichlet` is the concentration of the random weights, positive, not $dirichlet"))
        return new{typeof(window)}(window, Int(n_random), Float64(dirichlet))
    end
end

SliceEM(; window=[0], n_random=10, dirichlet=0.8) = SliceEM(window, n_random, dirichlet)

"""
    initialize([rng, ]m0::SmoothARPeriodicHMM, Y; n2t, past, method = SliceEM(), robust = true, ref_station = 1, maxiter = 100, tol = 1e-3, solver, solve_kwargs) -> SmoothARPeriodicHMM
    initialize([rng, ]m0::SmoothPeriodicHMM, Y; n2t, method = SliceEM(), kwargs...) -> SmoothPeriodicHMM

A starting point for fitting a model of the shape of `m0` to the `D × N` observations `Y`,
estimated by `method`. The result has the bases of `m0`; its coefficients are not read.

For `SmoothARPeriodicHMM`, `past = (; Y = Y_past)` holds the observations before `Y`, at least as
many as the memory. The slices are fitted by [`fit_mle_all_slices`](@ref) from a random model
whose states are sorted by the wetness of `ref_station`, then smoothed into the bases of `m0` by
nonlinear least squares, which needs `using SimpleNonlinearSolve` (`solver` and `solve_kwargs`
are those of that step). `maxiter` and `tol` are those of each EM run.

For `SmoothPeriodicHMM`, this is `fit_mle_all_slices(rng, m0, Y; n2t, Yₜ_extanted = window,
n_random_ini = n_random)` with the other keywords, which needs `using Optimization`;
`dirichlet` is not used.
"""
initialize(m0, Y; kwargs...) = initialize(GLOBAL_RNG, m0, Y; kwargs...)

function initialize(rng::AbstractRNG, m0::SmoothARPeriodicHMM, Y::AbstractMatrix{<:Bool};
    n2t=n_to_t(size(Y, 2), m0.nT)::AbstractVector{<:Integer}, past=nothing, method::SliceEM=SliceEM(),
    robust=true, ref_station::Integer=1, maxiter=100, tol=1e-3, solve_kwargs=(;), kwargs...)
    K, D, T, size_order = size(m0)
    @argcheck size(Y, 1) == D "`Y` has $(size(Y, 1)) rows for $D sites"
    @argcheck 1 <= ref_station <= D "`ref_station` is a site of 1:$D, not $ref_station"
    _check_n2t(n2t, size(Y, 2), T)
    Y_past, z0 = _hmm_past(m0, past; required=true)
    z0 === nothing || _no_past_z()
    hmm = randARPeriodicHMM(rng, K, T, D, Int(log2(size_order)); ref_station)
    slices = fit_mle_all_slices(rng, hmm, Y, Y_past; n2t, Yₜ_extanted=method.window,
        n_random_ini=method.n_random, Dirichlet_α=method.dirichlet, robust, ref_station, maxiter, tol)
    θᴬ, θᴮ = fit_θ!(slices, m0.basis_B; basis_A=m0.basis_A, basis_B=m0.basis_B, solve_kwargs, kwargs...)
    return SmoothARPeriodicHMM(slices.a, θᴬ, θᴮ, T; basis_A=m0.basis_A, basis_B=m0.basis_B)
end

function initialize(rng::AbstractRNG, m0::SmoothPeriodicHMM, Y::AbstractVecOrMat{<:Real};
    n2t=n_to_t(_ncols(Y), m0.nT)::AbstractVector{<:Integer}, past=nothing, method::SliceEM=SliceEM(),
    kwargs...)
    last(_hmm_past(m0, past; required=false)) === nothing || _no_past_z()
    return fit_mle_all_slices(rng, m0, Y; n2t, Yₜ_extanted=method.window,
        n_random_ini=method.n_random, kwargs...)
end
