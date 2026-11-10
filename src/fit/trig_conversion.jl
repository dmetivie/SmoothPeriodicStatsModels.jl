# The design matrix over the period indices `1:T`: from a basis of TimeVaryingParameters, or
# given as is. With `n` and `name`, checked to have `n` columns.
_design(X::AbstractMatrix, T) = X
_design(basis, T) = design_matrix(basis, 1:T)

function _design(basis, T, n, name)
    X = _design(basis, T)
    size(X) == (T, n) || throw(ArgumentError(
        "the basis of `$name` gives a $(size(X, 1)) × $(size(X, 2)) design matrix, and `$name` needs $T × $n"))
    return X
end

"""
    fit_θᴬ!(p, A; basis, solver, solve_kwargs) -> p

Fit the coefficients `p`, `(K-1) × nbasis`, of one row of a periodic transition matrix to the
probabilities `A[l, t]`, in place, by nonlinear least squares, with the softmax of
[`ARPeriodicHMM`](@ref). `basis` is a basis of TimeVaryingParameters or its `T × nbasis` design
matrix, by default `TrigBasis(T, d)` for `2d + 1` coefficients. `p` doubles as the starting
point, so the fit is warm-started from whatever it already holds. A one-state chain leaves `p`
untouched.
"""
function fit_θᴬ!(p::AbstractArray, A::AbstractArray{N,2} where {N};
    basis=TrigBasis(size(A, 2), _trig_order(size(p, 2), "p")),
    solver=default_nlls_solver(), solve_kwargs=(;))
    T, K = size(A, 2), size(A, 1)
    @assert K - 1 == size(p, 1)
    K == 1 && return p
    X = _design(basis, T, size(p, 2), "p")
    u = solve_nlls(transition_residual, collect(vec(p)), (A, K, X), K * T; solver, solve_kwargs)
    return p[:, :] = reshape(u, size(p))
end

"""
    fit_θᴮ!(p, B; basis, solver, solve_kwargs) -> p

Fit the coefficients `p` of a periodic Bernoulli success probability to `B[t]`, in place, by
nonlinear least squares, with the logistic of [`ARPeriodicHMM`](@ref). `basis` is as for
[`fit_θᴬ!`](@ref), and `p` doubles as the starting point.
"""
function fit_θᴮ!(p::AbstractVector, B::AbstractVector;
    basis=TrigBasis(length(B), _trig_order(length(p), "p")),
    solver=default_nlls_solver(), solve_kwargs=(;))
    T = size(B, 1)
    X = _design(basis, T, length(p), "p")
    return p[:] = solve_nlls(bernoulli_residual, collect(p), (B, X), T; solver, solve_kwargs)
end

"""
    fit_θ(hmm, basis; basis_A = basis, basis_B = basis, solver, solve_kwargs) -> (θᴬ, θᴮ)

Coefficients on `basis_A` and `basis_B` fitted by least squares to the transition matrices and
the emission probabilities of `hmm`, an `ARPeriodicHMM` or `sARPeriodicHMM`. An integer `basis`
is the degree `d` of the trigonometric basis `TrigBasis(T, d)`. The inverse of
`ARPeriodicHMM(a, θᴬ, θᴮ, T; basis_A, basis_B)`: on a model built that way, it returns the
coefficients it was built from.
"""
fit_θ(hmm::ARPeriodicHMM, 𝐃𝐞𝐠::Integer; kwargs...) = fit_θ(hmm, TrigBasis(size(hmm, 3), 𝐃𝐞𝐠); kwargs...)
fit_θ(hmm::sARPeriodicHMM, 𝐃𝐞𝐠::Integer; kwargs...) = fit_θ(hmm, TrigBasis(size(hmm, 3), 𝐃𝐞𝐠); kwargs...)

function fit_θ(hmm::ARPeriodicHMM, basis; basis_A=basis, basis_B=basis,
    solver=default_nlls_solver(), solve_kwargs=(;))
    K, D, T, size_order = size(hmm)
    XA, XB = _design(basis_A, T), _design(basis_B, T)
    θᴬ = zeros(K, K - 1, size(XA, 2))
    θᴮ = zeros(K, D, size_order, size(XB, 2))
    for k in 1:K
        fit_θᴬ!(@view(θᴬ[k, :, :]), hmm.A[k, :, :]; basis=XA, solver, solve_kwargs)
        for j in 1:D, m in 1:size_order
            fit_θᴮ!(@view(θᴮ[k, j, m, :]), succprob.(hmm.B[k, :, j, m]); basis=XB, solver, solve_kwargs)
        end
    end
    return θᴬ, θᴮ
end

function fit_θ(hmm::sARPeriodicHMM, basis; basis_A=basis, basis_B=basis,
    solver=default_nlls_solver(), solve_kwargs=(;))
    K, D, T, size_order = size(hmm)
    XA, XB = _design(basis_A, T), _design(basis_B, T)
    θᴬ = zeros(K, K - 1, size(XA, 2))
    θᴮ = [[zeros(size(XB, 2)) for m in 1:size_order[j]] for k in 1:K, j in 1:D]
    for k in 1:K
        fit_θᴬ!(@view(θᴬ[k, :, :]), hmm.A[k, :, :]; basis=XA, solver, solve_kwargs)
        for j in 1:D, m in 1:size_order[j]
            fit_θᴮ!(θᴮ[k, j][m], [succprob(hmm.B[k, t, j][m]) for t in 1:T]; basis=XB, solver, solve_kwargs)
        end
    end
    return θᴬ, θᴮ
end

"""
    fit_θ!(hmm, basis; basis_A = basis, basis_B = basis, solver, solve_kwargs) -> (θᴬ, θᴮ)

As [`fit_θ`](@ref), and additionally replaces `hmm`'s transition matrices and emission
probabilities by those the fitted coefficients encode.
"""
fit_θ!(hmm::ARPeriodicHMM, 𝐃𝐞𝐠::Integer; kwargs...) = fit_θ!(hmm, TrigBasis(size(hmm, 3), 𝐃𝐞𝐠); kwargs...)
fit_θ!(hmm::sARPeriodicHMM, 𝐃𝐞𝐠::Integer; kwargs...) = fit_θ!(hmm, TrigBasis(size(hmm, 3), 𝐃𝐞𝐠); kwargs...)

function fit_θ!(hmm::Union{ARPeriodicHMM,sARPeriodicHMM}, basis; basis_A=basis, basis_B=basis,
    solver=default_nlls_solver(), solve_kwargs=(;))
    T = size(hmm, 3)
    θᴬ, θᴮ = fit_θ(hmm, basis; basis_A, basis_B, solver, solve_kwargs)
    model = hmm isa ARPeriodicHMM ? ARPeriodicHMM : sARPeriodicHMM
    h = model(hmm.a, θᴬ, θᴮ, T; basis_A, basis_B)
    hmm.A[:] = h.A[:]
    hmm.B[:] = h.B[:]
    return θᴬ, θᴮ
end
