"""
    fit_θᴬ!(p, A; solver, solve_kwargs) -> p

Fit the trigonometric coefficients `p` of one row of a periodic transition matrix to the
probabilities `A[k, t]`, in place, by nonlinear least squares. `p` doubles as the starting point,
so the fit is warm-started from whatever it already holds. A one-state chain leaves `p` untouched.
"""
function fit_θᴬ!(p::AbstractArray, A::AbstractArray{N,2} where {N};
    solver=default_nlls_solver(), solve_kwargs=(;))
    T, K = size(A, 2), size(A, 1)
    @assert K - 1 == size(p, 1)
    K == 1 && return p
    u = solve_nlls(transition_residual, collect(vec(p)), (A, K, T), K * T; solver, solve_kwargs)
    return p[:, :] = reshape(u, size(p))
end

"""
    fit_θᴮ!(p, B; solver, solve_kwargs) -> p

Fit the trigonometric coefficients `p` of a seasonal Bernoulli success probability to `B[t]`, in
place, by nonlinear least squares. `p` doubles as the starting point.
"""
function fit_θᴮ!(p::AbstractVector, B::AbstractVector;
    solver=default_nlls_solver(), solve_kwargs=(;))
    T = size(B, 1)
    return p[:] = solve_nlls(bernoulli_residual, collect(p), (B, T), T; solver, solve_kwargs)
end

"""
    fit_θ(hmm::ARPeriodicHMM, 𝐃𝐞𝐠; solver, solve_kwargs) -> (θᴬ, θᴮ)

Trigonometric coefficients of degree `𝐃𝐞𝐠` fitted to the transition matrices and the emission
probabilities of `hmm`. The inverse of [`Trig2ARPeriodicHMM`](@ref): on an `hmm` that is already
trigonometric of that degree, it returns the coefficients it was built from.
"""
function fit_θ(hmm::ARPeriodicHMM, 𝐃𝐞𝐠; solver=default_nlls_solver(), solve_kwargs=(;))
    K, D, size_order = size(hmm)[[1, 2, 4]]
    θᴬ = zeros(K, K - 1, 2𝐃𝐞𝐠 + 1)
    θᴮ = zeros(K, D, size_order, 2𝐃𝐞𝐠 + 1)
    for k in 1:K
        fit_θᴬ!(@view(θᴬ[k, :, :]), hmm.A[k, :, :]; solver, solve_kwargs)
        for j in 1:D, m in 1:size_order
            fit_θᴮ!(@view(θᴮ[k, j, m, :]), succprob.(hmm.B[k, :, j, m]); solver, solve_kwargs)
        end
    end
    return θᴬ, θᴮ
end

"""
    fit_θ!(hmm::ARPeriodicHMM, 𝐃𝐞𝐠; solver, solve_kwargs) -> (θᴬ, θᴮ)

As [`fit_θ`](@ref), and additionally replaces `hmm`'s transition matrices and emission
probabilities by the trigonometric ones the fitted coefficients encode.
"""
function fit_θ!(hmm::ARPeriodicHMM, 𝐃𝐞𝐠; solver=default_nlls_solver(), solve_kwargs=(;))
    T = size(hmm, 3)
    θᴬ, θᴮ = fit_θ(hmm, 𝐃𝐞𝐠; solver, solve_kwargs)
    h = Trig2ARPeriodicHMM(hmm.a, θᴬ, θᴮ, T)
    hmm.A[:] = h.A[:]
    hmm.B[:] = h.B[:]
    return θᴬ, θᴮ
end
function fit_θ(hmm::sARPeriodicHMM, 𝐃𝐞𝐠; solver=default_nlls_solver(), solve_kwargs=(;))
    K, D, T, size_order = size(hmm)
    θᴬ = zeros(K, K - 1, 2𝐃𝐞𝐠 + 1)
    θᴮ = [[zeros(2𝐃𝐞𝐠 + 1) for d in 1:size_order[j]] for k in 1:K, j in 1:D]
    for k in 1:K
        fit_θᴬ!(@view(θᴬ[k, :, :]), hmm.A[k, :, :]; solver, solve_kwargs)
        for j in 1:D
            for m in 1:size_order[j]
                fit_θᴮ!(θᴮ[k, j][m], [succprob(hmm.B[k, t, j][m]) for t in 1:T]; solver, solve_kwargs)
            end
        end
    end
    h = Trig2ARPeriodicHMM(hmm.a, deepcopy(θᴬ), deepcopy(θᴮ), T)
    return h, deepcopy(θᴬ), deepcopy(θᴮ)
end