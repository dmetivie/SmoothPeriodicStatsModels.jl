function fit_mle(mix::AbstractArray{<:MixtureModel}, θ_α::AbstractArray, θ_Y::AbstractArray, y::AbstractVecOrMat, n2t::AbstractVector;
    display=:none, maxiter=100, tol=1e-3, robust=false,
    solver=default_solver(), adtype=default_adtype(), solve_kwargs=(;),
    silence=nothing, warm_start=nothing)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0
    _warn_legacy_solver_kwargs(silence, warm_start)

    N, K, T = size_sample(y), ncomponents(mix[1]), size(mix, 1)
    # TODO deal with K=1 case
    @argcheck K > 1
    d = (length(θ_α[1, :]) - 1) ÷ 2 #
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))
    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    s_γ = zeros(T, K)

    c = zeros(N)

    # Initial parameters
    θ_α = copy(θ_α)
    θ_Y = copy(θ_Y)
    n_in_t = [findall(n2t .== t) for t = 1:T]

    α = copy(hcat(probs.(mix)...))
    dists = copy(hcat(components.(mix)...))
    types = typeof.(dists[:, 1])
    dists = [convert(Vector{types[k, 1]}, dists[k, :]) for k = 1:K] # I have to separate the array to preserve types

    basis_α = trig_basis(T, (size(θ_α, 2) - 1) ÷ 2)
    basis_Y = [trig_basis(T, (length(θ_Y[k]) - 1) ÷ 2) for k = 1:K]

    # E-step
    # evaluate likelihood for each type k
    E_step!(LL, c, γ, dists, α, y, n2t; robust=robust)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Method = $(method)\nIteration 0: Loglikelihood = ", logtot)

    for it = 1:maxiter

        # M-step
        # with γ in hand, maximize (update) the parameters
        s_γα!(s_γ, γ, n_in_t)
        update_α!(α, θ_α, s_γ, basis_α; solver, adtype, solve_kwargs)
        for k = 1:K
            dists[k][:], θ_Y[k] = update_dist(dists[k], θ_Y[k], γ[:, k], y, n_in_t, basis_Y[k]; solver, adtype, solve_kwargs)
        end

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y, n2t; robust=robust)

        # Loglikelihood
        logtotp = sum(c)
        (display == :iter) && println("Iteration $(it): loglikelihood = ", logtotp)

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in ", it, " iterations, final loglikelihood = ", logtotp)
            history["converged"] = true
            break
        end

        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println(
                "EM has not converged after $(history["iterations"]) iterations, final loglikelihood = $logtot",
            )
        end
    end

    return [MixtureModel([dists[k][t] for k = 1:K], [α[k, t] for k = 1:K]) for t = 1:T], θ_α, θ_Y, history
end

"""
    update_dist(dist, θ_Y, γ, observations, n_in_t, basis; solver, adtype, solve_kwargs)

M-step for one seasonal `Exponential` component: refit its trigonometric coefficients against the
posterior weights `γ`, and return the refitted distributions together with the coefficients.
"""
function update_dist(dist::AbstractVector{F} where {F<:Exponential}, θ_Y::AbstractArray, γ::AbstractVector, observations::AbstractArray, n_in_t, basis::AbstractMatrix;
    solver=default_solver(), adtype=default_adtype(), solve_kwargs=(;))
    @argcheck size(γ, 1) == size(observations, 1)
    dist = copy(dist)
    θ_Y = copy(θ_Y)
    T = size(dist, 1)

    # sufficient statistics per period index: summed weight, and weighted sum of observations
    s_π = [sum(γ[n] for n in n_in_t[t]; init=0.0) for t = 1:T]
    s_πy = [sum(γ[n] * observations[n] for n in n_in_t[t]; init=0.0) for t = 1:T]

    θ_Y[:] = solve_unconstrained(exponential_nll, collect(θ_Y[:]), (s_π, s_πy, basis);
        solver, adtype, solve_kwargs)

    p = [exp(polynomial_trigo(t / T, θ_Y[:])) for t = 1:T]
    dist[:] = Exponential.(p)
    return dist, θ_Y
end

# Uppdate γ
"""
    s_γα!(s_ξ, ξ, n_in_t)
Give an aggregated weight to each category `k` for each "day" `t` which is the sum of the weight `ξ`
"""
function s_γα!(s_ξ, ξ, n_in_t)
    T, K = size(s_ξ, 1), size(s_ξ, 2)
    for t = 1:T
        for k = 1:K
            s_ξ[t, k] = sum(ξ[n, k] for n in n_in_t[t]; init=0)
        end
    end
end

# Uppdate α
"""
    update_α!(α, θ, s_γ, basis; solver, adtype, solve_kwargs)

M-step for the seasonal mixture weights: refit the multinomial-logit coefficients `θ` against the
aggregated posteriors `s_γ`, and write the resulting weights into `α`. Same objective as the HMM
transition M-step, see [`multinomial_logit_nll`](@ref).
"""
function update_α!(
    α::AbstractArray{F},
    θ::AbstractArray{F},
    s_γ::AbstractMatrix,
    basis::AbstractMatrix;
    solver=default_solver(), adtype=default_adtype(), solve_kwargs=(;)
) where {F}
    T, K = size(s_γ)
    u = solve_unconstrained(multinomial_logit_nll, collect(vec(θ)), (s_γ, basis, K);
        solver, adtype, solve_kwargs)
    θ[:, :] = reshape(u, K - 1, :)
    [α[k, t] = exp(polynomial_trigo(t / T, θ[k, :])) for k = 1:K-1, t = 1:T]
    [α[K, t] = 1 for t = 1:T] # last column is 1/normalization (one could do otherwise)
    normalization_polynomial = [1 + sum(α[k, t] for k = 1:K-1) for t = 1:T]
    [α[k, t] /= normalization_polynomial[t] for k = 1:K, t = 1:T]
end

function E_step!(
    LL::AbstractMatrix{T},
    c::AbstractVector{T},
    γ::AbstractMatrix{T},
    dists::AbstractArray,
    α::AbstractMatrix,
    y::AbstractVector{<:Real}, n2t::AbstractVector;
    robust=false
) where {T<:AbstractFloat}
    # evaluate likelihood for each type k
    for k in eachindex(dists)
        for (n, t) in enumerate(n2t)
            LL[n, k] = log(α[k, t]) + logpdf(dists[k][t], y[n])
        end
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    logsumexp!(c, LL) # ⇔ c[:] = logsumexp(LL, dims=2)
    γ[:, :] .= exp.(LL .- c)
end

function E_step!(
    LL::AbstractMatrix{T},
    c::AbstractVector{T},
    γ::AbstractMatrix{T},
    dists::AbstractArray,
    α::AbstractMatrix,
    y::AbstractMatrix, n2t::AbstractVector;
    robust=false
) where {T<:AbstractFloat}
    # evaluate likelihood for each type k
    for k in eachindex(dists)
        for (n, t) in enumerate(n2t)
            LL[n, k] = log(α[k, t]) + logpdf(dists[k][t], y[:, n])
        end
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    c[:] = logsumexp(LL, dims=2)
    γ[:, :] = exp.(LL .- c)
end

# Utilities 

size_sample(y::AbstractMatrix) = size(y, 2)
size_sample(y::AbstractVector) = length(y)
