function γₛ!(γₛ, γ, n_all)
    K, D, size_order, T, rain_cat = size(γₛ)
    for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)
        for k = 1:K
            γₛ[k, tup...] = sum(γ[n, k] for n in n_all[tup...]; init = 0)
        end
    end
end

function s_ξ!(s_ξ, ξ, n_in_t)
    T, K = size(s_ξ)
    for t = 1:T
        for (k, l) in Iterators.product(1:K, 1:K)
            s_ξ[t, k, l] = sum(ξ[n, k, l] for n in n_in_t[t])
        end
    end
    # * We add ξ[N, k, l] but it should be zeros
end

"""
    update_B!(B, θᴮ, γ, γₛ, Y, n_all, basis; solver, adtype, ensemble, solve_kwargs)

M-step for the emission probabilities: refit the trigonometric coefficients `θᴮ[k, s, h, :]` of
every (state, site, memory) triple against the aggregated posteriors `γₛ`, then write the
resulting probabilities into `B`.

The triples are independent problems differing only in their starting point and their data, so
they are solved as one `EnsembleProblem`; pass `ensemble = EnsembleThreads()` to run them in
parallel.
"""
function update_B!(B::AbstractArray{T,4} where {T}, θᴮ::AbstractArray{N,4} where {N}, γ::AbstractMatrix, γₛ::AbstractArray, Y, n_all, basis::AbstractMatrix;
    solver=default_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;))
    @argcheck size(γ, 1) == size(Y, 2)
    @argcheck size(γ, 2) == size(B, 1)
    K, T, D, size_order = size(B)

    ## For periodicHMM only the n Y corresponding to B(t) are used to update B(t)
    γₛ!(γₛ, γ, n_all)

    idx = vec(collect(Iterators.product(1:K, 1:D, 1:size_order)))
    emission_data(k, s, h) = (view(γₛ, k, s, h, :, :), basis)
    base = OptimizationProblem(OptimizationFunction(bernoulli_logit_nll, adtype),
        θᴮ[idx[1]..., :], emission_data(idx[1]...))
    eprob = EnsembleProblem(base;
        prob_func=(prob, ctx) -> remake(prob; u0=θᴮ[idx[ctx.sim_id]..., :],
            p=emission_data(idx[ctx.sim_id]...)))
    sol = solve(eprob, solver, ensemble; trajectories=length(idx), solve_kwargs...)

    for (i, (k, s, h)) in enumerate(idx)
        successful_retcode(sol.u[i].retcode) || @warn "emission ($k, $s, $h): $(sol.u[i].retcode)"
        θᴮ[k, s, h, :] = sol.u[i].u
    end

    p = [1 / (1 + exp(polynomial_trigo(t / T, θᴮ[k, s, h, :]))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    _assign_B!(B, p)
end

_assign_B!(B::AbstractArray{<:Real,4}, p) = B .= p
_assign_B!(B::AbstractArray{<:Distribution,4}, p) = B .= Bernoulli.(p)


"""
    update_A!(A, θᴬ, ξ, s_ξ, α, β, LL, n2t, n_in_t, basis; solver, adtype, ensemble, solve_kwargs)

M-step for the transition matrices: refit the trigonometric coefficients `θᴬ[k, :, :]` of every
state against the aggregated transition posteriors `s_ξ`, then write the resulting softmax
probabilities into `A`.

One independent problem per state, solved as one `EnsembleProblem`.
"""
function update_A!(
    A::AbstractArray{<:AbstractFloat,3},
    θᴬ::AbstractArray{<:AbstractFloat,3},
    ξ::AbstractArray,
    s_ξ::AbstractArray,
    α::AbstractMatrix,
    β::AbstractMatrix,
    LL::AbstractMatrix,
    n2t::AbstractArray{Int},
    n_in_t,
    basis::AbstractMatrix;
    solver=default_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;)
)
    @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1)
    @argcheck size(α, 2) ==
              size(β, 2) ==
              size(LL, 2) ==
              size(A, 1) ==
              size(A, 2) ==
              size(ξ, 2) ==
              size(ξ, 3)

    N, K = size(LL)
    T = size(A, 3)
    @inbounds for n in OneTo(N - 1)
        t = n2t[n] # periodic t
        m = vec_maximum(view(LL, n + 1, :))
        c = 0

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] = α[n, i] * A[i, j, t] * exp(LL[n+1, j] - m) * β[n+1, j]
            c += ξ[n, i, j]
        end

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] /= c
        end
    end
    ## 
    # ξ are the filtering probablies
    s_ξ!(s_ξ, ξ, n_in_t)

    transition_data(k) = (view(s_ξ, :, k, :), basis, K)
    base = OptimizationProblem(OptimizationFunction(multinomial_logit_nll, adtype),
        collect(vec(θᴬ[1, :, :])), transition_data(1))
    eprob = EnsembleProblem(base;
        prob_func=(prob, ctx) -> remake(prob; u0=collect(vec(θᴬ[ctx.sim_id, :, :])),
            p=transition_data(ctx.sim_id)))
    sol = solve(eprob, solver, ensemble; trajectories=K, solve_kwargs...)

    for k = 1:K
        successful_retcode(sol.u[k].retcode) || @warn "transition $k: $(sol.u[k].retcode)"
        θᴬ[k, :, :] = reshape(sol.u[k].u, K - 1, :)
    end

    for k = 1:K, l = 1:K-1, t = 1:T
        A[k, l, t] = exp(polynomial_trigo(t / T, θᴬ[k, l, :]))
    end
    for k = 1:K, t = 1:T
        A[k, K, t] = 1  # last colum is 1/normalization (one could do otherwise)
    end
    normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
    for k = 1:K, l = 1:K, t = 1:T
        A[k, l, t] /= normalization_polynomial[k, t]
    end
end

# A one-state chain has no transition to fit
update_A!(
    A::AbstractArray{<:AbstractFloat,3},
    θᴬ::AbstractArray{<:AbstractFloat,3},
    ξ::AbstractArray,
    s_ξ::AbstractArray,
    α::AbstractMatrix,
    β::AbstractMatrix,
    LL::AbstractMatrix,
    n2t::AbstractArray{Int},
    n_in_t,
    basis::Nothing;
    kwargs...
) = nothing


function fit_mle!(
    hmm::ARPeriodicHMM,
    θᴬ::AbstractArray{<:AbstractFloat,3},
    θᴮ::AbstractArray{<:AbstractFloat,4},
    Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool};
    n2t=n_to_t(size(Y, 2), size(hmm, 3))::AbstractVector{<:Integer},
    display=:none,
    maxiter=100,
    tol=1e-3,
    robust=false,
    solver=default_solver(),
    adtype=default_adtype(),
    ensemble=default_ensemble(),
    solve_kwargs=(;),
    silence=nothing,
    warm_start=nothing
)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0
    _warn_legacy_solver_kwargs(silence, warm_start)

    N, K, T, size_order, D = size(Y, 2), size(hmm, 1), size(hmm, 3), size(hmm, 4), size(hmm, 2)

    deg_θᴬ = (size(θᴬ, 3) - 1) ÷ 2
    deg_θᴮ = (size(θᴮ, 4) - 1) ÷ 2
    rain_cat = 2 # dry or wet
    @argcheck T == size(hmm.B, 2)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => Float64[])

    all_θᴬᵢ = [copy(θᴬ)]
    all_θᴮᵢ = [copy(θᴮ)]
    # Allocate order for in-place updates
    c = zeros(N)
    α = zeros(N, K)
    β = zeros(N, K)
    γ = zeros(N, K) # regular smoothing proba
    γₛ = zeros(K, D, size_order, T, rain_cat) # summed smoothing proba
    ξ = zeros(N, K, K)
    s_ξ = zeros(T, K, K)
    LL = zeros(N, K)

    # assign category for observation depending in the Y_past Y
    order = Int(log2(size_order))
    lag_cat = conditional_to(Y, Y_past)
    # @show lag_cat[1,:]
    # @show lag_cat[end,:]

    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(view(Y, j, :) .== y, view(lag_cat, j, :) .== h)) for j = 1:D, h = 1:size_order, y = 0:1] # dry or wet
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history) for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]

    basis_A = K ≥ 2 ? trig_basis(T, deg_θᴬ) : nothing # `nothing` disables the transition M-step
    basis_B = trig_basis(T, deg_θᴮ)

    loglikelihoods!(LL, hmm, Y, lag_cat; n2t=n2t)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL; n2t=n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL; n2t=n2t)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")
    # @show γ[1:3,:]
    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, θᴬ, ξ, s_ξ, α, β, LL, n2t, n_in_t, basis_A; solver, adtype, ensemble, solve_kwargs)
        update_B!(hmm.B, θᴮ, γ, γₛ, Y, n_all, basis_B; solver, adtype, ensemble, solve_kwargs)
        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely Y.
        robust && (hmm.A .+= eps())
    
        @check isprobvec(hmm.a)
        @check all(t -> istransmat(hmm.A[:, :, t]), OneTo(T))
    
        push!(all_θᴬᵢ, copy(θᴬ))
        push!(all_θᴮᵢ, copy(θᴮ))
    
        # loglikelihoods!(LL, hmm, Y, n2t)
        loglikelihoods!(LL, hmm, Y, lag_cat; n2t=n2t)
    
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    
        forwardlog!(α, c, hmm.a, hmm.A, LL; n2t=n2t)
        backwardlog!(β, c, hmm.a, hmm.A, LL; n2t=n2t)
        posteriors!(γ, α, β)
    
        logtotp = sum(c)

        if display == :iter
            ΔmaxA = round(maximum(abs, all_θᴬᵢ[it+1] - all_θᴬᵢ[it]), digits=5)
            ΔmaxB = round(maximum(abs, all_θᴮᵢ[it+1] - all_θᴮᵢ[it]), digits=5)
            println("Iteration $it: logtot = $(round(logtotp, digits = 6)), max(|θᴬᵢ-θᴬᵢ₋₁|) = ", ΔmaxA, " & max(|θᴮᵢ-θᴮᵢ₋₁|) = ", ΔmaxB)
            # flush(stdout)
        end
    
        push!(history["logtots"], logtotp)
        history["iterations"] += 1
    
        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history["converged"] = true
            break
        end
    
        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println("EM has not converged after $(history["iterations"]) iterations, logtot = $logtot")
        end
    end

    history, all_θᴬᵢ, all_θᴮᵢ
end

"""
    fit_mle(hmm::ARPeriodicHMM, θᴬ::AbstractArray{<:AbstractFloat,3}, θᴮ::AbstractArray{<:AbstractFloat,4}, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}; θ_iters=false, kwargs...)
Fit with EM algortihm for HMM (Baum-Welch) with seasonal parameters and local memory.
"""
function fit_mle(hmm::ARPeriodicHMM,
    θᴬ::AbstractArray{<:AbstractFloat,3},
    θᴮ::AbstractArray{<:AbstractFloat,4},
    Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool}; 
    θ_iters=false, kwargs...)

    hmm = copy(hmm)
    θᴬ = copy(θᴬ)
    θᴮ = copy(θᴮ)
    history, all_θᴬᵢ, all_θᴮᵢ = fit_mle!(hmm, θᴬ, θᴮ, Y, Y_past; kwargs...)
    if θ_iters == true
        return hmm, θᴬ, θᴮ, history, all_θᴬᵢ, all_θᴮᵢ
    else
        return hmm, θᴬ, θᴮ, history
    end
end