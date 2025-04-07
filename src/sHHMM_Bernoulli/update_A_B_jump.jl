function γₛ!(γₛ::AbstractVector, γ, n_all::AbstractVector)
    D = length(γₛ)
    for j in 1:D
        K, size_order, T, rain_cat = size(γₛ[j])
        for tup in Iterators.product(1:size_order, 1:T, 1:rain_cat)
            for k = 1:K
                γₛ[j][k, tup...] = sum(γ[n, k] for n in n_all[j][tup...]; init=0)
            end
        end
    end
end

function update_B!(B::AbstractArray{T,3} where {T}, θᴮ::AbstractMatrix, γ::AbstractMatrix, γₛ::AbstractVector, Y, n_all, model_B::Model; warm_start=true)
    @argcheck size(γ, 1) == size(Y, 1)
    @argcheck size(γ, 2) == size(B, 1)
    K, T, D = size(B)
    ## For periodicHMM only the n Y corresponding to B(t) are used to update B(t)
    ## Update the smoothing parameters in the JuMP model

    SmoothPeriodicStatsModels.γₛ!(γₛ, γ, n_all) # update coefficient in JuMP model

    all_iter = vcat(vec([[tuple(k,j,h) for h in 1:(length(θᴮ[k, j]))] for j in 1:D, k in 1:K])...)
    θ_res = pmap(all_iter) do tup
        k, s, h = tup
        SmoothPeriodicStatsModels.fit_mle_one_B(θᴮ[k, s][h], model_B, γₛ[s][k, h, :, :]; warm_start=warm_start)
    end
    for (i, tup) in enumerate(all_iter)
        k, s, h = tup
        θᴮ[k, s][h] = θ_res[i]
    end

    for k = 1:K, t = 1:T, s = 1:D
        B[k, t, s][:] .= Bernoulli.([1 / (1 + exp(polynomial_trigo(t, θh, T))) for θh in θᴮ[k, s]])
    end
end

function fit_mle!(
    hmm::sARPeriodicHMM,
    θᴬ::AbstractArray{<:AbstractFloat,3},
    θᴮ::AbstractMatrix,
    Y::AbstractMatrix{<:Bool},
    Y_past::AbstractVector;
    n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer},
    display=:none,
    maxiter=100,
    tol=1e-3,
    robust=false,
    silence=true,
    warm_start=true
)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, T, size_orders, D = size(Y, 1), size(hmm, 1), size(hmm, 3), size(hmm, 4), size(hmm, 2)

    deg_θᴬ = (size(θᴬ, 3) - 1) ÷ 2
    rain_cat = 2 # dry or wet
    @argcheck T == size(hmm.B, 2)
    @assert  2 .^ length.(Y_past) == size_orders "2 .^ length.(Y_past) = $(2 .^ length.(Y_past)) while size(hmm, 4) == $(size_orders)"
    history = Dict("converged" => false, "iterations" => 0, "logtots" => Float64[])

    all_θᴬᵢ = [copy(θᴬ)]
    all_θᴮᵢ = [copy(θᴮ)]
    # Allocate order for in-place updates
    c = zeros(N)
    α = zeros(N, K)
    β = zeros(N, K)
    γ = zeros(N, K) # regular smoothing proba
    γₛ = [zeros(K, size_orders[j], T, rain_cat) for j in 1:D] # summed smoothing proba
    ξ = zeros(N, K, K)
    s_ξ = zeros(T, K, K)
    LL = zeros(N, K)

    # assign category for observation depending in the Y_past Y
    lag_cat = SmoothPeriodicStatsModels.conditional_to(Y, Y_past)
    # @show lag_cat[1,:]
    # @show lag_cat[end,:]

    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [[findall(.&(Y[:, j] .== y, lag_cat[:, j] .== h)) for h = 1:size_orders[j], y = 0:1] for j = 1:D] # dry or wet
    n_all = [[SmoothPeriodicStatsModels.n_per_category(tup..., n_in_t, n_occurence_history[j]) for tup in Iterators.product(1:size_orders[j], 1:T, 1:rain_cat)] for j in 1:D]

    model_A = K ≥ 2 ? SmoothPeriodicStatsModels.model_for_A(s_ξ[:, 1, :], deg_θᴬ, silence=silence) : nothing # JuMP Model for transition matrix

    deg_θᴮ = (length(θᴮ[1, 1][1]) - 1) ÷ 2
    model_B = SmoothPeriodicStatsModels.model_for_B(γₛ[1][1, 1, :, :], deg_θᴮ, silence=silence) # JuMP Model for Emmission distribution

    SmoothPeriodicStatsModels.loglikelihoods!(LL, hmm, Y, lag_cat; n2t=n2t)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    SmoothPeriodicStatsModels.forwardlog!(α, c, hmm.a, hmm.A, LL; n2t=n2t)
    SmoothPeriodicStatsModels.backwardlog!(β, c, hmm.a, hmm.A, LL; n2t=n2t)
    SmoothPeriodicStatsModels.posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")
    # @show γ[1:3,:]

    for it = 1:maxiter
        SmoothPeriodicStatsModels.update_a!(hmm.a, α, β)
        SmoothPeriodicStatsModels.update_A!(hmm.A, θᴬ, ξ, s_ξ, α, β, LL, n2t, n_in_t, model_A; warm_start=warm_start)
        update_B!(hmm.B, θᴮ, γ, γₛ, Y, n_all, model_B; warm_start=warm_start)
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
            # ΔmaxB = round(maximum(abs, all_θᴮᵢ[it+1] - all_θᴮᵢ[it]), digits=5)
            # println("Iteration $it: logtot = $(round(logtotp, digits = 6)), max(|θᴬᵢ-θᴬᵢ₋₁|) = ", ΔmaxA, " & max(|θᴮᵢ-θᴮᵢ₋₁|) = ", ΔmaxB)
            println("Iteration $it: logtot = $(round(logtotp, digits = 6)), max(|θᴬᵢ-θᴬᵢ₋₁|) = ", ΔmaxA, " & max(|θᴮᵢ-θᴮᵢ₋₁|) = ?")
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

function fit_mle(hmm::sARPeriodicHMM,
    θᴬ::AbstractArray,
    θᴮ::AbstractArray,
    Y::AbstractMatrix{<:Bool},
    Y_past::AbstractVector;
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