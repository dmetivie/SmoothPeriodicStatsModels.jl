# A and B are updated using the code from HHMM - update_A_B_jump.jl.

function fit_mle!(
    hmm::ARPeriodicHMMSpatial,
    thetaA::AbstractArray{<:AbstractFloat,3},
    thetaB::AbstractArray{<:AbstractFloat,4},
    thetaR::AbstractArray{<:AbstractFloat,2}, Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool}; solver,
    n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer},
    display=:iter,
    maxiter=100,
    tol=1e-3,
    robust=false,
    silence=true,
    warm_start=true,
    tdist=1,
    QMC_m=30,
    maxiters_R=10, QMC_E=1, wp=1.0 .* (hmm.h .< maximum(hmm.h) * tdist)
)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0
    # println("tdist = ",tdist)
    N, K, T, size_order, D = size(Y, 1), size(hmm, 1), size(hmm, 3), size(hmm, 4), size(hmm, 2)
    # @show N, K, T, size_order, D  # debug

    deg_A = (size(thetaA, 3) - 1) ÷ 2
    deg_B = (size(thetaB, 4) - 1) ÷ 2
    # println("wp= ",wp)
    rain_cat = 2 # dry or wet
    @argcheck T == size(hmm.B, 2)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => Float64[])

    all_thetaA_iterations = [copy(thetaA)]
    all_thetaB_iterations = [copy(thetaB)]

    # add new param : range !

    all_thetaR_iterations = [copy(thetaR)]

    # Allocate order for in-place updates
    c = zeros(N)
    α = zeros(N, K) #forward ?
    β = zeros(N, K) #backward ?
    γ = zeros(N, K) # regular smoothing proba
    γₛ = zeros(K, D, size_order, T, rain_cat) # summed smoothing proba
    ξ = zeros(N, K, K) # pi_kl(t) ?
    s_ξ = zeros(T, K, K) #? somme pi_kl(t) pour t de même périodicité
    LL = zeros(N, K) # stock the loglikelihoods for each state at each time ? (completely unwanted in the M step)

    # assign category for observation depending in the Y_past Y
    order = Int(log2(size_order))
    lag_cat = conditional_to(Y, Y_past)

    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(Y[:, j] .== y, lag_cat[:, j] .== h)) for j = 1:D, h = 1:size_order, y = 0:1] # dry or wet
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history) for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]

    model_A = K ≥ 2 ? model_for_A(s_ξ[:, 1, :], deg_A, silence=silence) : nothing # JuMP Model for transition matrix

    #Done check  model_B
    model_B = model_for_B(γₛ[1, 1, 1, :, :], deg_B, silence=silence) # JuMP Model for Emmission distribution


    # Precompute situation bucket index for each (n, i, j), computed once and reused across EM iterations.
    # Encoding: block = (size_order - lag_cat_i) * size_order * 4 + (size_order - lag_cat_j) * 4
    #           within block: 1=(1,1), 2=(1,0), 3=(0,1), 4=(0,0)
    # For size_order == 1, lag_cat is all-ones, so the block offset is 0 and only the current-obs index (1..4) matters.
    SituationIdx = zeros(Int, N, D, D)
    for n in 1:N
        for i in 1:D
            for j in 1:D
                lci = lag_cat[n, i]
                lcj = lag_cat[n, j]
                cur_idx = 2 * (1 - Int(Y[n, i])) + (1 - Int(Y[n, j])) + 1
                SituationIdx[n, i, j] = (size_order - lci) * size_order * 4 + (size_order - lcj) * 4 + cur_idx
            end
        end
    end
    println("Situations generated")

    loglikelihoods!(LL, hmm, Y, lag_cat; n2t=n2t, QMC_m=QMC_m * QMC_E)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL; n2t=n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL; n2t=n2t)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")


    for it = 1:maxiter
        update_a!(hmm.a, α, β)

        # DONE :need to check update_A
        update_A!(hmm.A, thetaA, ξ, s_ξ, α, β, LL, n2t, n_in_t, model_A; warm_start=warm_start)



        update_B!(hmm.B, thetaB, γ, γₛ, Y, n_all, model_B; warm_start=warm_start)

        update_R!(hmm, thetaR, γ, wp, Y, SituationIdx; n2t=n2t, solver, maxiters=maxiters_R)


        push!(all_thetaA_iterations, copy(thetaA))
        push!(all_thetaB_iterations, copy(thetaB))

       
        robust && (hmm.A .+= eps())

        @check isprobvec(hmm.a)
        @check all(t -> istransmat(hmm.A[:, :, t]), 1:T)


        push!(all_thetaR_iterations, copy(thetaR))

        # loglikelihoods!(LL, hmm, Y, n2t)
        loglikelihoods!(LL, hmm, Y, lag_cat; n2t=n2t, QMC_m=QMC_m * QMC_E)

        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

        forwardlog!(α, c, hmm.a, hmm.A, LL; n2t=n2t)
        backwardlog!(β, c, hmm.a, hmm.A, LL; n2t=n2t)
        posteriors!(γ, α, β)

        logtotp = sum(c)

        ΔmaxA = round(maximum(abs, (all_thetaA_iterations[it+1] - all_thetaA_iterations[it]) ./ all_thetaA_iterations[it]), digits=5)
        ΔmaxB = round(maximum(abs, (all_thetaB_iterations[it+1] - all_thetaB_iterations[it]) ./ all_thetaB_iterations[it]), digits=5)
        ΔmaxR = round(maximum(abs, (all_thetaR_iterations[it+1] - all_thetaR_iterations[it]) ./ all_thetaR_iterations[it]), digits=5)
   
        if display == :iter
                println("Iteration $it: logtot = $(round(logtotp, digits = 6)), max(|θᴬᵢ-θᴬᵢ₋₁|/|θᴬᵢ₋₁|) = ", ΔmaxA, " & max(|θᴮᵢ-θᴮᵢ₋₁|/|θᴮᵢ₋₁|) = ", ΔmaxB, " & max(|θRᵢ-θRᵢ₋₁|/|θRᵢ₋₁|) = ", ΔmaxR)
            # flush(stdout)
        end

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if (ΔmaxA < tol && ΔmaxB < tol && ΔmaxR < tol)
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history["converged"] = true
            break
        end

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history["converged"] = true
            break
        end
        if abs((logtotp - logtot) / logtotp) > tol && logtotp < logtot
            (display in [:iter, :final]) &&
                println("stop the loglikelihood has deacreased dramatically")
            history["converged"] = false
            break
        end
        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println("EM has not converged after $(history["iterations"]) iterations, logtot = $logtot")
        end
    end
    history, all_thetaA_iterations, all_thetaB_iterations, all_thetaR_iterations

end

function fit_mle_one_R!(theta_R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(B, 1))::AbstractVector{<:Integer}, solver, return_sol=false, solkwargs...)
    T = size(B, 1)
    pairwise_indices2 = Tuple.(findall(wp .> 0))
    size_order = size(B, 3)  # 1 for no memory, 2^order for memory >= 1

    function optimfunction2(u, p)
        Rt = similar(u, T)
        for t in 1:T
            Rt[t] = exp(polynomial_trigo(t, u, T))
        end
        if size_order == 1
            return -pairwise_loglikelihood(Rt, B[:, :, 1], p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2)
        else
            return -pairwise_loglikelihood_memory(Rt, B, p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2)
        end
    end
    optf2 = OptimizationFunction((u, p) -> optimfunction2(u, p), AutoForwardDiff())
    u0 = collect(theta_R)

    prob = OptimizationProblem(optf2, u0, [Y, n_pair, h, wp])

    sol = solve(prob, solver; solkwargs...)

    if !SciMLBase.successful_retcode(sol.retcode)
        @warn "sol.retcode = $(sol.retcode)"
    end

    theta_R[:] .= view(sol.u, :)
end

function update_R!(hmm::ARPeriodicHMMSpatial,
    Range_θ::AbstractArray{N,2} where {N},
    γ::AbstractMatrix, wp, Y, SituationIdx::AbstractArray{<:Integer,3}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, solver, maxiters=10)
    @argcheck size(γ, 1) == size(Y, 1)
    N_obs = size(γ, 1)
    R = hmm.R
    D = size(hmm, 2)
    K = size(R, 1)
    T = size(R, 2)
    size_order = size(hmm, 4)
    n_situations = 4 * size_order^2
    pairwise_indices2 = Tuple.(findall(wp .> 0))

    @threads for k in 1:K
        B = hmm.B[k, :, :, :]  # B[t, d, lag_cat]
        h = hmm.h
        w = γ[:, k]
        n_pair = zeros(eltype(R), n_situations, D, D, T)

        @inbounds for tk in 1:N_obs
            t = n2t[tk]
            for (i, j) in pairwise_indices2
                n_pair[SituationIdx[tk, i, j], i, j, t] += w[tk]
            end
        end

        fit_mle_one_R!(view(Range_θ, k, :), B, h, Y, wp, n_pair; n2t=n2t, solver, maxiters=maxiters)
    end

    for k in 1:K
        R[k, :] .= [exp(polynomial_trigo(t, view(Range_θ, k, :), T)) for t = 1:T]
    end
end
