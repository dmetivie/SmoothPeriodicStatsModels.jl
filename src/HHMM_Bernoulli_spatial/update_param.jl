# A and B are updated using the code from HHMM - update_A_B_jump.jl.

function my_update_B!(B::AbstractArray{T,4} where {T}, θᴮ::AbstractArray{N,4} where {N}, γ::AbstractMatrix, γₛ::AbstractArray, Y, n_all, model_B::Model; warm_start=true)
    @argcheck size(γ, 1) == size(Y, 1)
    @argcheck size(γ, 2) == size(B, 1)
    N = size(γ, 1)
    K = size(B, 1)
    T = size(B, 2)
    D = size(B, 3)
    size_order = size(B, 4)
    ## For periodicHMM only the n Y corresponding to B(t) are used to update B(t)
    ## Update the smoothing parameters in the JuMP model

    γₛ!(γₛ, γ, n_all) # update coefficient in JuMP model

    all_iter = Iterators.product(1:K, 1:D, 1:size_order)
    #! TODO pmap option
    θ_res = pmap(tup -> fit_mle_one_B(θᴮ[tup..., :], model_B, γₛ[tup..., :, :]; warm_start=warm_start), all_iter)

    for (k, s, h) in all_iter
        θᴮ[k, s, h, :] = θ_res[k, s, h]
    end

    p = [1 / (1 + exp(polynomial_trigo(t, θᴮ[k, s, h, :], T))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    B[:, :, :, :] .= p
end

function fit_mle!(
    hmm::PeriodicHMMSpaMemory,
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


    # generate situations
    if size_order == 1
        Situations = zeros(Int, 4, N, D, D)

        for k in 1:N
            for i in 1:D
                for j in 1:D
                    Situations[1, k, i, j] = (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[2, k, i, j] = (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[3, k, i, j] = (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[4, k, i, j] = (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0
                end
            end
        end
    elseif size_order == 2
        # generate situations
        Situations = zeros(Int, 16, N, D, D)

        for k in 2:N
            for i in 1:D
                for j in 1:D
                    Situations[1, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[2, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[3, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[4, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0

                    Situations[5, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[6, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[7, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[8, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0

                    Situations[9, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[10, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[11, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[12, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0

                    Situations[13, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[14, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[15, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[16, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0
                end
            end
        end
        for i in 1:D
            for j in 1:D
                Situations[1, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[2, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[3, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[4, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0

                Situations[5, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[6, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[7, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[8, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0

                Situations[9, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[10, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[11, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[12, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0

                Situations[13, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[14, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[15, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[16, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0
            end
        end
    elseif size_order > 2
        println("memory of more than 2 not yet implemented for the mle estimation")
        return

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



        my_update_B!(hmm.B, thetaB, γ, γₛ, Y, n_all, model_B; warm_start=warm_start)

        if size_order == 1
            update_R!(hmm, thetaR, γ, wp, Y, Situations; n2t=n2t,solver, maxiters=maxiters_R)
        elseif size_order == 2
            update_R_memory1!(hmm, thetaR, γ, wp, Y, Situations; n2t=n2t,solver, maxiters=maxiters_R)
        end


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

function fit_mle_one_R!(theta_R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, solver, return_sol=false, solkwargs...)
    T = size(B, 1)
    # println("size(B,1) = T ? =",T)
    # println("inside updateR! - inside fit - before estim: ", theta_R)
    pairwise_indices2 = Tuple.(findall(wp .> 0))

    # Branch: Optimize only `range`, fixing `order`
    function optimfunction2(u, p)
        Rt = similar(u, T)
        for t in 1:T
            Rt[t] = exp(mypolynomial_trigo(t, u, T))
        end
        # println("u inside optimfun",u)
        # println("R inside optimfun",Rt)

        # println("B inside optimfun is called ",B) 
        # @show wp 
        # @show h

        return -my_loglikelihood(Rt, B, p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2
        )
    end
    optf2 = OptimizationFunction((u, p) -> optimfunction2(u, p), AutoForwardDiff())
    u0 = collect(theta_R) 
    # println("inside updateR! - inside fit - u0: ", u0)

    prob = OptimizationProblem(optf2, u0, [Y, n_pair, h, wp])

    # Solve the problem
    sol = solve(prob, solver; solkwargs...)

    # Check solution status
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn "sol.retcode = $(sol.retcode)"
    end
    # println("inside updateR! - inside fit - after estim sol.u: ", sol.u)

    # Return the result
    theta_R[:] .= view(sol.u, :)
    # @show theta_R
end


function update_R!(hmm::PeriodicHMMSpaMemory,
    Range_θ::AbstractArray{N,2} where {N},
    γ::AbstractMatrix, wp, Y, Situations::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer},solver, maxiters=10)
    @argcheck size(γ, 1) == size(Y, 1)
    N = size(γ, 1)
    R = hmm.R

    K = size(R, 1)
    T = size(R, 2)
    D = size(hmm, 2)
    # @show K, T  # debug
    # println("inside updateR! - before fit: ", Range_θ)
    pairwise_indices = findall(wp .> 0)
    pairwise_indices2 = [(pairwise_indices[i][1], pairwise_indices[i][2]) for i in 1:length(pairwise_indices)]

    # Parallelized loop
    @threads for k in 1:K
        # @show k  # debug
        B = hmm.B[k, :, :, 1]  # B[k,t]
        h = hmm.h
        w = γ[:, k]
        # println("B,h,w ok")
        n_pair = zeros(eltype(R), 4, D, D, T)

        @inbounds for tk in 1:N
            t = n2t[tk]
            for (i, j) in pairwise_indices2
                w_k = w[tk]
                @views begin
                    n_pair[1, i, j, t] += w_k * Situations[1, tk, i, j]
                    n_pair[2, i, j, t] += w_k * Situations[2, tk, i, j]
                    n_pair[3, i, j, t] += w_k * Situations[3, tk, i, j]
                    n_pair[4, i, j, t] += w_k * Situations[4, tk, i, j]
                end
            end
        end
        # println("weight pairs ok")
        # Fix: Use `view` to pass mutable references
        # @show (Range_θ[ k, :])
        fit_mle_one_R!(view(Range_θ, k, :), B, h, Y, wp, n_pair; n2t=n2t,solver, maxiters=maxiters)
        # @show (Range_θ[ k, :])
    end
    # println("inside updateR! - after fit: ", Range_θ)


    # Ensure in-place modification of R
    for k in 1:K
        R[k, :] .= [exp(mypolynomial_trigo(t, view(Range_θ, k, :), T)) for t = 1:T]
    end
end

function fit_mle_one_R_memory1!(theta_R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, solver, return_sol=false, solkwargs...)
    T = size(B, 1)
    # println("size(B,1) = T ? =",T)
    # println("inside updateR! - inside fit - before estim: ", theta_R)
    pairwise_indices2 = Tuple.(findall(wp .> 0))

    # Branch: Optimize only `range`, fixing `order`
    function optimfunction2(u, p)
        Rt = similar(u, T)
        for t in 1:T
            Rt[t] = exp(mypolynomial_trigo(t, u, T))
        end
        # println("u inside optimfun",u)
        # println("R inside optimfun",Rt)

        # println("B inside optimfun is called ",B) 
        # @show wp 
        # @show h

        return -my_loglikelihood_memory1(Rt, B, p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2)
    end
    optf2 = OptimizationFunction((u, p) -> optimfunction2(u, p), AutoForwardDiff())
    u0 = collect(theta_R)
    # println("inside updateR! - inside fit - u0: ", u0)

    prob = OptimizationProblem(optf2, u0, [Y, n_pair, h, wp])

    # Solve the problem
    sol = solve(prob, solver; solkwargs...)

    # Check solution status
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn "sol.retcode = $(sol.retcode)"
    end
    # println("inside updateR! - inside fit - after estim sol.u: ", sol.u)

    # Return the result
    theta_R[:] .= view(sol.u, :)
    # @show theta_R
end


function update_R_memory1!(hmm::PeriodicHMMSpaMemory,
    Range_θ::AbstractArray{N,2} where {N},
    γ::AbstractMatrix, wp, Y, Situations::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer},solver, maxiters=10)
    @argcheck size(γ, 1) == size(Y, 1)
    N = size(γ, 1)
    R = hmm.R
    D = size(hmm, 2)

    K = size(R, 1)
    T = size(R, 2)
    # println("inside updateR! - before fit: ", Range_θ)
    pairwise_indices = findall(wp .> 0)
    pairwise_indices2 = [(pairwise_indices[i][1], pairwise_indices[i][2]) for i in 1:length(pairwise_indices)]

    # Parallelized loop
    @threads for k in 1:K
        B = hmm.B[k, :, :, :]  # B[k,t,h]
        h = hmm.h
        w = γ[:, k]
        n_pair = zeros(eltype(R), 16, D, D, T)

        @inbounds for tk in 1:N
            t = n2t[tk]
            for (i, j) in pairwise_indices2
                w_k = w[tk]
                @views begin
                    for s in 1:16
                        n_pair[s, i, j, t] += w_k * Situations[s, tk, i, j]

                    end

                end
            end
        end

        # Fix: Use `view` to pass mutable references
        # @show (Range_θ[ k, :])
        fit_mle_one_R_memory1!(view(Range_θ, k, :), B, h, Y, wp, n_pair; n2t=n2t,solver, maxiters=maxiters)
        # @show (Range_θ[ k, :])
    end
    # println("inside updateR! - after fit: ", Range_θ)


    # Ensure in-place modification of R
    for k in 1:K
        R[k, :] .= [exp(mypolynomial_trigo(t, view(Range_θ, k, :), T)) for t = 1:T]
    end
end