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

function model_for_B(γₛ::AbstractMatrix, d::Int; silence = true, max_cpu_time = 60.0, max_iter = 100)

    T, rain_cat = size(γₛ)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_cpu_time", max_cpu_time)
    set_optimizer_attribute(model, "max_iter", max_iter)

    silence && set_silent(model)
    f = 2π / T
    cos_nj = [cos(f * j * t) for t = 1:T, j = 1:d]
    sin_nj = [sin(f * j * t) for t = 1:T, j = 1:d]
    trig = [[1; interleave2(cos_nj[t, :], sin_nj[t, :])] for t = 1:T]

    @variable(model, θ_jump[j = 1:(2d+1)])
    # Polynomial P
    @NLexpression(model, Pₙ[t = 1:T], sum(trig[t][j] * θ_jump[j] for j = 1:length(trig[t])))

    @NLparameter(model, πₛ[t = 1:T, y = 1:rain_cat] == γₛ[t, y])
    
    @NLexpression(model, mle,
        -sum(πₛ[t, 1] * log1p(exp(-Pₙ[t])) for t = 1:T) - sum(πₛ[t, 2] * log1p(exp(+Pₙ[t])) for t = 1:T)
    ) # 1 is where it did not rain # 2 where it rained
    @NLobjective(
        model, Max,
        mle
    )
    # I don't know if it is the best but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:πₛ] = πₛ
    return model
end

function update_B!(B::AbstractArray{T,4} where {T}, θᴮ::AbstractArray{N,4} where {N}, γ::AbstractMatrix, γₛ::AbstractArray, 𝐘, n_all, model_B::Model; warm_start = true)
    @argcheck size(γ, 1) == size(𝐘, 1)
    @argcheck size(γ, 2) == size(B, 1)
    N = size(γ, 1)
    K = size(B, 1)
    T = size(B, 2)
    D = size(B, 3)
    size_order = size(B, 4)
    ## For periodicHMM only the n 𝐘 corresponding to B(t) are used to update B(t)
    ## Update the smoothing parameters in the JuMP model

    γₛ!(γₛ, γ, n_all) # update coefficient in JuMP model

    all_iter = Iterators.product(1:K, 1:D, 1:size_order)
    #! TODO pmap option
    θ_res = map(tup -> fit_mle_one_B(θᴮ[tup..., :], model_B, γₛ[tup..., :, :]; warm_start=warm_start), all_iter)

    for (k, s, h) in all_iter
        θᴮ[k, s, h, :] = θ_res[k, s, h]
    end

    p = [1 / (1 + exp(polynomial_trigo(t, θᴮ[k, s, h, :], T))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    B[:, :, :, :] = Bernoulli.(p)
end

function fit_mle_one_B(θᴮ, model_B, γₛ; warm_start = true)
    T, rain_cat = size(γₛ)
    θ_jump = model_B[:θ_jump]
    warm_start && set_start_value.(θ_jump, θᴮ[:])
    πₛ = model_B[:πₛ]

    for t = 1:T, y = 1:rain_cat
        set_value(πₛ[t, y], γₛ[t, y])
    end
    optimize!(model_B)
    return value.(θ_jump)
end

# JuMP model use to increase R(θ,θ^i) for the Q(t) matrix
function model_for_A(s_ξ::AbstractArray, d::Int; silence = true)
    T, K = size(s_ξ)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 200)
    silence && set_silent(model)
    f = 2π / T
    cos_nj = [cos(f * j * t) for t = 1:T, j = 1:d]
    sin_nj = [sin(f * j * t) for t = 1:T, j = 1:d]

    trig = [[1; interleave2(cos_nj[t, :], sin_nj[t, :])] for t = 1:T]

    @variable(model, pklj_jump[l = 1:(K-1), j = 1:(2d+1)], start = 0.01)
    # Polynomial P_kl
    @NLexpression(model, Pkl[t = 1:T, l = 1:K-1], sum(trig[t][j] * pklj_jump[l, j] for j = 1:length(trig[t])))

    @NLparameter(model, s_πkl[t = 1:T, l = 1:K-1] == s_ξ[t, l])
    #TODO? is it useful to define the extra parameter for the sum?
    @NLparameter(model, s_πk[t = 1:T] == sum(s_ξ[t, l] for l = 1:K))

    @NLobjective(
        model,
        Max,
        sum(sum(s_πkl[t, l] * Pkl[t, l] for l = 1:K-1) - s_πk[t] * log1p(sum(exp(Pkl[t, l]) for l = 1:K-1)) for t = 1:T)
    )
    # To add NL parameters to the model for later use https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:s_πkl] = s_πkl
    model[:s_πk] = s_πk
    return model
end

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
    model_A::Model;
    warm_start = true
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
    #! TODO pmap option
    θ_res = map(k -> fit_mle_one_A(θᴬ[k, :, :], model_A, s_ξ[:, k, :]; warm_start = warm_start), 1:K)

    for k = 1:K
        θᴬ[k, :, :] = θ_res[k][:, :]
    end

    for k = 1:K, l = 1:K-1, t = 1:T
        A[k, l, t] = exp(polynomial_trigo(t, θᴬ[k, l, :], T))
    end
    for k = 1:K, t = 1:T
        A[k, K, t] = 1  # last colum is 1/normalization (one could do otherwise)
    end
    normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
    for k = 1:K, l = 1:K, t = 1:T
        A[k, l, t] /= normalization_polynomial[k, t]
    end
end

function fit_mle_one_A(θᴬ, model, s_ξ; warm_start = true)
    T, K = size(s_ξ)
    pklj_jump = model[:pklj_jump]
    s_πk = model[:s_πk]
    s_πkl = model[:s_πkl]
    ## Update the smoothing parameters in the JuMP model
    for t = 1:T
        set_value(s_πk[t], sum(s_ξ[t, l] for l = 1:K))
        for l = 1:K-1
            set_value(s_πkl[t, l], s_ξ[t, l])
        end
    end
    warm_start && set_start_value.(pklj_jump, θᴬ[:, :])
    # Optimize the updated model
    optimize!(model)
    # Obtained the new parameters
    return value.(pklj_jump)
end

function fit_mle!(
    hmm::HierarchicalPeriodicHMM,
    θᴬ::AbstractArray{<:AbstractFloat,3},
    θᴮ::AbstractArray{<:AbstractFloat,4},
    𝐘::AbstractArray{<:Bool},
    𝐘_past::AbstractArray{<:Bool}
    ;
    n2t=n_to_t(size(𝐘, 1), size(hmm, 3))::AbstractVector{<:Integer},
    display=:none,
    maxiter=100,
    tol=1e-3,
    robust=false,
    silence=true,
    warm_start=true
)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, T, size_order, D = size(𝐘, 1), size(hmm, 1), size(hmm, 3), size(hmm, 4), size(hmm, 2)

    deg_θᴬ = (size(θᴬ, 3) - 1) ÷ 2
    deg_θᴮ = (size(θᴮ, 4) - 1) ÷ 2
    rain_cat = 2 # dry or wet
    @argcheck T == size(hmm.B, 2)
    history = EMHistory(false, 0, [])

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

    # assign category for observation depending in the 𝐘_past 𝐘
    order = Int(log2(size_order))
    lag_cat = conditional_to(𝐘, 𝐘_past)

    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(𝐘[:, j] .== y, lag_cat[:, j] .== h)) for j = 1:D, h = 1:size_order, y = 0:1] # dry or wet
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history) for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]

    model_A = model_for_A(s_ξ[:, 1, :], deg_θᴬ, silence=silence) # JuMP Model for transition matrix
    model_B = model_for_B(γₛ[1, 1, 1, :, :], deg_θᴮ, silence=silence) # JuMP Model for Emmission distribution

    loglikelihoods!(LL, hmm, 𝐘, lag_cat; n2t=n2t)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL; n2t=n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL; n2t=n2t)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, θᴬ, ξ, s_ξ, α, β, LL, n2t, n_in_t, model_A; warm_start=warm_start)
        update_B!(hmm.B, θᴮ, γ, γₛ, 𝐘, n_all, model_B; warm_start=warm_start)
        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely 𝐘.
        robust && (hmm.A .+= eps())
    
        @check isprobvec(hmm.a)
        @check all(t -> istransmat(hmm.A[:, :, t]), OneTo(T))
    
        push!(all_θᴬᵢ, copy(θᴬ))
        push!(all_θᴮᵢ, copy(θᴮ))
    
        # loglikelihoods!(LL, hmm, 𝐘, n2t)
        loglikelihoods!(LL, hmm, 𝐘, lag_cat; n2t=n2t)
    
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
    
        push!(history.logtots, logtotp)
        history.iterations += 1
    
        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history.converged = true
            break
        end
    
        logtot = logtotp
    end

    if !history.converged
        if display in [:iter, :final]
            println("EM has not converged after $(history.iterations) iterations, logtot = $logtot")
        end
    end

    history, all_θᴬᵢ, all_θᴮᵢ
end

function fit_mle(hmm::HierarchicalPeriodicHMM,
    θᴬ::AbstractArray{<:AbstractFloat,3},
    θᴮ::AbstractArray{<:AbstractFloat,4},
    𝐘::AbstractArray{<:Bool},
    𝐘_past::AbstractArray{<:Bool}; 
    θ_iters=false, kwargs...)

    hmm = copy(hmm)
    θᴬ = copy(θᴬ)
    θᴮ = copy(θᴮ)
    history, all_θᴬᵢ, all_θᴮᵢ = fit_mle!(hmm, θᴬ, θᴮ, 𝐘, 𝐘_past; kwargs...)
    if θ_iters == true
        return hmm, θᴬ, θᴮ, history, all_θᴬᵢ, all_θᴮᵢ
    else
        return hmm, θᴬ, θᴮ, history
    end
end

#TODO add possibility of order size_memories = Vector different at each site
# function fit_mle!(
#     hmm::HierarchicalPeriodicHMM,
#     𝐘::AbstractArray,
#     n2t::AbstractArray{Int},
#     θᴬ::AbstractArray{TQ,3} where {TQ},
#     θᴮ::AbstractArray{TY,4} where {TY},
#     size_memories::AbstractVector # Vector of all local order when there are not indentical
#     ;
#     display = :none,
#     maxiter = 100,
#     tol = 1e-3,
#     robust = false,
#     silence = true,
#     warm_start = true,
#     𝐘_past = [0 1 0 1 1 0 1 0 0 0
#         1 1 0 1 1 1 1 1 1 1
#         1 1 0 1 1 1 0 1 1 1
#         1 1 0 1 1 0 0 0 1 0
#         1 1 0 1 1 0 0 1 0 1]
# )
#     @argcheck display in [:none, :iter, :final]
#     @argcheck maxiter >= 0

#     N, K, T, D = size(𝐘, 1), size(hmm, 1), size(hmm, 3), size(hmm, 2)
#     @argcheck length(size_memories) == D
#     max_size_order = maximum(size_memories)

#     deg_θᴬ = (size(θᴬ, 3) - 1) ÷ 2
#     deg_θᴮ = (size(θᴮ, 4) - 1) ÷ 2
#     rain_cat = 2
#     @argcheck T == size(hmm.B, 2)
#     history = EMHistory(false, 0, [])

#     all_θᴬᵢ = [copy(θᴬ)]
#     all_θᴮᵢ = [copy(θᴮ)]
#     # Allocate order for in-place updates
#     c = zeros(N)
#     α = zeros(N, K)
#     β = zeros(N, K)
#     γ = zeros(N, K) # regular smoothing proba
#     γₛ = zeros(K, D, max_size_order, T, rain_cat) # summed smoothing proba
#     ξ = zeros(N, K, K)
#     s_ξ = zeros(T, K, K)
#     LL = zeros(N, K)

#     # assign category for observation depending in the 𝐘_past 𝐘
#     memories = Int.(log.(size_memories) / log(2))
#     lag_cat = conditional_to(𝐘, 𝐘_past)

#     n_in_t = [findall(n2t .== t) for t = 1:T]
#     n_occurence_history = [findall(.&(𝐘[:, j] .== y, lag_cat[:, j] .== h)) for j = 1:D, h = 1:max_size_order, y = 0:1]
#     n_all = [n_per_category(tup..., n_in_t, n_occurence_history) for tup in Iterators.product(1:D, 1:max_size_order, 1:T, 1:rain_cat)]

#     model_A = model_for_A(s_ξ[:, 1, :], deg_θᴬ, silence = silence) # JuMP Model for transition matrix
#     model_B = model_for_B(γₛ[1, 1, 1, :, :], deg_θᴮ, silence = silence) # JuMP Model for Emmission distribution

#     loglikelihoods!(LL, hmm, 𝐘, n2t, lag_cat)
#     robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

#     forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
#     backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
#     posteriors!(γ, α, β)

#     logtot = sum(c)
#     (display == :iter) && println("Iteration 0: logtot = $logtot")

#     for it = 1:maxiter
#         update_a!(hmm.a, α, β)
#         update_A!(hmm.A, θᴬ, ξ, s_ξ, α, β, LL, n2t, n_in_t, model_A; warm_start = warm_start)
#         update_B!(hmm.B, θᴮ, γ, γₛ, 𝐘, n_all, model_B; warm_start = warm_start)
#         # Ensure the "connected-ness" of the states,
#         # this prevents case where there is no transitions
#         # between two extremely likely 𝐘.
#         robust && (hmm.A .+= eps())

#         @check isprobvec(hmm.a)
#         @check istransmats(hmm.A)

#         push!(all_θᴬᵢ, copy(θᴬ))
#         push!(all_θᴮᵢ, copy(θᴮ))

#         # loglikelihoods!(LL, hmm, 𝐘, n2t)
#         loglikelihoods!(LL, hmm, 𝐘, n2t, lag_cat)

#         robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

#         forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
#         backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
#         posteriors!(γ, α, β)

#         logtotp = sum(c)
#         (display == :iter) && println("Iteration $it: logtot = $logtotp")
#         flush(stdout)

#         push!(history.logtots, logtotp)
#         history.iterations += 1

#         if abs(logtotp - logtot) < tol
#             (display in [:iter, :final]) &&
#                 println("EM converged in $it iterations, logtot = $logtotp")
#             history.converged = true
#             break
#         end

#         logtot = logtotp
#     end

#     if !history.converged
#         if display in [:iter, :final]
#             println("EM has not converged after $(history.iterations) iterations, logtot = $logtot")
#         end
#     end

#     history, all_θᴬᵢ, all_θᴮᵢ
# end