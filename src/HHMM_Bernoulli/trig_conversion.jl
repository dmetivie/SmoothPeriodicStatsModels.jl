"""
    Trig2HierarchicalPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:AbstractFloat,3}, θᴮ::AbstractArray{<:AbstractFloat,4}, T::Integer)
Takes trigonometric parameters `θᴬ[k∈[1,K], l∈[1,K-1]`, `d∈[1,𝐃𝐞𝐠]`, `θᴬ[k∈[1,K]`, `l∈[1,K-1]`, `d∈[1,𝐃𝐞𝐠]`
"""
function Trig2HierarchicalPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:AbstractFloat,3}, θᴮ::AbstractArray{<:AbstractFloat,4}, T::Integer)
    K, D, size_order = size(θᴮ)
    @assert K == size(θᴬ, 1)

    A = zeros(K, K, T)
    for k = 1:K, l = 1:K-1, t = 1:T
        #TODO use μ, α, θ functions
        A[k, l, t] = exp(polynomial_trigo(t, θᴬ[k, l, :], T))
    end
    for k = 1:K, t = 1:T
        A[k, K, t] = 1  # last colum is 1/normalization (one could do otherwise)
    end
    normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
    for k = 1:K, l = 1:K, t = 1:T
        A[k, l, t] /= normalization_polynomial[k, t]
    end
    #TODO use μ, α, θ functions
    p = [1 / (1 + exp(polynomial_trigo(t, θᴮ[k, s, h, :], T))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]

    return HierarchicalPeriodicHMM(a, A, Bernoulli.(p))
end

Trig2HierarchicalPeriodicHMM(θᴬ::AbstractArray{<:AbstractFloat,3}, θᴮ::AbstractArray{<:AbstractFloat,4}, T::Integer) = Trig2HierarchicalPeriodicHMM(ones(size(θᴬ, 1)) ./ size(θᴬ, 1), θᴬ, θᴮ, T)

function fit_θᴬ!(p::AbstractArray, A::AbstractArray{N,2} where {N}; silence=true)
    T, K = size(A, 2), size(A, 1)
    @assert K - 1 == size(p, 1)
    d = (size(p, 2) - 1) ÷ 2
    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)
    f = 2π / T
    cos_nj = [cos(f * j * t) for t = 1:T, j = 1:d]
    sin_nj = [sin(f * j * t) for t = 1:T, j = 1:d]

    trig = [[1; interleave2(cos_nj[t, :], sin_nj[t, :])] for t = 1:T]

    @variable(model, p_jump[k=1:(K-1), j=1:(2d+1)])
    set_start_value.(p_jump, p)
    # Polynomial P_kl

    @NLexpression(model, Pol[t=1:T, k=1:K-1], sum(trig[t][j] * p_jump[k, j] for j = 1:length(trig[t])))
    #TODO use μ, α, θ functions
    @NLobjective(
        model,
        Min,
        sum((A[k, t] - exp(Pol[t, k]) / (1 + sum(exp(Pol[t, l]) for l = 1:K-1)))^2 for k = 1:K-1, t = 1:T)
        +
        sum((A[K, t] - 1 / (1 + sum(exp(Pol[t, l]) for l = 1:K-1)))^2 for t = 1:T)
    )
    optimize!(model)
    p[:, :] = value.(p_jump)
end

#TODO use α(t, p) instead
m_Bernoulli(t, p, T) = 1 ./ (1 .+ exp.(polynomial_trigo(t, p, T)))
# Fit (faster than JuMP) with LsqFit
function fit_θᴮ!(p::AbstractVector, B::AbstractVector)
    T = size(B, 1)
    p[:] = curve_fit((t, p) -> m_Bernoulli(t, p, T), collect(1:T), B, convert(Vector, p)).param
end

function fit_θ(hmm::HierarchicalPeriodicHMM, 𝐃𝐞𝐠)
    K, D, size_order = size(hmm)[[1, 2, 4]]
    θᴬ = zeros(K, K - 1, 2𝐃𝐞𝐠 + 1)
    θᴮ = zeros(K, D, size_order, 2𝐃𝐞𝐠 + 1)
    for k in 1:K
        fit_θᴬ!(@view(θᴬ[k, :, :]), hmm.A[k, :, :])
        for j in 1:D, m in 1:size_order
            fit_θᴮ!(@view(θᴮ[k, j, m, :]), succprob.(hmm.B[k, :, j, m]))
        end
    end
    return θᴬ, θᴮ
end

function fit_θ!(hmm::HierarchicalPeriodicHMM, 𝐃𝐞𝐠)
    K, D, T, size_order = size(hmm)
    θᴬ = zeros(K, K - 1, 2𝐃𝐞𝐠 + 1)
    θᴮ = zeros(K, D, size_order, 2𝐃𝐞𝐠 + 1)
    for k in 1:K
        fit_θᴬ!(@view(θᴬ[k, :, :]), hmm.A[k, :, :])
        for j in 1:D, m in 1:size_order
            fit_θᴮ!(@view(θᴮ[k, j, m, :]), succprob.(hmm.B[k, :, j, m]))
        end
    end
    h = Trig2HierarchicalPeriodicHMM(hmm.a, θᴬ, θᴮ, T)
    hmm.A[:] = h.A[:]
    hmm.B[:] = h.B[:]
    return θᴬ, θᴮ
end