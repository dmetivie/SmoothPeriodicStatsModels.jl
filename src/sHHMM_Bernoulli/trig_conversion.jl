"""
    Trig2ARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:AbstractFloat,3}, θᴮ::AbstractArray{<:AbstractFloat,4}, T::Integer)
Takes trigonometric parameters `θᴬ[k∈[1,K], l∈[1,K-1]`, `d∈[1,𝐃𝐞𝐠]`, `θᴬ[k∈[1,K]`, `l∈[1,K-1]`, `d∈[1,𝐃𝐞𝐠]`
"""
function Trig2ARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:AbstractFloat,3}, θᴮ::AbstractMatrix, T::Integer)
    K, D = size(θᴮ)

    @assert K == size(θᴬ, 1)
    if K == 1
        A = ones(K, K, T)
    else
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
    end
    #TODO use μ, α, θ functions
    B = [Bernoulli.([1 / (1 + exp(polynomial_trigo(t, θh, T))) for θh in θᴮ[k, s]]) for k = 1:K, t = 1:T, s = 1:D]

    return sARPeriodicHMM(a, A, B)
end

Trig2ARPeriodicHMM(θᴬ::AbstractArray{<:AbstractFloat,3}, θᴮ::AbstractMatrix, T::Integer) = Trig2ARPeriodicHMM(ones(size(θᴬ, 1)) ./ size(θᴬ, 1), θᴬ, θᴮ, T)

function fit_θ(hmm::sARPeriodicHMM, 𝐃𝐞𝐠)
    K, D, T, size_order = size(hmm)
    θᴬ = zeros(K, K - 1, 2𝐃𝐞𝐠 + 1)
    θᴮ = [[zeros(2𝐃𝐞𝐠 + 1) for d in 1:size_order[j]] for k in 1:K, j in 1:D]
    for k in 1:K
        fit_θᴬ!(@view(θᴬ[k, :, :]), hmm.A[k, :, :])
        for j in 1:D
            for m in 1:size_order[j]
                fit_θᴮ!(θᴮ[k, j][m], [succprob(hmm.B[k, t, j][m]) for t in 1:T])
            end
        end
    end
    h = Trig2ARPeriodicHMM(hmm.a, deepcopy(θᴬ), deepcopy(θᴮ), T)
    return h, deepcopy(θᴬ), deepcopy(θᴮ)
end