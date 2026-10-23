"""
    Trig2ARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:AbstractFloat,3}, θᴮ::AbstractArray{<:AbstractFloat,4}, T::Integer)
Takes trigonometric parameters `θᴬ[k∈[1,K], l∈[1,K-1]`, `d∈[1,𝐃𝐞𝐠]`, `θᴬ[k∈[1,K]`, `l∈[1,K-1]`, `d∈[1,𝐃𝐞𝐠]`
"""
function Trig2ARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:AbstractFloat,3}, θᴮ::AbstractArray{<:AbstractFloat,4}, T::Integer)
    K, D, size_order = size(θᴮ)
    @assert K == size(θᴬ, 1)
    if K == 1
        A = ones(K, K, T)
    else
        A = zeros(K, K, T)
        for k = 1:K, l = 1:K-1, t = 1:T
            #TODO use μ, α, θ functions
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
    #TODO use μ, α, θ functions
    p = [1 / (1 + exp(polynomial_trigo(t / T, θᴮ[k, s, h, :]))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]

    return ARPeriodicHMM(a, A, Bernoulli.(p))
end

Trig2ARPeriodicHMM(θᴬ::AbstractArray{<:AbstractFloat,3}, θᴮ::AbstractArray{<:AbstractFloat,4}, T::Integer) = Trig2ARPeriodicHMM(ones(size(θᴬ, 1)) ./ size(θᴬ, 1), θᴬ, θᴮ, T)

"""
    transition_residual(p, (A, K, T)) -> Vector

Residual of the softmax trig fit of one row of a transition matrix: entry `(t, k)` is
`A(t)[k] - A[k, t]`, where `A(t)` is the row the trigonometric coefficients `p` encode. The
`K`-th column is the normalization `1 / (1 + Σₗ exp(Pol[t, l]))`, matching
[`Trig2ARPeriodicHMM`](@ref).
"""
function transition_residual(p, (A, K, T))
    pm = reshape(p, K - 1, :)
    r = similar(p, K * T)
    for t = 1:T
        e = ntuple(l -> exp(polynomial_trigo(t / T, @view(pm[l, :]))), K - 1)
        den = 1 + sum(e)
        for k = 1:K-1
            r[(t-1)*K+k] = e[k] / den - A[k, t]
        end
        r[(t-1)*K+K] = 1 / den - A[K, t]
    end
    return r
end

"""
    bernoulli_residual(p, (B, T)) -> Vector

Residual of the inverse-logit trig fit of a seasonal success probability: entry `t` is
`1 / (1 + exp(P(t/T))) - B[t]`. Note the sign — there is no minus in the exponent, matching the
convention of [`Trig2ARPeriodicHMM`](@ref).
"""
bernoulli_residual(p, (B, T)) = [1 / (1 + exp(polynomial_trigo(t / T, p))) for t = 1:T] .- B

