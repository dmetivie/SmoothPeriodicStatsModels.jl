function Trig2ARPeriodicHMMSpatial(a::AbstractVector, trans_θ::AbstractArray{<:AbstractFloat,3}, Bernoulli_θ::AbstractArray{<:AbstractFloat,4}, Range_θ::AbstractArray{<:AbstractFloat,2}, T::Integer, h::AbstractMatrix)
    K, D, size_order = size(Bernoulli_θ)
    @assert K == size(trans_θ, 1)

    # make transition matrices as function of time
    if K == 1
        A = ones(K, K, T)
    else
        A = zeros(K, K, T)
        for k = 1:K, l = 1:K-1, t = 1:T
            A[k, l, t] = exp(polynomial_trigo(t, trans_θ[k, l, :], T))
        end
        for k = 1:K, t = 1:T
            A[k, K, t] = 1  # last colum is 1/normalization (one could do otherwise)
        end
        normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
        for k = 1:K, l = 1:K, t = 1:T
            A[k, l, t] /= normalization_polynomial[k, t]
        end
    end

    # A is a K*K* T matrix of transition.

    #make emission parameters
    p = [1 / (1 + exp(polynomial_trigo(t, Bernoulli_θ[k, s, h, :], T))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    # p is a K (states)* T(period) *  D (stations) * m+1 (memory) vector.
    range = [exp(polynomial_trigo(t, Range_θ[k, :], T)) for k = 1:K, t = 1:T]
    # range is a K (states)* T(period)  * m+1 (memory) vector.
    # return (A, p, range)

    model = ARPeriodicHMMSpatial(a, A, range, p, h)
    return model
end

