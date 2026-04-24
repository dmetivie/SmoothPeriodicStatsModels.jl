
#-----------------equivalent to trig_conversion.jl  ---------------------------#

function Trig2PeriodicHMMspaMemory(a::AbstractVector, my_trans_θ::AbstractArray{<:AbstractFloat,3}, Bernoulli_θ::AbstractArray{<:AbstractFloat,4}, Range_θ::AbstractArray{<:AbstractFloat,2}, my_T::Integer, my_h::AbstractMatrix)
    my_K, my_D, my_size_order = size(Bernoulli_θ)
    @assert my_K == size(my_trans_θ, 1)

    # make transition matrices as function of time
    if my_K == 1
        my_A = ones(my_K, my_K, my_T)
    else
        my_A = zeros(my_K, my_K, my_T)
        for k = 1:my_K, l = 1:my_K-1, t = 1:my_T
            my_A[k, l, t] = exp(polynomial_trigo(t, my_trans_θ[k, l, :], my_T))
        end
        for k = 1:my_K, t = 1:my_T
            my_A[k, my_K, t] = 1  # last colum is 1/normalization (one could do otherwise)
        end
        normalization_polynomial = [1 + sum(my_A[k, l, t] for l = 1:my_K-1) for k = 1:my_K, t = 1:my_T]
        for k = 1:my_K, l = 1:my_K, t = 1:my_T
            my_A[k, l, t] /= normalization_polynomial[k, t]
        end
    end
    my_A
    # A is a K*K* T matrix of transition.

    #make emission parameters
    my_p = [1 / (1 + exp(polynomial_trigo(t, Bernoulli_θ[k, s, h, :], my_T))) for k = 1:my_K, t = 1:my_T, s = 1:my_D, h = 1:my_size_order]
    # p is a K (states)* T(period) *  D (stations) * m+1 (memory) vector.
    my_range = [exp(polynomial_trigo(t, Range_θ[k, :], my_T)) for k = 1:my_K, t = 1:my_T]
    # range is a K (states)* T(period)  * m+1 (memory) vector.
    # return (my_A, p, range)


    model = PeriodicHMMSpaMemory(a, my_A, my_range, my_p, my_h)
    return model
end

