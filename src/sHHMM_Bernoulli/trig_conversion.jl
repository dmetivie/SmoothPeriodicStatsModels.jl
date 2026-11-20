"""
    sARPeriodicHMM(a, θᴬ, θᴮ::AbstractMatrix, T; basis_A, basis_B) -> sARPeriodicHMM

The [`sARPeriodicHMM`](@ref) of period `T` given by coefficients. `θᴬ` and the bases are as for
the coefficient method of [`ARPeriodicHMM`](@ref); `θᴮ` is a `K × D` matrix whose entry
`θᴮ[k, j]` holds one coefficient vector per memory category of site `j`: the success
probability is `logistic(X_B[t, :] ⋅ θᴮ[k, j][h])`.
"""
function sARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:Real,3}, θᴮ::AbstractMatrix,
    T::Integer; basis_A=TrigBasis(T, _trig_order(size(θᴬ, 3), "θᴬ")),
    basis_B=TrigBasis(T, _trig_order(length(first(first(θᴮ))), "θᴮ")))
    K, D = size(θᴮ)
    @argcheck size(θᴬ, 1) == K "`θᴬ` and `θᴮ` disagree on the number of states"
    _check_basis(basis_A, size(θᴬ, 3), "θᴬ")
    foreach(θh -> _check_basis(basis_B, length(θh), "θᴮ"), Iterators.flatten(θᴮ))
    A = _transitions!(zeros(K, K, T), θᴬ, design_matrix(basis_A, 1:T))
    X = design_matrix(basis_B, 1:T)
    B = [Bernoulli.([logistic(_linear_predictor(X, t, θh)) for θh in θᴮ[k, s]])
         for k = 1:K, t = 1:T, s = 1:D]
    return sARPeriodicHMM(collect(float.(a)), A, B)
end

Trig2ARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:Real,3}, θᴮ::AbstractMatrix, T::Integer) =
    sARPeriodicHMM(a, θᴬ, θᴮ, T)
Trig2ARPeriodicHMM(θᴬ::AbstractArray{<:Real,3}, θᴮ::AbstractMatrix, T::Integer) =
    sARPeriodicHMM(fill(1 / size(θᴬ, 1), size(θᴬ, 1)), θᴬ, θᴮ, T)
