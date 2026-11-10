"""
    ARPeriodicHMMSpatial(a, θᴬ, θᴮ, θᴿ, T, h; QMC_m = 30, basis_A, basis_B, basis_R) -> ARPeriodicHMMSpatial

The [`ARPeriodicHMMSpatial`](@ref) of period `T` given by coefficients: `θᴬ`, `θᴮ` and their bases
as for the coefficient method of [`ARPeriodicHMM`](@ref), and `θᴿ` (`K × nR`), the coefficients
of the spatial range `exp(X_R[t, :] ⋅ θᴿ[k, :])` of each state. `h` is the `D × D` distance
matrix between sites. The bases default to trigonometric polynomials.
"""
function ARPeriodicHMMSpatial(a::AbstractVector, θᴬ::AbstractArray{<:Real,3},
    θᴮ::AbstractArray{<:Real,4}, θᴿ::AbstractMatrix{<:Real}, T::Integer, h::AbstractMatrix;
    QMC_m::Integer=30,
    basis_A=TrigBasis(T, _trig_order(size(θᴬ, 3), "θᴬ")),
    basis_B=TrigBasis(T, _trig_order(size(θᴮ, 4), "θᴮ")),
    basis_R=TrigBasis(T, _trig_order(size(θᴿ, 2), "θᴿ")))
    K, D, size_order = size(θᴮ, 1), size(θᴮ, 2), size(θᴮ, 3)
    @argcheck size(θᴬ, 1) == K == size(θᴿ, 1) "`θᴬ`, `θᴮ` and `θᴿ` disagree on the number of states"
    _check_basis(basis_A, size(θᴬ, 3), "θᴬ")
    _check_basis(basis_B, size(θᴮ, 4), "θᴮ")
    _check_basis(basis_R, size(θᴿ, 2), "θᴿ")
    A = _transitions!(zeros(K, K, T), θᴬ, design_matrix(basis_A, 1:T))
    XB, XR = design_matrix(basis_B, 1:T), design_matrix(basis_R, 1:T)
    p = [logistic(_linear_predictor(XB, t, view(θᴮ, k, s, hh, :)))
         for k = 1:K, t = 1:T, s = 1:D, hh = 1:size_order]
    range = [exp(_linear_predictor(XR, t, view(θᴿ, k, :))) for k = 1:K, t = 1:T]
    return ARPeriodicHMMSpatial(collect(float.(a)), A, range, p, h, Int(QMC_m))
end

"""
    Trig2ARPeriodicHMMSpatial(a, θᴬ, θᴮ, θᴿ, T, h; QMC_m = 30) -> ARPeriodicHMMSpatial

The model of trigonometric coefficients: `ARPeriodicHMMSpatial(a, θᴬ, θᴮ, θᴿ, T, h; QMC_m)`.
"""
Trig2ARPeriodicHMMSpatial(a::AbstractVector, θᴬ::AbstractArray{<:Real,3}, θᴮ::AbstractArray{<:Real,4},
    θᴿ::AbstractMatrix{<:Real}, T::Integer, h::AbstractMatrix; QMC_m::Integer=30) =
    ARPeriodicHMMSpatial(a, θᴬ, θᴮ, θᴿ, T, h; QMC_m)
