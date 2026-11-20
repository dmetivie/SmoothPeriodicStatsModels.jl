"""
    ARPeriodicHMM(a, θᴬ, θᴮ, T; basis_A, basis_B) -> ARPeriodicHMM

The [`ARPeriodicHMM`](@ref) of period `T` whose transitions and emissions are basis expansions
over the period index `t = 1:T`, `X[t, :] ⋅ θ` with `X = design_matrix(basis, 1:T)`:

- `θᴬ`, `K × (K-1) × nA`: row `k` of `A(t)` is a softmax over `X_A[t, :] ⋅ θᴬ[k, l, :]`,
  `l = 1:K-1`, with state `K` as the reference.
- `θᴮ`, `K × D × 2^order × nB`: the success probability of site `j` in state `k` given memory
  category `h` is `logistic(X_B[t, :] ⋅ θᴮ[k, j, h, :])`.

The bases default to trigonometric polynomials, `TrigBasis(T, d)` for `2d + 1` coefficients.
Any basis of TimeVaryingParameters works, for example `CyclicBSplineBasis(T, n)` for a periodic
index, or `BSplineBasis(1, T, n)` for an index that is not periodic, such as a position.
"""
function ARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:Real,3}, θᴮ::AbstractArray{<:Real,4},
    T::Integer; basis_A=TrigBasis(T, _trig_order(size(θᴬ, 3), "θᴬ")),
    basis_B=TrigBasis(T, _trig_order(size(θᴮ, 4), "θᴮ")))
    K, D, size_order = size(θᴮ, 1), size(θᴮ, 2), size(θᴮ, 3)
    @argcheck size(θᴬ, 1) == K "`θᴬ` and `θᴮ` disagree on the number of states"
    _check_basis(basis_A, size(θᴬ, 3), "θᴬ")
    _check_basis(basis_B, size(θᴮ, 4), "θᴮ")
    A = _transitions!(zeros(K, K, T), θᴬ, design_matrix(basis_A, 1:T))
    X = design_matrix(basis_B, 1:T)
    p = [logistic(_linear_predictor(X, t, view(θᴮ, k, j, h, :)))
         for k = 1:K, t = 1:T, j = 1:D, h = 1:size_order]
    return ARPeriodicHMM(collect(float.(a)), A, Bernoulli.(p))
end

"""
    Trig2ARPeriodicHMM([a, ]θᴬ, θᴮ, T)

The model of trigonometric coefficients: `ARPeriodicHMM(a, θᴬ, θᴮ, T)`, or
`sARPeriodicHMM(a, θᴬ, θᴮ, T)` when `θᴮ` is a matrix of coefficient vectors. `a` defaults to
uniform.
"""
Trig2ARPeriodicHMM(a::AbstractVector, θᴬ::AbstractArray{<:Real,3}, θᴮ::AbstractArray{<:Real,4}, T::Integer) =
    ARPeriodicHMM(a, θᴬ, θᴮ, T)
Trig2ARPeriodicHMM(θᴬ::AbstractArray{<:Real,3}, θᴮ::AbstractArray{<:Real,4}, T::Integer) =
    ARPeriodicHMM(fill(1 / size(θᴬ, 1), size(θᴬ, 1)), θᴬ, θᴮ, T)

"""
    transition_residual(p, (A, K, X)) -> Vector

Residual of the softmax fit of one row of a transition matrix. Entry `(t, k)` is
`A(t)[k] - A[k, t]`, where `A(t)` is the row that the coefficients `p`, `(K-1) × nbasis`, encode
on the `T × nbasis` design matrix `X`. The `K`-th column is the normalization
`1 / (1 + Σₗ exp(X[t, :] ⋅ p[l, :]))`, matching [`ARPeriodicHMM`](@ref).
"""
function transition_residual(p, (A, K, X))
    pm = reshape(p, K - 1, :)
    T = size(A, 2)
    r = similar(p, K * T)
    for t = 1:T
        η = ntuple(l -> _linear_predictor(X, t, @view(pm[l, :])), K - 1)
        shift = reduce(max, η; init=zero(eltype(r)))   # the reference's predictor is 0
        e = map(x -> exp(x - shift), η)                 # so that `exp` cannot overflow
        e_ref = exp(-shift)
        den = e_ref + sum(e)
        for k = 1:K-1
            r[(t-1)*K+k] = e[k] / den - A[k, t]
        end
        r[(t-1)*K+K] = e_ref / den - A[K, t]
    end
    return r
end

"""
    bernoulli_residual(p, (B, X)) -> Vector

Residual of the inverse-logit fit of a periodic success probability: entry `t` is
`logistic(X[t, :] ⋅ p) - B[t]` for the `T × nbasis` design matrix `X`, the convention of
[`ARPeriodicHMM`](@ref).
"""
bernoulli_residual(p, (B, X)) = [logistic(_linear_predictor(X, t, p)) for t in axes(X, 1)] .- B

