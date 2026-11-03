# # A reference forward-backward, for cross-checking HiddenMarkovModels.jl
#
# These are the scaled forward and backward passes this package used to take from
# `PeriodicHiddenMarkovModels` v0.1 (itself from HMMBase), kept here after v4 moved the models
# onto `HiddenMarkovModels.AbstractHMM` and deleted the package's own inference.
#
# They are what makes the interface testsets a real check rather than a tautology: they index
# `A[:, :, n2t[n-1]]` directly, in this package's own convention — `A(t)` governs the step
# *leaving* period `t` — while the library is handed `transition_matrix(hmm, (t, h))`, which
# shifts by `mod1(t - 1, T)` because HiddenMarkovModels indexes by the step a control *arrives*
# at. Agreement between the two is the evidence that the shift in the accessor is the right one.

function ref_vec_maximum(v::AbstractVector)
    m = v[1]
    @inbounds for i in eachindex(v)
        v[i] > m && (m = v[i])
    end
    return m
end

"Scaled forward pass: fills `α` and the per-step log scaling `c`, so `sum(c)` is the marginal loglikelihood."
function ref_forwardlog!(α::AbstractMatrix, c::AbstractVector, a::AbstractVector,
    A::AbstractArray{<:Any,3}, LL::AbstractMatrix; n2t)
    N, K = size(LL)
    fill!(α, 0)
    fill!(c, 0)

    m = ref_vec_maximum(view(LL, 1, :))
    for j = 1:K
        α[1, j] = a[j] * exp(LL[1, j] - m)
        c[1] += α[1, j]
    end
    for j = 1:K
        α[1, j] /= c[1]
    end
    c[1] = log(c[1]) + m

    @inbounds for n = 2:N
        tₙ₋₁ = n2t[n-1]
        m = ref_vec_maximum(view(LL, n, :))
        for j = 1:K
            for i = 1:K
                α[n, j] += α[n-1, i] * A[i, j, tₙ₋₁]
            end
            α[n, j] *= exp(LL[n, j] - m)
            c[n] += α[n, j]
        end
        for j = 1:K
            α[n, j] /= c[n]
        end
        c[n] = log(c[n]) + m
    end
    return α
end

"Scaled backward pass, using the `c` that [`ref_forwardlog!`](@ref) produced."
function ref_backwardlog!(β::AbstractMatrix, c::AbstractVector, a::AbstractVector,
    A::AbstractArray{<:Any,3}, LL::AbstractMatrix; n2t)
    N, K = size(LL)
    L = zeros(K)
    fill!(β, 0)
    fill!(c, 0)

    for j = 1:K
        β[end, j] = 1
    end

    @inbounds for n = N-1:-1:1
        t = n2t[n]
        m = ref_vec_maximum(view(LL, n + 1, :))
        for i = 1:K
            L[i] = exp(LL[n+1, i] - m)
        end
        for j = 1:K
            for i = 1:K
                β[n, j] += β[n+1, i] * A[j, i, t] * L[i]
            end
            c[n+1] += β[n, j]
        end
        for j = 1:K
            β[n, j] /= c[n+1]
        end
        c[n+1] = log(c[n+1]) + m
    end

    m = ref_vec_maximum(view(LL, 1, :))
    for j = 1:K
        c[1] += a[j] * exp(LL[1, j] - m) * β[1, j]
    end
    c[1] = log(c[1]) + m
    return β
end

"State posteriors `γ ∝ α .* β`, normalised per step."
function ref_posteriors!(γ::AbstractMatrix, α::AbstractMatrix, β::AbstractMatrix)
    N, K = size(α)
    for n = 1:N
        c = 0.0
        for i = 1:K
            γ[n, i] = α[n, i] * β[n, i]
            c += γ[n, i]
        end
        for i = 1:K
            γ[n, i] /= c
        end
    end
    return γ
end
