# # Decoding
#
# The three HMM families here each build their own `N × K` loglikelihood matrix — the spatial one
# batched and threaded — and then decode it with this kernel, on loose `a`, `A` and `LL`.
#
# `HiddenMarkovModels.viterbi` decodes the same path (the test suite asserts it on every family),
# but it is the one generic algorithm this package does not use: it takes `log.(A)` afresh at
# every step and pays its per-step bookkeeping on top, which measured 5× the cost of this decode
# even with the logs cached. Here the `T` log matrices are computed once.
#
# `A[i, j, t]` is the probability of the step *leaving* period `t`, this package's convention
# throughout, so the step into `n` reads `A[:, :, n2t[n-1]]`.

"""
    viterbilog!(T1, T2, z, a, A, LL; n2t) -> z

Fill `z` with the most likely state sequence for the loglikelihood matrix `LL` (`N × K`), given
the initial distribution `a` and the periodic transition arrays `A` (`K × K × T`). `T1`, `T2`
and `z` are the working arrays, allocated beforehand.
"""
function viterbilog!(
    T1::AbstractMatrix,
    T2::AbstractMatrix,
    z::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{<:Any,3},
    LL::AbstractMatrix;
    n2t=n_to_t(size(LL, 1), size(A, 3))::AbstractVector{<:Integer}
)
    @argcheck size(T1) == size(T2) == size(LL)
    @argcheck length(z) == size(LL, 1)
    @argcheck size(LL, 2) == length(a) == size(A, 1) == size(A, 2)

    N, K = size(LL)
    (N == 0) && return z

    fill!(T1, 0)
    fill!(T2, 0)

    al = log.(a)
    Al = log.(A)

    for i in OneTo(K)
        T1[1, i] = al[i] + LL[1, i]
    end

    @inbounds for n = 2:N
        tₙ₋₁ = n2t[n-1]
        for j in OneTo(K)
            amax = 0
            vmax = -Inf

            for i in OneTo(K)
                v = T1[n-1, i] + Al[i, j, tₙ₋₁]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end

            T1[n, j] = vmax + LL[n, j]
            T2[n, j] = amax
        end
    end

    z[N] = argmax(view(T1, N, :))
    for n = N-1:-1:1
        z[n] = T2[n+1, z[n+1]]
    end
    return z
end

"""
    viterbi(a, A, LL; n2t) -> Vector{Int}

Most likely state sequence for the loglikelihood matrix `LL` (`N × K`), the initial distribution
`a` and the periodic transition arrays `A` (`K × K × T`).

The model-level methods `viterbi(hmm, Y, Y_past)` build `LL` and call this.
"""
function viterbi(a::AbstractVector, A::AbstractArray, LL::AbstractMatrix;
    n2t=n_to_t(size(LL, 1), size(A, 3))::AbstractVector{<:Integer})
    T1 = Matrix{Float64}(undef, size(LL))
    T2 = Matrix{Int}(undef, size(LL))
    z = Vector{Int}(undef, size(LL, 1))
    return viterbilog!(T1, T2, z, a, A, LL; n2t=n2t)
end
