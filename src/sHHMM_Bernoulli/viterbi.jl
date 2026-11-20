"""
    viterbi(hmm::sARPeriodicHMM, Y, Y_past; n2t, robust = false) -> Vector{Int}

Most likely hidden state sequence for the `D × N` observations `Y`. `Y_past` is one initial
condition per site, since the sites differ in autoregressive order.
"""
function viterbi(hmm::sARPeriodicHMM, Y::AbstractMatrix{<:Bool}, Y_past::AbstractVector; robust=false, n2t=n_to_t(size(Y, 2), size(hmm.B, 2))::AbstractVector{<:Integer})
    LL = loglikelihoods(hmm, Y, Y_past; n2t=n2t, robust=robust)
    return viterbi(hmm.a, hmm.A, LL; n2t=n2t)
end
