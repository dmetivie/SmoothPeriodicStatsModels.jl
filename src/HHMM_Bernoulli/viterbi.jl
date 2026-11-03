"""
    viterbi(hmm::ARPeriodicHMM, Y, Y_past; n2t, robust = false) -> Vector{Int}

Most likely hidden state sequence for the `D × N` observations `Y` with initial condition
`Y_past`.
"""
function viterbi(hmm::ARPeriodicHMM, Y::AbstractMatrix{<:Bool}, Y_past::AbstractMatrix{<:Bool}; robust=false, n2t=n_to_t(size(Y, 2), size(hmm.B, 2))::AbstractVector{<:Integer})
    LL = loglikelihoods(hmm, Y, Y_past; n2t=n2t, robust=robust)
    return _viterbi_cached(hmm, Y, LL, conditional_to(Y, Y_past), n2t)
end
