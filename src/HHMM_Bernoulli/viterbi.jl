function viterbi(hmm::ARPeriodicHMM, Y::AbstractMatrix{<:Bool}, Y_past::AbstractMatrix{<:Bool}; robust = false, n2t=n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer})
    LL = loglikelihoods(hmm, Y, Y_past; n2t=n2t, robust=robust)
    return viterbi(hmm.a, hmm.A, LL; n2t=n2t)
end