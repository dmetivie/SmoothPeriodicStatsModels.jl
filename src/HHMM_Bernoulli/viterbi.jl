function viterbi(hmm::HierarchicalPeriodicHMM, 𝐘::AbstractArray{<:Bool}, 𝐘_past::AbstractArray{<:Bool}; robust = false, n2t=n_to_t(size(𝐘, 1), size(hmm.B, 2))::AbstractVector{<:Integer})
    LL = loglikelihoods(hmm, 𝐘, 𝐘_past; n2t=n2t, robust=robust)
    return viterbi(hmm.a, hmm.A, LL; n2t=n2t)
end