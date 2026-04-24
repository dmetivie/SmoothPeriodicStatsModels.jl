function viterbi(hmm::PeriodicHMMSpaMemory, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}; robust=false, n2t=n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m=30)
    LL = loglikelihoods(hmm, Y, Y_past; n2t=n2t, robust=robust, QMC_m=QMC_m)
    return viterbi(hmm.a, hmm.A, LL; n2t=n2t)
end