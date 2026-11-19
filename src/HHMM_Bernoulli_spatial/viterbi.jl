"""
    viterbi(hmm::ARPeriodicHMMSpatial, Y, Y_past; n2t, robust = false, QMC_m = hmm.QMC_m) -> Vector{Int}

Most likely hidden state sequence for the `D × N` observations `Y` with initial condition
`Y_past`.

The emission loglikelihoods are quasi-Monte-Carlo estimates with `QMC_m` points per site (see
[`loglikelihoods`](@ref)), so the decoded path can vary with that budget.
"""
function viterbi(hmm::ARPeriodicHMMSpatial, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}; robust=false, n2t=n_to_t(size(Y, 2), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m=hmm.QMC_m)
    LL = loglikelihoods(hmm, Y, Y_past; n2t=n2t, robust=robust, QMC_m=QMC_m)
    return viterbi(hmm.a, hmm.A, LL; n2t=n2t)
end
