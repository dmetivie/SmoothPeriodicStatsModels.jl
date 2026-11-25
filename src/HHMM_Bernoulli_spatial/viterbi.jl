"""
    viterbi(hmm::ARPeriodicHMMSpatial, Y, Y_past; n2t, robust = false, QMC_m = hmm.QMC_m, rng = nothing) -> Vector{Int}

Most likely hidden state sequence for the `D × N` observations `Y` with initial condition
`Y_past`.

The emission loglikelihoods are quasi-Monte-Carlo estimates with `QMC_m` points per site (see
[`loglikelihoods`](@ref)), so the decoded path can vary with that budget. `rng` seeds those
estimates.
"""
function viterbi(hmm::ARPeriodicHMMSpatial, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}; robust=false, n2t=n_to_t(size(Y, 2), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m=hmm.QMC_m, rng=nothing)
    LL = loglikelihoods(hmm, Y, Y_past; n2t=n2t, robust=robust, QMC_m=QMC_m, rng)
    return viterbi(hmm.a, hmm.A, LL; n2t=n2t)
end
