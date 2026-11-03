"""
    viterbi(hmm::ARPeriodicHMMSpatial, Y, Y_past; n2t, robust = false, QMC_m) -> Vector{Int}

Most likely hidden state sequence for the `D × N` observations `Y` with initial condition
`Y_past`.

The emission is a `D`-dimensional orthant probability estimated by quasi-Monte Carlo, so the
decoded path is only as stable as that estimate: `QMC_m` is the per-site budget handed to
[`loglikelihoods`](@ref), which evaluates the whole `N × K` matrix in one threaded sweep.
"""
function viterbi(hmm::ARPeriodicHMMSpatial, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}; robust=false, n2t=n_to_t(size(Y, 2), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m=30)
    LL = loglikelihoods(hmm, Y, Y_past; n2t=n2t, robust=robust, QMC_m=QMC_m)
    return _viterbi_cached(hmm, Y, LL, conditional_to(Y, Y_past), n2t)
end
