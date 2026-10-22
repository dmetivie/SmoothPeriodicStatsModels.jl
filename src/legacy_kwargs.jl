"""
    _warn_legacy_solver_kwargs(silence, warm_start)

Warn once for each of the two JuMP-era keywords still accepted by the fitting entry points.

`silence` silenced Ipopt; verbosity is now whatever the chosen solver understands, passed through
`solve_kwargs`. `warm_start` toggled `set_start_value`; every M-step now starts from the previous
iterate unconditionally, which is what `warm_start = true` did.
"""
function _warn_legacy_solver_kwargs(silence, warm_start)
    silence === nothing ||
        @warn "`silence` is deprecated and ignored: pass solver options through `solve_kwargs`, e.g. `solve_kwargs = (print_level = 0,)`." maxlog = 1
    warm_start === nothing ||
        @warn "`warm_start` is deprecated and ignored: every M-step now starts from the previous iterate." maxlog = 1
    return nothing
end

