"""
    fit_mle_RO(Y[, Y_past], z, n2t, deg_θᴮ, K = length(unique(z)), T = length(unique(n2t)); solver, adtype, ensemble, solve_kwargs) -> (B, θᴮ)

Fit the periodic autoregressive Bernoulli emissions of `D` sites given a known state sequence
`z`. The success probability of site `j` in state `k` with memory category `h` is
`logistic(P(t))`, a trigonometric polynomial of degree `deg_θᴮ`. Use it to add sites to an HMM
whose hidden states are already estimated.

`Y` is `D × N`; the optional `Y_past` is `D × order` and sets the memory order (`0` without it).
Returns the `K × T × D × 2^order` array of `Bernoulli` and the coefficients
`θᴮ[k, j, h, :]`.
"""
function fit_mle_RO(Y::AbstractMatrix{<:Bool}, Y_past::AbstractMatrix{<:Bool}, z, n2t, deg_θᴮ, K = length(unique(z)), T = length(unique(n2t));
    solver=_required_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;))
    @assert K ≥ length(unique(z)) "The provided `K` is smaller than the number of states in the provided sequence `z`"
    @assert T ≥ length(unique(n2t)) "The provided `T` is smaller than the number of days in the provided sequence `n2t`"

    order = size(Y_past, 2)
    size_order = 2^order #! Bernoulli
    rain_cat = 2 #! bernoulli
    D, N = size(Y)
    γₛ = zeros(K, D, size_order, T, rain_cat) # summed smoothing proba
    ν = [Bernoulli() for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    lag_cat = conditional_to(Y, Y_past)
    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(view(Y, j, :) .== y, view(lag_cat, j, :) .== h)) for j = 1:D, h = 1:size_order, y = 0:1] # dry or wet
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history) for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]
    basis_B = trig_basis(T, deg_θᴮ)
    γ = zeros(eltype(z), N, K) # regular smoothing proba
    for (n, k) in enumerate(z)
        γ[n, k] = 1
    end
    θᴮ = zeros(K,D,size_order,2deg_θᴮ+1)
    update_B!(ν, θᴮ, γ, γₛ, Y, n_all, basis_B; solver, adtype, ensemble, solve_kwargs)
    return ν, θᴮ
end

function fit_mle_RO(Y::AbstractMatrix{<:Bool}, z, n2t, deg_θᴮ, K = length(unique(z)), T = length(unique(n2t));
    solver=_required_solver(), adtype=default_adtype(), ensemble=default_ensemble(), solve_kwargs=(;))
    D, N = size(Y)
    Y_past = falses(D, 0)
    order = size(Y_past, 2)
    size_order = 2^order #! Bernoulli
    rain_cat = 2 #! bernoulli
    γₛ = zeros(K, D, size_order, T, rain_cat) # summed smoothing proba
    ν = [Bernoulli() for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    lag_cat = conditional_to(Y, Y_past)
    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(view(Y, j, :) .== y, view(lag_cat, j, :) .== h)) for j = 1:D, h = 1:size_order, y = 0:1] # dry or wet
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history) for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]
    basis_B = trig_basis(T, deg_θᴮ)
    γ = zeros(eltype(z), N, K) # regular smoothing proba
    for (n, k) in enumerate(z)
        γ[n, k] = 1
    end
    θᴮ = zeros(K,D,size_order,2deg_θᴮ+1)
    update_B!(ν, θᴮ, γ, γₛ, Y, n_all, basis_B; solver, adtype, ensemble, solve_kwargs)
    return ν, θᴮ
end