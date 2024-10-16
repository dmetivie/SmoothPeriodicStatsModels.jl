"""
    fit_mle_YY(Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}, z, n2t, deg_θᴮ, K = length(unique(z)), T = length(unique(n2t)); silence=true, warm_start = true)
Given the hidden states `z`, this function fits the smooth rain occurrences `Y` Bernoulli distribution at one location.
Note that this new station is currently added independently of pre-existing stations (beside the hidden state correlations).
Return smooth matrix of Bernoulli `[Bernoulli(pₖₛₕ(t)) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]` and coeff `θᴮ`
`K` and `T` can be provided if different from `K = length(unique(z))` and `T = length(unique(n2t))`.
"""
function fit_mle_YY(Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}, z, n2t, deg_θᴮ, K = length(unique(z)), T = length(unique(n2t)); silence=true, warm_start = true)
    order = size(Y_past, 1) #! Table convention
    size_order = 2^order #! Bernoulli
    rain_cat = 2 #! bernoulli
    N, D = size(Y) #! old convention
    γₛ = zeros(K, D, size_order, T, rain_cat) # summed smoothing proba
    ν = [Bernoulli() for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    lag_cat = conditional_to(Y, Y_past)
    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(Y[:, j] .== y, lag_cat[:, j] .== h)) for j = 1:D, h = 1:size_order, y = 0:1] # dry or wet
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history) for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]
    model_B = model_for_B(γₛ[1, 1, 1, :, :], deg_θᴮ, silence=silence) # JuMP Model for Emmission distribution
    γ = zeros(eltype(z), N, K) # regular smoothing proba
    for (n, k) in enumerate(z)
        γ[n, k] = 1
    end
    θᴮ = zeros(K,D,size_order,2deg_θᴮ+1)
    update_B!(ν, θᴮ, γ, γₛ, Y, n_all, model_B; warm_start=warm_start)
    return ν, θᴮ
end
