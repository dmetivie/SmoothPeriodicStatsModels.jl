"""
    fit_mle_RO(Y::AbstractArray{<:Bool}[, Y_past::AbstractArray{<:Bool}], z, n2t, deg_θᴮ, K = length(unique(z)), T = length(unique(n2t)); silence=true, warm_start = true)
This function fits at one location the observed rain occurrences (RO) `Y` with a (smooth) periodic Bernoulli distribution. 
The autoregressive order is determined by the size of the initial (past) input `Y_past`. If not provided autoregressive order is `0`.
`K` and `T` can be provided if different from `K = length(unique(z))` and `T = length(unique(n2t))`.
# Output
Matrix of Bernoulli `[Bernoulli(pₖₛₕ(t)) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]` and coefficients `θᴮ` of the trigonometric expansion of `pₖₛₕ(t) = αₜ(t, θᴮ)`

Currently, the station Rain Occurences are only dependant of the hidden states (conditional independdance) and time of the year. 
TODO: possibility to add stations with a conditional dependance to pre-existing stations (useful for close stations).
"""
function fit_mle_RO(Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}, z, n2t, deg_θᴮ, K = length(unique(z)), T = length(unique(n2t)); silence=true, warm_start = true)
    @assert K ≥ length(unique(z)) "The provided `K` is smaller than the number of states in the provided sequence `z`"
    @assert T ≥ length(unique(n2t)) "The provided `T` is smaller than the number of days in the provided sequence `n2t`"

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

function fit_mle_RO(Y::AbstractArray{<:Bool}, z, n2t, deg_θᴮ, K = length(unique(z)), T = length(unique(n2t)); silence=true, warm_start = true)
    N, D = size(Y) #! Table convention
    Y_past = zeros(Int, 0, D)
    order = size(Y_past, 1) #! Table convention
    size_order = 2^order #! Bernoulli
    rain_cat = 2 #! bernoulli
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