#TODO: K and T could be infered (minus extrem cases where one category/day are missing)
"""
    fit_mle_stations(𝐘::AbstractArray{<:Bool}, 𝐘_past::AbstractArray{<:Bool}, z, n2t, k, K, T)
return smooth matrix of Bernoulli `[Bernoulli(pₖₛₕ(t)) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]` and coeff `θᴮ`
"""
function fit_mle_stations(𝐘::AbstractArray{<:Bool}, 𝐘_past::AbstractArray{<:Bool}, z, n2t, K, T, deg_θᴮ; silence=true, warm_start = true)
    order = size(𝐘_past, 1) #! old convention
    size_order = 2^order #! Bernoulli
    rain_cat = 2 #! bernoulli
    N, D = size(𝐘) #! old convention
    γₛ = zeros(K, D, size_order, T, rain_cat) # summed smoothing proba
    ν = [Bernoulli() for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    lag_cat = conditional_to(𝐘, 𝐘_past)
    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(𝐘[:, j] .== y, lag_cat[:, j] .== h)) for j = 1:D, h = 1:size_order, y = 0:1] # dry or wet
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history) for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]
    model_B = model_for_B(γₛ[1, 1, 1, :, :], deg_θᴮ, silence=silence) # JuMP Model for Emmission distribution
    γ = zeros(eltype(z), N, K) # regular smoothing proba
    for (n, k) in enumerate(z)
        γ[n, k] = 1
    end
    θᴮ = zeros(K,D,size_order,2deg_θᴮ+1)
    update_B!(ν, θᴮ, γ, γₛ, 𝐘, n_all, model_B; warm_start=warm_start)
    return ν, θᴮ
end
