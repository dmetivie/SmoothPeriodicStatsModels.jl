
function conditional_to(Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool})
    order = size(Y_past, 1)
    if order == 0
        return ones(Int, size(Y))
    else
        lag_obs = [copy(lag(Y, m)) for m = 1:order]  # transform dry=0 into 1 and wet=1 into 2 for array indexing
        for m = 1:order
            lag_obs[m][1:m, :] .= reverse(Y_past[1:m, :], dims=1) # avoid the missing first row
        end
        return dayx(lag_obs)
    end
end
