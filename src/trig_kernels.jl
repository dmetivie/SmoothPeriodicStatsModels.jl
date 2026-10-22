# # M-step objectives of the seasonal EM algorithms
#
# Each kernel is the negative of the quantity the corresponding JuMP model used to maximize, so
# that it can be handed straight to `solve_unconstrained`. They all share the same shape: the
# decision variable is a vector of trigonometric coefficients, and the data are the posterior
# weights aggregated per period index `t`.

"""
    trig_basis(T, d) -> Matrix{Float64}

`T × (2d+1)` matrix whose row `t` is `[1, cos(2π t/T), sin(2π t/T), …, cos(2π d t/T), sin(2π d t/T)]`,
so that `basis * β` evaluates [`polynomial_trigo`](@ref) at every `t = 1:T` at once.

Built once per model instead of inside every objective evaluation.
"""
function trig_basis(T::Integer, d::Integer)
    basis = Matrix{Float64}(undef, T, 2d + 1)
    for t = 1:T
        basis[t, 1] = 1
        for l = 1:d
            basis[t, 2l] = cos(2π * l * t / T)
            basis[t, 2l+1] = sin(2π * l * t / T)
        end
    end
    return basis
end

"""
    log1pexp_sum(x) -> Number

`log(1 + Σᵢ exp(xᵢ))`, computed by shifting out the largest exponent so that it does not overflow
for large `xᵢ`. Equal to `logsumexp` over `x` with an extra zero entry.
"""
function log1pexp_sum(x)
    m = zero(eltype(x))
    for xi in x
        xi > m && (m = xi)
    end
    s = exp(-m)                 # the implicit zero entry
    for xi in x
        s += exp(xi - m)
    end
    return m + log(s)
end

"""
    bernoulli_logit_nll(θ, (γ, basis)) -> Number

Negative of the seasonal Bernoulli loglikelihood maximized by the emission M-step.

`γ[t, 1]` and `γ[t, 2]` are the posterior weights of a failure and of a success at period index
`t`, and the success probability is `p(t) = 1 / (1 + exp(P(t)))` with `P = basis * θ` — the sign
convention of [`Trig2ARPeriodicHMM`](@ref). The objective is then
`-Σₜ γ[t,1] log(1 - p(t)) + γ[t,2] log p(t)`.
"""
function bernoulli_logit_nll(θ, (γ, basis))
    P = basis * θ
    nll = zero(eltype(P))
    for t in eachindex(P)
        # log(1 - p) = -log1pexp(-P) and log(p) = -log1pexp(P); `log1pexp` does not overflow
        # for large |P|, where the `log1p(exp(·))` of the JuMP model did
        nll += γ[t, 1] * log1pexp(-P[t]) + γ[t, 2] * log1pexp(P[t])
    end
    return nll
end

"""
    multinomial_logit_nll(θ, (s, basis, K)) -> Number

Negative of the seasonal multinomial-logit loglikelihood maximized by the transition M-step and
by the mixture-weight M-step — the two are the same objective.

`θ` holds the `(K-1) × (2d+1)` coefficients of the `K-1` free categories, flattened. `s[t, l]` is
the posterior weight of category `l` at period index `t`, over all `K` categories; the `K`-th is
the reference whose polynomial is fixed to zero.
"""
function multinomial_logit_nll(θ, (s, basis, K))
    T = size(basis, 1)
    P = basis * transpose(reshape(θ, K - 1, :))   # T × (K-1)
    nll = zero(eltype(P))
    for t = 1:T
        total = zero(eltype(s))
        for l = 1:K
            total += s[t, l]
        end
        acc = zero(eltype(P))
        for l = 1:K-1
            acc += s[t, l] * P[t, l]
        end
        nll += total * log1pexp_sum(view(P, t, :)) - acc
    end
    return nll
end

"""
    exponential_nll(θ, (s_π, s_πy, basis)) -> Number

Negative of the seasonal `Exponential` loglikelihood maximized by the mixture component M-step.

With scale `λ(t) = exp(P(t))`, `P = basis * θ`, the loglikelihood of the observations attached to
period index `t` is `-s_π[t] P(t) - s_πy[t] exp(-P(t))`, where `s_π[t]` is the summed posterior
weight and `s_πy[t]` the posterior-weighted sum of the observations.
"""
function exponential_nll(θ, (s_π, s_πy, basis))
    P = basis * θ
    nll = zero(eltype(P))
    for t in eachindex(P)
        nll += s_π[t] * P[t] + s_πy[t] * exp(-P[t])
    end
    return nll
end
