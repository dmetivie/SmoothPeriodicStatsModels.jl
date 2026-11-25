#-----------------will soon be replace by using SpatialBernoulli, Caroline's own package.  ---------------------------#

"""
    SpatialBernoulli(range, λ, h)
    SpatialBernoulli(range, λ, h, ΣU)

A multivariate Bernoulli distribution on `D` correlated sites. Site `s` is a success with
probability `λ[s]`, and the sites are correlated through a latent Gaussian vector with covariance
`ΣU = expkernel.(h; range)`, `h` being the `D × D` distance matrix between sites.
"""
struct SpatialBernoulli{TR<:Real,AV<:AbstractVector,AM<:AbstractMatrix,AAM<:AbstractMatrix} <: DiscreteMultivariateDistribution
    range::TR
    λ::AV
    h::AM
    ΣU::AAM
end

Base.length(d::SpatialBernoulli) = length(d.λ)
Base.eltype(::Type{<:SpatialBernoulli}) = Bool   # draws are `Bool` vectors, which `pdf` takes

function SpatialBernoulli(range, λ, h)
    C_GS = expkernel.(h; range=range)
    return SpatialBernoulli(range, λ, h, C_GS)
end

"""
    expkernel(h; range) -> Real

The exponential correlation `exp(-h / range)` at distance `h`, and `1` at `h = 0`.
"""
function expkernel(h; range)
    iszero(h) && return one(h)
    arg = h / range
    return exp(-arg)
end

function Distributions._rand!(rng::AbstractRNG, d::SpatialBernoulli, x::AbstractVector{T}) where {T<:Real}
    u = rand(rng, MvNormal(d.ΣU))
    thresholds = quantile.(Normal(), d.λ)
    x[:] .= u .< thresholds
end

"""
    pdf(d::SpatialBernoulli, y; m = 500length(d), return_error = false, rng = nothing)

The probability of the `Bool` vector `y`: the probability that the latent Gaussian vector lies in
the orthant `y` selects, estimated by quasi-Monte-Carlo with `m` points
(`MvNormalCDF.mvnormcdf`). `return_error = true` returns `(estimate, error)`. `rng` draws the
randomization of the estimate, the default generator of the task without it. `logpdf` is its
logarithm.
"""
function Distributions.pdf(d::SpatialBernoulli, y::AbstractVector{<:Bool}; m=length(d) * 500, return_error=false, a=zeros(length(d)), b=zeros(length(d)), finite_bounds=zeros(length(d)), zerosvec=fill(0, length(d)), rng=nothing)
    finite_bounds .= quantile.(Normal(), d.λ)
    a .= ifelse.(y .== 0, finite_bounds, -Inf)
    b .= ifelse.(y .== 1, finite_bounds, Inf)
    hy = _mvnormcdf(zerosvec, d.ΣU, a, b, m, rng)
    return return_error ? hy : hy[1]
end

# `mvnormcdf` with `rng`, or exactly as before without one.
_mvnormcdf(μ, Σ, a, b, m, ::Nothing) = mvnormcdf(μ, Σ, a, b; m)
_mvnormcdf(μ, Σ, a, b, m, rng::AbstractRNG) = mvnormcdf(μ, Σ, a, b; m, rng)

# Log of `pdf`. Defined because `Distributions`' generic fallback for a multivariate
# distribution goes through `_logpdf`, which this type did not provide — without it
# `logpdf(::SpatialBernoulli, y)` is a `MethodError`, which is why
# `complete_loglikelihood(::ARPeriodicHMMSpatial, …)` could never have run.
#
# The orthant probability is a quasi-Monte-Carlo estimate, so taking the log afterwards is the
# best available: there is no log-domain version of the integral.
function Distributions._logpdf(d::SpatialBernoulli, y::AbstractVector{<:Bool}; kwargs...)
    return log(pdf(d, y; kwargs...))
end

function Distributions.logpdf(d::SpatialBernoulli, y::AbstractVector{<:Bool}; kwargs...)
    return Distributions._logpdf(d, y; kwargs...)
end
