#-----------------will soon be replace by using SpatialBernoulli, Caroline's own package.  ---------------------------#

"""
    SpatialBernoulli{TR<:Real, AV<:AbstractVector, AM<:AbstractMatrix, AAM<:AbstractMatrix}
Defines a discrete multivariate distribution `SpatialBernoulli` using a latend Gaussian process. The latent covarience matrix is definied by a Matern covariance (range). 
The marginal Bernoulli probabilities are given by `λ`.
"""
struct SpatialBernoulli{TR<:Real,AV<:AbstractVector,AM<:AbstractMatrix,AAM<:AbstractMatrix} <: DiscreteMultivariateDistribution
    range::TR
    λ::AV
    h::AM
    ΣU::AAM
end

Base.length(d::SpatialBernoulli) = length(d.λ)

"""
    SpatialBernoulli(range, λ, h)
Constructor for a discrete multivariate distribution `SpatialBernoulli` using a latent Gaussian process. The latent covarience matrix is definied by a exp covariance (range). 
The marginal Bernoulli probabilities are given by `λ`.
The constructor used the distance martix to compute the covariance matrix.
"""
function SpatialBernoulli(range, λ, h)
    C_GS = expkernel.(h; range=range)
    return SpatialBernoulli(range, λ, h, C_GS)
end

"""
expkernel(h;range)
Define the exponential kernel. 
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

    pdf(d::SpatialBernoulli, y::AbstractVector{<:Real}; m=length(d) * 100, return_error=false)
pdf(y) = CDF(N(mu,sigma))(0,...,0) is all there is to compute !
``h(y) = \\int_{-infty}^0 \\dots \\int_{-infty}^0 pdf(\\mathcal{N}((-1)^{y_s}\\sigma_s \\phi^{-1}(\\lambda_s)),\\Sigma U)``
recommended to use m= d*1000 inside of MvNormalCDF, set to d*100 as a starting point for this work (is quite ok)
"""
function Distributions.pdf(d::SpatialBernoulli, y::AbstractVector{<:Real}; m=length(d) * 500, return_error=false, a=zeros(length(d)), b=zeros(length(d)), finite_bounds=zeros(length(d)), zerosvec=fill(0, length(d)))
    finite_bounds .= quantile.(Normal(), d.λ)
    a .= ifelse.(y .== 0, finite_bounds, -Inf)
    b .= ifelse.(y .== 1, finite_bounds, Inf)
    hy = mvnormcdf(zerosvec, d.ΣU, a, b; m=m)
    return return_error ? hy : hy[1]
end
