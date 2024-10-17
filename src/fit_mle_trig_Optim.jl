abstract type AbstractMLE end

"""
    OptimMLE{T,V} <: AbstractMLE where {T<:AbstractOptimizationAlgorithm, V<:AbstractVector}
- ℓ::Function is `-loglikelihood`
- solver
- θ₀::AbstractVector Starting point of the Optimization algo
"""
struct OptimMLE{T,V} <: AbstractMLE where {T, V<:AbstractVector}
    ℓ::Function # -loglikelihood
    solver::T # <:AbstractOptimizationAlgorithm
    θ₀::V 
end
#TODO figure out SecondOrder when needed to supress warning (and have better perf?)
function Distributions.fit_mle(Opt::OptimMLE, y::AbstractArray; solvekwargs...)
    OptFunc = OptimizationFunction(Opt.ℓ, Optimization.SecondOrder(Optimization.AutoForwardDiff(), Optimization.AutoZygote()))
    prob = OptimizationProblem(OptFunc, Opt.θ₀, y)

    return solve(prob, Opt.solver; solvekwargs...)
end

fit_loss_optim(ℓ, y, θ0, method = :Ipopt; kwargs...) = fit_mle(OptimMLE(ℓ, Ipopt.Optimizer(), vec(θ0)), y; kwargs...)