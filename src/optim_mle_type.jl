abstract type AbstractMLE end

# All three fields are parametric so that the type stays concrete: with `ℓ::Function` the compiler
# cannot see through `OptimizationFunction(opt.ℓ, adtype)`.
"""
    OptimMLE(ℓ, solver, θ₀)
    OptimMLE(ℓ, θ₀)

A maximum likelihood problem stated directly as a loss to minimize, for models the package has no
dedicated `fit_mle` for.

- `ℓ(θ, y)` is the **negative** loglikelihood, taking the parameters and the data
- `solver` is the algorithm handed to `solve`; it defaults to [`default_solver`](@ref) and can
  still be overridden per call
- `θ₀` is the starting point

The solver stored here is a default, not a constraint: `fit_mle(opt, y; solver = …)` wins.
"""
struct OptimMLE{F,S,V<:AbstractVector} <: AbstractMLE
    ℓ::F # -loglikelihood
    solver::S
    θ₀::V
end

OptimMLE(ℓ, θ₀::AbstractVector) = OptimMLE(ℓ, default_solver(), θ₀)

