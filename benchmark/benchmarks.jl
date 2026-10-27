using BenchmarkTools
using Distributions
using Random
using StableRNGs
using SmoothPeriodicStatsModels
using Ipopt, OptimizationMOI, OptimizationOptimJL
using Optimization, SimpleNonlinearSolve, ForwardDiff # with Optimization: triggers the v3 fitting extension
using LinearAlgebra: BLAS

BLAS.set_num_threads(1)

const SPSM = SmoothPeriodicStatsModels

# ── Running against both v2 and v3 ───────────────────────────────────────────
# AirspeedVelocity benchmarks two revisions with a *single* script, so this file has to speak
# the API of the branch *and* of whatever is on master. Three things differ; everything else is
# common. (`silence`/`warm_start` are not passed at all: v3 ignores them, v4 removed them.)
#
# Delete this block, and the uses below, once both features below are on the default branch.
const V3 = isdefined(SPSM, :fit_mle_AR1)

# v3 renamed this; the old name survives as a deprecated alias, but benchmark the real one.
const ar1_fit = V3 ? SPSM.fit_mle_AR1 : SPSM.model_for_loglikelihood_AR1_full

# v3 funnels solver options through `solve_kwargs` instead of a `kwargs...` slurp.
const ipopt_quiet = V3 ? (; solve_kwargs=(print_level=0,)) : (; print_level=0)

# Observations became sites-first: `D × N` instead of `N × D`, and `D × order` instead of
# `order × D` for the initial condition. This came *after* the v3 release, so neither `V3` nor
# the version number tells the two layouts apart — ask the package instead. A 2-site, 4-step
# draw is unambiguous, and `order = 0` means no initial condition to orient in the first place.
const DXN = let
    hmm = Trig2ARPeriodicHMM(fill(1 / 2, 2), zeros(2, 1, 1), zeros(2, 2, 1, 1), 2)
    size(rand(StableRNG(0), hmm, n_to_t(4, 2); z_ini=1)) == (2, 4)
end

# Observation matrices are written sites-first below and transposed when the loaded version wants
# them time-first, so both revisions are timed on the very same data.
const as_obs = DXN ? identity : permutedims
const SUITE = BenchmarkGroup()

"""
    toy_hmm(rng, K, D, T, order, deg)

An `ARPeriodicHMM` with `K` states, `D` sites, period `T`, local memory `order` and trigonometric
degree `deg`, together with the trigonometric parameters it was built from.
"""
function toy_hmm(rng, K, D, T, order, deg)
    θᴬ = 2 * (rand(rng, K, K - 1, 2deg + 1) .- 1 / 2)
    θᴮ = 2 * (rand(rng, K, D, 2^order, 2deg + 1) .- 1 / 2)
    return Trig2ARPeriodicHMM(fill(1 / K, K), θᴬ, θᴮ, T), θᴬ, θᴮ
end

"""
    flat_guess(θᴬ, θᴮ)

The usual starting point of the Baum-Welch runs: the seasonal mean of each parameter, i.e. the
trigonometric coefficients of order ≥ 1 set to zero.
"""
function flat_guess(θᴬ, θᴮ)
    θᴬ0 = zeros(size(θᴬ))
    θᴬ0[:, :, 1] = θᴬ[:, :, 1]
    θᴮ0 = zeros(size(θᴮ))
    θᴮ0[:, :, :, 1] = θᴮ[:, :, :, 1]
    return θᴬ0, θᴮ0
end

# Seasonal two-component exponential mixture, the model used in the `Mixture` tests.
# `θ` is a 3×3 matrix: one column of trigonometric coefficients per parameter (σ₁, σ₂, α).
mixture_exp2(t, θ::AbstractMatrix) =
    MixtureModel([Exponential(σₜ(t, θ[:, 1])), Exponential(σₜ(t, θ[:, 2]))],
        [αₜ(t, θ[:, 3]), 1 - αₜ(t, θ[:, 3])])
mixture_exp2(t, θ::AbstractVector) = mixture_exp2(t, reshape(θ, 3, 3))

# ── Trigonometric utilities ──────────────────────────────────────────────────
# These are called once per observation inside every likelihood, so they are worth watching.
SUITE["utilities"] = BenchmarkGroup()

let
    β = [0.3, -0.2, 0.5, 0.1, -0.4]   # degree 2

    SUITE["utilities"]["polynomial_trigo_scalar"] = BenchmarkGroup()
    SUITE["utilities"]["polynomial_trigo_vector"] = BenchmarkGroup()
    for T in [100, 10_000]
        ts = collect(1:T)
        SUITE["utilities"]["polynomial_trigo_scalar"][T] =
            @benchmarkable [SPSM.polynomial_trigo(t / $T, $β) for t = 1:$T]
        SUITE["utilities"]["polynomial_trigo_vector"][T] =
            @benchmarkable SPSM.polynomial_trigo($ts ./ $T, $β)
    end

    # `conditional_to` turns the raw observations into the memory category of every (n, site)
    SUITE["utilities"]["conditional_to"] = BenchmarkGroup()
    for order in [1, 2, 3]
        N, D = 50_000, 20
        Y = as_obs(rand(StableRNG(1), Bool, D, N))
        Y_past = as_obs(rand(StableRNG(2), Bool, D, order))
        SUITE["utilities"]["conditional_to"]["order=$order"] =
            @benchmarkable SPSM.conditional_to($Y, $Y_past)
    end
end

# ── ARPeriodicHMM: simulation ────────────────────────────────────────────────
SUITE["hmm"] = BenchmarkGroup()
SUITE["hmm"]["rand"] = BenchmarkGroup()

let
    T, deg, N = 11, 1, 10_000
    for (K, D, order) in [(2, 5, 0), (2, 5, 1), (4, 20, 1), (4, 20, 2)]
        hmm, = toy_hmm(StableRNG(1), K, D, T, order, deg)
        n2t = n_to_t(N, T)
        y_ini = as_obs(rand(StableRNG(2), Bool, D, order))
        SUITE["hmm"]["rand"]["K=$K D=$D order=$order"] = @benchmarkable(
            rand(rng, $hmm, $n2t; y_ini=$y_ini, z_ini=1),
            setup = (rng = StableRNG(3)),
            evals = 1
        )
    end
end

# ── ARPeriodicHMM: inference on a fixed model ────────────────────────────────
SUITE["hmm"]["loglikelihoods"] = BenchmarkGroup()
SUITE["hmm"]["viterbi"] = BenchmarkGroup()
SUITE["hmm"]["complete_loglikelihood"] = BenchmarkGroup()

let
    T, deg = 11, 1
    for (K, D, order, N) in [(2, 5, 1, 10_000), (4, 20, 1, 10_000), (4, 20, 2, 50_000)]
        hmm, = toy_hmm(StableRNG(1), K, D, T, order, deg)
        n2t = n_to_t(N, T)
        y_ini = as_obs(rand(StableRNG(2), Bool, D, order))
        z, y = rand(StableRNG(3), hmm, n2t; y_ini=y_ini, z_ini=1, seq=true)

        key = "K=$K D=$D order=$order N=$N"
        SUITE["hmm"]["loglikelihoods"][key] =
            @benchmarkable loglikelihoods($hmm, $y, $y_ini; n2t=$n2t)
        SUITE["hmm"]["viterbi"][key] =
            @benchmarkable viterbi($hmm, $y, $y_ini; n2t=$n2t)
        SUITE["hmm"]["complete_loglikelihood"][key] =
            @benchmarkable complete_loglikelihood($hmm, $y, $y_ini, $z; n2t=$n2t)
    end
end

# ── ARPeriodicHMM: trigonometric <-> seasonal conversions ────────────────────
SUITE["hmm"]["Trig2ARPeriodicHMM"] = BenchmarkGroup()
SUITE["hmm"]["fit_θ"] = BenchmarkGroup()

let
    T = 11
    for (K, D, order, deg) in [(2, 5, 1, 1), (4, 20, 1, 1), (4, 20, 2, 2)]
        hmm, θᴬ, θᴮ = toy_hmm(StableRNG(1), K, D, T, order, deg)
        a = fill(1 / K, K)
        key = "K=$K D=$D order=$order deg=$deg"
        SUITE["hmm"]["Trig2ARPeriodicHMM"][key] =
            @benchmarkable Trig2ARPeriodicHMM($a, $θᴬ, $θᴮ, $T)
        # `fit_θ` is the inverse direction: least squares back to the trigonometric coefficients
        SUITE["hmm"]["fit_θ"][key] = @benchmarkable(fit_θ($hmm, $deg), evals = 1)
    end
end

# ── ARPeriodicHMM: Baum-Welch ────────────────────────────────────────────────
# `tol = 0` never triggers, so every sample runs exactly `maxiter` iterations: a change in
# convergence speed cannot be mistaken here for a change in the cost of an iteration.
SUITE["hmm"]["fit_mle"] = BenchmarkGroup()

let
    T, deg, maxiter = 11, 1, 5
    for (K, D, order, N) in [(2, 5, 1, 5_000), (3, 6, 1, 10_000), (4, 10, 1, 10_000), (3, 6, 2, 10_000)]
        hmm, θᴬ, θᴮ = toy_hmm(StableRNG(1), K, D, T, order, deg)
        n2t = n_to_t(N, T)
        y_ini = as_obs(rand(StableRNG(2), Bool, D, order))
        y = rand(StableRNG(3), hmm, n2t; y_ini=y_ini, z_ini=1)

        θᴬ0, θᴮ0 = flat_guess(θᴬ, θᴮ)
        hmm0 = Trig2ARPeriodicHMM(fill(1 / K, K), θᴬ0, θᴮ0, T)

        SUITE["hmm"]["fit_mle"]["K=$K D=$D order=$order N=$N"] = @benchmarkable(
            fit_mle($hmm0, $θᴬ0, $θᴮ0, $y, $y_ini;
                n2t=$n2t, maxiter=$maxiter, tol=0.0, robust=true),
            evals = 1
        )
    end
end

# ── ARPeriodicHMMSpatial ─────────────────────────────────────────────────────
# The spatial emission law needs a bivariate normal CDF per pair of sites, so even a short
# series is worth watching here.
SUITE["hmm_spatial"] = BenchmarkGroup()
SUITE["hmm_spatial"]["rand"] = BenchmarkGroup()
SUITE["hmm_spatial"]["loglikelihoods"] = BenchmarkGroup()

let
    K, T, order, deg = 2, 10, 1, 1
    for (D, N) in [(10, 300), (20, 300)]
        rng = StableRNG(1)
        locations = 500 * rand(rng, D, 2)
        distance = [sqrt(sum(abs2, locations[i, :] - locations[j, :])) for i = 1:D, j = 1:D]

        θᴬ = 0.5 * (rand(rng, K, K - 1, 2deg + 1) .- 1 / 2)
        θᴮ = rand(rng, K, D, 2^order, 2deg + 1) .- 1 / 2
        θᴿ = rand(rng, K, 2deg + 1) .- 1 / 2
        θᴿ[:, 1] = log.(100.0 .* (1:K))

        hmm = Trig2ARPeriodicHMMSpatial(fill(1 / K, K), θᴬ, θᴮ, θᴿ, T, distance)
        n2t = n_to_t(N, T)
        y_ini = as_obs(rand(StableRNG(2), Bool, D, order))
        y = rand(StableRNG(3), hmm, n2t; y_ini=y_ini, z_ini=1)

        key = "D=$D N=$N"
        SUITE["hmm_spatial"]["rand"][key] = @benchmarkable(
            rand(rng, $hmm, $n2t; y_ini=$y_ini, z_ini=1),
            setup = (rng = StableRNG(3)),
            evals = 1
        )
        SUITE["hmm_spatial"]["loglikelihoods"][key] = @benchmarkable(
            loglikelihoods($hmm, $y, $y_ini; n2t=$n2t, QMC_m=10),
            evals = 1
        )
    end
end

# ── Seasonal mixture: EM ─────────────────────────────────────────────────────
# As for the HMM, `tol = 0` pins the number of EM iterations.
SUITE["mixture"] = BenchmarkGroup()
SUITE["mixture"]["fit_mle_EM"] = BenchmarkGroup()

let
    θtrue = hcat([1.0, 1.0, -0.2], [3.0, -0.5, 0.6], [0.1, -0.5, 1.0])
    θ0 = hcat([0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.2, 0.0])
    maxiter = 20

    for T in [50, 100], N in [10_000, 50_000]
        n2t = repeat(1:T, N ÷ T)
        rng = StableRNG(1)
        y = [rand(rng, mixture_exp2(t / T, θtrue)) for t in n2t]

        mix0 = [mixture_exp2(t / T, θ0) for t = 1:T]
        θ_α0 = permutedims(θ0[:, 3])
        θ_Y0 = [θ0[:, 1], θ0[:, 2]]

        SUITE["mixture"]["fit_mle_EM"]["T=$T N=$N"] = @benchmarkable(
            fit_mle($mix0, $θ_α0, $θ_Y0, $y, $n2t;
                maxiter=$maxiter, tol=0.0, robust=false),
            evals = 1
        )
    end
end

# ── Seasonal mixture: direct maximization of the likelihood ──────────────────
SUITE["mixture"]["OptimMLE"] = BenchmarkGroup()

let
    θtrue = vec(hcat([1.0, 1.0, -0.2], [3.0, -0.5, 0.6], [0.1, -0.5, 1.0]))
    θ0 = vec(hcat([0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.2, 0.0]))

    for (T, N) in [(50, 2_000), (100, 10_000)]
        n2t = repeat(1:T, N ÷ T)
        rng = StableRNG(1)
        y = [rand(rng, mixture_exp2(t / T, θtrue)) for t in n2t]

        ℓ = (θ, x) -> -sum(logpdf(mixture_exp2(t / T, θ), x[n]) for (n, t) in enumerate(n2t))

        SUITE["mixture"]["OptimMLE"]["Ipopt T=$T N=$N"] = @benchmarkable(
            fit_mle(OptimMLE($ℓ, opt, $θ0), $y; $ipopt_quiet...),
            setup = (opt = Ipopt.Optimizer()),
            evals = 1
        )
        SUITE["mixture"]["OptimMLE"]["NewtonTrustRegion T=$T N=$N"] = @benchmarkable(
            fit_mle(OptimMLE($ℓ, NewtonTrustRegion(), $θ0), $y),
            evals = 1
        )
    end
end

# ── Periodic AR(1) ───────────────────────────────────────────────────────────
SUITE["ar1"] = BenchmarkGroup()
SUITE["ar1"]["rand"] = BenchmarkGroup()
SUITE["ar1"]["fit_mle"] = BenchmarkGroup()

let
    T, d = 20, 1
    μ = [μₜ(t / T, [0.5, 0.3, -0.2]) for t = 1:T]
    ρ = [ρₜ(t / T, [0.4, 0.2, 0.1]) for t = 1:T]
    σ = [σₜ(t / T, [-0.5, 0.1, 0.2]) for t = 1:T]
    AR = AR1(μ, ρ, σ)

    for N in [10_000, 100_000]
        SUITE["ar1"]["rand"][N] = @benchmarkable rand($AR, $N; y₁=0.0)
    end

    # The fit reduces the data to per-period sufficient statistics first, so the objective is
    # O(T) however long the series is; only the one-off statistics pass scales with N.
    for N in [2_000, 5_000]
        Random.seed!(1)
        y = rand(AR, N; y₁=0.0)
        SUITE["ar1"]["fit_mle"][N] = @benchmarkable($ar1_fit($y, $d, $T), evals = 1)
    end
end
