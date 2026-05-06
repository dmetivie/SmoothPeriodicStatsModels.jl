using Test
using SmoothPeriodicStatsModels
using Distributions, Random
using Optimization, OptimizationOptimJL
using Ipopt, OptimizationMOI

# EM for Mixture 
@testset "Mixture: fit_mle_trig_exp2_EM" begin

    #TODO: for now it just test that this runs, the results are not tested (but so far it reaches a local minima so...)
    Random.seed!(1234)
    # K = 2
    # 𝔻𝕖𝕘 = 1
    f(θ) = MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
    f(t, θ) = f([σₜ(t, θ[:, 1]), σₜ(t, θ[:, 2]), αₜ(t, θ[:, 3])])

    θσ1 = [1, 1, -0.2]
    θσ2 = [3, -0.5, 0.6]
    θα = [0.1, -0.5, 1]
    θtrue = hcat(θσ1, θσ2, θα)

    # Data
    T = 100
    N = 20000
    n2t = repeat(1:T, N ÷ T)
    y = [rand(f(t / T, θtrue)) for t in n2t]

    #* guess
    θσ10 = [0, 0, 0]
    θσ20 = [5, 0, 0.0]
    θα0 = [0.0, 0.2, 0]
    θ0 = hcat(θσ10, θσ20, θα0)
    mix0 = [f(t / T, θ0) for t in 1:T]
    θσ0 = [θσ10, θσ20]

    mixt, θ_α, θ_Y, history = fit_mle(mix0, permutedims(θα0), θσ0, y, n2t;
        display=:none, maxiter=1000, tol=1e-5, robust=false, silence=true, warm_start=true)

    @test all(diff(history["logtots"]) .> 0) # increasing loglikelihood
end

# Optim for Mixture 
@testset "Mixture: fit_mle_trig_exp2_Optim" begin
    Random.seed!(1234)

    f(θ) = MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
    f(t, θ) = f([σₜ(t, θ[1:3]), σₜ(t, θ[4:6]), αₜ(t, θ[7:9])])

    θσ1 = [1, 1, -0.2]
    θσ2 = [3, -0.5, 0.6]
    θα = [0.1, -0.5, 1]
    θtrue = hcat(θσ1, θσ2, θα)

    # Data
    T = 100
    N = 20000
    n2t = repeat(1:T, N ÷ T)
    y = [rand(f(t / T, vec(θtrue))) for t in n2t]

    ℓ(θ, x) = -sum(logpdf(f(t / T, θ), x[n]) for (n, t) in enumerate(n2t)) # = -loglikelihood

    #* guess
    θσ10 = [0, 0, 0]
    θσ20 = [5, 0, 0.0]
    θα0 = [0.0, 0.2, 0]
    θ0 = hcat(θσ10, θσ20, θα0)

    sol_Ipopt = fit_mle(OptimMLE(ℓ, Ipopt.Optimizer(), vec(θ0)), y)
    sol_NewtonTR = fit_mle(OptimMLE(ℓ, NewtonTrustRegion(), vec(θ0)), y) #TODO figure out SecondOrder to supress warning (and have better perf?)

    @test sol_Ipopt.u ≈ vec(θtrue) rtol = 5e-2
    @test sol_NewtonTR.u ≈ vec(θtrue) rtol = 5e-2
end

# HMM
@testset "Sample from HMM" begin
    K = 4
    T = 20
    D = 10
    N = 100
    ξ = [1; zeros(K - 1)]
    ref_station = 1

    # Test that the HMM is well definied with different order of chain (= "local memory" in my jargon)
    for order in 0:3
        hmm_random = randARPeriodicHMM(K, T, D, order; ξ=ξ, ref_station=ref_station)

        z, y = rand(hmm_random, N, seq=true, z_ini=1, y_ini=zeros(Int, order, D))

        y = rand(hmm_random, N)
    end

    #TODO add test comparing order = 0 to PeriodicHMM (it should be exactly the same)
end

@testset "PeriodicHMM" begin
    Random.seed!(2020)
    K = 3 # Number of Hidden states
    T = 11 # Period
    N = 49_586 # Length of observation
    D = 6 # dimension of observed variables
    autoregressive_order = 1
    size_order = 2^autoregressive_order
    degree_of_P = 1
    size_degree_of_P = 2 * degree_of_P + 1
    trans_θ = 4 * (rand(K, K - 1, size_degree_of_P) .- 1 / 2)
    Bernoulli_θ = 2 * (rand(K, D, size_order, size_degree_of_P) .- 1 / 2)
    hmm = Trig2ARPeriodicHMM([1 / 3, 1 / 6, 1 / 2], trans_θ, Bernoulli_θ, T)
    z_ini = 1
    y_past = rand(Bool, autoregressive_order, D)
    n2t = n_to_t(N, T)
    z, y = rand(hmm, n2t; y_ini=y_past, z_ini=z_ini, seq=true)

    trans_θ_guess = rand(K, K - 1, size_degree_of_P)
    trans_θ_guess[:, :, 1] .= trans_θ[:, :, 1]
    Bernoulli_θ_guess = zeros(K, D, size_order, size_degree_of_P)
    Bernoulli_θ_guess[:, :, :, 1] = Bernoulli_θ[:, :, :, 1]
    hmm_guess = Trig2ARPeriodicHMM([1 / 4, 1 / 4, 1 / 2], trans_θ_guess, Bernoulli_θ_guess, T)

    @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_guess, trans_θ_guess, Bernoulli_θ_guess, y, y_past, maxiter=10000, robust=true; display=:iter, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)
    @test θq_fit ≈ trans_θ rtol = 20e-2
    @test θy_fit ≈ Bernoulli_θ rtol = 20e-2
end

@testset "Hierachical PeriodicHMM for many hyperparams" begin
    Random.seed!(2020)
    T = 5 # Period
    N = 100 # Length of observation
    D = 6 # dimension of observed variables
    for K in 1:2, autoregressive_order in 0:2, degree_of_P in 0:2
        size_order = 2^autoregressive_order
        size_degree_of_P = 2 * degree_of_P + 1
        trans_θ = 4 * (rand(K, K - 1, size_degree_of_P) .- 1 / 2)
        Bernoulli_θ = 2 * (rand(K, D, size_order, size_degree_of_P) .- 1 / 2)
        ξ = (1:K) / sum(1:K)
        hmm = Trig2ARPeriodicHMM(ξ, trans_θ, Bernoulli_θ, T)
        z_ini = 1
        y_past = rand(Bool, autoregressive_order, D)
        n2t = n_to_t(N, T)
        z, y = rand(hmm, n2t; y_ini=y_past, z_ini=z_ini, seq=true)

        trans_θ_guess = rand(K, K - 1, size_degree_of_P)
        trans_θ_guess[:, :, 1] .= trans_θ[:, :, 1]
        Bernoulli_θ_guess = zeros(K, D, size_order, size_degree_of_P)
        Bernoulli_θ_guess[:, :, :, 1] = Bernoulli_θ[:, :, :, 1]
        hmm_guess = Trig2ARPeriodicHMM(ξ, trans_θ_guess, Bernoulli_θ_guess, T)

        @time "FitMLE SHMM (Baum Welch) K = $K, autoregressive_order = $autoregressive_order, degree_of_P = $degree_of_P" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_guess, trans_θ_guess, Bernoulli_θ_guess, y, y_past, maxiter=10000, robust=true; silence=true, tol=1e-3, θ_iters=true, n2t=n2t)
        z_hat = viterbi(hmm_fit, y, y_past; n2t=n2t)
        ll_fit = complete_loglikelihood(hmm_fit, y, y_past, z_hat; n2t=n2t)
        ll_true = complete_loglikelihood(hmm, y, y_past, z; n2t=n2t)

        # @test ll_fit ≈ ll_true rtol = 20e-2
        # @test z_hat ≈ z rtol = 20e-2
        # @test θq_fit ≈ trans_θ rtol = 20e-2
        # @test θy_fit ≈ Bernoulli_θ rtol = 20e-2
    end
end

using Distributed
addprocs(2)
@everywhere using SmoothPeriodicStatsModels
@testset "Distributed PeriodicHMM" begin
    Random.seed!(2020)
    K = 4 # Number of Hidden states
    T = 11 # Period
    N = 49_586 # Length of observation
    D = 10 # dimension of observed variables
    autoregressive_order = 1
    size_order = 2^autoregressive_order
    degree_of_P = 1
    size_degree_of_P = 2 * degree_of_P + 1
    trans_θ = 4 * (rand(K, K - 1, size_degree_of_P) .- 1 / 2)
    Bernoulli_θ = 2 * (rand(K, D, size_order, size_degree_of_P) .- 1 / 2)
    ξ = (1:K) / sum(1:K)
    hmm = Trig2ARPeriodicHMM(ξ, trans_θ, Bernoulli_θ, T)
    z_ini = 1
    y_past = rand(Bool, autoregressive_order, D)
    n2t = n_to_t(N, T)
    z, y = rand(hmm, n2t; y_ini=y_past, z_ini=z_ini, seq=true)

    trans_θ_guess = rand(K, K - 1, size_degree_of_P)
    trans_θ_guess[:, :, 1] .= trans_θ[:, :, 1]
    Bernoulli_θ_guess = zeros(K, D, size_order, size_degree_of_P)
    Bernoulli_θ_guess[:, :, :, 1] = Bernoulli_θ[:, :, :, 1]
    hmm_guess = Trig2ARPeriodicHMM(ξ, trans_θ_guess, Bernoulli_θ_guess, T)

    @time "pmap worker" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_guess, trans_θ_guess, Bernoulli_θ_guess, y, y_past, maxiter=10000, robust=true; silence=true, tol=1e-3, θ_iters=true, n2t=n2t)
    # pmap 0 worker: 31.229542 seconds (161.79 M allocations: 17.413 GiB, 2.89% gc time, 0.01% compilation time)
    # map alone: 32.788183 seconds (161.73 M allocations: 17.411 GiB, 3.53% gc time)
    # pmap 10 worker: 19.509379 seconds (165.01 M allocations: 18.707 GiB, 7.18% gc time)
end

include("test_hmmspa.jl")