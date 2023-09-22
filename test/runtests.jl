using SmoothPeriodicStatsModels
using Test
using Distributions, Random
using Optimization, OptimizationOptimJL
using Ipopt, OptimizationMOI

# EM for Mixture 
@testset "fit_mle_trig_exp2_EM" begin

    #TODO: for now it just test that this runs, the results are not tested (but so far it reaches a local minima so...)
    Random.seed!(1234)
    # K = 2
    # 𝔻𝕖𝕘 = 1
    f(θ) = MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1-θ[3]])
    f(t, θ) = f([σₜ(t, θ[:,1]), σₜ(t, θ[:,2]), αₜ(t, θ[:,3])])

    θσ1 = [1, 1, -0.2]
    θσ2 = [3, -0.5, 0.6]
    θα = [0.1, -0.5, 1]
    θtrue = hcat(θσ1, θσ2, θα)
    
    # Data
    T = 100
    N = 20000
    n2t = repeat(1:T, N÷T)
    y = [rand(f(t/T, θtrue)) for t in n2t]
    
    #* guess
    θσ10 = [0, 0, 0]
    θσ20 = [5, 0, 0.]
    θα0 = [0.0, 0.2, 0]
    θ0 = hcat(θσ10, θσ20, θα0)
    mix0 = [f(t/T, θ0) for t in 1:T]
    θσ0 = [θσ10, θσ20]
    
    mixt, θ_α, θ_Y, history = fit_mle(mix0, permutedims(θα0), θσ0, y, n2t;
        display=:none, maxiter=1000, tol=1e-5, robust=false, silence=true, warm_start=true)
    
    @test all(diff(history["logtots"]) .> 0) # increasing loglikelihood
    # @btime fit_mle($mix0, $permutedims(θα0), $θσ0, $y, $n2t;display=:none, maxiter=1000, tol=1e-5, robust=false, silence=true, warm_start=true)
    # 765.973 ms (1833703 allocations: 330.39 MiB)
    # local minima though
    # ([θ_Y2[1] θ_Y2[2] θ_α2'])
    # 3×3 Matrix{Float64}:
    #   1.01778    3.03886   0.106579
    #   1.05301   -0.509514  0.234709
    #  -0.280736   0.649009  0.901227
    # 
    # rangeT = (1:T)/T
    # plot(rangeT,t̃-> σₜ(t̃, θσ1))
    # plot!(rangeT,t̃-> σₜ(t̃, θσ2))
    # plot!(rangeT,t̃-> σₜ(t̃, θ_Y[1]), c= 1,s=:dot)
    # plot!(rangeT,t̃-> σₜ(t̃, θ_Y[2]), c= 2,s=:dot)
    # plot!(rangeT,t̃-> σₜ(t̃, θ_Y2[1]), c= 1,s=:dot)
    # plot!(rangeT,t̃-> σₜ(t̃, θ_Y2[2]), c= 2,s=:dot)
    # plot(rangeT,t̃-> αₜ(t̃, θα))
    # plot!(rangeT,t̃-> αₜ(t̃, permutedims(θ_α)))
end

# Optim for Mixture 
@testset "fit_mle_trig_exp2_Optim" begin
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
    n2t = repeat(1:T, N÷T)
    y = [rand(f(t/T, vec(θtrue))) for t in n2t]
    
    ℓ(θ, x) = -sum(logpdf(f(t / T, θ), x[n]) for (n, t) in enumerate(n2t)) # = -loglikelihood

    #* guess
    θσ10 = [0, 0, 0]
    θσ20 = [5, 0, 0.]
    θα0 = [0.0, 0.2, 0]
    θ0 = hcat(θσ10, θσ20, θα0)

    sol_Ipopt = fit_mle(OptimMLE(ℓ, Ipopt.Optimizer(), vec(θ0)), y)
    sol_NewtonTR = fit_mle(OptimMLE(ℓ, NewtonTrustRegion(), vec(θ0)), y)

    @test sol_Ipopt.u ≈ vec(θtrue) rtol = 5e-2
    @test sol_NewtonTR.u ≈ vec(θtrue) rtol = 5e-2
end

# HMM
@testset "Sample from HMM" begin
    K = 4
    T = 20
    D = 10
    N = 100
    ξ = [1; zeros(K-1)]
    ref_station = 1

    # Test that the HMM is well definied with different order of chain (= "local memory" in my jargon)
    for order in 0:3
        hmm_random = randhierarchicalPeriodicHMM(K, T, D, order; ξ=ξ, ref_station=ref_station)

        z, y = rand(hmm_random, N, seq = true, z_ini = 1, y_ini = zeros(Int, order, D))

        y = rand(hmm_random, N)
    end

    #TODO add test comparing order = 0 to PeriodicHMM (it should be exactly the same)
end
