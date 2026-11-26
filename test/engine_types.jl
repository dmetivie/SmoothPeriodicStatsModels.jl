# The engine types of 4.2: `SmoothDistribution`, the size constructor, `initialize` and
# `SliceEM`, the fit of an HMM given state weights, `curve` and `coef` on every model, and an
# `rng` for the spatial HMM. Each is checked against the code it replaces, to the last bit where
# the arithmetic is the same.

import SciMLBase

@testset "SmoothDistribution: construction and evaluation" begin
    T = 20
    sd = SmoothDistribution(Gamma, T)
    @test size(parameters(sd).α) == (3, 1)
    @test size(curve(sd, :α)) == (1, T)
    @test curve(sd, :α; state=1) == fill(1.0, T)        # zero coefficients
    @test sd(3) isa Gamma
    sd3 = SmoothDistribution(Gamma, T; K=3, season=2)
    @test size(parameters(sd3).α) == (5, 3)
    @test size(curve(sd3, :θ)) == (3, T)
    @test curve(sd3, :θ, 1:5; state=2) == curve(sd3, :θ)[2, 1:5]
    @test sd3(3; state=2) isa Gamma
    @test_throws ArgumentError sd3(3)                    # which state?
    @test_throws ArgumentError sd3(3; state=0)
    @test_throws ArgumentError curve(sd3, :θ; state=4)
    @test_throws ArgumentError curve(sd3, :nosuch)

    # `season`: one basis for every parameter, or one each
    sdn = SmoothDistribution(Gamma, T; season=(α=0, θ=2))
    @test size(parameters(sdn).α, 1) == 1 && size(parameters(sdn).θ, 1) == 5
    @test size(parameters(SmoothDistribution(Gamma, T; season=CyclicBSplineBasis(T, 6))).α, 1) == 6
    @test_throws ArgumentError SmoothDistribution(Gamma, T; season=(α=0,))
    @test_throws ArgumentError SmoothDistribution(Gamma, T; season=(α=0, θ=1, β=2))
    @test_throws ArgumentError SmoothDistribution(Gamma, T; season=-1)
    @test_throws ArgumentError SmoothDistribution(Gamma, T; season="a")
    @test_throws ArgumentError SmoothDistribution(Gamma, T; K=0)
    @test_throws ArgumentError SmoothDistribution(Gamma, 0)
    @test_throws ArgumentError SmoothDistribution(Pareto, T)   # no `seasonal_family` for it

    # `init`, one start for every state or one each
    two = SmoothDistribution(Gamma, T; K=2, init=[(α=2.0, θ=1.0), (α=3.0, θ=0.5)])
    @test curve(two, :α) ≈ repeat([2.0, 3.0], 1, T)
    @test curve(SmoothDistribution(Gamma, T; K=2, init=(α=2.0, θ=1.0)), :θ) ≈ fill(1.0, 2, T)
    @test_throws ArgumentError SmoothDistribution(Gamma, T; init=(α=0.0, θ=1.0))   # outside the domain
    @test_throws ArgumentError SmoothDistribution(Gamma, T; K=2, init=[(α=2.0, θ=1.0)])
    @test_throws ArgumentError SmoothDistribution(Gamma, T; init="x")
    fam = seasonal_family(Normal, TrigBasis(T, 1))
    @test size(curve(SmoothDistribution(fam, T; K=2), :σ)) == (2, T)

    N = 300
    rng = StableRNG(31)
    n2t = n_to_t(N, T)
    z = rand(rng, 1:2, N)
    y = rand(rng, two, n2t; z)
    @test length(y) == N && rand(StableRNG(32), two, n2t; z) == rand(StableRNG(32), two, n2t; z)
    LL = loglikelihoods(two, y; n2t)
    @test size(LL) == (N, 2)
    @test LL[5, 2] ≈ logpdf(two(n2t[5]; state=2), y[5])
    @test loglikelihood(two, y, OneHot(z, 2); n2t) ≈ sum(LL[n, z[n]] for n = 1:N)
    @test loglikelihoods(two, y; n2t, past=1.0) == LL
    @test sum(loglikelihoods(sd, y; n2t)) ≈ loglikelihood(sd, y; n2t)
    @test_throws ArgumentError loglikelihood(two, y; n2t)               # which state?
    @test_throws ArgumentError loglikelihoods(two, y; n2t=n2t[1:end-1])
    @test_throws ArgumentError loglikelihoods(two, y; n2t=n2t .+ T)
    @test_throws ArgumentError rand(two, n2t)
    @test_throws ArgumentError rand(two, n2t; z=z[1:end-1])
    @test_throws ArgumentError rand(two, n2t; z=fill(3, N))
    @test_throws ArgumentError rand(sd, n2t; z=fill(2, N))
    @test mean(rand(StableRNG(33), two, fill(4, 40_000); z=fill(2, 40_000))) ≈ mean(two(4; state=2)) rtol = 0.02

    # `coef` is the live container
    @test coef(two) === parameters(two)
    coef(two).α[1, 2] = log(5.0)
    @test curve(two, :α; state=2) ≈ fill(5.0, T)
end

# The fit of SWG's seasonal Gamma of the temperature range given the states, written out as it is
# there (`ext/SmoothPeriodicStatsModelsExt/fit_mle_TN.jl`, `_fit_seasonal_gamma`): one fit per
# state of the average `PeriodicNLL` over that state's observations, from zero.
function swg_seasonal_gamma(y, n2t, z, K, T, d)
    fam = seasonal_family(Gamma, TrigBasis(T, d))
    design = PeriodicDesign(fam.spec, 1:T)
    θ0 = zeros(ncoef(fam.spec))
    return map(1:K) do k
        idx = findall(==(k), z)
        nll = PeriodicNLL(fam, design, y[idx], group_by_time(n2t[idx], T))
        Nk = length(idx)
        ℓ(θ, _x) = nll(θ) / Nk
        fit_loss_optim(ℓ, y[idx], θ0; solver=SOLVER).u
    end
end

@testset "SmoothDistribution: the fit is SWG's seasonal Gamma" begin
    T, K, N, d = 50, 2, 4000, 1
    rng = StableRNG(34)
    n2t = n_to_t(N, T)
    z = ones(Int, N)
    for n = 2:N
        z[n] = rand(rng) < (z[n-1] == 1 ? 0.9 : 0.8) ? z[n-1] : 3 - z[n-1]
    end
    truth = SmoothDistribution(Gamma, T; K, season=d, init=[(α=3.0, θ=1.0), (α=1.5, θ=4.0)])
    parameters(truth).θ[2, :] .= [0.3, -0.2]
    y = rand(rng, truth, n2t; z)

    m0 = SmoothDistribution(Gamma, T; K, season=d)
    m = fit_mle(m0, y, OneHot(z, K); n2t, solver=SOLVER)
    @test m isa SmoothDistribution
    ref = swg_seasonal_gamma(y, n2t, z, K, T, d)
    for k = 1:K
        @test column(m.family.spec, parameters(m), k) == ref[k]
    end
    # one state, no weights
    m1 = fit_mle(SmoothDistribution(Gamma, T; season=d), y; n2t, solver=SOLVER)
    @test column(m1.family.spec, parameters(m1), 1) == only(swg_seasonal_gamma(y, n2t, ones(Int, N), 1, T, d))
    @test maximum(abs, curve(m, :α) - curve(truth, :α)) < 0.5

    # dense weights, and weights scaled by two, give the same fit to the last bit
    W = Float64.(OneHot(z, K))
    @test parameters(fit_mle(m0, y, W; n2t, solver=SOLVER)) == parameters(m)
    @test parameters(fit_mle(m0, y, 2.0 .* W; n2t, solver=SOLVER)) == parameters(m)
    # a weight of 2 counts an observation twice
    w2 = copy(W)
    w2[1, z[1]] = 2.0
    dup = fit_mle(m0, [y[1]; y], Float64.(OneHot([z[1]; z], K)); n2t=[n2t[1]; n2t], solver=SOLVER)
    @test collect(parameters(fit_mle(m0, y, w2; n2t, solver=SOLVER))) ≈ collect(parameters(dup)) rtol = 1e-6
    # an observation of zero weight is left out, even outside the support
    @test parameters(fit_mle(m0, [y; -1.0], [W; 0.0 0.0]; n2t=[n2t; 1], solver=SOLVER)) == parameters(m)
    # a state of zero weight keeps its start
    m3 = SmoothDistribution(Gamma, T; K=3, season=d, init=(α=2.0, θ=2.0))
    f3 = @test_logs (:warn, r"no weight") fit_mle(m3, y, OneHot(z, 3); n2t, solver=SOLVER)
    @test column(f3.family.spec, parameters(f3), 3) == column(m3.family.spec, parameters(m3), 3)

    # an additional loss of zero changes nothing; a stiff one pins the coefficients, being added to
    # the summed negative loglikelihood
    @test parameters(fit_mle(m0, y, W; n2t, solver=SOLVER, additional_loss=(phi, θ, p) -> 0.0)) == parameters(m)
    c = [0.5, 0.0, 0.0, 0.2, 0.0, 0.0]
    mp = fit_mle(m0, y, W; n2t, solver=SOLVER, additional_loss=(phi, θ, p) -> 1e7 * sum(abs2, θ .- c))
    @test column(mp.family.spec, parameters(mp), 1) ≈ c atol = 1e-2

    # without derivatives, for a solver that needs none or a family that cannot take dual numbers
    c0 = SmoothDistribution(Gamma, T; K, season=0)
    lb = fit_mle(c0, y, W; n2t, solver=SOLVER)
    for adtype in (SciMLBase.NoAD(), NoAutoDiff())
        nm = fit_mle(c0, y, W; n2t, solver=NelderMead(), adtype)
        @test curve(nm, :α) ≈ curve(lb, :α) rtol = 1e-3
    end
    no_duals(α, θ) = α isa Float64 && θ isa Float64 ? Gamma(α, θ) : error("no dual numbers")
    plain = TimeVaryingFamily(no_duals, ParamSpec(α=ConstantBasis() => LogLink(), θ=ConstantBasis() => LogLink()))
    pm = fit_mle(SmoothDistribution(plain, T; K), y, W; n2t, solver=NelderMead(), adtype=NoAutoDiff())
    @test curve(pm, :α) ≈ curve(lb, :α) rtol = 1e-3
    @test_throws ErrorException fit_mle(SmoothDistribution(plain, T; K), y, W; n2t, solver=SOLVER)

    f = fit(SeasonalFit, m0, y, OneHot(z, K); n2t, solver=SOLVER)
    @test f.model isa SmoothDistribution && parameters(f.model) == parameters(m)
    @test nobs(f) == N && dof(f) == nparams(m0) == 2 * 6
    @test loglikelihood(f) ≈ loglikelihood(m, y, OneHot(z, K); n2t)
    @test converged(f)
    @test nobs(fit(SeasonalFit, m0, [y; -1.0], [W; 0.0 0.0]; n2t=[n2t; 1], solver=SOLVER)) == N

    # wrong weights, each with its reason
    err = try
        fit_mle(m0, y, z; n2t, solver=SOLVER)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("OneHot", sprint(showerror, err))
    @test_throws ArgumentError fit_mle(m0, y, fill(NaN, N, K); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, fill(Inf, N, K); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, -W; n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, zeros(N, K); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, W[1:end-1, :]; n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, OneHot(z, K + 1); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y; n2t, solver=SOLVER)              # K > 1 needs weights
    @test_throws ArgumentError fit_mle(m0, y, ones(N); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, [NaN; y[2:end]], W; n2t, solver=SOLVER)
    @test_throws UndefKeywordError fit_mle(m0, y, W; solver=SOLVER)            # `n2t` is required
end

@testset "SmoothARPeriodicHMM(K, D, T; memory, season)" begin
    m = SmoothARPeriodicHMM(3, 4, 10; memory=1)
    @test size(m) == (3, 4, 10, 2)
    @test curve(m, :A) ≈ fill(1 / 3, 3, 3, 10)
    @test all(==(0.5), curve(m, :p))
    @test HMMs.initialization(m) ≈ fill(1 / 3, 3)
    @test degrees(m) == (1, 1)
    @test size(SmoothARPeriodicHMM(2, 4, 10; memory=0)) == (2, 4, 10, 1)
    @test degrees(SmoothARPeriodicHMM(2, 4, 10; memory=1, season=(A=0, B=2))) == (0, 2)
    mb = SmoothARPeriodicHMM(2, 4, 10; memory=1, season=(A=1, B=CyclicBSplineBasis(10, 5)))
    @test size(parameters(mb).B, 4) == 5 && mb.basis_B == CyclicBSplineBasis(10, 5)
    @test curve(SmoothARPeriodicHMM(1, 2, 10; memory=1), :A) == ones(1, 1, 10)
    @test_throws ArgumentError SmoothARPeriodicHMM(0, 4, 10; memory=1)
    @test_throws ArgumentError SmoothARPeriodicHMM(2, 0, 10; memory=1)
    @test_throws ArgumentError SmoothARPeriodicHMM(2, 4, 0; memory=1)
    @test_throws ArgumentError SmoothARPeriodicHMM(2, 4, 10; memory=-1)
    @test_throws UndefKeywordError SmoothARPeriodicHMM(2, 4, 10)
    @test_throws ArgumentError SmoothARPeriodicHMM(2, 4, 10; memory=1, season=(A=1,))
    @test_throws ArgumentError SmoothARPeriodicHMM(2, 4, 10; memory=1, season="x")
end

@testset "initialize and SliceEM" begin
    K, D, T, order, deg, N = 2, 4, 10, 1, 1, 1000
    rng = StableRNG(35)
    truth = SmoothARPeriodicHMM(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1),
        randn(rng, K, D, 2^order, 2deg + 1), T)
    n2t = n_to_t(N, T)
    Y_past = rand(rng, Bool, D, order)
    Y = rand(StableRNG(36), truth, n2t; past=(; Y=Y_past)).Y
    m0 = SmoothARPeriodicHMM(K, D, T; memory=order, season=deg)
    method = SliceEM([-1, 0, 1], 2, 0.8)
    mi = initialize(MersenneTwister(5), m0, Y; n2t, past=(; Y=Y_past), method)

    # the chain of the tutorials, by hand, from the same generator
    rng5 = MersenneTwister(5)
    hr = randARPeriodicHMM(rng5, K, T, D, order; ref_station=1)
    hs = fit_mle_all_slices(rng5, hr, Y, Y_past; n2t, Yₜ_extanted=[-1, 0, 1], n_random_ini=2,
        Dirichlet_α=0.8, robust=true)
    θA, θB = fit_θ!(hs, deg)
    legacy = SmoothARPeriodicHMM(hs.a, θA, θB, T)
    @test parameters(mi) == parameters(legacy)
    @test HMMs.initialization(mi) == HMMs.initialization(legacy)
    @test mi.basis_A == m0.basis_A && mi.basis_B == m0.basis_B

    # the bases of `m0` are kept, a spline included
    m0b = SmoothARPeriodicHMM(K, D, T; memory=order, season=(A=1, B=CyclicBSplineBasis(T, 4)))
    mib = initialize(MersenneTwister(5), m0b, Y; n2t, past=Y_past, method)
    @test mib.basis_B == CyclicBSplineBasis(T, 4) && size(parameters(mib).B, 4) == 4

    s = SliceEM()
    @test (s.window, s.n_random, s.dirichlet) == ([0], 10, 0.8)
    s2 = SliceEM(; window=-2:2, n_random=3)
    @test (s2.window, s2.n_random, s2.dirichlet) == (-2:2, 3, 0.8)
    @test_throws ArgumentError SliceEM(Int[], 1, 0.8)
    @test_throws ArgumentError SliceEM([0], 0, 0.8)
    @test_throws ArgumentError SliceEM([0], 1, 0.0)
    @test_throws ArgumentError SliceEM([0], 1, NaN)
    @test_throws ArgumentError initialize(m0, Y; n2t)                              # memory 1, no past
    @test_throws ArgumentError initialize(m0, Y; n2t, past=falses(D, 0))
    @test_throws ArgumentError initialize(m0, Y; n2t, past=(; Y=Y_past, z=1))
    @test_throws ArgumentError initialize(m0, Y; n2t, past=Y_past, ref_station=D + 1)
    @test_throws ArgumentError initialize(m0, Y[1:end-1, :]; n2t, past=Y_past)

    # the model with any emission: `fit_mle_all_slices` with the method's window
    mh = SmoothPeriodicHMM(Normal, 2, 1, T; init=[(μ=0.0, σ=1.0), (μ=3.0, σ=1.0)])
    yh = rand(StableRNG(37), mh, n2t; past=(; z=1)).Y
    kw = (; n2t, solver=SOLVER)
    @test parameters(initialize(MersenneTwister(6), mh, yh; method=SliceEM([0], 2, 0.8), kw...)) ==
          parameters(fit_mle_all_slices(MersenneTwister(6), mh, yh; Yₜ_extanted=[0], n_random_ini=2, kw...))
    @test_throws ArgumentError initialize(mh, yh; past=(; z=1), kw...)
end

@testset "fit_mle(hmm, Y, W): the fit given the states" begin
    K, D, T, order, deg, N = 2, 4, 12, 1, 1, 2000
    rng = StableRNG(38)
    truth = SmoothARPeriodicHMM(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1),
        randn(rng, K, D, 2^order, 2deg + 1), T)
    n2t = n_to_t(N, T)
    Y_past = rand(rng, Bool, D, order)
    (; z, Y) = rand(StableRNG(39), truth, n2t; past=(; Y=Y_past))

    m0 = SmoothARPeriodicHMM(K, D, T; memory=order, season=deg)
    mw = fit_mle(m0, Y, OneHot(z, K); n2t, past=(; Y=Y_past), solver=SOLVER)
    @test mw isa SmoothARPeriodicHMM
    B_RO, θ_RO = fit_mle_RO(Y, Y_past, z, n2t, deg, K, T; solver=SOLVER)
    @test parameters(mw).B == θ_RO
    @test succprob.(ARPeriodicHMM(mw).B) == succprob.(B_RO)
    @test curve(mw, :A) == curve(m0, :A)                       # the transitions are fixed
    @test parameters(fit_mle(m0, Y, Float64.(OneHot(z, K)); n2t, past=Y_past, solver=SOLVER)) == parameters(mw)
    # from another start, the same optimum
    @test parameters(fit_mle(truth, Y, OneHot(z, K); n2t, past=Y_past, solver=SOLVER)).B ≈ θ_RO rtol = 1e-4

    # the transitions, from a constant basis, are the frequencies of the steps of `z`
    mA0 = SmoothARPeriodicHMM(K, D, T; memory=order, season=(A=0, B=1))
    mA = fit_mle(mA0, Y, OneHot(z, K); n2t, past=Y_past, fixed=(), solver=SOLVER)
    counts = [count(n -> z[n] == k && z[n+1] == l, 1:N-1) for k = 1:K, l = 1:K]
    @test curve(mA, :A)[:, :, 1] ≈ counts ./ sum(counts; dims=2) atol = 1e-5
    @test HMMs.initialization(mA) == [k == z[1] for k = 1:K]
    @test parameters(mA).B == θ_RO
    @test parameters(fit_mle(m0, Y, OneHot(z, K); n2t, past=Y_past, fixed=(:A, :B), solver=SOLVER)) == parameters(m0)
    @test parameters(fit_mle(m0, Y, OneHot(z, K); n2t, past=Y_past, fixed=:A, solver=SOLVER)) == parameters(mw)

    @test_throws ArgumentError fit_mle(m0, Y, Float64.(OneHot(z, K)); n2t, past=Y_past, fixed=(), solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, Y, OneHot(z, K); n2t, past=Y_past, fixed=(:C,), solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, Y, ones(N - 1, K); n2t, past=Y_past, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, Y, OneHot(z, K + 1); n2t, past=Y_past, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, Y, OneHot(z, K); n2t, solver=SOLVER)            # memory 1, no past
    @test_throws ArgumentError fit_mle(m0, Y, OneHot(z, K); n2t, past=(; Y=Y_past, z=1), solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, Y, fill(NaN, N, K); n2t, past=Y_past, solver=SOLVER)
    # a `Bool` matrix is still the `Y_past` of Baum-Welch; a `OneHot` is not
    @test fit_mle(m0, Y, Y_past; n2t, solver=SOLVER, maxiter=1) isa Tuple
    @test_throws ArgumentError fit(SeasonalFit, m0, Y, OneHot(z, K); n2t, solver=SOLVER)
end

@testset "curve and coef on every model" begin
    K, D, T, deg = 2, 3, 8, 1
    rng = StableRNG(40)
    m = SmoothARPeriodicHMM(fill(1 / K, K), randn(rng, K, K - 1, 2deg + 1), randn(rng, K, D, 2, 2deg + 1), T)
    hmm = ARPeriodicHMM(m)
    @test curve(m, :A) == hmm.A
    @test curve(m, :p) == permutedims(succprob.(hmm.B), (1, 3, 4, 2))
    @test curve(m, :A, 2:3) == hmm.A[:, :, 2:3]
    @test curve(m, :A; state=2) == hmm.A[2, :, :]
    @test curve(m, :p, [5]; state=1) == curve(m, :p)[1, :, :, 5:5]
    @test_throws ArgumentError curve(m, :nosuch)
    @test_throws ArgumentError curve(m, :A; state=3)

    θᴮs = [[randn(rng, 2deg + 1) for h = 1:2^o] for k = 1:K, o in [0, 1, 2]]
    ms = SmoothsARPeriodicHMM(fill(1 / K, K), randn(rng, K, K - 1, 2deg + 1), θᴮs, T)
    sh = sARPeriodicHMM(ms)
    @test curve(ms, :A) == sh.A
    @test size(curve(ms, :p)) == (K, D) && size(curve(ms, :p)[2, 3]) == (4, T)
    @test curve(ms, :p)[2, 3][4, 5] == succprob(sh.B[2, 5, 3][4])
    @test curve(ms, :p; state=2) == curve(ms, :p)[2, :]

    h = [0.0 1.0 2.0; 1.0 0.0 1.5; 2.0 1.5 0.0]
    msp = SmoothARPeriodicHMMSpatial(fill(1 / K, K), randn(rng, K, K - 1, 2deg + 1),
        randn(rng, K, D, 2, 2deg + 1), randn(rng, K, 2deg + 1), T, h)
    hsp = ARPeriodicHMMSpatial(msp)
    @test curve(msp, :A) == hsp.A
    @test curve(msp, :p) == permutedims(hsp.B, (1, 3, 4, 2))
    @test curve(msp, :R) == hsp.R
    @test curve(msp, :R, 2:3; state=1) == hsp.R[1, 2:3]
    @test_throws ArgumentError curve(msp, :nosuch)

    mh = SmoothPeriodicHMM(Normal, 2, 1, T; init=[(μ=0.0, σ=1.0), (μ=3.0, σ=1.0)])
    models = (SmoothAR1(1, T), SmoothSeasonalMixture(Exponential, 2, 1, T; init=[1.0, 5.0]),
        SmoothDistribution(Gamma, T; K=2), m, ms, msp, mh)
    for x in models
        @test coef(x) === parameters(x)
    end
    @test coef(m, :A) === parameters(m).A
    # a write through `coef` reaches the curves, the evaluated model and the likelihood
    n2t = n_to_t(40, T)
    Y_past = rand(rng, Bool, D, 1)
    Y = rand(StableRNG(41), m, n2t; past=(; Y=Y_past)).Y
    ll = loglikelihood(m, Y; n2t, past=Y_past)
    coef(m).B[1, 1, 1, 1] += 1.0
    @test curve(m, :p)[1, 1, 1, 1] != hmm.B[1, 1, 1, 1].p
    @test ARPeriodicHMM(m).B[1, 1, 1, 1] != hmm.B[1, 1, 1, 1]
    @test loglikelihood(m, Y; n2t, past=Y_past) != ll
end

@testset "Spatial HMM: a seeded rng" begin
    K, D, T, order, deg, N = 2, 4, 5, 1, 1, 60
    rng = StableRNG(42)
    xy = rand(rng, D, 2) .* 100
    h = [hypot(xy[i, 1] - xy[j, 1], xy[i, 2] - xy[j, 2]) for i = 1:D, j = 1:D]
    θR = randn(rng, K, 2deg + 1)
    θR[:, 1] .= log.(50.0 .* (1:K))
    m = SmoothARPeriodicHMMSpatial(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1),
        0.8randn(rng, K, D, 2^order, 2deg + 1), θR, T, h; QMC_m=5)
    n2t = n_to_t(N, T)
    Y_past = rand(rng, Bool, D, order)
    (; z, Y) = rand(StableRNG(43), m, n2t; past=(; Y=Y_past))

    same(f) = f(Xoshiro(1)) == f(Xoshiro(1)) && f(Xoshiro(1)) != f(Xoshiro(2))
    @test same(r -> loglikelihoods(m, Y, Y_past; n2t, rng=r))
    @test same(r -> loglikelihoods(m, Y; n2t, past=Y_past, rng=r))
    @test loglikelihoods(m, Y; n2t, past=Y_past, rng=Xoshiro(1)) == loglikelihoods(m, Y, Y_past; n2t, rng=Xoshiro(1))
    @test same(r -> loglikelihood(m, Y, Y_past; n2t, rng=r))
    @test same(r -> loglikelihood(m, Y; n2t, past=Y_past, rng=r))
    @test same(r -> complete_loglikelihood(m, Y, Y_past, z; n2t, rng=r))
    @test viterbi(m, Y, Y_past; n2t, rng=Xoshiro(1)) == viterbi(m, Y; n2t, past=Y_past, rng=Xoshiro(1))
    kw = (; n2t, solver=SOLVER, maxiter=2, tol=0.0)
    f1, h1 = fit_mle(m, Y, Y_past; rng=Xoshiro(1), kw...)
    f2, h2 = fit_mle(m, Y; past=Y_past, rng=Xoshiro(1), kw...)
    @test parameters(f1) == parameters(f2) && h1.logtots == h2.logtots
    @test_throws ArgumentError fit_mle(m, Y, OneHot(z, K); kw...)

    # the same numbers whatever the number of threads
    code = """
    using SmoothPeriodicStatsModels, StableRNGs, Random
    rng = StableRNG(42)
    K, D, T, N = 2, 4, 5, 60
    xy = rand(rng, D, 2) .* 100
    h = [hypot(xy[i, 1] - xy[j, 1], xy[i, 2] - xy[j, 2]) for i = 1:D, j = 1:D]
    θR = randn(rng, K, 3)
    θR[:, 1] .= log.(50.0 .* (1:K))
    m = SmoothARPeriodicHMMSpatial(fill(1 / K, K), 0.5randn(rng, K, K - 1, 3), 0.8randn(rng, K, D, 2, 3), θR, T, h; QMC_m = 5)
    n2t = n_to_t(N, T)
    Y_past = rand(rng, Bool, D, 1)
    (; z, Y) = rand(StableRNG(43), m, n2t; past = (; Y = Y_past))
    print(repr((vec(loglikelihoods(m, Y, Y_past; n2t, rng = Xoshiro(1))),
        SmoothPeriodicStatsModels.complete_loglikelihood(m, Y, Y_past, z; n2t, rng = Xoshiro(1)))))
    """
    run_with(t) = read(pipeline(`$(Base.julia_cmd()) --startup-file=no -t $t --project=$(Base.active_project()) -e $code`; stderr=devnull), String)
    one, four = run_with(1), run_with(4)
    @test !isempty(one) && one == four
end

@testset "No method ambiguities in the package" begin
    mods = (SmoothPeriodicStatsModels,
        Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt),
        Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsSimpleNonlinearSolveExt))
    ours = filter(Test.detect_ambiguities(SmoothPeriodicStatsModels)) do (a, b)
        a.module in mods && b.module in mods
    end
    @test isempty(ours)
end
