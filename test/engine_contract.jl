# The engine contract of 4.2 on the models of version 4: state weights, `past`, `loglikelihoods`
# with one column per state, the keyword `rand`, and the extension split. Every new form is
# checked against the positional form it stands for, to the last bit where the arithmetic is the
# same.

@testset "OneHot" begin
    W = OneHot([1, 3, 2, 3], 3)
    @test size(W) == (4, 3)
    @test eltype(W) == Bool
    @test W[2, 3] && !W[2, 1]
    @test Matrix(W) == Bool[1 0 0; 0 0 1; 0 1 0; 0 0 1]
    @test sum(W; dims=1) == [1 1 2]
    @test size(OneHot(Int[], 2)) == (0, 2)
    @test OneHot(ones(Int, 3), 1) == trues(3, 1)
    z = [1, 2, 2]
    @test OneHot(z, 2).z === z            # not copied
    @test_throws BoundsError W[5, 1]
    @test_throws BoundsError W[1, 4]
    @test_throws ArgumentError OneHot([1, 0], 2)
    @test_throws ArgumentError OneHot([1, 3], 2)
    @test_throws ArgumentError OneHot([-1], 2)
    @test_throws ArgumentError OneHot([1, 1], 0)
    @test_throws ArgumentError OneHot([true, true], 1)
end

@testset "AR1: state weights, OneHot and the integer path agree" begin
    T, K, N = 12, 2, 3000
    μ = [0.5 .+ 0.3cospi.(2(1:T) / T)'; -0.5 .+ 0.2sinpi.(2(1:T) / T)']
    ρ = [fill(0.6, 1, T); fill(-0.2, 1, T)]
    σ = [fill(0.5, 1, T); fill(1.2, 1, T)]
    rng = StableRNG(11)
    n2t = n_to_t(N, T)
    z = rand(rng, 1:K, N)
    y = rand(rng, AR1(μ, ρ, σ), n2t, z; y₁=0.0)

    m0 = SmoothAR1(1, T; K)
    mz, hz = fit_mle(m0, y, z; n2t, solver=SOLVER)
    mW, hW = fit_mle(m0, y, OneHot(z, K); n2t, solver=SOLVER)
    @test parameters(mW) == parameters(mz)
    @test hW.objective == hz.objective
    # without `K` on the model, `OneHot` supplies it
    @test parameters(fit_mle(SmoothAR1(1, T), y, OneHot(z, K); n2t, solver=SOLVER)[1]) ==
          parameters(fit_mle(SmoothAR1(1, T), y, z; n2t, solver=SOLVER)[1])
    # dense weights of zeros and ones are the same statistics
    mF, hF = fit_mle(m0, y, Float64.(OneHot(z, K)); n2t, solver=SOLVER)
    @test parameters(mF) == parameters(mz)
    @test loglikelihood(hF) == loglikelihood(hz)
    @test loglikelihood(mz, y, OneHot(z, K); n2t) == loglikelihood(mz, y, z; n2t)

    # a state of zero weight keeps its start exactly while the others are fitted
    m3 = SmoothAR1(1, T; K=3, init=(μ=0.1, ρ=0.2, σ=1.5))
    f3, _ = fit_mle(m3, y, Float64.(OneHot(z, 3)); n2t, solver=SOLVER)
    @test column(f3.spec, parameters(f3), 3) == column(m3.spec, parameters(m3), 3)
    @test column(f3.spec, parameters(f3), 1) ≈ column(mz.spec, parameters(mz), 1) rtol = 1e-6

    # one state: vector weights of ones, `OneHot(ones, 1)` and no weights give the same fit
    m1 = SmoothAR1(1, T)
    u, hu = fit_mle(m1, y; n2t, solver=SOLVER)
    @test parameters(fit_mle(m1, y, ones(N); n2t, solver=SOLVER)[1]) == parameters(u)
    @test vec(collect(parameters(fit_mle(m1, y, OneHot(ones(Int, N), 1); n2t, solver=SOLVER)[1]))) ≈ collect(parameters(u)) rtol = 1e-10
    @test vec(collect(parameters(fit_mle(m1, y, ones(N, 1); n2t, solver=SOLVER)[1]))) ≈ collect(parameters(u)) rtol = 1e-10
    # weights of 2 count each transition twice: the same optimum
    @test collect(parameters(fit_mle(m1, y, fill(2.0, N); n2t, solver=SOLVER)[1])) ≈ collect(parameters(u)) rtol = 1e-5
    # a transition of zero weight is left out, whatever its value
    yb = copy(y)
    yb[10] = 1e6
    wb = ones(N)
    wb[10] = wb[11] = 0.0
    fb, _ = fit_mle(m1, yb, wb; n2t, solver=SOLVER)
    @test all(isfinite, parameters(fb))
    @test nobs(fit(SeasonalFit, m1, yb, wb; n2t, solver=SOLVER)) == N - 3
    @test nobs(fit(SeasonalFit, m0, y, OneHot(z, K); n2t, solver=SOLVER)) == N - 1

    # wrong weights are refused, with what to pass instead
    @test_throws ArgumentError fit_mle(m0, y, fill(NaN, N, K); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, fill(Inf, N, K); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, -Float64.(OneHot(z, K)); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, zeros(N, K); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, ones(N - 1, K); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, ones(N, K + 1); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, OneHot(z, K + 1); n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m0, y, ones(N); n2t, solver=SOLVER)          # one state only
    @test_throws ArgumentError fit_mle(m1, y, Any[1.0 for _ = 1:N]; n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m1, y, Union{Missing,Float64}[1.0 for _ = 1:N]; n2t, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m1, y, ones(N); n2t=n2t[1:end-1], solver=SOLVER)
    err = try
        fit_mle(m0, y, ones(N); n2t, solver=SOLVER)
    catch e
        e
    end
    @test occursin("OneHot", sprint(showerror, err))
end

@testset "AR1: past" begin
    T, N = 10, 2000
    rng = StableRNG(12)
    n2t = n_to_t(N, T)
    y = rand(rng, AR1(fill(0.3, T), fill(0.5, T), fill(0.8, T)), n2t; y₁=0.0)
    m1 = SmoothAR1(1, T)
    full, hfull = fit_mle(m1, y; n2t, solver=SOLVER)
    tail = (y[2:end], n2t[2:end])
    for past in (y[1], (; Y=y[1]), (; Y=[0.7, y[1]]), reshape([y[1]], 1, 1), [y[1]], (; Y=y[1], z=2))
        f, h = fit_mle(m1, tail[1]; n2t=tail[2], past, solver=SOLVER)
        @test parameters(f) == parameters(full)
        @test loglikelihood(h) == loglikelihood(hfull)
    end
    @test loglikelihood(full, tail[1]; n2t=tail[2], past=y[1]) == loglikelihood(full, y; n2t)
    @test nobs(fit(SeasonalFit, m1, tail[1]; n2t=tail[2], past=y[1], solver=SOLVER)) == N - 1
    @test nobs(fit(SeasonalFit, m1, y; n2t, solver=SOLVER)) == N - 1

    # given the states, the transition into the first observation follows its own state
    z = rand(StableRNG(13), 1:2, N)
    mK = SmoothAR1(1, T; K=2)
    @test parameters(fit_mle(mK, y[2:end], z[2:end]; n2t=n2t[2:end], past=y[1], solver=SOLVER)[1]) ==
          parameters(fit_mle(mK, y, z; n2t, solver=SOLVER)[1])
    @test loglikelihood(fit_mle(mK, y, z; n2t, solver=SOLVER)[1], y[2:end], z[2:end]; n2t=n2t[2:end], past=y[1]) ≈
          loglikelihood(fit_mle(mK, y, z; n2t, solver=SOLVER)[1], y, z; n2t)

    @test_throws ArgumentError fit_mle(m1, y; n2t, past=Float64[], solver=SOLVER)   # shorter than the memory
    @test_throws ArgumentError fit_mle(m1, y; n2t, past=NaN, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m1, y; n2t, past=missing, solver=SOLVER)
    @test_throws ArgumentError fit_mle(m1, y; n2t, past=[1.0, missing], solver=SOLVER)
    @test_throws ArgumentError fit_mle(m1, y; n2t, past=(; X=1.0), solver=SOLVER)
    @test_throws ArgumentError fit_mle(m1, y; n2t, past="a", solver=SOLVER)
    @test_throws ArgumentError fit_mle(m1, y; n2t, past=[1.0 2.0; 3.0 4.0], solver=SOLVER)
end

@testset "AR1: loglikelihoods" begin
    T, K, N = 8, 2, 400
    rng = StableRNG(14)
    n2t = n_to_t(N, T)
    z = rand(rng, 1:K, N)
    mK = SmoothAR1(1, T; K, init=[(μ=0.2, ρ=0.5, σ=0.7), (μ=-0.3, ρ=-0.1, σ=1.3)])
    parameters(mK).μ[2, :] .= [0.1, -0.2]
    y = rand(rng, mK, n2t; z, y₁=0.0)
    LL = loglikelihoods(mK, y; n2t)
    @test size(LL) == (N, K)
    @test all(iszero, LL[1, :])
    @test sum(LL[n, z[n]] for n = 1:N) ≈ loglikelihood(mK, y, z; n2t)
    AR = AR1(mK)
    n, k = 7, 2
    t = n2t[n]
    @test LL[n, k] ≈ logpdf(Normal(AR.μ[k, t] + AR.ρ[k, t] * y[n-1], AR.σ[k, t]), y[n])
    @test loglikelihoods(AR, y; n2t) == LL
    # with `past`, the first row is the transition from it
    LLp = loglikelihoods(mK, y[2:end]; n2t=n2t[2:end], past=y[1])
    @test LLp == LL[2:end, :]
    @test LLp[1, 1] ≈ logpdf(Normal(AR.μ[1, n2t[2]] + AR.ρ[1, n2t[2]] * y[1], AR.σ[1, n2t[2]]), y[2])

    m1 = SmoothAR1(1, T; init=(μ=0.2, ρ=0.5, σ=0.7))
    L1 = loglikelihoods(m1, y; n2t)
    @test size(L1) == (N, 1)
    @test sum(L1) ≈ loglikelihood(m1, y; n2t)
    @test size(loglikelihoods(m1, Float64[]; n2t=Int[])) == (0, 1)
    @test_throws ArgumentError loglikelihoods(m1, y; n2t=n2t[1:end-1])
    @test_throws ArgumentError loglikelihoods(m1, y; n2t=n2t .+ T)
end

@testset "AR1: rand with keywords" begin
    T, K, N = 8, 2, 50
    n2t = n_to_t(N, T)
    z = rand(StableRNG(15), 1:K, N)
    mK = SmoothAR1(1, T; K, init=[(μ=0.2, ρ=0.5, σ=0.7), (μ=-0.3, ρ=-0.1, σ=1.3)])
    m1 = SmoothAR1(1, T; init=(μ=0.2, ρ=0.5, σ=0.7))
    AR = AR1(m1)

    # the positional forms draw what they always drew: `y₁` first, then one innovation per step
    rng = StableRNG(16)
    y = rand(copy(rng), m1, n2t)
    y1 = rand(rng, Normal(0, AR.σ[1]))
    @test y[1] == y1
    @test y[2] == AR.μ[n2t[2]] + AR.ρ[n2t[2]] * y1 + rand(rng, Normal(0, AR.σ[n2t[2]]))
    @test rand(StableRNG(16), m1, N) == y
    # the keyword `z` is the positional one
    @test rand(StableRNG(17), mK, n2t; z) == rand(StableRNG(17), mK, n2t, z)
    @test rand(StableRNG(17), mK, N; z) == rand(StableRNG(17), mK, n2t, z)
    @test rand(StableRNG(17), AR1(mK), N; z) == rand(StableRNG(17), mK, n2t, z)
    # `past` makes the first value a step of the process, with the first innovation
    ARK = AR1(mK)
    yp = rand(StableRNG(18), mK, n2t; z, past=0.3)
    ε = randn(StableRNG(18), N)
    @test yp[1] ≈ ARK.μ[z[1], n2t[1]] + ARK.ρ[z[1], n2t[1]] * 0.3 + ARK.σ[z[1], n2t[1]] * ε[1]
    @test yp[2] ≈ ARK.μ[z[2], n2t[2]] + ARK.ρ[z[2], n2t[2]] * yp[1] + ARK.σ[z[2], n2t[2]] * ε[2]
    @test rand(StableRNG(18), mK, n2t; z, past=(; Y=[9.0, 0.3])) == yp
    ys = rand(StableRNG(19), m1, n2t; past=0.3)
    @test ys[1] ≈ AR.μ[n2t[1]] + AR.ρ[n2t[1]] * 0.3 + rand(StableRNG(19), Normal(0, AR.σ[n2t[1]]))
    # one process takes a path of ones only
    @test rand(StableRNG(20), m1, n2t; z=ones(Int, N)) == rand(StableRNG(20), m1, n2t)
    @test length(rand(m1, Int[]; past=0.1)) == 0

    @test_throws ArgumentError rand(m1, n2t; y₁=0.0, past=0.3)
    @test_throws ArgumentError rand(mK, n2t)                       # K processes need `z`
    @test_throws ArgumentError rand(mK, n2t; z=z[1:end-1])
    @test_throws ArgumentError rand(mK, n2t; z=[z[1:end-1]; K + 1])
    @test_throws ArgumentError rand(m1, n2t; z=fill(2, N))
    @test_throws ArgumentError rand(mK, n2t; z, past=NaN)
end

@testset "Mixture: weights as one column of state weights" begin
    Tm, Nm = 10, 400
    rng = StableRNG(21)
    n2t = repeat(1:Tm, Nm ÷ Tm)
    y = [rand(rng, Exponential(t <= 5 ? 1.0 : 4.0)) for t in n2t]
    mix0 = SmoothSeasonalMixture(Exponential, 2, 1, Tm; init=[1.0, 5.0])
    kw = (; n2t, solver=SOLVER, maxiter=3, tol=0.0)
    ref = parameters(first(fit_mle(mix0, y, ones(Nm); kw...)))
    @test parameters(first(fit_mle(mix0, y, OneHot(ones(Int, Nm), 1); kw...))) == ref
    @test parameters(first(fit_mle(mix0, y, ones(Nm, 1); kw...))) == ref
    @test parameters(first(fit_mle(mix0, y; past=1.0, kw...))) == parameters(first(fit_mle(mix0, y; kw...)))
    err = try
        fit_mle(mix0, y, ones(Nm, 2); kw...)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("W[:, k]", sprint(showerror, err))
    @test_throws ArgumentError fit_mle(mix0, y, OneHot(rand(rng, 1:2, Nm), 2); kw...)
    @test_throws ArgumentError fit_mle(mix0, y, fill(NaN, Nm, 1); kw...)
    @test_throws ArgumentError fit_mle(mix0, y, -ones(Nm, 1); kw...)
    @test_throws ArgumentError fit_mle(mix0, y, zeros(Nm, 1); kw...)
    @test_throws ArgumentError fit_mle(mix0, y, ones(Nm - 1, 1); kw...)

    LL = loglikelihoods(mix0, y; n2t)
    @test size(LL) == (Nm, 1)
    @test sum(LL) ≈ loglikelihood(mix0, y; n2t)
    @test loglikelihoods(mix0, y; n2t, past=0.5) == LL
    @test loglikelihood(mix0, y; n2t, past=0.5) == loglikelihood(mix0, y; n2t)
    @test rand(StableRNG(22), mix0, n2t; z=ones(Int, Nm)) == rand(StableRNG(22), mix0, n2t)
    @test_throws ArgumentError rand(mix0, n2t; z=fill(2, Nm))
end

@testset "HMMs: past, seq_ends and the keyword forms" begin
    K, D, T, order, deg, N = 2, 3, 10, 1, 1, 300
    rng = StableRNG(23)
    m = SmoothARPeriodicHMM(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1),
        0.8randn(rng, K, D, 2^order, 2deg + 1), T)
    hmm = ARPeriodicHMM(m)
    n2t = n_to_t(N, T)
    Y_past = rand(rng, Bool, D, order)
    z, Y = rand(StableRNG(24), hmm, n2t; Y_past, z_ini=1, seq=true)

    # the keyword forms are the positional ones
    @test loglikelihoods(hmm, Y; n2t, past=(; Y=Y_past)) == loglikelihoods(hmm, Y, Y_past; n2t)
    @test loglikelihoods(m, Y; n2t, past=Y_past) == loglikelihoods(m, Y, Y_past; n2t)
    @test loglikelihoods(m, Y; n2t, past=(; Y=Y_past), robust=true) == loglikelihoods(m, Y, Y_past; n2t, robust=true)
    @test viterbi(m, Y; n2t, past=(; Y=Y_past)) == viterbi(m, Y, Y_past; n2t)
    @test loglikelihood(m, Y; n2t, past=(; Y=Y_past)) == loglikelihood(m, Y, Y_past; n2t)
    @test loglikelihood(hmm, Y; n2t, past=(; Y=Y_past)) == loglikelihood(hmm, Y, Y_past; n2t)
    # more columns than the memory: the last ones are used
    long = [rand(rng, Bool, D, 3) Y_past]
    @test loglikelihood(m, Y; n2t, past=(; Y=long)) == loglikelihood(m, Y, Y_past; n2t)
    # `past.z` is read by viterbi, and accepted and unused by the emission loglikelihoods
    LL = loglikelihoods(hmm, Y, Y_past; n2t)
    @test loglikelihoods(hmm, Y; n2t, past=(; Y=Y_past, z=2)) == LL
    @test viterbi(m, Y; n2t, past=(; Y=Y_past, z=2)) == viterbi(hmm.A[2, :, mod1(n2t[1] - 1, T)], hmm.A, LL; n2t)

    # several sequences, end to end or as a vector, agree with the positional form
    Y1, Y2 = Y[:, 1:120], Y[:, 121:end]
    P2 = Y[:, 120:120]
    n2ts = [n2t[1:120], n2t[121:end]]
    @test loglikelihood(m, Y; n2t, seq_ends=[120, N], past=[Y_past, P2]) ≈
          loglikelihood(m, [Y1, Y2], [Y_past, P2]; n2t=n2ts)
    @test loglikelihood(m, [Y1, Y2]; n2t=n2ts, past=[Y_past, (; Y=P2)]) ≈
          loglikelihood(m, [Y1, Y2], [Y_past, P2]; n2t=n2ts)
    @test viterbi(m, Y; n2t, seq_ends=[120, N], past=[Y_past, P2]) ==
          vcat(viterbi(m, Y1, Y_past; n2t=n2ts[1]), viterbi(m, Y2, P2; n2t=n2ts[2]))
    @test viterbi(m, [Y1, Y2]; n2t=n2ts, past=[Y_past, P2]) == viterbi(m, [Y1, Y2], [Y_past, P2]; n2t=n2ts)
    # one sequence ending where the other starts is the whole series
    @test loglikelihoods(m, Y; n2t, seq_ends=[120, N], past=[Y_past, P2]) == LL

    kw = (; solver=SOLVER, maxiter=2, tol=0.0)
    mf, hf = fit_mle(m, Y; n2t, past=(; Y=Y_past), kw...)
    mp, hp = fit_mle(m, Y, Y_past; n2t, kw...)
    @test parameters(mf) == parameters(mp)
    @test hf.logtots == hp.logtots
    @test parameters(first(fit_mle(m, Y; n2t, seq_ends=[120, N], past=[Y_past, P2], kw...))) ==
          parameters(first(fit_mle(m, [Y1, Y2], [Y_past, P2]; n2t=n2ts, kw...)))
    @test parameters(first(fit_mle(m, [Y1, Y2]; n2t=n2ts, past=[Y_past, P2], kw...))) ==
          parameters(first(fit_mle(m, [Y1, Y2], [Y_past, P2]; n2t=n2ts, kw...)))
    f = fit(SeasonalFit, m, Y; n2t, past=Y_past, kw...)
    @test nobs(f) == N && coef(f) == parameters(mf)

    # wrong pasts are refused, each with its reason
    @test_throws ArgumentError loglikelihood(m, Y; n2t)                          # memory 1, no past
    @test_throws ArgumentError loglikelihood(m, Y; n2t, past=falses(D, 0))       # shorter than the memory
    @test_throws ArgumentError loglikelihood(m, Y; n2t, past=falses(D + 1, 1))   # wrong rows
    @test_throws ArgumentError loglikelihood(m, Y; n2t, past=zeros(D, 1))        # not Bool
    @test_throws ArgumentError loglikelihood(m, Y; n2t, past=(; Y=Y_past, x=1))   # unknown field
    @test_throws ArgumentError loglikelihood(m, Y; n2t, past=1.0)
    @test_throws ArgumentError viterbi(m, Y; n2t, past=(; Y=Y_past, z=K + 1))
    @test_throws ArgumentError viterbi(m, Y; n2t, past=(; Y=Y_past, z=0))
    @test_throws ArgumentError loglikelihood(m, Y; n2t, past=(; Y=Y_past, z=1))  # the fits refuse `z`
    @test_throws ArgumentError fit_mle(m, Y; n2t, past=(; Y=Y_past, z=1), kw...)
    @test_throws ArgumentError viterbi(m, Y; n2t, seq_ends=[120, N - 1], past=[Y_past, P2])
    @test_throws ArgumentError viterbi(m, Y; n2t, seq_ends=[120, 120, N], past=[Y_past, P2, P2])
    @test_throws ArgumentError viterbi(m, Y; n2t, seq_ends=Int[], past=[])
    @test_throws ArgumentError viterbi(m, Y; n2t, seq_ends=[0, N], past=[Y_past, P2])
    @test_throws ArgumentError viterbi(m, Y; n2t, seq_ends=[120, N], past=[Y_past])
    @test_throws ArgumentError viterbi(m, Y; n2t, seq_ends=[120, N], past=Y_past)

    # site-dependent memory: a D × m past is read per site
    orders = [0, 1, 2]
    θᴮs = [[0.5randn(rng, 2deg + 1) for h = 1:2^orders[j]] for k = 1:K, j = 1:D]
    ms = SmoothsARPeriodicHMM(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1), θᴮs, T)
    Pm = rand(rng, Bool, D, 2)
    per_site = [Pm[j, (end-orders[j]+1):end] for j = 1:D]
    @test viterbi(ms, Y; n2t, past=(; Y=Pm)) == viterbi(ms, Y, per_site; n2t)
    @test loglikelihood(ms, Y; n2t, past=Pm) == loglikelihood(ms, Y, per_site; n2t)
    @test loglikelihood(ms, Y; n2t, past=(; Y=per_site)) == loglikelihood(ms, Y, per_site; n2t)
    @test_throws ArgumentError loglikelihood(ms, Y; n2t, past=Pm[:, 1:1])
end

@testset "HMMs: rand with keywords" begin
    K, D, T, order, deg, N = 2, 3, 10, 1, 1, 60
    rng = StableRNG(25)
    m = SmoothARPeriodicHMM(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1),
        0.8randn(rng, K, D, 2^order, 2deg + 1), T)
    hmm = ARPeriodicHMM(m)
    n2t = n_to_t(N, T)
    Y_past = rand(rng, Bool, D, order)

    r = rand(StableRNG(26), hmm, n2t; past=(; Y=Y_past))
    zs, Ys = rand(StableRNG(26), hmm, n2t; Y_past, seq=true)
    @test r isa NamedTuple{(:z, :Y)}
    @test r.z == zs && r.Y == Ys
    (; z, Y) = rand(StableRNG(26), m, n2t; past=(; Y=Y_past))
    @test z == zs && Y == Ys
    @test rand(StableRNG(26), m, N; past=(; Y=Y_past)).Y == Ys
    # the positional form is unchanged
    @test rand(StableRNG(26), hmm, n2t; Y_past) == Ys
    @test rand(StableRNG(26), hmm, n2t; Y_past) isa Matrix{Bool}
    # `z` given: only `Y` is drawn
    r2 = rand(StableRNG(27), hmm, n2t; z=zs, past=(; Y=Y_past))
    @test r2.z === zs && r2.Y == rand(StableRNG(27), hmm, zs, n2t; Y_past)
    # without `past.Y`, the past is zeros, as the positional default
    @test rand(StableRNG(28), hmm, n2t; z=zs).Y == rand(StableRNG(28), hmm, zs, n2t)

    # `past.z` sets the state before the first: with transitions that always switch, `z[1]` is
    # the other state
    A = repeat([0.0 1.0; 1.0 0.0], 1, 1, T)
    flip = ARPeriodicHMM([0.5, 0.5], A, hmm.B)
    @test all(rand(StableRNG(s), flip, n2t; past=(; Y=Y_past, z=1)).z[1] == 2 for s = 1:20)
    @test all(rand(StableRNG(s), flip, n2t; past=(; Y=Y_past, z=2)).z[1] == 1 for s = 1:20)
    r0 = rand(StableRNG(29), hmm, Int[]; past=(; Y=Y_past))
    @test isempty(r0.z) && size(r0.Y) == (D, 0)

    @test_throws ArgumentError rand(hmm, n2t; z=zs, z_ini=1)
    @test_throws ArgumentError rand(hmm, n2t; past=(; Y=Y_past), seq=true)
    @test_throws ArgumentError rand(hmm, n2t; z=[zs[1:end-1]; K + 1])
    @test_throws ArgumentError rand(hmm, n2t; z=zs[1:end-1])
    @test_throws ArgumentError rand(hmm, n2t; past=(; z=K + 1))
    @test_throws ArgumentError rand(hmm, n2t; past=(; Y=falses(D, 0)))

    # the site-dependent and the memoryless models
    θᴮs = [[0.5randn(rng, 2deg + 1) for h = 1:2^o] for k = 1:K, o in [0, 1, 2]]
    sh = sARPeriodicHMM(SmoothsARPeriodicHMM(fill(1 / K, K), 0.5randn(rng, K, K - 1, 2deg + 1), θᴮs, T))
    Pm = rand(rng, Bool, D, 2)
    per_site = [Pm[j, (end-o+1):end] for (j, o) in enumerate([0, 1, 2])]
    rs = rand(StableRNG(30), sh, n2t; past=(; Y=Pm))
    @test (rs.z, rs.Y) == rand(StableRNG(30), sh, n2t; Y_past=per_site, seq=true)
    mh = SmoothPeriodicHMM(Normal, 2, 1, T; init=[(μ=0.0, σ=1.0), (μ=3.0, σ=1.0)])
    rh = rand(StableRNG(31), mh, n2t; past=(; z=1))
    @test rh isa NamedTuple && size(rh.Y) == (1, N)
    @test rand(StableRNG(31), mh, n2t; z=rh.z).Y == rand(StableRNG(31), mh, rh.z, n2t)
    @test viterbi(mh, rh.Y; n2t, past=(; z=2)) ==
          viterbi(mh.stored.A[2, :, mod1(n2t[1] - 1, T)], mh.stored.A, loglikelihoods(mh, rh.Y; n2t); n2t)
    @test loglikelihoods(mh, rh.Y; n2t, past=(; Y=rh.Y)) == loglikelihoods(mh, rh.Y; n2t)
    @test loglikelihood(mh, rh.Y; n2t, past=(; Y=rh.Y)) == loglikelihood(mh, rh.Y; n2t)
    @test_throws ArgumentError loglikelihood(mh, rh.Y; n2t, past=(; z=1))
    @test_throws ArgumentError fit_mle(mh, rh.Y; n2t, past=(; z=1), solver=SOLVER)
    @test_throws ArgumentError loglikelihoods(mh, rh.Y; n2t, past=(; Y=zeros(2, 3)))
end

# A fit without the solver extensions is a helpful error, and the two extensions load apart:
# Optimization without SimpleNonlinearSolve fits everything but `fit_θ*`. In a subprocess, since
# this one has loaded both.
@testset "Fits need an extension, and say which" begin
    @test !isnothing(Base.get_extension(SPSM, :SmoothPeriodicStatsModelsOptimizationExt))
    @test !isnothing(Base.get_extension(SPSM, :SmoothPeriodicStatsModelsSimpleNonlinearSolveExt))
    # once loaded, a call with wrong arguments is a `MethodError`, not a request to load it
    @test_throws MethodError fit_mle(SmoothAR1(1, 10), "x"; solver=SOLVER)
    @test_throws MethodError fit_θᴮ!(zeros(3), "x")
    code = """
    using SmoothPeriodicStatsModels, Distributions
    m = SmoothAR1(1, 10)
    y = rand(AR1(fill(0.5, 10), fill(0.3, 10), fill(1.0, 10)), 300; y₁ = 0.0)
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    @assert occursin("using Optimization", msg(() -> fit_mle(m, y; solver = nothing)))
    @assert occursin("using Optimization", msg(() -> fit(SeasonalFit, m, y)))
    @assert occursin("using SimpleNonlinearSolve", msg(() -> fit_θᴮ!(zeros(3), fill(0.5, 10))))
    using Optimization, OptimizationOptimJL
    @assert Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt) !== nothing
    @assert Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsSimpleNonlinearSolveExt) === nothing
    m1, h = fit_mle(m, y; solver = LBFGS())
    @assert converged(h)
    sd = fit_mle(SmoothDistribution(Gamma, 10), rand(Gamma(2.0, 1.0), 200); n2t = n_to_t(200, 10), solver = LBFGS())
    @assert sd isa SmoothDistribution
    @assert occursin("using SimpleNonlinearSolve", msg(() -> fit_θᴮ!(zeros(3), fill(0.5, 10))))
    print("OK")
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
    out = read(pipeline(cmd; stderr=devnull), String)
    @test endswith(out, "OK")
end
