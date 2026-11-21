# The tests of TimeVaryingParameters.jl, for its copy in `src/TimeVaryingParameters/` (whose first
# lines say where both come from). They are that package's `test/runtests.jl` at the same
# commit, except that:
# - each fit passes `solver = SOLVER`, which the package's extension made the default;
# - the Aqua testset is left out, since Aqua checks a package, not a submodule;
# - the gradient check uses the five-point difference `fd_gradient` below instead of
#   FiniteDifferences.jl.
# In a module of its own, so that its names stay apart from those of `runtests.jl`.
module TestTimeVaryingParameters

using SmoothPeriodicStatsModels.TimeVaryingParameters
using Test
using Distributions
using LinearAlgebra: I, dot
using LinkFunctions
using LogExpFunctions: logistic
using ForwardDiff
using StableRNGs
using Optimization, OptimizationOptimJL

const TVP = TimeVaryingParameters
const SOLVER = BFGS()

# five-point central differences, the `central_fdm(5, 1)` of FiniteDifferences.jl
function fd_gradient(f, x; h = 1.0e-3)
    return map(eachindex(x)) do i
        e = zero(x)
        e[i] = h
        (f(x - 2e) - 8f(x - e) + 8f(x + e) - f(x + 2e)) / 12h
    end
end

# `SmoothPeriodicStatsModels.trig_basis`, transcribed. Written out rather than depended on,
# so this package stays free of that one; the compatibility testset below is what guarantees
# coefficient vectors carry over between the two unchanged.
function spsm_trig_basis(T::Integer, d::Integer)
    B = Matrix{Float64}(undef, T, 2d + 1)
    for t in 1:T
        B[t, 1] = 1
        for l in 1:d
            B[t, 2l] = cos(2pi * l * t / T)
            B[t, 2l + 1] = sin(2pi * l * t / T)
        end
    end
    return B
end

@testset "TimeVaryingParameters.jl" begin

    @testset "Bases: features and shapes" begin
        @test nbasis(ConstantBasis()) == 1
        @test eval_basis(ConstantBasis(), 3.0) == [1.0]
        @test design_matrix(ConstantBasis(), 1:4) == ones(4, 1)

        p = PolynomialBasis(3)
        @test nbasis(p) == 3
        @test eval_basis(p, 2.0) == [2.0, 4.0, 8.0]
        # centring and scaling are applied before the powers
        pc = PolynomialBasis(2; center = 10, scale = 2)
        @test eval_basis(pc, 14.0) == [2.0, 4.0]

        f = FourierBasis(12, 2)
        @test nbasis(f) == 4
        @test eval_basis(f, 0.0) == [1.0, 0.0, 1.0, 0.0]
        @test eval_basis(f, 12.0) ≈ [1.0, 0.0, 1.0, 0.0] atol = 1e-12   # one full period

        ind = IndicatorBasis(t -> mod1(Int(t), 3), 1:3)
        @test nbasis(ind) == 3
        @test eval_basis(ind, 1.0) == [1.0, 0.0, 0.0]
        @test eval_basis(ind, 5.0) == [0.0, 1.0, 0.0]
        @test design_matrix(ind, 1:3) == Matrix(1.0I(3))

        # sums concatenate, in order
        s = ConstantBasis() + PolynomialBasis(2)
        @test nbasis(s) == 3
        @test eval_basis(s, 3.0) == [1.0, 3.0, 9.0]
        @test nbasis(ConstantBasis() + PolynomialBasis(2) + FourierBasis(10, 1)) == 5

        # `design_matrix` is time-major: row i is the basis at ts[i]
        ts = [0.5, 1.5, 2.5]
        M = design_matrix(s, ts)
        @test size(M) == (3, 3)
        @test all(M[i, :] == eval_basis(s, ts[i]) for i in 1:3)
    end

    @testset "Bases: input validation" begin
        @test_throws ArgumentError PolynomialBasis(0)
        @test_throws ArgumentError PolynomialBasis(2; scale = 0)
        # the positional form must not bypass the checks: these are inner constructors
        @test_throws ArgumentError PolynomialBasis(0, 1.0, 1.0)
        @test_throws ArgumentError FourierBasis(-1, 2)
        @test_throws ArgumentError FourierBasis(12, 0)
        @test_throws ArgumentError IndicatorBasis(identity, Int[])
        @test_throws ArgumentError IndicatorBasis(identity, [1, 1, 2])   # levels must be unique
    end

    @testset "Bases: IndicatorBasis survives being used away from the fit" begin
        # The original draft discovered levels inside `eval_basis` and never stored them, so a
        # fit succeeded and every later evaluation threw. Levels are now required up front,
        # which makes a single-point evaluation always well defined.
        b = IndicatorBasis(t -> mod1(Int(t), 4), 1:4)
        M = design_matrix(b, 1:8)
        @test M[1, :] == eval_basis(b, 1)
        @test eval_basis(b, 100) == eval_basis(b, 4)   # 100 and 4 agree mod 4
        @test sum(eval_basis(b, 7)) == 1               # exactly one level is active
    end

    @testset "Bases: element types are promoted, not forced to Float64" begin
        @test eltype(design_matrix(TrigBasis(12, 1), 1:12)) == Float64
        @test eval_basis(PolynomialBasis(2), 2.0f0) isa Vector{Float32}
        # a Dual flows through, which is what differentiating with respect to `t` needs
        d = ForwardDiff.Dual(2.0, 1.0)
        @test eval_basis(PolynomialBasis(2), d) isa Vector{<:ForwardDiff.Dual}
    end

    @testset "Bases: B-splines" begin
        # degree 1 is the hat functions, and a single interval of degree 3 the Bernstein basis
        b1 = BSplineBasis([0.0, 1.0, 2.0]; degree = 1)
        @test nbasis(b1) == 3
        @test eval_basis(b1, 0.5) == [0.5, 0.5, 0.0]
        @test eval_basis(b1, 1.5) == [0.0, 0.5, 0.5]
        x = 0.3
        @test eval_basis(BSplineBasis([0.0, 1.0]), x) ≈ [(1 - x)^3, 3x * (1 - x)^2, 3x^2 * (1 - x), x^3]

        b = BSplineBasis(0, 10, 6)                 # 6 knots, cubic
        @test nbasis(b) == 6 + 3 - 1
        ts = range(-1, 11; length = 501)
        M = design_matrix(b, ts)
        @test all(≥(0), M)
        @test all(sum(M; dims = 2) .≈ 1)           # a partition of unity: the intercept is inside
        @test eval_basis(b, 0.0)[1] == 1 && eval_basis(b, 10.0)[end] == 1
        @test eval_basis(b, -5.0) == eval_basis(b, 0.0)   # clamped outside the interval
        # a cubic is reproduced exactly
        tt = collect(range(0, 10; length = 200))
        X = design_matrix(b, tt)
        f = @. 1 - 2tt + 0.3tt^2 - 0.02tt^3
        @test maximum(abs, X * (X \ f) - f) < 1e-10
        for p in 0:4
            @test all(sum(design_matrix(BSplineBasis(0, 10, 7; degree = p), ts); dims = 2) .≈ 1)
        end

        @test_throws ArgumentError BSplineBasis([0.0, 2.0, 1.0])
        @test_throws ArgumentError BSplineBasis([1.0])
        @test_throws ArgumentError BSplineBasis([0.0, 1.0]; degree = -1)
    end

    @testset "Bases: cyclic B-splines" begin
        c = CyclicBSplineBasis(24, 8)
        @test nbasis(c) == 8
        # at a knot, a cubic B-spline basis takes the values 1/6, 2/3, 1/6
        @test sort(eval_basis(c, 3.0)) ≈ [0, 0, 0, 0, 0, 1 / 6, 1 / 6, 2 / 3]
        ts = range(0, 48; length = 501)
        M = design_matrix(c, ts)
        @test all(≥(0), M)
        @test all(sum(M; dims = 2) .≈ 1)
        @test design_matrix(c, ts .+ 24) ≈ M      # periodic
        @test eval_basis(c, 24 - 1e-9) ≈ eval_basis(c, 0.0) atol = 1e-8   # continuous across the wrap
        for p in 0:4
            @test all(sum(design_matrix(CyclicBSplineBasis(10, 6; degree = p), ts); dims = 2) .≈ 1)
        end
        # a smooth periodic curve with two peaks per period, which one harmonic cannot follow
        tt = 1:24
        f = @. sin(4π * tt / 24) + 0.5cos(2π * tt / 24)
        fit_err(B) = (X = design_matrix(B, tt); maximum(abs, X * (X \ f) - f))
        @test fit_err(CyclicBSplineBasis(24, 12)) < 0.05 < fit_err(TrigBasis(24, 1))

        @test_throws ArgumentError CyclicBSplineBasis(24, 3)    # needs more knots than its degree
        @test_throws ArgumentError CyclicBSplineBasis(-1, 8)
    end

    @testset "Bases: splines in a ParamSpec" begin
        spec = ParamSpec(μ = BSplineBasis(1, 50, 5) => IdentityLink(), σ = CyclicBSplineBasis(50, 6) => LogLink())
        @test ncoef(spec) == (5 + 3 - 1) + 6
        θ = initial_coefficients(spec; μ = 2.0, σ = 1.5)
        # every spline column sums to one, so constant starting values are exact everywhere
        @test all(parameters_at(spec, θ, t)[1] ≈ 2.0 for t in (1, 17, 50))
        @test all(parameters_at(spec, θ, t)[2] ≈ 1.5 for t in (1, 17, 50))
    end

    @testset "COMPATIBILITY: TrigBasis reproduces SmoothPeriodicStatsModels exactly" begin
        # Bit-identical, not merely approximate: stored coefficient vectors keep their meaning
        # and a refit reproduces the same numbers.
        for (T, d) in ((12, 2), (366, 3), (11, 1), (52, 4), (365, 1), (7, 0))
            @test design_matrix(TrigBasis(T, d), 1:T) == spsm_trig_basis(T, d)
        end
        @test nbasis(TrigBasis(366, 3)) == 2 * 3 + 1
        @test TrigBasis(10, 0) isa ConstantBasis
    end

    @testset "COMPATIBILITY: links reproduce the SPSM link functions" begin
        Ps = range(-8, 8; length = 201)
        @test all(linkinv(IdentityLink(), P) == P for P in Ps)                   # mu
        @test all(linkinv(LogLink(), P) == exp(P) for P in Ps)                   # sigma
        @test all(linkinv(LogitLink(), P) == logistic(P) for P in Ps)            # alpha
        @test all(linkinv(BoundedLink(-1, 1), P) ≈ 2logistic(P) - 1 for P in Ps) # rho
        # and they stay inside their domains at extreme predictors
        @test linkinv(LogLink(), -800.0) >= 0
        @test 0 <= linkinv(LogitLink(), 800.0) <= 1
        @test -1 <= linkinv(BoundedLink(-1, 1), -800.0) <= 1
    end

    @testset "IntervalLink is BoundedLink without the overflow" begin
        l = IntervalLink(-1, 1)
        Ps = range(-8, 8; length = 201)
        @test all(linkinv(l, P) ≈ linkinv(BoundedLink(-1, 1), P) for P in Ps)
        @test all(linkinv(l, P) == 2logistic(P) - 1 for P in Ps)

        # the reason it exists: `(a + b*exp(η))/(1 + exp(η))` is `Inf/Inf` past `η = 709`,
        # which the positive tail of a line search reaches
        @test isnan(linkinv(BoundedLink(-1, 1), 800.0))
        @test linkinv(l, 800.0) == 1
        @test linkinv(l, -800.0) == -1

        # round trip, and the four derivatives of the `Link` interface
        @test linkfun(l, 0.3) ≈ linkfun(BoundedLink(-1, 1), 0.3)
        @test linkinv(l, linkfun(l, 0.3)) ≈ 0.3
        @test theta_eta(l, 0.7) ≈ ForwardDiff.derivative(P -> linkinv(l, P), 0.7)
        @test theta2_eta2(l, 0.7) ≈ ForwardDiff.derivative(P -> theta_eta(l, P), 0.7)
        @test eta_theta(l, 0.3) ≈ ForwardDiff.derivative(t -> linkfun(l, t), 0.3)
        @test eta2_theta2(l, 0.3) ≈ ForwardDiff.derivative(t -> eta_theta(l, t), 0.3)

        @test_throws ArgumentError IntervalLink(1, 1)
        @test_throws ArgumentError IntervalLink(2, 1)
        @test IntervalLink(-1, 1.0) isa IntervalLink{Float64}   # promotes
    end

    @testset "ParamSpec: layout and views" begin
        spec = ParamSpec(
            mu = TrigBasis(12, 1) => IdentityLink(),
            sigma = ConstantBasis() => LogLink(),
        )
        @test paramnames(spec) == (:mu, :sigma)
        @test TVP.nparams(spec) == 2
        @test ncoef(spec) == 3 + 1

        theta = [1.0, 2.0, 3.0, 4.0]
        @test collect(coefs(spec, theta, :mu)) == [1.0, 2.0, 3.0]
        @test collect(coefs(spec, theta, :sigma)) == [4.0]

        # a view, so writing through it updates the flat vector
        coefs(spec, theta, :sigma)[1] = 9.0
        @test theta[4] == 9.0

        @test_throws ArgumentError coefs(spec, theta, :nope)
        @test_throws ArgumentError ParamSpec()
        @test_throws ArgumentError ParamSpec(mu = TrigBasis(12, 1))          # not a Pair
        @test_throws ArgumentError ParamSpec(mu = 3 => IdentityLink())       # not a basis
    end

    @testset "ParamSpec: parameters_at applies basis then link" begin
        spec = ParamSpec(
            mu = TrigBasis(12, 1) => IdentityLink(),
            sigma = TrigBasis(12, 1) => LogLink(),
        )
        theta = [0.5, 0.1, -0.2, 0.3, 0.0, 0.4]
        B = spsm_trig_basis(12, 1)
        for t in 1:12
            P_mu = dot(B[t, :], theta[1:3])
            P_sigma = dot(B[t, :], theta[4:6])
            m, s = parameters_at(spec, theta, t)
            @test m ≈ P_mu
            @test s ≈ exp(P_sigma)
        end
        @test parameters_at(spec, theta, 3) isa Tuple{Float64, Float64}   # concretely typed
    end

    @testset "initial_coefficients" begin
        spec = ParamSpec(
            mu = TrigBasis(12, 1) => IdentityLink(),
            sigma = TrigBasis(12, 1) => LogLink(),
        )
        # a scalar start goes on the intercept, through the link
        theta = initial_coefficients(spec; mu = 2.0, sigma = 3.0)
        @test theta[1] == 2.0
        @test theta[4] == log(3.0)
        @test theta[[2, 3, 5, 6]] == zeros(4)
        # and it round-trips to the value asked for
        m, s = parameters_at(spec, theta, 1)
        @test m ≈ 2.0
        @test s ≈ 3.0

        # a vector start is taken as coefficients
        theta2 = initial_coefficients(spec; mu = [1.0, 2, 3], sigma = [4.0, 5, 6])
        @test theta2 == 1.0:6.0

        @test_throws ArgumentError initial_coefficients(spec; mu = 2.0)                # missing sigma
        @test_throws ArgumentError initial_coefficients(spec; mu = [1.0], sigma = 3.0) # wrong length

        # A scalar is refused when column 1 is not an intercept. The draft silently wrote it
        # onto the first slope instead.
        noint = ParamSpec(
            p = PolynomialBasis(1) + IndicatorBasis(t -> mod1(Int(t), 3), 1:3) => LogitLink(),
        )
        @test_throws ArgumentError initial_coefficients(noint; p = 0.5)
        @test initial_coefficients(noint; p = zeros(4)) == zeros(4)
    end

    @testset "TimeVaryingFamily yields a Distributions.jl object" begin
        spec = ParamSpec(
            mu = TrigBasis(12, 1) => IdentityLink(),
            sigma = TrigBasis(12, 1) => LogLink(),
        )
        fam = TimeVaryingFamily(Normal, spec)
        theta = [0.5, 0.1, -0.2, 0.3, 0.0, 0.4]
        d = fam(3, theta)
        @test d isa Normal
        m, s = parameters_at(spec, theta, 3)
        @test mean(d) == m
        @test std(d) == s
        # everything time-independent comes from Distributions
        @test logpdf(d, 0.4) == logpdf(Normal(m, s), 0.4)
        @test rand(StableRNG(1), d) == rand(StableRNG(1), Normal(m, s))

        # a closure works too, e.g. to skip argument validation
        fam2 = TimeVaryingFamily((m, s) -> Normal(m, s; check_args = false), spec)
        @test fam2(3, theta) == d
    end

    @testset "Periodic and pointwise objectives agree" begin
        T, N = 12, 600
        spec = ParamSpec(
            mu = TrigBasis(T, 1) => IdentityLink(),
            sigma = TrigBasis(T, 1) => LogLink(),
        )
        fam = TimeVaryingFamily(Normal, spec)
        theta = [2.0, 0.5, -0.3, 0.0, 0.4, 0.2]
        rng = StableRNG(42)
        n2t = repeat(1:T, N ÷ T)
        y = [rand(rng, fam(t, theta)) for t in n2t]

        design = PeriodicDesign(spec, 1:T)
        groups = group_by_time(n2t, T)
        per = PeriodicNLL(fam, design, y, groups)
        pt = PointwiseNLL(fam, y, n2t)
        @test per(theta) ≈ pt(theta)

        # and with weights
        w = rand(StableRNG(7), N)
        @test PeriodicNLL(fam, design, y, groups, w)(theta) ≈ PointwiseNLL(fam, y, n2t, w)(theta)

        # the periodic design is the trig basis, once per parameter
        @test design.designs[1] == spsm_trig_basis(T, 1)
        @test length(linear_predictors(design, theta)) == 2
        @test linear_predictors(design, theta)[1] ≈ spsm_trig_basis(T, 1) * theta[1:3]
        @test length(distributions(fam, design, theta)) == T
        @test distributions(fam, design, theta)[3] == fam(3, theta)

        # grouping covers every observation exactly once
        @test sort(vcat(groups...)) == 1:N
        @test_throws ArgumentError group_by_time([0, 1], 3)
    end

    @testset "The objective is type stable with a bare distribution type" begin
        # Holding `Normal` directly would store a `UnionAll` that inference cannot see
        # through: the objective returned `Any`, allocated 3 MB per call and ran ~16x
        # slower than the identical hand-written closure. `TimeVaryingFamily(::Type, spec)`
        # wraps it in a `DistConstructor` singleton so the two are equivalent.
        T, N = 24, 480
        spec = ParamSpec(
            mu = TrigBasis(T, 1) => IdentityLink(),
            sigma = TrigBasis(T, 1) => LogLink(),
        )
        theta = [1.0, 0.3, -0.2, 0.1, 0.2, -0.1]
        bare = TimeVaryingFamily(Normal, spec)
        closure = TimeVaryingFamily((m, s) -> Normal(m, s), spec)
        @test bare.dist isa DistConstructor
        rng = StableRNG(9)
        n2t = repeat(1:T, N ÷ T)
        y = [rand(rng, bare(t, theta)) for t in n2t]
        design = PeriodicDesign(spec, 1:T)
        groups = group_by_time(n2t, T)
        o_bare = PeriodicNLL(bare, design, y, groups)
        o_clos = PeriodicNLL(closure, design, y, groups)

        @test o_bare(theta) == o_clos(theta)
        @test Base.return_types(o_bare, (Vector{Float64},))[1] === Float64
        # and it stays allocation-light: a handful of period-length vectors, not one
        # temporary per observation
        o_bare(theta)
        @test (@allocated o_bare(theta)) < 100 * T
    end

    @testset "Gradients: ForwardDiff matches finite differences" begin
        T, N = 10, 200
        spec = ParamSpec(
            mu = TrigBasis(T, 1) => IdentityLink(),
            sigma = TrigBasis(T, 1) => LogLink(),
        )
        fam = TimeVaryingFamily(Normal, spec)
        theta = [1.0, 0.3, -0.2, 0.1, 0.2, -0.1]
        rng = StableRNG(3)
        n2t = repeat(1:T, N ÷ T)
        y = [rand(rng, fam(t, theta)) for t in n2t]
        obj = PeriodicNLL(fam, PeriodicDesign(spec, 1:T), y, group_by_time(n2t, T))

        g_ad = ForwardDiff.gradient(obj, theta)
        g_fd = fd_gradient(obj, theta)
        @test g_ad ≈ g_fd rtol = 1e-6
        @test all(isfinite, g_ad)
    end

    @testset "fit: recovers known coefficients" begin
        T, N = 12, 4000
        spec = ParamSpec(
            mu = TrigBasis(T, 1) => IdentityLink(),
            sigma = TrigBasis(T, 1) => LogLink(),
        )
        fam = TimeVaryingFamily(Normal, spec)
        truth = [2.0, 0.8, -0.5, 0.0, 0.3, 0.2]
        rng = StableRNG(2024)
        n2t = repeat(1:T, N ÷ T)
        y = [rand(rng, fam(t, truth)) for t in n2t]

        res = fit(fam, y; n2t = n2t, nT = T, init = (mu = 0.0, sigma = 1.0), solver = SOLVER)
        @test coef(res) ≈ truth atol = 0.1
        @test collect(coef(res, :mu)) == coef(res)[1:3]
        @test res(3) isa Normal
        @test params(res, 3) == parameters_at(spec, coef(res), 3)
        @test length(curve(res, :sigma, 1:T)) == T
        @test all(>(0), curve(res, :sigma, 1:T))

        # the generic per-observation path finds the same optimum
        res_pt = fit(fam, y, n2t; init = (mu = 0.0, sigma = 1.0), solver = SOLVER)
        @test coef(res_pt) ≈ coef(res) rtol = 1e-4
    end

    @testset "fit: weights behave like repeated observations" begin
        # The sharpest check that weights are wired through: fitting `y` with integer weights
        # must equal fitting the sample with each observation repeated that many times.
        T = 6
        spec = ParamSpec(
            mu = TrigBasis(T, 1) => IdentityLink(),
            sigma = ConstantBasis() => LogLink(),
        )
        fam = TimeVaryingFamily(Normal, spec)
        rng = StableRNG(11)
        n2t = repeat(1:T, 40)
        y = [rand(rng, fam(t, [1.0, 0.4, -0.3, 0.0])) for t in n2t]
        reps = rand(StableRNG(5), 1:3, length(y))

        init = (mu = 0.0, sigma = 1.0)
        weighted = fit(fam, y; n2t = n2t, nT = T, weights = float(reps), init = init, solver = SOLVER)
        y_rep = vcat([fill(y[n], reps[n]) for n in eachindex(y)]...)
        t_rep = vcat([fill(n2t[n], reps[n]) for n in eachindex(y)]...)
        repeated = fit(fam, y_rep; n2t = t_rep, nT = T, init = init, solver = SOLVER)
        @test coef(weighted) ≈ coef(repeated) rtol = 1e-4

        # zero weight drops an observation entirely
        w0 = ones(length(y))
        w0[1] = 0.0
        dropped = fit(fam, y[2:end]; n2t = n2t[2:end], nT = T, init = init, solver = SOLVER)
        zeroed = fit(fam, y; n2t = n2t, nT = T, weights = w0, init = init, solver = SOLVER)
        @test coef(zeroed) ≈ coef(dropped) rtol = 1e-4

        @test_throws ArgumentError fit(fam, y; n2t = n2t, nT = T, weights = ones(3), init = init, solver = SOLVER)
        @test_throws ArgumentError fit(
            fam, y; n2t = n2t, nT = T, weights = -ones(length(y)), init = init, solver = SOLVER)
    end

    @testset "fit: the arbitrary-loss escape hatch" begin
        # Least squares against a seasonal curve: no distribution involved.
        T = 12
        spec = ParamSpec(rho = TrigBasis(T, 1) => BoundedLink(-1, 1))
        truth = [0.2, 0.5, -0.3]
        B = spsm_trig_basis(T, 1)
        target = [2logistic(dot(B[t, :], truth)) - 1 for t in 1:T]
        loss(p, x) = abs2(only(p) - x)
        sol = fit(spec, loss, target, 1:T; init = (rho = zeros(3),), solver = SOLVER)
        @test sol.u ≈ truth atol = 1e-3
    end

    @testset "fit: input validation" begin
        spec = ParamSpec(mu = ConstantBasis() => IdentityLink())
        fam = TimeVaryingFamily(m -> Normal(m, 1.0), spec)
        y = randn(StableRNG(1), 10)
        @test_throws ArgumentError fit(fam, y; init = (mu = 0.0,), solver = SOLVER)                      # no ts, no n2t
        @test_throws ArgumentError fit(fam, y; n2t = ones(Int, 10), init = (mu = 0.0,), solver = SOLVER) # n2t without nT
    end
end

end # module TestTimeVaryingParameters
