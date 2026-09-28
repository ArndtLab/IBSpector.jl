using IBSpector
using IBSpector.Spectra
using Test
using HistogramBinnings
using StatsBase
using IBSpector.Spectra.PreallocationTools
using ForwardDiff
using LinearAlgebra
using Random
using Distributions
using Statistics

const SMCp = IBSpector.Spectra.SMCpIntegrals
using IBSpector.Spectra.SMCpIntegrals: TimeGrid, timegrid, timenodes!, ndt,
    npanels, sepkernel!, transition!, TIMEGRID_DEFAULTS

@testset "TimeGrid" begin
    g = TimeGrid(5; msub = 8, nfin = 4, ntail = 16)
    @test g.K == 5 && g.msub == 8 && g.nfin == 4
    @test npanels(g) == 4 * (length(g.fedge) - 1) + (length(g.uedge) - 1)
    @test ndt(g) == npanels(g) * 8
    # a one-epoch history has no finite panels, only the tail
    @test npanels(TimeGrid(1; msub = 8, nfin = 4, ntail = 16)) == length(g.uedge) - 1

    # finite-epoch mesh: fixed fractions of the epoch, geometric towards its
    # left endpoint, so a boundary layer of unknown scale is resolved there
    @test g.fedge[1] == 0.0 && g.fedge[end] == 1.0
    @test issorted(g.fedge)
    @test g.fedge[2] <= 1e-6 + 1e-15
    @test length(g.fedge) - 1 == g.nfin + 1

    # Gauss-Legendre weights on (-1,1) sum to the interval length
    @test sum(g.wleg) ≈ 2 rtol = 1e-12
    @test length(g.zleg) == 8 && all(-1 .< g.zleg .< 1)

    # tail mesh: graded towards u = 0 by the algebraic map, truncated at umax,
    # then split so no sub-panel exceeds dumax
    @test g.uedge[1] == 0.0
    @test issorted(g.uedge)
    @test maximum(diff(g.uedge)) <= 1.0 + 1e-12
    @test g.uedge[end] == 25.0
    # the mesh must still reach small u: that density is what resolves the
    # recombination decay at the largest r when K = 1 and the tail is everything
    @test g.uedge[2] < 0.1

    # dumax and umax are honoured
    @test maximum(diff(TimeGrid(1; ntail = 16, dumax = 0.25).uedge)) <= 0.25 + 1e-12
    @test TimeGrid(1; ntail = 16, umax = 12.0).uedge[end] == 12.0
end

@testset "Lpw integrates the local interpolant exactly" begin
    for msub in (4, 8, 12)
        g = TimeGrid(1; msub = msub)
        # Lpw[q,i]*wleg[i] = int_{-1}^{z_q} l_i(z) dz, so applied to the nodal
        # values of any polynomial of degree < msub it gives the exact partial
        # integral. Checked on the monomial basis.
        for d in 0:msub-1, q in 1:msub
            exact = (g.zleg[q]^(d+1) - (-1.0)^(d+1)) / (d + 1)
            got = sum(g.Lpw[q,i] * g.wleg[i] * g.zleg[i]^d for i in 1:msub)
            @test got ≈ exact atol = 1e-12
        end
        # constant function: the partial integral is the interval length
        for q in 1:msub
            @test sum(g.Lpw[q,i] * g.wleg[i] for i in 1:msub) ≈ g.zleg[q] + 1 atol = 1e-13
        end
    end
end

# the real 5-epoch history where the production fit stalls; N3 = 9.99e7 sits at
# the upper bound, giving the worst adjacent-N ratio the box permits (~14300)
const TNSTALL = [3.003e9, 12388.8, 28302.1, 6975.85, 6214.37,
                 9.99002e7, 3066.44, 2754.27, 215.101, 21782.5]

@testset "timenodes! tiles the epochs" begin
    TN = TNSTALL
    K = length(TN) ÷ 2
    g = TimeGrid(K; msub = 8, nfin = 2, ntail = 8)
    n = ndt(g); np = npanels(g)
    ts = zeros(n); om = zeros(n); EE = zeros(n); EB = zeros(np)
    timenodes!(ts, om, EE, EB, g, TN)

    @test all(diff(ts) .> 0)
    @test all(ts .> 0)
    @test all(isfinite, ts) && all(isfinite, om)
    @test all(om .> 0)

    # every finite epoch's nodes lie strictly inside it, and its sub-panel
    # weights sum to the epoch width
    per = (length(g.fedge) - 1) * g.msub
    for k in 1:K-1
        lo = Spectra.getts(TN, k); hi = Spectra.getts(TN, k+1)
        blk = ts[(k-1)*per+1 : k*per]
        @test all(lo .< blk .< hi)
        @test sum(om[(k-1)*per+1 : k*per]) ≈ hi - lo rtol = 1e-12
    end
    # tail nodes are past the last epoch time and stop at the truncation
    tailrange = (K-1)*per+1 : n
    @test all(ts[tailrange] .> Spectra.getts(TN, K))
    @test maximum(ts) < Spectra.getts(TN, K) + 2 * Spectra.getns(TN, K) * g.uedge[end]
end

@testset "timenodes! rescalings match cumcr" begin
    TN = TNSTALL
    K = length(TN) ÷ 2
    g = TimeGrid(K; msub = 8, nfin = 3, ntail = 8)
    n = ndt(g); np = npanels(g)
    ts = zeros(n); om = zeros(n); EE = zeros(n); EB = zeros(np)
    timenodes!(ts, om, EE, EB, g, TN)

    # EE[j] = exp((C(t_j) - C(a_p))/2) with a_p the sub-panel's left edge, and
    # EB[p] = exp((C(b_p) - C(a_p))/2). Both are built from the affine maps
    # rather than from cumcr, so this is a real cross-check.
    for p in 1:np
        s0 = (p-1)*g.msub
        for q in 2:g.msub
            ref = exp(Spectra.cumcr(ts[s0+1], ts[s0+q], TN) / 2)
            @test EE[s0+q] / EE[s0+1] ≈ ref rtol = 1e-12
        end
        # EB[p] covers the whole sub-panel, so it exceeds every EE inside it
        @test EB[p] >= EE[s0+g.msub] * (1 - 1e-12)
    end
    # panels are contiguous: chaining EB across all of them reproduces C(t)/2
    acc = 0.0
    for p in 1:np-1
        acc += log(EB[p])
    end
    tlast = Spectra.getts(TN, K) + 2 * Spectra.getns(TN, K) * g.uedge[end-1]
    @test acc ≈ Spectra.cumcr(0.0, tlast, TN) / 2 rtol = 1e-10
end

@testset "timenodes! integrates the tail exactly" begin
    # single epoch: the whole domain is the tail, and int_0^inf exp(-t/2N) dt = 2N
    # up to the exp(-umax) truncation
    N = 12345.0
    TN = [3.0e9, N]
    g = TimeGrid(1; msub = 8, ntail = 16)
    n = ndt(g); np = npanels(g)
    ts = zeros(n); om = zeros(n); EE = zeros(n); EB = zeros(np)
    timenodes!(ts, om, EE, EB, g, TN)
    @test sum(om .* exp.(-ts ./ (2N))) ≈ 2N rtol = 1e-9
    # and a polynomial-times-exponential moment
    @test sum(om .* ts .* exp.(-ts ./ (2N))) ≈ (2N)^2 rtol = 1e-9
    # the truncation is where the grid says it is
    @test sum(om) ≈ 2N * g.uedge[end] rtol = 1e-12
end

@testset "timenodes! is exactly affine in TN" begin
    # THE invariant that the old tolegendre/tolaguerre map violated: there, a node
    # crossing an epoch boundary kept t continuous but changed dt/dz by the ratio
    # of adjacent N (14300x at this TN), producing a kink in the likelihood.
    # Here every node is affine in the epoch parameters, so along any straight
    # line in TN space each ts[j] must be exactly linear, to roundoff.
    TN0 = TNSTALL
    K = length(TN0) ÷ 2
    g = TimeGrid(K; msub = 8, nfin = 4, ntail = 8)
    n = ndt(g); np = npanels(g)
    for idx in (3, 5, 7, 9, 2, 10)          # durations and sizes alike
        d = zeros(length(TN0)); d[idx] = TN0[idx]   # relative perturbation
        as = range(-1e-3, 1e-3, length = 51)
        T = zeros(length(as), n)
        Om = zeros(length(as), n)
        ts = zeros(n); om = zeros(n); EE = zeros(n); EB = zeros(np)
        for (q, a) in enumerate(as)
            timenodes!(ts, om, EE, EB, g, TN0 .+ a .* d)
            T[q, :] .= ts
            Om[q, :] .= om
        end
        for j in 1:n
            lo, hi = T[1, j], T[end, j]
            for q in 1:length(as)
                lin = lo + (hi - lo) * (q - 1) / (length(as) - 1)
                @test T[q, j] ≈ lin atol = 1e-9 * max(abs(lo), abs(hi)) + 1e-12
            end
            olo, ohi = Om[1, j], Om[end, j]
            for q in 1:length(as)
                lin = olo + (ohi - olo) * (q - 1) / (length(as) - 1)
                @test Om[q, j] ≈ lin atol = 1e-9 * max(abs(olo), abs(ohi)) + 1e-12
            end
        end
    end
end

@testset "sweep runs on the panel grid and matches the order loop" begin
    mu, rho = 1.0e-8, 2.0e-8
    TN = TNSTALL
    K = length(TN) ÷ 2
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = 30_000, nbins = 120)
    edges = collect(Float64, ev); rs = collect(Float64, midpoints(ev))
    grid = TimeGrid(K; msub = 8, nfin = 4, ntail = 8)

    bagf = IntegralArrays(grid, length(rs), rs, edges, Val{length(TN)})
    @test bagf.n_dt == ndt(grid)

    # npicard = 6 (the design-target rule for this alpha) only contracts the
    # error to ~3e-3 by the fixed exponential-Euler bin count here; np = 25
    # drives the Picard iteration itself well under the 1e-6 bound, confirming
    # fused and order-loop converge to the same integral.
    SMCp.fusedsweep!(bagf, mu, rho, TN; npicard = 25)
    yf = copy(get_tmp(bagf.ys, Float64))
    yo = orderref(rs, edges, mu, rho, grid, TN, 60)

    @test all(isfinite, yf) && all(yf .> 0)
    # converged Picard vs converged order loop: same integral, same grid
    @test maximum(abs.(yf .- yo) ./ yo) < 1e-6
end

# quadrature estimate of the order-1 terminal integral, whose exact value is
# firstorder(r, rate, TN)
function firstorder_quad(r, mu, rho, grid, TN)
    rate = mu + rho
    n = ndt(grid)
    ts = zeros(n); om = zeros(n); EE = zeros(n); EB = zeros(npanels(grid))
    timenodes!(ts, om, EE, EB, grid, TN)
    s = 0.0
    for j in 1:n
        s += rate * exp(-2rate * r * ts[j]) * SMCp.pt(ts[j], TN) * 2 * ts[j] * om[j]
    end
    s
end

@testset "quadrature matches the analytic firstorder" begin
    mu, rho = 1.0e-8, 2.0e-8
    rate = mu + rho
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = 3_000_000, nbins = 200)
    rs = collect(Float64, midpoints(ev))

    histories = Dict(
        "stationary"          => [3.0e9, 20000.0],
        "real 5-epoch"        => TNSTALL,
        "N at upper bound"    => [3.0e9, 12000.0, 5000.0, 1.0e8, 3000.0, 7000.0],
        "T at lower floor"    => [3.0e9, 12000.0, 10.0, 3000.0, 10.0, 20000.0],
        "near-empty epoch"    => [3.0e9, 15000.0, 6000.0, 9.9e7, 4000.0, 8000.0],
    )
    for (name, TN) in histories
        K = length(TN) ÷ 2
        g = TimeGrid(K)                       # the shipped defaults
        err = maximum(abs(firstorder_quad(r, mu, rho, g, TN) -
                          firstorder(r, rate, TN)) / firstorder(r, rate, TN)
                      for r in rs)
        @test err < 1e-6
    end
end

@testset "firstorder error converges geometrically in m" begin
    mu, rho = 1.0e-8, 2.0e-8
    rate = mu + rho
    TN = TNSTALL
    K = length(TN) ÷ 2
    rs = [1.0, 100.0, 10_000.0, 1_000_000.0]
    # hold the tail mesh fixed and generous so this measures the FINITE panels only
    errs = map((4, 6, 8)) do ms
        g = TimeGrid(K; msub = ms, nfin = 4, ntail = 32)
        maximum(abs(firstorder_quad(r, mu, rho, g, TN) -
                    firstorder(r, rate, TN)) / firstorder(r, rate, TN) for r in rs)
    end
    # Adding nodes per sub-panel must gain at least an order of magnitude UNTIL
    # the error reaches the double-precision floor, after which no further gain
    # is possible. Asserting a ratio without the floor guard is unsatisfiable.
    FLOOR = 1e-10
    @test errs[2] < errs[1] / 10 || errs[2] < FLOOR
    @test errs[3] < errs[2] / 10 || errs[3] < FLOOR
end

@testset "FitOptions carries the sub-panel counts" begin
    fop = FitOptions(3.0e9, 100_000, 1.0e-8, 2.0e-8; nepochs = 5)
    @test fop.msub > 0 && fop.nfin > 0 && fop.ntail > 0
    @test !hasproperty(fop, :ndt)     # renamed, so a stale `ndt = 800` cannot pass silently
    @test !hasproperty(fop, :mpanel)  # ditto for the pre-correction names
    @test !hasproperty(fop, :order)   # the order loop is gone
    g = timegrid(fop.nepochs; msub = fop.msub, nfin = fop.nfin, ntail = fop.ntail)
    @test ndt(g) == npanels(g) * fop.msub
    @test (fop.msub, fop.nfin, fop.ntail) ==
          (TIMEGRID_DEFAULTS.msub, TIMEGRID_DEFAULTS.nfin, TIMEGRID_DEFAULTS.ntail)
end

@testset "likelihood is smooth along a line" begin
    mu, rho = 1.0e-8, 2.0e-8
    TN = TNSTALL
    K = length(TN) ÷ 2
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = 3_000_000, nbins = 200)
    edges = collect(Float64, ev); rs = collect(Float64, IBSpector.midpoints(ev))

    # Poisson counts from the model itself, so the surface has a real optimum
    Random.seed!(20260819)
    grid = TimeGrid(K)
    bag = IntegralArrays(grid, length(rs), rs, edges, Val{K * 2}, 3)
    SMCp.fusedsweep!(bag, mu, rho, TN)
    w0 = get_tmp(bag.ys, Float64) .* diff(edges)
    counts = [rand(Poisson(max(w, 0.0))) for w in w0]

    function f(v)
        SMCp.fusedsweep!(bag, mu, rho, v)
        w = get_tmp(bag.ys, eltype(v)) .* diff(edges)
        s = zero(eltype(v))
        for i in eachindex(counts)
            (!(w[i] > 0) || isnan(w[i])) && continue
            s += counts[i] * log(w[i]) - w[i]
        end
        s
    end

    Random.seed!(3)
    d = zeros(length(TN))   # pre-declared so it survives the loop (soft-scope in a hard-scope testset)
    for trial in 1:3
        d = TN .* normalize(randn(length(TN)))
        # detrended residual must sit at the float floor at EVERY window size;
        # the old global map climbed like h (1e-8 at h=3e-7 -> 0.42 at h=1e-5)
        for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
            n = 41
            ss = collect(range(-h, h, length = n))
            fs = [f(TN .+ a .* d) for a in ss]
            A = hcat(ones(n), ss, ss .^ 2, ss .^ 3)
            resid = maximum(abs, fs .- A * (A \ fs))
            @test resid < 1e-6 * max(1.0, maximum(abs, fs))
        end
    end

    # (a) derivative-jump ratio: was 5.85e5 with the old map, must be O(1)
    n = 241; hw = 1e-4
    as = collect(range(-hw, hw, length = n))
    ps = [f(TN .+ a .* d) for a in as]
    slope = [(ps[i+1] - ps[i]) / (as[i+1] - as[i]) for i in 1:n-1]
    d2 = [abs(slope[i+1] - slope[i]) for i in 1:n-2]
    @test maximum(d2) < 50 * median(d2)

    # (b) central FD must agree with AD at EVERY h, not only below 1e-7
    g = ForwardDiff.gradient(f, TN)
    gd = dot(g, d)
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
        fd = (f(TN .+ h .* d) - f(TN .- h .* d)) / (2h)
        @test abs(fd - gd) < 1e-3 * abs(gd)
    end
end

@testset "timenodes! rejects a non-monotone TN" begin
    g = TimeGrid(3; msub = 8, nfin = 2, ntail = 8)
    n = ndt(g); np = npanels(g)
    ts = zeros(n); om = zeros(n); EE = zeros(n); EB = zeros(np)

    # zero-width panel: without the guard this silently gives min(om) == 0.0
    @test_throws ArgumentError timenodes!(ts, om, EE, EB, g, [3e9, 1e4, 0.0, 2e4, 100.0, 3e4])
    # descending panel: without the guard this silently gives a negative weight
    @test_throws ArgumentError timenodes!(ts, om, EE, EB, g, [3e9, 1e4, -50.0, 2e4, 100.0, 3e4])
    # a valid monotone TN does not throw
    timenodes!(ts, om, EE, EB, g, [3e9, 1e4, 50.0, 2e4, 100.0, 3e4])
    @test all(diff(ts) .> 0)
end

@testset "the corrected sweep is allocation-free" begin
    TN = TNSTALL
    K = length(TN) ÷ 2
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = 3_000_000, nbins = 200)
    edges = collect(Float64, ev); rs = collect(Float64, IBSpector.midpoints(ev))
    grid = TimeGrid(K)
    bag = IntegralArrays(grid, length(rs), rs, edges, Val{K * 2}, 3)
    SMCp.fusedsweep!(bag, 1.0e-8, 2.0e-8, TN)          # warm up
    @test (@allocated SMCp.fusedsweep!(bag, 1.0e-8, 2.0e-8, TN)) == 0

    # and under ForwardDiff, where the DiffCache buffers have to be reused too
    f(v) = (SMCp.fusedsweep!(bag, 1.0e-8, 2.0e-8, v);
            sum(get_tmp(bag.ys, eltype(v))))
    TNd = [ForwardDiff.Dual{Nothing}(v, 0.0, 0.0, 0.0) for v in TN]
    f(TNd)                                                         # warm up
    @test (@allocated f(TNd)) < 4096
end

@testset "the corrected sweep converges in the sub-panel counts" begin
    # The pre-correction rule was O(1/n) here: doubling the nodes only halved
    # the error, on every history and every node map (see the 2026-08-20 spike).
    # With the corner handled, refining the mesh must leave the answer alone.
    mu, rho = 1.0e-8, 2.0e-8
    TN = TNSTALL
    K = length(TN) ÷ 2
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = 3_000_000, nbins = 200)
    edges = collect(Float64, ev); rs = collect(Float64, IBSpector.midpoints(ev))
    dr = diff(edges)

    function sweepat(msub, nfin, ntail)
        g = TimeGrid(K; msub, nfin, ntail)
        bag = IntegralArrays(g, length(rs), rs, edges, Val{length(TN)})
        SMCp.fusedsweep!(bag, mu, rho, TN)
        copy(get_tmp(bag.ys, Float64))
    end

    ref = sweepat(12, 20, 32)
    wref = ref .* dr
    # Poisson sigma against the refined reference, at whole-genome counts.
    # Measured 4.4e-6 / 1.6e-8 / 7.6e-11 at 616 / 744 / 930 nodes, against the
    # pre-correction rule's 2.55 sigma at 640 and the old global map's 0.42.
    for (ms, nf, nt) in ((8, 8, 16), (8, 12, 16), (10, 12, 16))
        w = sweepat(ms, nf, nt) .* dr
        @test maximum(abs.(w .- wref) ./ sqrt.(wref)) < 1e-4
    end
    # the shipped defaults are on that list
    @test (TIMEGRID_DEFAULTS.msub, TIMEGRID_DEFAULTS.nfin, TIMEGRID_DEFAULTS.ntail) ==
          (8, 12, 16)
end
