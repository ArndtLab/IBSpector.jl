using IBSpector.Spectra
using IBSpector.Spectra.PreallocationTools
using IBSpector.Spectra.SMCpIntegrals: sepkernel!, transition!, ptt, TimeGrid,
    ndt, npanels, timenodes!
using IBSpector.Spectra.CoalescentBase: Nt, cumcr, getts, getns
using ForwardDiff
using Test

const SMCp = IBSpector.Spectra.SMCpIntegrals

# Node setup for a grid, returning everything transition! needs.
function kernelsetup(grid, TN)
    n = ndt(grid); np = npanels(grid)
    ts = zeros(n); om = zeros(n); EE = zeros(n); EB = zeros(np)
    timenodes!(ts, om, EE, EB, grid, TN)
    Phi = zeros(n); dgn = zeros(n); Gc = zeros(n); Ninv = zeros(n)
    sepkernel!(Phi, dgn, Gc, Ninv, ts, TN)
    (; ts, om, EE, EB, Phi, dgn, Gc, Ninv, n)
end

# The plain Nystrom rule the panel scheme shipped with before the diagonal
# correction: one global partial sum split at the row index, using whole-panel
# weights on what is really a partial interval. Kept as the contrast arm — the
# corrected apply must beat it by orders, not by a factor.
function plain_apply(x, K, ts, om, TN)
    n = length(ts)
    [sum(max(ptt(ts[j], ts[k], TN), 0.0) * (k == j ? 1.0 : om[k]) * x[k] for k in 1:n)
     for j in 1:n]
end

# Reference value of (M x)(t_j) for an analytic x, integrated to high accuracy.
# ptt has a corner at t' = t_j, and its own kinks at every epoch boundary, so the
# reference splits on all of them and applies a composite Gauss rule to each
# smooth piece. Splitting where the integrand is non-smooth is exactly what the
# rule under test does not do for free — which is what makes this a reference and
# not a restatement of it.
function exact_apply(xf, tj, tmax, TN; nsub = 20, mq = 24)
    K = length(TN) ÷ 2
    zz, ww = SMCp.gausslegendre(mq)
    brk = sort(unique(vcat(0.0, tj, tmax, [getts(TN, k) for k in 2:K])))
    filter!(t -> 0.0 <= t <= tmax, brk)
    s = 0.0
    for p in 1:length(brk)-1
        a, b = brk[p], brk[p+1]
        b > a || continue
        for k in 1:nsub
            aa = a + (b - a) * (k - 1) / nsub; bb = a + (b - a) * k / nsub
            c = (aa + bb) / 2; h = (bb - aa) / 2
            for i in eachindex(zz)
                t = c + h * zz[i]
                s += ww[i] * h * max(ptt(tj, t, TN), 0.0) * xf(t)
            end
        end
    end
    return s + max(ptt(tj, tj, TN), 0.0) * xf(tj)
end

const SEPTNS = [
    [3.0e9, 10000.0],
    [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0],
    [3.0e9, 20000.0, 60000.0, 8000.0, 8000.0, 16000.0, 1600.0, 2000.0, 400.0, 10000.0],
]

@testset "sepkernel! invariants" begin
    for TN in SEPTNS
        grid = TimeGrid(length(TN) ÷ 2; msub = 8, nfin = 4, ntail = 8)
        s = kernelsetup(grid, TN)
        @test all(diff(s.ts) .> 0)
        @test all(isfinite, s.Phi) && all(isfinite, s.Gc)
        @test all(s.Phi .>= 0) && all(s.dgn .>= 0) && all(s.Gc .>= 0)
        @test all(s.EE .>= 1) && all(s.EB .>= 1)   # C is monotone increasing
        @test all(isfinite, s.EE) && all(isfinite, s.EB)
    end
end

@testset "transition! integrates across the moving corner" begin
    # A smooth analytic x, so the only error left is the quadrature's own.
    for TN in SEPTNS
        K = length(TN) ÷ 2
        grid = TimeGrid(K; msub = 10, nfin = 8, ntail = 16)
        s = kernelsetup(grid, TN)
        tmax = getts(TN, K) + 2 * getns(TN, K) * grid.uedge[end]
        tau = tmax / 6
        xf(t) = exp(-t / tau) * (1 + t / tmax)

        x = xf.(s.ts)
        out = zeros(s.n)
        transition!(out, x, s.Phi, s.dgn, s.Gc, s.Ninv, s.EE, s.EB, s.om, grid)
        plain = plain_apply(x, nothing, s.ts, s.om, TN)

        # score on interior rows; the outermost nodes of the truncated tail carry
        # values at the exp(-umax) floor where a relative score is meaningless
        rows = 1:3:(s.n - grid.msub)
        ref = [exact_apply(xf, s.ts[j], tmax, TN) for j in rows]
        ecorr = maximum(abs(out[j] - ref[q]) / abs(ref[q]) for (q, j) in enumerate(rows))
        eplain = maximum(abs(plain[j] - ref[q]) / abs(ref[q]) for (q, j) in enumerate(rows))

        # Measured 3.7e-8 to 1.5e-7 across these histories, against the plain
        # rule's 2.2e-2 to 2.8e-2. The absolute floor here is the reference's
        # own composite rule, not the apply's.
        @test ecorr < 1e-6
        # the whole point of the correction: it is not a constant-factor win
        @test ecorr < eplain / 1e4
    end
end

@testset "transition! converges geometrically in the nodes per sub-panel" begin
    TN = SEPTNS[2]
    K = length(TN) ÷ 2
    errs = map((4, 6, 8)) do ms
        grid = TimeGrid(K; msub = ms, nfin = 2, ntail = 16)
        s = kernelsetup(grid, TN)
        tmax = getts(TN, K) + 2 * getns(TN, K) * grid.uedge[end]
        tau = tmax / 6
        xf(t) = exp(-t / tau) * (1 + t / tmax)
        out = zeros(s.n)
        transition!(out, xf.(s.ts), s.Phi, s.dgn, s.Gc, s.Ninv, s.EE, s.EB, s.om, grid)
        rows = 1:3:(s.n - grid.msub)
        maximum(begin
            ref = exact_apply(xf, s.ts[j], tmax, TN)
            abs(out[j] - ref) / abs(ref)
        end for j in rows)
    end
    # For a smooth integrand the accuracy is set by msub, not by the sub-panel
    # count: the sub-panels already place the corner correctly, and what is left
    # is the local interpolant's error. The floor is the reference's own
    # composite rule, around 4e-8 here.
    FLOOR = 1e-7
    @test issorted(errs, rev = true) || errs[end] < FLOOR
    @test errs[2] < errs[1] / 10 || errs[2] < FLOOR
    @test errs[3] < errs[2] / 10 || errs[3] < FLOOR
end

@testset "transition! is ForwardDiff-differentiable" begin
    TN0 = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    grid = TimeGrid(length(TN0) ÷ 2; msub = 8, nfin = 4, ntail = 8)
    n = ndt(grid); np = npanels(grid)

    function total(TN)
        T = eltype(TN)
        ts = zeros(T, n); om = zeros(T, n); EE = zeros(T, n); EB = zeros(T, np)
        timenodes!(ts, om, EE, EB, grid, TN)
        Phi = zeros(T, n); dgn = zeros(T, n); Gc = zeros(T, n); Ninv = zeros(T, n)
        sepkernel!(Phi, dgn, Gc, Ninv, ts, TN)
        x = [exp(-t / 5000) for t in ts]
        out = zeros(T, n)
        transition!(out, x, Phi, dgn, Gc, Ninv, EE, EB, om, grid)
        sum(out)
    end

    g = ForwardDiff.gradient(total, TN0)
    @test length(g) == length(TN0)
    @test all(isfinite, g)
    @test any(!iszero, g)

    for k in (2, 4, 6)
        h = 1e-3 * TN0[k]
        tp = copy(TN0); tp[k] += h
        tm = copy(TN0); tm[k] -= h
        fd = (total(tp) - total(tm)) / 2h
        @test isapprox(g[k], fd; rtol = 1e-4, atol = 1e-8 * abs(g[k]))
    end
end
