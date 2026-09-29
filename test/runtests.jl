using IBSpector
using IBSpector: npar, setinit!, initialize!, fit_model_epochs!, #PInit, 
    setnepochs!, timesplitter, integral_ws, next!,
    reset_perturb!, perturb_fit!, residstructure, compute_residuals,
    isonlyN, setonlyN!, freemask, midlineagetime, epochfinder!
using PopSim
using HistogramBinnings
using Distributions
using StatsBase, StatsAPI
using Test
using IBSpector.Spectra

include("Aqua.jl")
include("spectra.jl")
include("smcp_semiseparable.jl")
include("smcp_fused.jl")
include("time_quadrature.jl")

const LOCAL = false

TNs = [
    [3000000000, 10000],
    [3000000000, 20000, 60000, 8000, 4000, 16000, 2000, 8000],
    [3000000000, 20000, 60000, 8000, 8000, 16000, 1600, 2000, 400, 10000],
    [3000000000, 20000, 60000, 8000, 8000, 16000, 1600, 2000, 400, 8000, 60, 300]
]
mus = [2.36e-8, 1e-8, 5e-9]
rhos = [1e-8]
itr = Base.Iterators.product(mus,rhos,TNs)

@testset "Test FitOptions" begin
    fop = FitOptions(30, 10, 1.0, 1.0)
    @test npar(fop) == 2
    @test fop.nepochs == 1
    @test all(fop.init .== zeros(npar(fop)))
    setinit!(fop, ones(npar(fop)))
    @test all(fop.init .!= ones(npar(fop)))
    @test all(fop.init .> fop.low)
    @test all(fop.upp .!= zeros(npar(fop)))
    @test all(fop.low .!= zeros(npar(fop)))
    h = Histogram([1,2,3,4])
    append!(h, [1,1,1,2,3,1,2])
    initialize!(fop, h.weights)
    @test any(fop.init .!= ones(npar(fop)))
    @test all(fop.init .> zeros(npar(fop)))
    @test all(fop.init .> fop.low)
    @test all(fop.init .< fop.upp)
    @test !any(fop.perturb)
    @test all(fop.low .< rand.(fop.prior) .< fop.upp)
    setnepochs!(fop, 5)
    @test npar(fop) == 10
    @test fop.init == zeros(npar(fop))
    initialize!(fop, h.weights)
    @test fop.perturb == falses(npar(fop))
    @test length(fop.low) == npar(fop)
    @test length(fop.upp) == npar(fop)
    @test all(fop.low .<= fop.init .<= fop.upp)
end

# @testset "Test PInit" begin
#     fop = FitOptions(30, 10, 1.0, 1.0)
#     p = PInit(fop)
#     @test fop.delta.state == 0
#     @test length(p) == npar(fop)
#     @test all(p .== fop.init)
#     @test all(fop.perturb .== false)
#     fop.perturb .= trues(npar(fop))
#     setinit!(fop, ones(npar(fop)))
#     next!(fop.delta)
#     @test length(p) == npar(fop)
#     @test any(p .!= fop.init)
#     @test all(fop.low .<= p .<= fop.upp)
#     @test fop.delta.state == 1
#     reset_perturb!(fop)
#     @test all(fop.perturb .== false)
# end

@testset "Test fit" begin
    h = Histogram([1,2,3,4])
    append!(h, [1,1,1,2,3,1,2])
    fop = FitOptions(11, 7, 1.0, 1.0; msub = 6, nfin = 4, ntail = 8, locut = 1)
    f = fit_model_epochs!(fop, h.edges[1], h.weights, Val(true))
    f = fit_model_epochs!(fop, h)
    @test f.converged
    @test f.opt.optflag
    perturb_fit!(f, fop, h)
    IBSpector.setnaive!(fop, false)
    IBSpector.setOptimOptions!(fop, g_tol=1e-3)
    # The smcp branch reads its buffers off fop.diffcache, which only demoinfer
    # populates. Build one the same way here: a theory grid at least as fine as
    # the observed one, mapped down onto h's edges by map_fine_to_coarse!.
    lo_edge, hi_edge = h.edges[1][1], h.edges[1][end]
    eth = IBSpector.CustomEdgeVector(; lo = lo_edge, hi = hi_edge - 1,
                                       nbins = length(h.weights))
    fop.diffcache = IntegralArrays(
        timegrid(fop.nepochs; msub = fop.msub, nfin = fop.nfin, ntail = fop.ntail),
        length(h.weights), midpoints(eth), eth, Val{npar(fop)}, 2,
    )
    fit_model_epochs!(fop, h)
end

function smcp_diffcache(fop, h)
    lo_edge, hi_edge = h.edges[1][1], h.edges[1][end]
    eth = IBSpector.CustomEdgeVector(; lo = lo_edge, hi = hi_edge - 1,
                                       nbins = length(h.weights))
    return IntegralArrays(
        timegrid(fop.nepochs; msub = fop.msub, nfin = fop.nfin, ntail = fop.ntail),
        length(h.weights), midpoints(eth), eth, Val{npar(fop)}, 2,
    )
end

@testset "Test fitNs! / sampleNs_posterior (N-only)" begin
    h = Histogram([1,2,3,4])
    append!(h, [1,1,1,2,3,1,2])
    fop = FitOptions(11, 7, 1.0, 1.0; msub = 6, nfin = 4, ntail = 8, locut = 1, nepochs = 2)
    @test !isonlyN(fop)
    @test freemask(fop) == trues(4)
    initialize!(fop, h.weights)

    f = fitNs!(fop, h)
    @test isonlyN(fop)
    mask = freemask(fop)
    @test mask == Bool[1, 1, 0, 1]
    @test f.free == mask
    @test free(f) == mask
    @test f.para[3] == fop.init[3] # T fixed
    @test f.stderrors[3] == 0.0
    @test f.opt.ci_low[3] == f.opt.ci_high[3] == f.para[3]
    @test size(f.opt.hess) == (3, 3)

    chain = sampleNs_posterior(fop, h, f; nsamples = 10)
    @test size(chain, 1) == 10

    # naive == false (SMC') path
    fop2 = FitOptions(11, 7, 1.0, 1.0; msub = 6, nfin = 4, ntail = 8, locut = 1, nepochs = 2)
    initialize!(fop2, h.weights)
    IBSpector.setnaive!(fop2, false)
    IBSpector.setOptimOptions!(fop2, g_tol=1e-3)
    fop2.diffcache = smcp_diffcache(fop2, h)
    f2 = fitNs!(fop2, h)
    @test f2.free == freemask(fop2)
    @test f2.para[3] == fop2.init[3]
    @test size(f2.opt.hess) == (3, 3)

    # regression: default (onlyN = false) TN-mode behavior is unaffected
    fop3 = FitOptions(11, 7, 1.0, 1.0; locut = 1)
    @test !isonlyN(fop3)
    f3 = fit_model_epochs!(fop3, h)
    @test all(f3.free)
    @test size(f3.opt.hess) == (2, 2)
end

@testset "Test setinit! with onlyN" begin
    fop = FitOptions(11, 7, 1.0, 1.0; nepochs = 3)
    # L below its lower bound, T1 below Tlow, N1 above Nupp
    TN = [5.0, 5.0, 5.0, 1e9, 20.0, 1000.0]

    setinit!(fop, TN)
    @test !isonlyN(fop)
    @test fop.init[1] > TN[1] # L truncated
    @test fop.init[3] > TN[3] # T1 truncated
    @test fop.init[5] == TN[5]
    @test all(fop.low .<= fop.init .<= fop.upp)

    setonlyN!(fop, true)
    setinit!(fop, TN)
    @test fop.init[1] > TN[1] # L is free, truncated
    @test fop.init[3] == TN[3] # Ts kept
    @test fop.init[5] == TN[5]
    @test fop.low[2] < fop.init[2] < fop.upp[2]
    @test fop.low[4] < fop.init[4] < fop.upp[4]
    @test fop.init[6] == TN[6]

    # the onlyN flag is reset by the full TN entry points
    h = Histogram([1,2,3,4])
    append!(h, [1,1,1,2,3,1,2])
    fop2 = FitOptions(11, 7, 1.0, 1.0; nepochs = 2)
    initialize!(fop2, h.weights)
    fitNs!(fop2, h; stats = false)
    @test isonlyN(fop2)
    fit_model_epochs!(fop2, h; stats = false)
    @test !isonlyN(fop2)
end

@testset "Test epochfinder! with onlyN" begin
    # a split of the oldest epoch, less than 1000 generations above its lower
    # boundary: the new duration is floored when the Ts are fitted, kept as is
    # when only the Ns are
    old = [3e9, 10000.0, 5000.0, 20000.0]
    t = 5500.0

    fop = FitOptions(3e9, 10, 1.0, 1.0; nepochs = 3)
    init = copy(old)
    epochfinder!(init, t, fop)
    @test length(init) == npar(fop)
    @test init[3] == 1000
    @test Spectra.getts(init, 3) == 6000

    setonlyN!(fop, true)
    init = copy(old)
    epochfinder!(init, t, fop)
    @test init[3] == t - 5000
    @test Spectra.getts(init, 3) == t
    @test init[4] == old[2] # the split epoch keeps its size
    @test init[[1,2,5,6]] == old
end

@testset "Test midlineagetime" begin
    TN = [3e9, 10000.0]
    rho = 1e-8
    tmax = 1e9
    t = midlineagetime(0, tmax, TN, rho)
    @test 0 < t < tmax
    half = cumulative_lineages(tmax, TN, rho) / 2
    @test cumulative_lineages(t, TN, rho) ≈ half rtol=0.05
    # degenerate intervals resolve no split
    @test midlineagetime(0, 0, TN, rho) == 0
    @test midlineagetime(100, 50, TN, rho) == 0
end

@testset "Compare models" begin
    m1 = FitResult(1,0,0,0,[],[],"",false,-1e4,-1e4,trues(0),nothing)
    m2 = FitResult(2,0,0,0,[],[],"",true,-1e3,-1e3,trues(0),nothing)
    m3 = FitResult(3,0,0,0,[],[],"",true,-1e2,-1e2,trues(0),nothing)
    m4 = FitResult(4,0,0,0,[],[],"",true,-1e1,-1e1,trues(0),nothing)
    flags = [true,true,true,false]
    b = compare_models([m1, m2, m3, m4], flags)
    @test !isnothing(b)
end

function get_sim(params::Vector, mu::Float64, rho::Float64)

    tnv = map(x -> ceil(Int, x), params)
    pop = VaryingPopulation(; TNvector = tnv, mutation_rate = mu, recombination_rate = rho)

    map(IBSIterator(PopSim.SMCprimeapprox.IBDIterator(pop), mu)) do ibs_segment
        length(ibs_segment)
    end
end

@testset "Test core functionality" begin
    mu, rho, TN = mus[1], rhos[1], TNs[1]

    ibs_segments = get_sim(TN, mu, rho)
    h = adapt_histogram(ibs_segments; nbins = 200)
    @test length(h.weights) == 200
    @test h.weights[end] > 0

    wcoarse = IBSpector.map_fine_to_coarse(h.weights, h.edges[1], h.edges[1])
    @test all(wcoarse .== h.weights)

    fop = FitOptions(sum(ibs_segments), length(ibs_segments), mu, rho)
    stat = pre_fit!(fop, h, 2)
    @test isassigned(stat, 1)
    stat = stat[1]

    ts = timesplitter(h, get_para(stat), fop; frame = 10)
    @test length(ts) >= 1

    fop = FitOptions(sum(ibs_segments), length(ibs_segments), mu, rho; msub=6, nfin=4, ntail=8)
    # `iters` is gone: the correction loop count is now derived from `ramp`,
    # and the optimiser budget is what a smoke test can bound.
    res = demoinfer(ibs_segments, 1:length(TN)÷2, mu, rho;
        nbins = 30, th_discr = 30, maxiters = 100, maxtime = 10
    )
    @test length(res.chains) == length(TN)÷2
    @test length(res.resid) == length(TN)÷2
    @test all(length.(res.chains) .>= 1)
    @test all(length.(res.corrections) .>= 1)
    @test all(length.(res.lls) .>= 1)
    @test all(length.(res.resid) .>= 1)
    @test !any(isinf.(evd.(res.fits)))
    best = compare_models(res.fits)
    @test !isnothing(best)
    @test !any(best.opt.at_lboundary)
    @test !any(best.opt.at_uboundary[2:end])
    covar = get_covar(best)
    # correctestimate! is gone: getFitResult now recomputes the smcp branch's
    # Hessian itself, through getHessian(edges, counts, options, TN).
    chain = sample_model_epochs(fop, h, best; nsamples = 10)
    fl = flags(best)

    resid = compute_residuals(h, mu, rho, TN)
    @test !any(isnan.(resid))
    resid = compute_residuals(h, mu, rho, TN; naive=false)
    @test !any(isnan.(resid))
    ws = integral_ws(h.edges[1], mu, TN)
    @test !any(isnan.(ws))
    @test !any(ws .< 0)
    resid = compute_residuals(h, ws./diff(h.edges[1]))
    @test !any(isnan.(resid))
    p = residstructure(h.weights, ws)

    ibs2 = get_sim(TN, mu, rho)
    h2 = Histogram(h.edges)
    append!(h2, ibs2)
    resid2 = compute_residuals(h, h2)
    @test !any(isnan.(resid2))
end

@testset "Test refine_model!" begin
    mu, rho, TN = mus[1], rhos[1], TNs[2]

    ibs_segments = get_sim(TN, mu, rho)
    h = adapt_histogram(ibs_segments; nbins = 200)
    fop = FitOptions(sum(ibs_segments), length(ibs_segments), mu, rho)

    # seed with a cheap two epochs model
    fits = pre_fit!(fop, h, 2)
    seed = get_para(fits[end])
    nepochs0 = length(seed) ÷ 2

    fits = refine_model!(fop, h, seed)
    @test fits isa Vector{FitResult}
    @test 1 < length(fits) <= nepochs0 + 1
    neps = [f.nepochs for f in fits]
    # ordered from the most to the least complex model, the input being the last
    @test all(diff(neps) .== -1)
    @test neps[1] <= 2nepochs0
    @test neps[end] == nepochs0
    # each model nests the following one
    @test fits[1].lp >= fits[end].lp
    for f in fits
        # L and the Ns are estimated, the Ts are held fixed at the proposal
        @test f.free[1]
        @test f.free[2:2:end] == trues(f.nepochs)
        @test f.free[3:2:end] == falses(f.nepochs - 1)
        @test f.para[3:2:end-1] == f.opt.init[3:2:end-1]
        @test !isnothing(f.opt.hess)
        @test !isnothing(flags(f))
    end
    @test fop.nepochs == fits[end].nepochs
    @test isonlyN(fop)
end

if LOCAL
    @testset "fitting procedure" begin
        @testset "exhaustive pre-fit $(length(TN)÷2) epochs,  mu $mu, rho $rho" for (mu,rho,TN) in itr
            ibs_segments = get_sim(TN, mu, rho)
            h = adapt_histogram(ibs_segments)
            Ltot = sum(ibs_segments)
            fop = FitOptions(Ltot, length(ibs_segments), mu, rho; maxnts = 8, force = false, locut = 1)
            fits = pre_fit!(fop, h, 8)
            nepochs = length(fits)
            bestll = argmax(i->fits[i].lp, 1:nepochs)
            residuals = compute_residuals(h, mu, rho, get_para(fits[bestll]); naive = true)
            @test abs(mean(residuals)) < 3/sqrt(length(residuals))
            @test std(residuals) - 1 < 3/sqrt(length(residuals))
        end

        @testset "Iterative fit" begin
            mu, rho, TN = mus[1], rhos[1], TNs[3]
            ibs_segments = get_sim(TN, mu, rho)
            h = adapt_histogram(ibs_segments)
            Ltot = sum(ibs_segments)
            fop = FitOptions(Ltot, length(ibs_segments), mu, rho)
            pfits = pre_fit!(fop, h, 5)
            res = demoinfer(h, 4:5, fop)
            best = compare_models(res.fits)
            @test !isnothing(best)
            @test best.nepochs == 5
            m = 2
            for i in 1:length(res.chains[m])
                p = get_para(res.chains[m][i])
                wth = integral_ws(h.edges[1], mu, p)
                ws = wth .+ res.corrections[m][i]
                ws = max.(0,ws)
                resid = (h.weights .- ws) ./ sqrt.(h.weights .+ ws)
                resid[ws .== 0 .& h.weights .== 0] .= 0
                resid = resid[fop.locut:end]
                @test std(resid) - 1 < 3/sqrt(length(resid))
            end
        end
    end
end