using IBSpector
using IBSpector: npar, setinit!, initialize!, fit_model_epochs!, PInit, 
    setnepochs!, timesplitter, integral_ws, next!,
    reset_perturb!, perturb_fit!, residstructure, compute_residuals
using PopSim
using HistogramBinnings
using Distributions
using StatsBase, StatsAPI
using Test
using IBSpector.Spectra

include("Aqua.jl")
include("spectra.jl")

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

@testset "Test PInit" begin
    fop = FitOptions(30, 10, 1.0, 1.0)
    p = PInit(fop)
    @test fop.delta.state == 0
    @test length(p) == npar(fop)
    @test all(p .== fop.init)
    @test all(fop.perturb .== false)
    fop.perturb .= trues(npar(fop))
    setinit!(fop, ones(npar(fop)))
    next!(fop.delta)
    @test length(p) == npar(fop)
    @test any(p .!= fop.init)
    @test all(fop.low .<= p .<= fop.upp)
    @test fop.delta.state == 1
    reset_perturb!(fop)
    @test all(fop.perturb .== false)
end

@testset "Test fit" begin
    h = Histogram([1,2,3,4])
    append!(h, [1,1,1,2,3,1,2])
    fop = FitOptions(11, 7, 1.0, 1.0; order = 2, ndt = 10)
    f = fit_model_epochs!(fop, h.edges[1], h.weights, Val(true))
    f = fit_model_epochs!(fop, h)
    @test f.converged
    perturb_fit!(f, fop, h)
    IBSpector.setnaive!(fop, false)
    IBSpector.setOptimOptions!(fop, g_tol=1e-3)
    fit_model_epochs!(fop, h)
end

@testset "Compare models" begin
    m1 = FitResult(1,0,0,0,[],[],"",false,-1e4,-1e4,nothing)
    m2 = FitResult(2,0,0,0,[],[],"",true,-1e3,-1e3,nothing)
    m3 = FitResult(3,0,0,0,[],[],"",true,-1e2,-1e2,nothing)
    m4 = FitResult(4,0,0,0,[],[],"",true,-1e1,-1e1,nothing)
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

    fop = FitOptions(sum(ibs_segments), length(ibs_segments), mu, rho)
    stat = pre_fit!(fop, h, 2)
    @test isassigned(stat, 1)
    stat = stat[1]

    ts = timesplitter(h, get_para(stat), fop; frame = 10)
    @test length(ts) >= 1

    fop = FitOptions(sum(ibs_segments), length(ibs_segments), mu, rho; order=2, ndt=10)
    res = demoinfer(ibs_segments, 1:length(TN)÷2, mu, rho;
        iters = 1, nbins=10
    )
    @test length(res.chains) == length(TN)÷2
    @test length(res.yth) == length(TN)÷2
    @test all(length.(res.chains) .>= 1)
    @test all(length.(res.corrections) .>= 1)
    @test all(length.(res.deltas) .>= 1)
    @test all(length.(res.yth) .>= 1)
    @test !any(isinf.(evd.(res.fits)))
    best = compare_models(res.fits)
    @test !isnothing(best)
    @test !any(best.opt.at_lboundary)
    @test !any(best.opt.at_uboundary[2:end])
    covar = get_covar(best)
    fcor = correctestimate!(fop, best, h)
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
    p = residstructure(resid)

    ibs2 = get_sim(TN, mu, rho)
    h2 = Histogram(h.edges)
    append!(h2, ibs2)
    resid2 = compute_residuals(h, h2)
    @test !any(isnan.(resid2))
end

# Test the fitting procedure on multiple simulated datasets. 
# This is a long test, so we allow it to be skipped by setting 
# the environment variable OMIT_LONG_TESTS.

if haskey(ENV, "OMIT_LONG_TESTS")
    @info "Omitting long tests"
else

    @testset verbose = true "fitting procedure" begin
        @testset "exhaustive pre-fit $(length(TN)÷2) epochs,  mu $mu, rho $rho" for (mu,rho,TN) in itr
            ibs_segments = get_sim(TN, mu, rho)
            h = adapt_histogram(ibs_segments; nbins = 200)
            Ltot = sum(ibs_segments)
            fop = FitOptions(Ltot, length(ibs_segments), mu, rho; maxnts = 8, force = false)
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
            h = adapt_histogram(ibs_segments; nbins = 200)
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
                @test std(resid) - 1 < 3/sqrt(length(resid))
            end
        end
    end

end