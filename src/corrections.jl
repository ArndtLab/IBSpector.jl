function ramp(iter, mu, rho)
    min(mu/10 * iter, rho, mu)
end

"""
    demoinfer(segments::AbstractVector{<:Integer}, epochrange::AbstractRange{<:Integer}, mu::Float64, rho::Float64; kwargs...)

Make an histogram with IBS `segments` and infer demographic models with
piece-wise constant epochs where the number of epochs is in `epochrange`.

Return a named tuple which contains the fields:
- `fits`: a vector of `FitResult` (see [`FitResult`](@ref))
- `h_obs`: the histogram of the observed segments.
- `resid`: a vector of vectors of residuals, one for each model.
- `p`: a vector of p-values for the autocorrelation of residuals, one for each model.
  These are Ljung-Box tests, and might be affected by the number of bins included in
  the fit relative to the estimated parameters. Additionally in the tail counts are low,
  then a Poisson model implies that many (autocorrelated) zeros can be expected.
- `chains`: a vector of vectors of `FitResult`, one for each iteration
  of the correction procedure, and one chain per model. For diagnostics.
- `corrections`: a vector of vectors of corrections, one for each iteration
  of the correction procedure, and one vector of corrections per model.
  Corrections are histogram counts, therefore they have the same shape.
  For diagnostics.
- `lls`: a vector of vectors of log-likelihoods, one for each iteration and
  for each model. For diagnostics.
- `prefin`: a vector of `FitResult` for the pre-fit stage, one for each model.
  For diagnostics.
- `prefend`: a vector of `FitResult` for the pre-fit stage, one for each model,
  after the last correction iteration. For diagnostics.

# Optional Arguments
- `fop::FitOptions = FitOptions(sum(segments), mu, rho)`: the fit options, see [`FitOptions`](@ref).
- `lo::Int=1`: The lowest segment length to be considered in the histogram
- `hi::Int=50_000_000`: The highest segment length to be considered in the histogram
- `nbins::Int=200`: The number of bins to use in the histogram.
- `th_discr::Int=800`: number of discrete points for the numerical integration over the IBS length.
"""
function demoinfer(segments::AbstractVector{<:Integer}, epochrange::AbstractRange{<:Integer},
    mu::Float64, rho::Float64;
    fop::FitOptions = FitOptions(sum(segments), length(segments), mu, rho),
    lo::Int = 1, hi::Int = 50_000_000, nbins::Int = 200,
    kwargs...
)
    h = adapt_histogram(segments; lo, hi, nbins)
    if sum(segments) != fop.Ltot
        @warn "inconsistent Ltot and segments, taking sum(segments)"
        fop.Ltot = sum(segments)
    end
    return demoinfer(h, epochrange, fop; kwargs...)
end

"""
    demoinfer(h::Histogram, epochrange, fop::FitOptions; th_discr=800, maxiters=6000, maxtime=7200)
    demoinfer(h, epochs, fop; th_discr=800, maxiters=6000, maxtime=7200)

Take an histogram of IBS segments, fit options, and infer demographic models with
piece-wise constant epochs where the number of epochs is in `epochrange` or `epochs`.
Return a named tuple as above.

If `epochs` is passed (an integer), then it fits only the model with that number of epochs.
In this case the returned named tuple contains only one element per field, instead of a vector.
"""
function demoinfer(h_obs::Histogram{T,1,E}, epochrange::AbstractRange{<:Integer}, fop_::FitOptions;
    kwargs...
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    @assert length(epochrange) > 0
    results = Vector{NamedTuple}(undef, length(epochrange))
    @threads for i in eachindex(epochrange)
        results[i] = demoinfer(h_obs, epochrange[i], fop_; kwargs...)
    end
    return (;
        fits = map(r->r.f, results),
        prefin = map(r->r.prefin, results),
        prefend = map(r->r.prefend, results),
        chains = map(r->r.chain, results),
        corrections = map(r->r.corrections, results),
        h_obs = results[1].h_obs,
        resid = map(r->r.resid, results),
        p = map(r->r.p, results),
        lls = map(r->r.lls, results)
    )
end

function demoinfer(h_obs::Histogram{T,1,E}, epochs::Int, fop_::FitOptions;
    th_discr::Int = 800, maxiters::Int = 6000, maxtime::Int = 7200
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    @assert !isempty(h_obs.weights) "histogram is empty"
    @assert epochs > 0 "epochs must be strictly positive"
    @assert th_discr >= length(h_obs.weights) "th_discr must be at least the number of bins in the histogram"
    @assert fop_.locut < length(h_obs.weights) "locut must be less than the number of bins in the histogram"

    h_mod = Histogram(h_obs.edges)
    h_mod.weights .= h_obs.weights

    fop = deepcopy(fop_)
    lo_edge = h_obs.edges[1].edges[1]
    hi_edge = h_obs.edges[1].edges[end]
    eth = CustomEdgeVector(; lo = lo_edge, hi = hi_edge - 1, nbins = th_discr)
    rs = midpoints(eth)
    bag = IntegralArrays(
        timegrid(epochs; msub = fop.msub, nfin = fop.nfin, ntail = fop.ntail),
        length(h_obs.weights),
        rs,
        eth,
        Val{2epochs},
        2,
    )
    fastbag = IntegralArrays(
        timegrid(epochs; msub = fop.msub, nfin = 6, ntail = fop.ntail),
        length(h_obs.weights),
        midpoints(h_obs.edges[1]),
        h_obs.edges[1],
        Val{2epochs},
        2,
    )

    chain = []
    corrections = []
    lls = []

    corr = zeros(Float64, length(h_obs.weights))
    warmup = 1
    for i in 100:-1:2
        nx = ramp(i, fop.mu, fop.rho)
        if nx != ramp(i-1, fop.mu, fop.rho)
            warmup = i
            break
        end
    end
    for iter in 1:warmup
        fits = pre_fit!(fop, h_mod, epochs; getStats=false)
        f = fits[end]
        if f.nepochs != epochs
            push!(chain, f)
            break
        end
        setinit!(fop, f.para)
        push!(chain, f)
        push!(corrections, corr)

        rho = ramp(iter, fop.mu, fop.rho)
        mldsmcp!(bag, fop.mu, rho, fop.init)
        map_fine_to_coarse!(bag, h_obs.edges[1], eltype(fop.init))

        w = get_tmp(bag.wcoarse, eltype(fop.init))
        ll = llsmcp(w, h_obs.weights, fop.locut)
        push!(lls, ll)

        h_mod.weights .= h_obs.weights

        corr = correcthistogram!(h_mod.weights, h_obs.edges[1], fop.mu, fop.locut, w, fop.init)
    end

    setnaive!(fop, false)
    fop.diffcache = fastbag
    N0 = 1/(4*fop.mu*(fop.Ltot/sum(h_obs.weights)))

    f = chain[1]
    init = get_para(f)
    regularizetn!(init, N0)
    setinit!(fop, init)
    setOptimOptions!(fop; maxiters=6000, maxtime=1800)
    prefin = fit_model_epochs!(fop, h_obs; stats=false)

    f = chain[end]
    init = get_para(f)
    regularizetn!(init, N0)
    setinit!(fop, init)
    setOptimOptions!(fop; maxiters=6000, maxtime=1800)
    prefend = fit_model_epochs!(fop, h_obs; stats=false)

    best = prefin.lp > prefend.lp ? prefin : prefend

    setinit!(fop, best.para)
    fop.diffcache = bag
    setOptimOptions!(fop; maxiters, maxtime)
    best = fit_model_epochs!(fop, h_obs)

    resid = compute_residuals(h_obs, fop.mu, fop.rho, best.para; naive=false,
        msub = fop.msub, nfin = fop.nfin, ntail = fop.ntail, finebins = th_discr
    )
    dof = length(get_para(f))
    ze = length(resid)
    for j in fop.locut:length(resid)-1
        if h_obs.weights[j] == 0
            ze = j
            break
        end
    end
    lag = max(10, length(resid[fop.locut:ze]) ÷ 5, dof+5)
    p = pvalue(LjungBoxTest(resid[fop.locut:ze], lag, dof))

    (;
        f = best,
        prefin,
        prefend,
        chain,
        corrections,
        h_obs,
        resid,
        p,
        lls
    )
end

function correcthistogram!(weights::AbstractVector{<:Real}, edges::AbstractVector{<:Real},
    mu::Real, locut::Int, wth::AbstractVector{<:Real}, para::AbstractVector{<:Real}
)
    @assert length(wth) == length(weights)
    weightsnaive = integral_ws(edges, mu, para)
    corr = wth .- weightsnaive
    corr[1:locut-1] .= 0.
    lim = findfirst(corr .> weights)
    if isnothing(lim)
        lim = length(corr) + 1
    end
    corr[lim:end] .= 0.
    temp = weights .- corr
    temp .= round.(Int, temp)
    weights .= max.(temp, 0)
    @assert all(isfinite, weights)
    @assert all(!isnan, weights)
    return corr # just diagnostic, maybe drop
end

function regularizetn!(para, N0; minrel = 0.1, maxrel = 10.0)
    sumt = 0
    for i in length(para):-2:4
        if para[i] < minrel * N0
            para[i] = minrel * N0
        elseif para[i] > maxrel * N0
            para[i] = maxrel * N0
        end
        if para[i-1] < minrel * sumt
            para[i-1] = sumt
        end
        sumt += para[i-1]
    end
    return nothing
end

"""
    compare_models(models[, mask])

Compare the models parameterized by `FitResult`s and return the best one.
Takes an iterable of `FitResult` as input and optionally a boolean mask
to reflect prior knowledge on models to discard.
"""
function compare_models(models, mask=trues(length(models)))
    ms = copy(models)
    ms = ms[mask]
    if isempty(ms)
        @warn "none of the models is meaningful"
        return nothing
    end
    best = 1
    lp = ms[1].lp
    ev = evd(ms[1])
    monotonic = true
    for i in eachindex(ms)
        if evd(ms[i]) > ev && ms[i].lp >= lp
            best = i
            lp = ms[i].lp
            ev = evd(ms[i])
        elseif ms[i].lp < lp && monotonic
            @warn """
                log-likelihood is not monotonic in the number of epochs.
                This means that at least one likelihood optimization
                has probably failed. See diagnostics.
            """
            monotonic = false
        end
    end
    if ms[best].converged == false
        @warn "the best model's optimization did not converge"
    end
    return ms[best]
end