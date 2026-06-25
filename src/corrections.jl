function ramp(iter, mu, rho)
    min(mu/10 * iter, rho, mu)
end

"""
    demoinfer(segments::AbstractVector{<:Integer}, epochrange::AbstractRange{<:Integer}, mu::Float64, rho::Float64; kwargs...)

Make an histogram with IBS `segments` and infer demographic models with
piece-wise constant epochs where the number of epochs is `epochrange`.

Return a named tuple which contains the fields:
- `fits`: a vector of `FitResult` (see [`FitResult`](@ref))
- `chains`: a vector of vectors of `FitResult`, one for each iteration
  of the correction procedure, and one chain per model
- `yth`: a vector of vectors of the expected weights, one for each model
  and one vector of expected weights per iteration of the correction procedure
- `corrections`: a vector of vectors of corrections, one for each iteration
  of the correction procedure, and one vector of corrections per model.
  Corrections are histogram counts, therefore they have the same shape.
- `h_obs`: the histogram of the observed segments
- `h_mods`: a vector of modified histograms, one for each model, with
  higher order corrections applied.
- `ybest`: a vector of expected weights corresponding 
  to the best fit, one for each model
- `resid`: a vector of vectors of residuals, one for each model
- `p`: a vector of right-tail p-values for the autocorrelation of residuals, one for each model
- `llbest`: a vector of the best log-likelihoods, one for each model
- `deltas`: a vector of vectors of the maximum absolute difference between
  corrections in consecutive iterations, and for each model.
- `lls`: a vector of vectors of log-likelihoods, one for each iteration and
  for each model. The output estimate is the one with the highest log-likelihood.
- `conv`: a vector of booleans, one for each model, indicating whether the
  maximum iterations were reached (false) or whether the procedure 
  converged before (true).


# Optional Arguments
- `fop::FitOptions = FitOptions(sum(segments), mu, rho)`: the fit options, see [`FitOptions`](@ref).
- `lo::Int=1`: The lowest segment length to be considered in the histogram
- `hi::Int=50_000_000`: The highest segment length to be considered in the histogram
- `nbins::Int=fop.ndt`: The number of bins to use in the histogram
- `iters::Int=20`: The number of iterations to perform after warmup. It might converge earlier.
  Warmup iterations are proportional to the `rho`/`mu` ratio.
- `reltol::Float64=1e-2`: The relative tolerance to use for convergence,
  i.e. the maximum absolute difference between corrections in consecutive iterations.
  The convergence condition test this or `relchange`.
- `relchange::Float64=1e-4`: The relative change in parameters to use for convergence.
  This is the maximum relative change in parameters between consecutive iterations.
  The convergence condition test this or `reltol`.
- `th_discr::Int=fop.ndt`: number of discrete points for numerical integration when
 computing the expected weights. Default is set automatically.
"""
function demoinfer(segments::AbstractVector{<:Integer}, epochrange::AbstractRange{<:Integer}, mu::Float64, rho::Float64;
    fop::FitOptions = FitOptions(sum(segments), length(segments), mu, rho),
    lo::Int = 1, hi::Int = 50_000_000, nbins::Int = fop.ndt,
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
    demoinfer(h::Histogram, epochrange, fop::FitOptions; iters=20, reltol=1e-2, relchange=1e-4)
    demoinfer(h, epochs, fop; iters=20, reltol=1e-2, relchange=1e-4)

Take an histogram of IBS segments, fit options, and infer demographic models with
piece-wise constant epochs where the number of epochs is `epochrange`.
Return a named tuple as above.

If `epochrange` is a integer, then it fits only the model with that number of epochs.
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
        yth = map(r->r.yth, results),
        chains = map(r->r.chain, results),
        corrections = map(r->r.corrections, results),
        h_obs = results[1].h_obs,
        h_mods = map(r->r.h_mod, results),
        ybest = map(r->r.ybest, results),
        resid = map(r->r.resid, results),
        p = map(r->r.p, results),
        llbest = map(r->r.llbest, results),
        deltas = map(r->r.deltas, results),
        lls = map(r->r.lls, results),
        conv = map(r->r.conv, results)
    )
end

function map_fine_to_coarse(wth_fine, fine_edges, coarse_edges)
    wth = zeros(eltype(wth_fine), length(coarse_edges) - 1)
    k = 1  # current coarse bin index
    for j in eachindex(wth_fine)
        a = fine_edges[j]
        b = fine_edges[j + 1]
        fine_width = b - a
        # advance coarse pointer past bins that end before this fine bin starts
        while k <= length(wth) && coarse_edges[k + 1] <= a
            k += 1
        end
        # distribute weight to all coarse bins overlapping [a, b)
        kk = k
        while kk <= length(wth) && coarse_edges[kk] < b
            overlap = min(b, coarse_edges[kk + 1]) - max(a, coarse_edges[kk])
            wth[kk] += wth_fine[j] * overlap / fine_width
            kk += 1
        end
    end
    return wth
end

function demoinfer(h_obs::Histogram{T,1,E}, epochs::Int, fop_::FitOptions;
    iters::Int = 20, reltol::Float64 = 1e-2, relchange::Float64=1e-4,
    th_discr::Int = fop_.ndt
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    @assert !isempty(h_obs.weights) "histogram is empty"
    @assert epochs > 0 "epochrange has to be strictly positive"
    @assert iters > 0 "number of iterations has to be strictly positive"
    @assert th_discr >= 1 "th_discr must be at least 1"
    if fop_.mu < fop_.rho
        @warn "the method is currently designed for mu >= rho, results may be biased"
    end

    h_mod = Histogram(h_obs.edges)

    fop = deepcopy(fop_)
    lo_edge = h_obs.edges[1].edges[1]
    hi_edge = h_obs.edges[1].edges[end]
    hth = CustomEdgeVector(; lo = lo_edge, hi = hi_edge - 1, nbins = th_discr)
    rs_th = midpoints(hth)
    bag = IntegralArrays(fop.order, fop.ndt, length(rs_th), Val{2epochs})

    chain = []
    corrections = []
    yths = []
    deltas = [Inf]
    lls = []

    h_mod.weights .= h_obs.weights
    corr = zeros(Float64, length(h_obs.weights))
    conv = false
    warmup = 1
    for i in 100:-1:2
        nx = ramp(i, fop.mu, fop.rho)
        if nx != ramp(i-1, fop.mu, fop.rho)
            warmup = i
            break
        end
    end
    for iter in 1:iters+warmup
        fits = pre_fit!(fop, h_mod, epochs)
        f = fits[end]
        if f.nepochs != epochs
            push!(chain, f)
            break
        end
        init = get_para(f)
        push!(chain, f)
        push!(corrections, corr)

        rho = ramp(iter, fop.mu, fop.rho)
        mldsmcp!(bag, 1:fop.order, rs_th, hth, fop.mu, rho, init)
        yth_fine = get_tmp(bag.ys, eltype(init))
        wth_fine = yth_fine .* diff(hth)
        wth = map_fine_to_coarse(wth_fine, hth, h_obs.edges[1])
        yth = wth ./ diff(h_obs.edges[1])

        ll = llsmcp(wth, h_obs.weights, fop.locut)
        push!(lls, ll)
        push!(yths, copy(yth))

        h_mod.weights .= h_obs.weights

        weightsnaive = integral_ws(h_obs.edges[1], fop.mu, init)
        corr = wth .- weightsnaive
        corr[1:fop.locut-1] .= 0.
        lim = findfirst(corr .> h_mod.weights)
        if isnothing(lim)
            lim = length(corr) + 1
        end
        corr[lim:end] .= 0.
        temp = h_mod.weights .- corr
        temp .= round.(Int, temp)
        h_mod.weights .= max.(temp, 0)
        @assert all(isfinite, h_mod.weights)
        @assert all(!isnan, h_mod.weights)

        if iter > warmup
            deltaw = (yths[iter] .- yths[iter-1]) .* diff(h_obs.edges[1])
            delta = maximum(abs.(deltaw))
            deltapars = maximum(
                abs.((get_para(chain[end]) .- get_para(chain[end-1])) ./ get_para(chain[end-1]))
            )
            push!(deltas, delta)
            if delta < reltol || deltapars < relchange
                conv = true
                break
            end
        end
    end

    best = argmax(lls[warmup:end]) + warmup - 1
    ybest = yths[best]
    f = chain[best]
    ll = lls[best]
    resid = compute_residuals(h_obs, ybest)
    lim = findfirst(h_obs.weights .== 0)
    isnothing(lim) && (lim = length(resid))
    p = residstructure(resid[fop.locut:lim])

    temp = h_obs.weights .- corrections[best]
    temp .= round.(Int, temp)
    h_mod.weights .= max.(temp, 0)

    (;
        f,
        chain,
        yth = yths,
        corrections,
        h_obs,
        h_mod,
        ybest,
        resid,
        p,
        llbest = ll,
        deltas,
        lls,
        conv
    )
end

function correctestimate!(fop::FitOptions, fit::FitResult, h::Histogram)
    rs = midpoints(h.edges[1])
    bag = IntegralArrays(fop.order, fop.ndt, length(rs), Val{length(fit.para)}, 3)

    setnepochs!(fop, length(fit.para)÷2)
    setinit!(fop, fit.para)

    he = ForwardDiff.hessian(
        tn -> -llsmcp!(bag, rs, h.edges[1].edges, h.weights, fop.mu, fop.rho, fop.locut, tn),
        get_para(fit)
    )
    return getFitResult(he, fit.para, fit.lp, fit.opt.optim_result, fop, h.edges[1], h.weights, true)
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