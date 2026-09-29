function tcondr(r::Number, mu::Number)
    return 1 / (mu * r)
end

function timesplitter(h::Histogram, prev_para::Vector{T}, fop::FitOptions;
    frame::Number = 5
) where {T <: Number}

    # find approximate time of positive (negative) deviation from previous fit
    r = midpoints(h.edges[1])
    residuals = compute_residuals(h, fop.mu, fop.rho, prev_para, naive = isnaive(fop))

    found = zeros(1)
    j = fop.locut
    while j < length(residuals)
        z = j + 1
        while z < length(residuals) && residuals[j] * residuals[z] > 0
            z += 1
        end
        if z - j >= frame || (j == fop.locut) || (z == length(residuals))
            t1 = tcondr(r[j], fop.mu)
            t2 = tcondr(r[z], fop.mu)
            @debug "identified deviation " r[j] r[z]
            append!(found, t1, t2)
        end
        j = z
    end
    @debug "time splits results " found
    return found
end

function epochfinder!(init::Vector{T}, t, fop::FitOptions) where {T <: Number}
    nep = fop.nepochs - 1 # previous model
    # these are the absolute times of epochs changes
    # ordered from ancient to recent
    ts = [Spectra.getts(init,i) for i in nep:-1:1]
    split_epoch = findfirst(ts .< t)
    isnothing(split_epoch) && (split_epoch = 1)

    if split_epoch == 1
        newT = t - ts[1]
        # the floor keeps the new duration within the prior support of the
        # oldest T; when only the Ns are fitted the Ts are held fixed at the
        # proposed split, so it is used as is
        isonlyN(fop) || (newT = max(newT, 1000))
        newN = init[2]
        insert!(init, 3, newN)
        insert!(init, 3, newT)
    else
        newT1 = ts[split_epoch-1] - t
        newT2 = t - ts[split_epoch]
        newN = init[2split_epoch]
        init[2split_epoch-1] = newT1
        insert!(init, 2split_epoch, newT2)
        insert!(init, 2split_epoch, newN)
    end
    return nothing
end

"""
    midlineagetime(t0::Real, t1::Real, TN::AbstractVector{<:Real}, rho::Real; ngrid::Int = 1000)

Find the time `t` in `(t0, t1)` such that the number of lineages coalescing in
`[t0, t]` is roughly equal to the number of lineages coalescing in `[t, t1]`,
for the demographic scenario `TN` and recombination rate `rho`, see
[`cumulative_lineages`](@ref). All times are absolute, in generations before
present.

Return `0` when no such time is resolved on the logarithmic grid of `ngrid`
points used for the search.
"""
function midlineagetime(t0::Real, t1::Real, TN::AbstractVector{<:Real}, rho::Real;
    ngrid::Int = 1000
)
    t1 <= t0 && return 0.0
    cum0 = cumulative_lineages(t0, TN, rho)
    nlin = (cumulative_lineages(t1, TN, rho) - cum0) / 2
    nlin <= 0 && return 0.0
    for t in logrange(max(1, t0), t1, ngrid)
        if cumulative_lineages(t, TN, rho) - cum0 >= nlin
            return t
        end
    end
    return 0.0
end

function perturb_fit!(f::FitResult, fop::FitOptions, h::Histogram)
    f_ = deepcopy(f)
    reset_perturb!(fop)
    set_perturb!(fop, f)
    if any(fop.perturb)
        # pinit = PInit(fop)
        for fct in fop.delta.factors
            next!(fop.delta)
            setinit!(fop, f.para)
            set_perturb!(fop, f)
            pinit = getPinit(fop)
            setinit!(fop, pinit)
            f = fit_model_epochs!(fop, h; stats = false)
            if f.converged && !any(f.opt.at_lboundary[1:end-2])
                break
            end
        end
    end
    if f.lp < f_.lp || any(isnan, f.para)
        return f_
    end
    return f
end

"""
    pre_fit!(fop::FitOptions, h::Histogram, nfits; getStats=true)

Preliminarily fit `h` with an approximate model of piece-wise constant 
epochs for each number of epochs from 1 to `nfits`.

See [`FitOptions`](@ref) for how to specify them.
It modifies `fop` in place to adapt it to all the requested
epochs.
Return a vector of `FitResult`, one for each number of epochs,
see also [`FitResult`](@ref).
"""
function pre_fit!(fop::FitOptions, h::Histogram{T,1,E}, nfits::Int;
    getStats::Bool = true
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    fits = FitResult[]
    @assert nfits > 0 "number of fits has to be strictly positive"
    for i in 1:nfits
        setnepochs!(fop, i)
        if i == 1
            f = fit_model_epochs!(fop, h; stats = getStats)
        else
            ts = timesplitter(h, get_para(fits[i-1]), fop)
            if iszero(ts)
                @info "pre_fit: no split found, epoch $i"
                if !fop.force
                    return fits
                else
                    ts = [Spectra.getts(get_para(fits[i-1]), j) for j in 1:i-1]
                    push!(ts, 1e9)
                    ts[1] = 1
                    @debug ts
                    ts = sqrt.(ts[1:end-1] .* ts[2:end])
                end
            else
                filter!(t->t!=0, ts)
                push!(ts, 15.0)
                sort!(ts)
                unique!(ts)
                maxnts_ = min(fop.maxnts, length(ts))
                ts = ts[range(start=1, stop=length(ts), step=length(ts)÷maxnts_)]
            end
            fs = Vector{FitResult}(undef, length(ts))
            fops = Vector{FitOptions}(undef, length(ts))
            for j in eachindex(fops)
                fops[j] = deepcopy(fop)
            end
            @threads for j in eachindex(ts)
                init = get_para(fits[i-1])
                epochfinder!(init, ts[j], fops[j])
                setinit!(fops[j], init)
                f = fit_model_epochs!(fops[j], h; stats = false)
                fs[j] = f
            end
            lps = map(f->f.lp, fs)
            f = fs[argmax(lps)]
            @debug "best " ts[argmax(lps)] f.lp f.converged
            f = perturb_fit!(f, fop, h)
            if getStats
                p = 1 .+ (rand(length(f.para)) .- 0.5) * 0.001
                setinit!(fop, get_para(f) .* p) # perturb slightly to avoid linesearch failure
                f = fit_model_epochs!(fop, h)
            end
            if (f.lp < fits[i-1].lp) && f.converged
                @error "epoch $i ll not improved. Please report an issue: $(f.lp) < $(fits[i-1].lp)"
            end
            @assert all(!isnan, f.para) """
                NaN parameters $(f.para)
                $(f.lp)
                $(f.opt.init)
                $(fop.upp)
                $(fs[argmax(lps)])
                $(f.para)
            """
        end
        push!(fits, f)
    end
    return fits
end

"""
    refine_model!(fop::FitOptions, h::Histogram, TN::AbstractVector{<:Real})

Refine the demographic model `TN` on the observed histogram `h` by splitting its
epochs, fitting only the genome length and population sizes, see [`fitNs!`](@ref).

Every epoch `[t0, t1)` of `TN` proposes one new epoch boundary, placed where half
of the lineages coalescing within that epoch have coalesced, see
[`midlineagetime`](@ref). The most complex model keeps all of these splits, i.e.
it has twice the epochs of `TN`. Splits are then dropped one at a time, starting
from the one lying in the epoch with the fewest coalescing lineages, see
[`cumulative_lineages`](@ref), down to the model keeping only the split of the
epoch with the most lineages. Every model of this ladder is fitted, together with
`TN` itself.

Only the total genome length and the population sizes are ever estimated: all
epoch durations stay fixed at the proposed values. No model is discarded, the
selection is left to the caller, see [`evd`](@ref).

`fop` is modified in place to describe the last returned model (`nepochs`, `init`
and `onlyN`). Return a vector of `FitResult` with stats computed, ordered from
the most to the least complex model, the input `TN` being the last entry, see
also [`FitResult`](@ref).
"""
function refine_model!(fop::FitOptions, h::Histogram{T,1,E}, TN::AbstractVector{<:Real}
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    nepochs0 = length(TN) ÷ 2
    @assert nepochs0 > 0 "TN has to contain at least one epoch"
    tmax = 1e9 # the oldest epoch has no upper boundary

    # one candidate split and the number of coalescing lineages
    # for each epoch of the input model
    bounds = [Spectra.getts(TN, i) for i in 1:nepochs0]
    ts = Float64[]
    ws = Float64[]
    for i in 1:nepochs0
        t0 = bounds[i]
        t1 = i < nepochs0 ? bounds[i+1] : tmax
        t = midlineagetime(t0, t1, TN, fop.rho)
        if t0 + 1 < t < t1 - 1 && !(t in ts)
            push!(ts, t)
            push!(ws, cumulative_lineages(t1, TN, fop.rho) -
                cumulative_lineages(t0, TN, fop.rho))
        end
    end
    # splits are dropped starting from the epoch with the fewest lineages
    perm = sortperm(ws)
    ts = ts[perm]
    nsplits = length(ts)
    nsplits == 0 && @info "refine_model: no split found, $nepochs0 epochs"
    @debug "refine_model: proposed splits " ts ws[perm]

    # the k-th model keeps the k splits lying in the richest epochs
    fops = Vector{FitOptions}(undef, nsplits)
    for k in 1:nsplits
        fops[k] = deepcopy(fop)
        init = collect(float.(TN))
        for (n, t) in enumerate(sort(ts[nsplits-k+1:end]))
            setnepochs!(fops[k], nepochs0 + n)
            epochfinder!(init, t, fops[k])
        end
        setonlyN!(fops[k], true) # before setinit!, to keep the proposed Ts
        setinit!(fops[k], init)
    end

    setnepochs!(fop, nepochs0)
    setonlyN!(fop, true)
    setinit!(fop, TN)
    fits = Vector{FitResult}(undef, nsplits + 1)
    fits[end] = fitNs!(fop, h)
    @threads for k in 1:nsplits
        fits[nsplits-k+1] = fitNs!(fops[k], h)
    end

    for f in fits
        @assert all(!isnan, f.para) """
            NaN parameters $(f.para)
            $(f.lp)
            $(f.opt.init)
        """
    end
    for i in 1:nsplits
        # each model nests the following, less complex, one
        if fits[i].lp < fits[i+1].lp
            @error "refine_model: ll not improved with $(fits[i].nepochs) epochs. Please report an issue"
        end
    end
    return fits
end