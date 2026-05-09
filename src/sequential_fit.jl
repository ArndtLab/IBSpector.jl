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

function epochfinder!(init::Vector{T}, N0, t, fop::FitOptions) where {T <: Number}
    # these are the absolute times separating epochs
    # ordered from ancient to recent
    ts = [Spectra.getts(init,i) for i in fop.nepochs-1:-1:1]
    split_epoch = findfirst(ts .< t)
    isnothing(split_epoch) && (split_epoch = 1)

    if split_epoch == 1
        newT = t - ts[1]
        newT = max(newT, 1000)
        newN = N0
        insert!(init, 3, newN)
        insert!(init, 3, newT)
    else
        newT1 = ts[split_epoch-1] - t
        newT1 = max(newT1, 20)
        newT2 = t - ts[split_epoch]
        newT2 = max(newT2, 20)
        newN = init[2split_epoch]
        init[2split_epoch-1] = newT1
        insert!(init, 2split_epoch, newT2)
        insert!(init, 2split_epoch, newN)
    end
    return init
end

function perturb_fit!(f::FitResult, fop::FitOptions, h::Histogram;
    by_pass::Bool = false
)
    f_ = deepcopy(f)
    reset_perturb!(fop)
    set_perturb!(fop, f)
    if any(fop.perturb)
        pinit = PInit(fop)
        for fct in fop.delta.factors
            next!(fop.delta)
            setinit!(fop, f.para)
            set_perturb!(fop, f)
            setinit!(fop, pinit)
            f = fit_model_epochs!(fop, h; stats = false)
            if f.converged
                if by_pass
                    break
                elseif !any(f.opt.at_lboundary[1:end-2])
                    break
                end
            end
        end
    end
    if f.lp < f_.lp || any(isnan, f.para)
        return f_
    end
    return f
end

"""
    pre_fit!(fop::FitOptions, h::Histogram, nfits)

Preliminarily fit `h` with an approximate model of piece-wise constant 
epochs for each number of epochs from 1 to `nfits`.

See [`FitOptions`](@ref) for how to specify them.
It modifies `fop` in place to adapt it to all the requested
epochs.
Return a vector of `FitResult`, one for each number of epochs,
see also [`FitResult`](@ref).
"""
function pre_fit!(fop::FitOptions, h::Histogram{T,1,E}, nfits::Int
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    fits = FitResult[]
    N0 = sum(h.weights) / fop.Ltot / 4fop.mu
    @assert nfits > 0 "number of fits has to be strictly positive"
    for i in 1:nfits
        setnepochs!(fop, i)
        if i == 1
            f = fit_model_epochs!(fop, h)
        else
            ts = timesplitter(h, get_para(fits[i-1]), fop)
            if iszero(ts)
                @info "pre_fit: no split found, epoch $i"
                if !fop.force
                    return fits
                else
                    r = midpoints(h.edges[1])
                    append!(ts, tcondr(rand(r), fop.mu))
                    append!(ts, tcondr(rand(r), fop.mu))
                end
            end
            filter!(t->t!=0, ts)
            push!(ts, 15.0)
            sort!(ts)
            unique!(ts)
            maxnts_ = min(fop.maxnts, length(ts))
            ts = ts[range(start=1, stop=length(ts), step=length(ts)÷maxnts_)]
            fs = Vector{FitResult}(undef, length(ts))
            fops = Vector{FitOptions}(undef, length(ts))
            for j in eachindex(fops)
                fops[j] = deepcopy(fop)
            end
            @threads for j in eachindex(ts)
                init = get_para(fits[i-1])
                epochfinder!(init, N0, ts[j], fops[j])
                setinit!(fops[j], init)
                f = fit_model_epochs!(fops[j], h; stats = false)
                fs[j] = f
            end
            lps = map(f->f.lp, fs)
            f = fs[argmax(lps)]
            @debug "best " ts[argmax(lps)] f.lp f.converged
            f2 = perturb_fit!(f, fop, h; by_pass=true)
            setinit!(fop, get_para(f2)*0.99)
            f = fit_model_epochs!(fop, h)
            @assert all(!isnan, f.para) """
                NaN parameters $(f.para)
                $(f.lp)
                $(f.opt.init)
                $(fop.upp)
                $(fs[argmax(lps)])
                $(f2.para)
            """
        end

        push!(fits, f)
    end
    return fits
end