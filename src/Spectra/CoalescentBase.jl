module CoalescentBase
using Distributions
using Random

export getts, getns,
    Nt, cumcr,
    coalescent, extbps,
    lineages, cumulative_lineages, crediblehistory,
    sampleN, quantilesN


function getts(TN::AbstractVector{T}, i::Int) where T
    # TN = [L, N0, T1, N1, T2, N2, ...]
    # returns the ordered times in reverse order
    (i < 1 || i > length(TN) ÷ 2) && throw(ArgumentError("index out of bounds"))
    s = zero(T)
    for j in 2:i
        s += TN[end-1-2*(j-2)]
    end
    return s
end

function getns(TN::AbstractVector{T}, i::Int) where T
    # TN = [L, N0, T1, N1, T2, N2, ...]
    # returns the ordered population sizes in reverse order
    (i < 1 || i > length(TN) ÷ 2) && throw(ArgumentError("index out of bounds"))
    return TN[end-2*(i-1)]
end

function Nt(t::Real, TN::AbstractVector{<:Real})
    pnt = 1
    while pnt < length(TN)÷2 && getts(TN, pnt+1) <= t
        pnt += 1
    end
    return getns(TN, pnt)
end

# t is assumed to be sorted
function Nt(t::AbstractVector{<:Real}, TN::AbstractVector{<:Real})
    res = zeros(length(t))
    pnt = 1
    for i in eachindex(t)
        while pnt < length(TN)÷2 && getts(TN, pnt+1) <= t[i]
            pnt += 1
        end
        res[i] = getns(TN, pnt)
    end
    return res
end

# cumulative coalescence rate between t1 and t2
function cumcr(t1::Real, t2::Real, TN::AbstractVector{<:Real})
    @assert t2 >= t1
    @assert t1 >= 0
    pnt = 1
    while pnt < length(TN)÷2 && getts(TN, pnt+1) <= t1
        pnt += 1
    end
    c = 0.
    while pnt < length(TN)÷2 && getts(TN, pnt) < t2
        gens = min(t2, getts(TN, pnt+1)) - max(t1, getts(TN, pnt))
        c += gens / getns(TN, pnt)
        pnt += 1
    end
    if getts(TN, pnt) < t2
        gens = t2 - max(t1, getts(TN, pnt))
        c += gens / getns(TN, pnt)
        pnt += 1
    end
    return c
end

# t2 assumed to be sorted
function cumcr(t1::Real, t2::AbstractVector{<:Real}, TN::AbstractVector{<:Real})
    res = zeros(length(t2))
    tmp = 0.
    tprev = t1
    for i in eachindex(t2)
        tmp += cumcr(tprev, t2[i], TN)
        res[i] = tmp
        tprev = t2[i]
    end
    return res
end

"""
    coalescent(t::Number, TN::Vector)

Calculate the probability of coalescence at time `t` generations in
the past.

It is computed for two alleles in the absence of recombinaiton 
and for a demographic scenario encoded in `TN`. The distribution 
of such `t`s is geometric as introduced by Hudson and Kingman.

### References
"""
function coalescent(t::Real, TN::AbstractVector{<:Real})
    return exp(-cumcr(0, t, TN)/2) / (2 * Nt(t, TN))
end

"""
    extbps(t::Real, TN::AbstractVector{<:Real})

Calculate the the expected number of basepairs that still have to
reach coalescence at time `t` generations in the past. 

The demographic scenario is encoded in `TN`.

### Reference
"""
function extbps(t::Real, TN::AbstractVector{<:Real})
    return round(TN[1]*exp(-cumcr(0, t, TN)/2))
end

"""
    lineages(t::Real, TN::AbstractVector{<:Real}, rho::Real; k::Real = 0)
    cumulative_lineages(t::Real, TN::AbstractVector{<:Real}, rho::Real; k::Real = 0)

Calculate the expected (cumulative) number of genomic segments which are Identical by Descent 
and coalesce (up to) at time `t` generations in the past having a genomic length longer
than `k` basepairs.

The demographic scenario is encoded in `TN` and the recombination rate is `rho`
in unit per bp per generation.
"""
function lineages(t::Real, TN::AbstractVector{<:Real}, rho::Real; k::Real = 0)
    return 2 * TN[1] * rho * t * exp(-2 * rho * t * k - cumcr(0, t, TN)/2) / (2 * Nt(t, TN))
end

# t assumed to be sorted
function lineages(t::AbstractVector{<:Real}, TN::AbstractVector{<:Real}, rho::Real; k::Real = 0)
    return 2 * TN[1] * rho .* t .* exp.(-2 * rho .* t .* k .- cumcr(0, t, TN)/2) ./ (2 * Nt(t, TN))
end

function cumulative_lineages(t::Real, TN::AbstractVector{<:Real}, rho::Real; k::Real = 0)
    s = 0.
    cum = 0.
    pnt = 1
    while pnt < length(TN)÷2 && getts(TN, pnt+1) < t
        pnt += 1
        t_ = getts(TN, pnt)
        aem = 1/2getns(TN, pnt-1)
        aep = 1/2getns(TN, pnt)
        cum += (t_ - getts(TN, pnt-1)) / 2getns(TN, pnt-1)
        s += ( 
            t_*(aep/(aep+2rho*k) - aem/(aem+2rho*k)) 
            + (aep/(aep+2rho*k)^2 - aem/(aem+2rho*k)^2)
        ) * exp(-2rho * k * t_ - cum)
    end
    ae = 1/2getns(TN, pnt)
    cum += (t - getts(TN, pnt)) / 2getns(TN, pnt)
    s -= ( 
        t*ae/(ae+2rho*k) + ae/(ae+2rho*k)^2
    ) * exp(-2rho * k * t - cum)
    s += 2 * getns(TN, 1) / (1 + 4*getns(TN, 1) * rho * k)^2
    return s * 2 * TN[1] * rho
end

function lin2N(t::AbstractVector{<:Real}, TN::AbstractVector{<:Real}, rho::Real)
    lint = lineages(t, TN, rho)
    int = 0.
    N = zeros(length(t))
    for i in eachindex(t)
        l = lint[i]
        int += l / 2t[i]
        N[i] = t[i] * (TN[1] * rho - int) / max(l, eps())
    end
    return N
end

function varN(t::AbstractVector{<:Real}, TN::AbstractVector{<:Real}, rho::Real)
    L = TN[1]
    lin = lineages(t, TN, rho)
    int = 0.
    s = zeros(length(t))
    cums = cumcr(0, t, TN)
    for i in eachindex(t)
        l = lin[i]
        int += l / (4 * t[i]^2)
        s[i] = t[i]^2 / max(l^2, eps()) * (int + (L*rho)^2 * exp(-cums[i]) / max(l, eps()))
    end
    return s
end

function harmeanN(t1::Real, t2::Real, TN::AbstractVector{<:Real})
    s = 0.
    for t in t1:t2
        s += 1 / Nt(t, TN)
    end
    return (t2 - t1 + 1) / s
end

function sampleH(TN::AbstractVector{<:Real}, cov::AbstractMatrix{<:Real}, nsamples)
    mvn = MvNormal(TN, cov)
    tns = [rand(mvn) for _ in 1:nsamples]
    for i in 1:nsamples
        again = 0
        while again < 10 && (any(!isfinite, tns[i]) || any(tns[i] .<= 0))
            rand!(mvn, tns[i])
            again += 1
        end
        tns[i][tns[i] .<= 0] .= 10
    end
    return tns
end

"""
    sampleN(t::AbstractVector{<:Real}, TN::AbstractVector{<:Real}, cov::AbstractMatrix{<:Real}, rho::Real; nsamples = 10)

Sample a demographic history with an expected number of coalescence events per generation given by `TN`.
Specifically, `TN` and it's covariance matrix `cov` are used to sample `TN` vectors which then define
the expected distribution of coalescence events per generation for each history.
The recombination rate `rho` is per bp per generation.
`t` is a unit range of generations in the past.
"""
function sampleN(t::AbstractVector{<:Real}, TN::AbstractVector{<:Real}, cov::AbstractMatrix{<:Real},
    rho::Real; nsamples = 10
)
    tns = sampleH(TN, cov, nsamples)
    return sampleN(t, tns, rho)
end

"""
    sampleN(t::AbstractVector{<:Real}, tns::Vector{<:AbstractVector{<:Real}}, rho::Real)

Alternatively to the above, a vector of `TN` vectors can be directly provided,
which are then used to sample demographic histories as above.
"""
function sampleN(t::AbstractVector{<:Real}, tns::Vector{<:AbstractVector{<:Real}}, rho::Real)
    nsamples = length(tns)
    int = zeros(nsamples)
    N = zeros(nsamples, length(t))
    l = zeros(nsamples, length(t))
    ccr = zeros(nsamples)
    for i in eachindex(t)
        for j in 1:nsamples
            ccr[j] += 1 / (2 * Nt(t[i], tns[j]))
            N[j, i] = 2 * tns[j][1] * rho * t[i] * exp(-ccr[j]) / (2 * Nt(t[i], tns[j]))
        end
        l[:, i] .= rand.(Poisson.(N[:,i]))
        int .+= l[:, i] ./ 2t[i]
        N[:, i] .= t[i] * (tns[1][1] * rho  .- int) ./ max.(l[:, i], eps())
    end
    return N, l
end

"""
    quantilesN(t::AbstractVector{<:Real}, TN::AbstractVector{<:Real}, cov::AbstractMatrix{<:Real}, rho::Real; qs=[0.025,0.975], nsamples = 10)
    quantilesN(t::AbstractVector{<:Real}, tns::Vector{<:AbstractVector{<:Real}}, rho::Real; qs=[0.025,0.975])

Works as the above `sampleN` functions but returns quantiles of the sampled demographic
histories instead of the samples themselves (i.e. it's faster and uses less memory).
"""
function quantilesN(t::AbstractVector{<:Real}, TN::AbstractVector{<:Real}, cov::AbstractMatrix{<:Real}, 
    rho::Real; qs=[0.025,0.975], nsamples = 10
)
    tns = sampleH(TN, cov, nsamples)
    return quantilesN(t, tns, rho; qs=qs)
end

function quantilesN(t::AbstractVector{<:Real}, tns::Vector{<:AbstractVector{<:Real}}, rho::Real;
    qs=[0.025,0.975]
)
    nsamples = length(tns)
    int = zeros(nsamples)
    l = zeros(nsamples)
    lint = zeros(nsamples)
    ccr = zeros(nsamples)
    Ntemp = zeros(nsamples)
    N = zeros(length(qs), length(t))
    for i in eachindex(t)
        for j in 1:nsamples
            ccr[j] += 1 / (2 * Nt(t[i], tns[j]))
            lint[j] = 2 * tns[j][1] * rho * t[i] * exp(-ccr[j]) / (2 * Nt(t[i], tns[j]))
        end
        l .= rand.(Poisson.(lint))
        int .+= l ./ 2t[i]
        Ntemp .= t[i] * (tns[1][1] * rho  .- int) ./ max.(l, eps())
        for j in eachindex(qs)
            N[j, i] = quantile(Ntemp, qs[j])
        end
    end
    return N
end

"""
    crediblehistory(TN::AbstractVector{<:Real}, rho::Real; tmax=1e7, level=0.975)

Return the minimum (and maximum) time before which (and after which) the 
expected cumulative number of coalescence events is 0 with at least confidence `level`
according to a Poisson distribution with mean given by the cumulative number of lineages
in the respective tail for a demographic scenario encoded in `TN` and recombination rate `rho`.
The times are in generations.
"""
function crediblehistory(TN::AbstractVector{<:Real}, rho::Real; tmax=1e7, level=0.975)
    mint = 1
    maxt = tmax
    q = 0
    t = 0
    while q == 0 && t <= tmax
        t += 1
        l = cumulative_lineages(t, TN, rho)
        q = quantile(Poisson(l), level)
    end
    mint = t
    t = tmax + 1
    ltot = cumulative_lineages(tmax, TN, rho)
    q = 0
    while q == 0 && t >= mint
        t -= 1
        l = ltot - cumulative_lineages(t, TN, rho)
        q = quantile(Poisson(l), level)
    end
    maxt = t
    return mint, maxt
end

end