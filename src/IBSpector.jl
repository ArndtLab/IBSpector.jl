module IBSpector

using StatsBase, Distributions, HistogramBinnings
using LinearAlgebra, Statistics
using Turing, Optim
using StatsAPI
using Printf
using DynamicPPL, ForwardDiff, Accessors
using Random
using Base.Threads
using Logging
using PreallocationTools
using MvNormalCDF

include("Spectra/Spectra.jl")
using .Spectra

include("utils.jl")
include("mle_optimization.jl")
include("sequential_fit.jl")
include("corrections.jl")
include("VCF.jl")



export pre_fit!, demoinfer, compare_models, sample_model_epochs,
    correctestimate!,
    get_para, evd, loglike, sds, pop_sizes, durations, times, get_covar, flags,
    adapt_histogram,
    FitResult, FitOptions, setOptimOptions!,
    laplacekingman, mldsmcp,
    extbps,
    lineages, cumulative_lineages, crediblehistory,
    sampleN, quantilesN,
    VCF


function integral_ws(edges::AbstractVector{<:Real}, mu::Float64, TN::Vector)
    a = 0.5
    last_hid_I = laplacekingmanint(edges[1] - a, mu, TN)
    weights = Vector{Float64}(undef, length(edges)-1)
    for i in eachindex(weights)
        @inbounds this_hid_I = laplacekingmanint(edges[i+1] - a, mu, TN)
        weights[i] = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
    end
    weights
end

"""
    compute_residuals(h::Histogram, mu, rho, TN::Vector; naive=true)

Compute the residuals between the observed and expected weights.
## Optional arguments
- `naive::Bool=true`: if true the expected weights are computed
  using the closed form integral, otherwise using higher order transition
  probabilities from SMC' theory.
- `order::Int=10`: maximum number of higher order corrections to use
  when `naive` is false, i.e. number of intermediate recombination events
  plus one.
- `ndt::Int=800`: number of Legendre nodes to use when `naive` is false.
"""
function compute_residuals(h::Histogram, mu::Float64, rho::Float64, TN::Vector; 
    naive=true, order=10, ndt=800
)
    if naive
        w_th = integral_ws(h.edges[1], mu, TN)
    else
        rs = midpoints(h.edges[1])
        bag = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
        mldsmcp!(bag, 1:bag.order, rs, h.edges[1].edges, mu, rho, TN)
        w_th = get_tmp(bag.ys, eltype(TN)) .* diff(h.edges[1])
    end
    residuals = (h.weights .- w_th) ./ sqrt.(w_th)
    @assert all(isfinite.(residuals))
    return residuals
end

"""
    compute_residuals(h::Histogram, yth::AbstractVector{<:Real})

Compute the residuals between the observed and expected weights.
"""
function compute_residuals(h::Histogram, yth::AbstractVector{<:Real})
    w_th = yth .* diff(h.edges[1])
    residuals = (h.weights .- w_th) ./ sqrt.(w_th)
    @assert all(isfinite.(residuals))
    return residuals
end

"""
    compute_residuals(h1::Histogram, h2::Histogram)

When two histograms are given the std error of the difference is
used for normalization.
"""
function compute_residuals(h1::Histogram, h2::Histogram; fc1 = 1.0, fc2 = 1.0)
    @assert length(h1.weights) == length(h2.weights)
    # when both observations are zero the residual is zero
    w_ = h1.weights / fc1 .+ h2.weights / fc2
    residuals = (h1.weights / fc1 - h2.weights / fc2) ./ sqrt.(w_)
    residuals[w_ .== 0] .= 0
    @assert all(isfinite.(residuals))
    return residuals
end

"""
    residstructure(residuals::AbstractVector{<:Real})

Compute the p-value for the autocorrelation of adjacent residuals.
The p-value is the right tail of the t-distribution.
"""
function residstructure(residuals::AbstractVector{<:Real})
    l = length(residuals)
    if l % 2 == 1
        l -= 1
    end
    x1 = 1:2:l-1
    x2 = 2:2:l
    c = cor(view(residuals, x1), view(residuals, x2))
    t = c * sqrt((l÷2 - 2)/(1-c^2))
    p = StatsAPI.pvalue(Distributions.TDist(l÷2 - 2), t; tail=:right)
    return p
end

function CustomEdgeVector(; lo = 1, hi = 10, nbins::Integer)
    @assert (lo > 0) && (hi > 0) && (nbins > 0) && (hi > lo)
    lo = floor(Int, lo)
    hi = ceil(Int, hi)
    edges = collect(logrange(lo, hi+1, length = nbins+1))
    for i in 2:nbins+1
        nw = ceil(Int, edges[i])
        delta = nw - edges[i]
        edges[i:end] .+= delta
    end
    edges = unique(round.(Int, edges))
    edges[1] = lo
    @assert all(diff(edges) .> 0)
    @assert length(edges) == nbins + 1
    LogEdgeVector(edges)
end

"""
    adapt_histogram(segments::AbstractVector{<:Integer}; lo::Int=1, hi::Int=50_000_000, nbins::Int=0, tailthr::Int=0)

Build an histogram from `segments` logbinned between `lo` and `hi`
with `nbins` bins. `nbins` is automatically determined by default.

The upper limit is adapted to ensure logspacing with the requested `nbins`. The adaptive strategy is such that the
last bin has at least `tailthr` segments.
"""
function adapt_histogram(segments::AbstractVector{<:Integer}; lo::Int=1, hi::Int=50_000_000, nbins::Int=0, tailthr::Int=0)
    if iszero(nbins)
        if length(segments) > 1e7
            nbins = 1600
        else
            nbins = 800
        end
    end
    @assert nbins > 0
    h_obs = Histogram(CustomEdgeVector(;lo, hi, nbins))
    @assert !isempty(segments)
    append!(h_obs, segments)
    l = findlast(h_obs.weights .> tailthr)
    isnothing(l) && return h_obs
    while l-1 > 0 && h_obs.weights[l-1] == 0
        l -= 1
    end
    while h_obs.edges[1].edges[l+1] < hi
        hi = h_obs.edges[1].edges[l+1]
        h_obs = Histogram(CustomEdgeVector(;lo, hi, nbins))
        append!(h_obs, segments)
        l = findlast(h_obs.weights .> tailthr)
        isnothing(l) && return h_obs
        while l-1 > 0 && h_obs.weights[l-1] == 0
            l -= 1
        end
    end
    return h_obs
end

"""
    compare_mlds(segs1::Vector{Int}, segs2::Vector{Int}; lo = 1, hi = 1_000_000, nbins = 200)

Build two histograms from `segs1` and `segs2`, rescale them for number and
average heterozygosity and return four vectors containing respectively
common midpoints for bins, the two rescaled weights and variances of
the difference between weights.
"""
function compare_mlds(segs1::AbstractVector{<:Integer}, segs2::AbstractVector{<:Integer}; lo = 1, hi = 1_000_000, nbins = 200)
    theta1 = 1/mean(segs1)
    theta2 = 1/mean(segs2)
    h1 = Histogram(LogEdgeVector(lo = lo, hi = hi, nbins = nbins))
    append!(h1, segs1)
    h2 = Histogram(LogEdgeVector(lo = lo, hi = hi, nbins = nbins))
    append!(h2, segs2)
    return compare_mlds!(h1, h2, theta1, theta2)
end

"""
    compare_mlds!(h1::Histogram, h2::Histogram, theta1::Float64, theta2::Float64)

The same as `compare_mlds`, except that it takes two histograms and mutates them.
Return values are the same as `compare_mlds`.
"""
function compare_mlds!(h1, h2, theta1, theta2)
    # 1 is the target lattice, i.e. with biggest theta
    length(h1.weights) == length(h2.weights) && @assert any(h1.weights .!= h2.weights)
    @assert theta1 != theta2
    @assert any(h1.weights .> 0) && any(h2.weights .> 0)
    swap = false
    if theta1 < theta2
        temp = deepcopy(h1)
        h1 = deepcopy(h2)
        h2 = temp
        theta1, theta2 = theta2, theta1
        swap = true
    end
    edges1 = h1.edges[1].edges * theta1
    edges2 = h2.edges[1].edges * theta2
    tw = zeros(Float64, length(h1.weights))
    w2 = h2.weights
    factor = sum(h1.weights) / sum(h2.weights)
    t = 1 # target
    f = 1 # following
    while t < length(edges1) && f < length(edges2)
        st, en = edges1[t], edges1[t+1]
        width = edges2[f+1] - edges2[f]
        if st <= edges2[f] < edges2[f+1] < en
            tw[t] += w2[f]
            f += 1
        elseif st <= edges2[f] < en < edges2[f+1]
            tw[t] += w2[f] * (en - edges2[f]) / width
            t += 1
        elseif edges2[f] < st <= edges2[f+1] < en
            if f == 1
                tw[t] += w2[f]
            else
                tw[t] += w2[f] * (edges2[f+1] - st) / width
            end
            f += 1
        elseif edges2[f] <= st < en < edges2[f+1]
            tw[t] += w2[f] * (en - st) / width
            t += 1
        else
            if edges2[f+1] < st && t == 1
                tw[t] += w2[f]
                f += 1
            else
                f > t ? t+=1 : f+=1
                @error "disjoint bins"
            end
        end
    end
    
    rs = midpoints(h1.edges[1]) * theta1
    sigmasq = h1.weights .+ tw * factor^2
    maxl = min(findlast(w2 .> 0), findlast(h1.weights .> 0))
    if swap
        return rs[1:maxl], tw[1:maxl] * factor, h1.weights[1:maxl], sigmasq[1:maxl]
    else
        return rs[1:maxl], h1.weights[1:maxl], tw[1:maxl] * factor, sigmasq[1:maxl]
    end
end

end
