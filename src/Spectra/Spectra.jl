module Spectra

using LinearAlgebra
using PreallocationTools

include("CoalescentBase.jl")
using .CoalescentBase

include("SMCpIntegrals.jl")
using .SMCpIntegrals

export
    firstorder, firstorderint,
	laplacekingman, laplacekingmanint,
	mldsmcp, mldsmcp!, fusedsweep!, IntegralArrays, getnpicard, TimeGrid, timegrid, ndt, npanels,
	extbps,
    lineages, cumulative_lineages, crediblehistory,
    sampleN, quantilesN,
	map_fine_to_coarse, map_fine_to_coarse!

"""
	map_fine_to_coarse(wthfine, fine_edges, coarse_edges)
	map_fine_to_coarse!(wthcoarse, coarse_edges, wthfine, fine_edges)
	map_fine_to_coarse!(bag::IntegralArrays, coarse_edges, T)

Rebin the **weights** `wthfine` from `fine_edges` onto `coarse_edges`, splitting a
fine bin that straddles a coarse edge in proportion to the overlap. Both edge
vectors must be ascending; zero-width fine bins carry no weight and are skipped.

The allocating form returns a fresh `length(coarse_edges) - 1` vector; the
in-place forms accumulate into `wthcoarse`, which the caller must zero first (the
`bag` method does that itself, writing `bag.wcoarse` from `bag.ys` over
`bag.edges`).
"""
function map_fine_to_coarse(wthfine, fine_edges, coarse_edges)
    wthcoarse = zeros(eltype(wthfine), length(coarse_edges) - 1)
    map_fine_to_coarse!(wthcoarse, coarse_edges, wthfine, fine_edges)
    return wthcoarse
end

function map_fine_to_coarse!(wthcoarse, coarse_edges, wthfine, fine_edges)
    @assert length(wthcoarse) == length(coarse_edges) - 1
    @assert length(wthfine) == length(fine_edges) - 1
    k = 1  # current coarse bin index
    for j in eachindex(wthfine)
        a = fine_edges[j]
        b = fine_edges[j + 1]
        fine_width = b - a
        # advance coarse pointer past bins that end before this fine bin starts
        while k <= length(wthcoarse) && coarse_edges[k + 1] <= a
            k += 1
        end
        # distribute weight to all coarse bins overlapping [a, b)
        kk = k
        while kk <= length(wthcoarse) && coarse_edges[kk] < b
            overlap = min(b, coarse_edges[kk + 1]) - max(a, coarse_edges[kk])
            wthcoarse[kk] += wthfine[j] * overlap / fine_width
            kk += 1
        end
    end
    return nothing
end

function map_fine_to_coarse!(bag::IntegralArrays, coarse_edges::AbstractVector{<:Real}, T::Type{<:Real})
    wc = get_tmp(bag.wcoarse, T)
    fill!(wc, zero(eltype(wc)))
    map_fine_to_coarse!(wc, coarse_edges, get_tmp(bag.ys, T), bag.edges)
    return nothing
end

"""
	mldsmcp(rs, edges, mu, rho, TN; msub = 0, nfin = 0, ntail = 0, npicard = 0)

Compute the expected number of segments **per bin** of the histogram `edges`,
evaluated at the representative lengths `rs` (the bin midpoints), given the
mutation rate `mu`, recombination rate `rho`, and population size history `TN`.

The result is a weight, use `fusedsweep!` if the density is what you want.

The time integration runs on a `timegrid(length(TN) ÷ 2; msub, nfin, ntail)`:
sub-panels pinned to the epoch boundaries, `msub` Gauss-Legendre nodes each,
`nfin` per finite epoch and `ntail` in the tail. Any of the three passed as zero
takes the `TimeGrid` default.
"""
function mldsmcp(rs, edges, mu, rho, TN;
	msub::Int = 0, nfin::Int = 0, ntail::Int = 0, npicard::Int = 0
)
	K = length(TN) ÷ 2
	bag = IntegralArrays(timegrid(K; msub, nfin, ntail), length(rs), rs, edges, Val{length(TN)})
	mldsmcp!(bag, mu, rho, TN; npicard)
	return get_tmp(bag.ys, eltype(TN))
end

"""
	mldsmcp!(bag, mu, rho, TN; npicard = 0)

In-place `mldsmcp`, writing the per-bin weights into `bag.ys` over `bag.edges`.
"""
function mldsmcp!(bag::IntegralArrays, mu::Real, rho::Real,
    TN::AbstractVector{<:Real}; npicard::Int = 0
)
	fusedsweep!(bag, mu, rho, TN; npicard)
    y = get_tmp(bag.ys, eltype(TN))
    for i in eachindex(y)
        y[i] *= bag.edges[i + 1] - bag.edges[i]
    end
	return nothing
end

"""
	laplacekingman(r, mu, TN)

Compute the approximate number of segments of length `r` 
using the Laplace transform of the Kingman coalescent at frequency `2mu r`,
given mutation rate `mu` and population size history `TN`.
"""
function laplacekingman(r::Real, mu::Real, TN::AbstractVector{<:Real})
    return firstorder(r, mu, TN) * 2 * mu * TN[1]
end

function laplacekingmanint(r::Real, mu::Real, TN::AbstractVector{<:Real})
    return firstorderint(r, mu, TN) * 2 * mu * TN[1]
end

end
