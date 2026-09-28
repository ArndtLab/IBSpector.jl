module SMCpIntegrals

using FastGaussQuadrature
using LinearAlgebra
using PreallocationTools

using ..CoalescentBase

export IntegralArrays, fusedsweep!, getnpicard,
    firstorder, firstorderint, TimeGrid, timegrid, timenodes!, ndt, npanels

# The scheme is documented in the latex notes.

# The Laplace transform of the coalescent density, summed epoch by epoch over the history `TN`.
function firstorder(r::Real, rate::Real, TN::AbstractVector{<:Real})
    s = 0.
    cum = 0.
    pnt = 1
    while pnt < length(TN)÷2
        pnt += 1
        t = getts(TN, pnt)
        aem = 2getns(TN, pnt-1)
        aep = 2getns(TN, pnt)
        cum += (t - getts(TN, pnt-1)) / 2getns(TN, pnt-1)
        s += (
            t^2*(1/(1+2aep*rate*r) - 1/(1+2aem*rate*r)) 
            + 2t*(aep/(1+2aep*rate*r)^2 - aem/(1+2aem*rate*r)^2) 
            + 2*(aep^2/(1+2aep*rate*r)^3 - aem^2/(1+2aem*rate*r)^3)
        ) * exp(-2rate * r * t - cum)
    end
    s += 8 * getns(TN, 1)^2 / (1 + 4*getns(TN, 1) * rate * r)^3
    return s * 2 * rate
end

# Integral of the above, for bin volumes
function firstorderint(r::Real, rate::Real, TN::AbstractVector{<:Real})
    s = 0.
    cum = 0.
    pnt = 1
    while pnt < length(TN)÷2
        pnt += 1
        t = getts(TN, pnt)
        aem = 2getns(TN, pnt-1)
        aep = 2getns(TN, pnt)
        cum += (t - getts(TN, pnt-1)) / 2getns(TN, pnt-1)
        s += ( 
            t*(1/(1+2*aep*rate*r) - 1/(1+2*aem*rate*r)) 
            + (aep/(1+2*aep*rate*r)^2 - aem/(1+2*aem*rate*r)^2)
        ) * exp(-2rate * r * t - cum)
    end
    s += 2 * getns(TN, 1) / (1 + 4*getns(TN, 1) * rate * r)^2
    return - s
end

function pt(t::Real, TN::AbstractVector{<:Real})
    # q(t) of the notes; the mean coalescent time cancels against the number of
    # segments it is later multiplied by, so it is not divided out here.
    return exp(-cumcr(0, t, TN)/2) * t / (2 * Nt(t, TN))
end

# R(t) of the notes (sec:sep).
function margrecomb(t::Real, TN::AbstractVector{<:Real})
    s = 0.
    pnt = 1
    while pnt < length(TN)÷2 && getts(TN, pnt+1) < t
        s += (getns(TN, pnt) - getns(TN, pnt+1)) * exp(-cumcr(getts(TN, pnt+1), t, TN))
        pnt += 1
    end
    return s
end

# smc' transition kernel at `ti` given `tj`, multiplied by 2*tj (the compensating
# 2t is applied once, in the terminal integral). Scalar reference form, used only
# by the regression tests; the production path is `sepkernel!` + `transition!`.
function ptt(ti::Real, tj::Real, TN::AbstractVector{<:Real})
    if ti == tj
        return ti - margrecomb(ti, TN) - Nt(ti, TN) + Nt(0, TN) * exp(-cumcr(0, ti, TN)) #/ 2tj
    elseif ti < tj
        return 1 + margrecomb(ti, TN)/Nt(ti, TN) - Nt(0, TN) / Nt(ti, TN) * exp(-cumcr(0, ti, TN)) #/ 2tj
    else
        return exp(-cumcr(tj, ti, TN)/2) * (Nt(tj, TN) + margrecomb(tj, TN) - Nt(0, TN) * exp(-cumcr(0, tj, TN))) / Nt(ti, TN) #/ 2tj
    end
end

# Semiseparable factorisation of the transition kernel at the nodes `ts`
# (ascending) for the history `TN`: writes the upper-branch factor `Phi`, the
# diagonal atom `dgn`, the clamped lower-branch factor `Gc` and `Ninv = 1/N(t)`.
# Derived in sec:sep of the notes; `dgn` is formed from the unclamped G.
function sepkernel!(Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real},
    Gc::AbstractVector{<:Real}, Ninv::AbstractVector{<:Real},
    ts::AbstractVector{<:Real}, TN::AbstractVector{<:Real}
)
    n = length(ts)
    n0 = Nt(0, TN)
    @inbounds for j in 1:n
        t = ts[j]
        c = cumcr(0, t, TN)
        nt = Nt(t, TN)
        g = nt + margrecomb(t, TN) - n0 * exp(-c)
        Gc[j] = max(g, zero(g))
        Phi[j] = Gc[j] / nt
        dgn[j] = max(t - g, zero(g))
        Ninv[j] = 1 / nt
    end
    return nothing
end

"""
    TimeGrid(K; msub = 8, nfin = 12, ntail = 16,
              fmin = 1e-6, umin = 1e-8, umax = 25.0, dumax = 1.0)
    timegrid(K; msub = 0, nfin = 0, ntail = 0)

Reference nodes, weights and partial-integral weights for the composite
panel-wise time quadrature of a `K`-epoch history. Holds nothing that depends on
the history `TN`: the panels are laid on the epoch times at evaluation time by
[`timenodes!`](@ref).

The time axis is tiled by `npanels(g)` sub-panels carrying `msub`
Gauss-Legendre nodes each: `nfin + 1` per finite epoch, graded towards the
epoch's left endpoint by `geomedges(nfin, fmin)`, and `tailedges(ntail, umin,
umax, dumax)` over the final epoch, worked in its local coalescent variable.
`Lpw` holds the partial integrals of the local Lagrange basis, which is what
lets [`transition!`](@ref) integrate across the kernel's moving delta corner.

`fmin`, `umin`, `umax` and `dumax` are grid constants applied once here, never a
runtime test on `TN`. Construction, the role of each constant, the node
calibration and the stability bound on `nfin` are in app:tquad and app:tquad-cal
of the notes.
"""
struct TimeGrid
    msub::Int
    nfin::Int
    ntail::Int
    K::Int
    zleg::Vector{Float64}
    wleg::Vector{Float64}
    Lpw::Matrix{Float64}
    fedge::Vector{Float64}
    uedge::Vector{Float64}
end

# Partial integrals of the Lagrange basis on the Gauss-Legendre nodes `z`, from
# -1 up to each node, normalised by the weight `w`. Closed form via the Legendre
# expansion of the basis; derived in app:lpw.
function partialweights(z::Vector{Float64}, w::Vector{Float64})
    m = length(z)
    Lpw = zeros(Float64, m, m)
    P  = zeros(Float64, m + 1)          # P_0 .. P_m at the current point
    Pz = zeros(Float64, m, m)           # Pz[k+1,i] = P_k(z_i)
    for i in 1:m
        legendre!(P, z[i], m)
        for k in 0:m-1
            Pz[k+1, i] = P[k+1]
        end
    end
    for q in 1:m
        legendre!(P, z[q], m)
        for k in 0:m-1
            Ik = k == 0 ? z[q] + 1 : (P[k+2] - P[k]) / (2k + 1)
            c = (2k + 1) / 2 * Ik
            for i in 1:m
                Lpw[q, i] += c * Pz[k+1, i]
            end
        end
    end
    return Lpw
end

# Legendre polynomials P_0 .. P_n at `x`, written into `P`, by the three-term
# recurrence.
function legendre!(P::Vector{Float64}, x::Float64, n::Int)
    P[1] = 1.0
    n >= 1 && (P[2] = x)
    for k in 1:n-1
        P[k+2] = ((2k + 1) * x * P[k+1] - k * P[k]) / (k + 1)
    end
    return P
end

"""
Default sub-panel splits, calibrated in app:tquad-cal of the notes.
"""
const TIMEGRID_DEFAULTS = (msub = 8, nfin = 12, ntail = 16, fmin = 1.0e-6)

function TimeGrid(K::Int;
    msub::Int = TIMEGRID_DEFAULTS.msub,
    nfin::Int = TIMEGRID_DEFAULTS.nfin,
    ntail::Int = TIMEGRID_DEFAULTS.ntail,
    fmin::Float64 = TIMEGRID_DEFAULTS.fmin, umin::Float64 = 1.0e-8, umax::Float64 = 25.0,
    dumax::Float64 = 1.0
)
    @assert K >= 1 "need at least one epoch"
    @assert msub >= 2 "need at least 2 nodes per sub-panel"
    @assert nfin >= 1 "need at least 1 sub-panel per finite epoch"
    @assert ntail >= 1 "need at least 1 sub-panel in the tail"
    @assert 0 < fmin < 1 "fmin must lie in (0,1)"
    @assert 0 < umin < 1 "umin must lie in (0,1)"
    @assert umax > 1 "tail truncation must exceed 1"
    @assert dumax > 0 "dumax must be positive"
    z, w = gausslegendre(msub)
    return TimeGrid(msub, nfin, ntail, K, z, w, partialweights(z, w),
                    geomedges(nfin, fmin), tailedges(ntail, umin, umax, dumax))
end

# Sub-panel edges as fractions of a finite epoch's width, geometric from `fmin`
# towards the epoch's left endpoint and closed at 1. Fixed fractions, so every
# node stays exactly affine in TN. Grading argument in app:tquad.
function geomedges(nfin::Int, fmin::Float64)
    f = [0.0, fmin]
    ratio = (1 / fmin)^(1 / nfin)
    for q in 1:nfin
        push!(f, fmin * ratio^q)
    end
    f[end] = 1.0
    return f
end

# Tail sub-panel edges in the local coalescent variable: geometric from `umin` to
# 1 in `ntail` steps, then steps of at most `dumax` up to the truncation `umax`,
# with any wider step split so `dumax` binds everywhere. Grading and the role of
# `dumax` in app:tquad.
function tailedges(ntail::Int, umin::Float64, umax::Float64, dumax::Float64)
    uedge = [0.0, umin]
    ratio = (1 / umin)^(1 / ntail)
    for q in 1:ntail
        push!(uedge, umin * ratio^q)
    end
    uedge[end] = 1.0
    while uedge[end] < umax
        push!(uedge, min(umax, uedge[end] + dumax))
    end
    # honour dumax in the geometric part too (it only binds if umin is large)
    out = [0.0]
    for q in 1:length(uedge)-1
        a, b = uedge[q], uedge[q+1]
        nsplit = max(1, ceil(Int, (b - a) / dumax))
        for i in 1:nsplit
            push!(out, a + (b - a) * i / nsplit)
        end
    end
    return out
end

"""
    npanels(g::TimeGrid)

Number of sub-panels tiling the time axis: `nfin + 1` per finite epoch plus the
tail mesh, whose count `dumax` may raise above `ntail`.
"""
npanels(g::TimeGrid) = (g.K - 1) * (length(g.fedge) - 1) + length(g.uedge) - 1

function timegrid(K::Int; msub::Int = 0, nfin::Int = 0, ntail::Int = 0)
    TimeGrid(K;
        msub  = iszero(msub)  ? TIMEGRID_DEFAULTS.msub  : msub,
        nfin  = iszero(nfin)  ? TIMEGRID_DEFAULTS.nfin  : nfin,
        ntail = iszero(ntail) ? TIMEGRID_DEFAULTS.ntail : ntail,
    )
end

"""
    ndt(g::TimeGrid)

Total number of quadrature nodes, `npanels(g) * msub`.
"""
ndt(g::TimeGrid) = npanels(g) * g.msub

"""
    timenodes!(ts, om, EE, EB, g::TimeGrid, TN)

Lay the grid `g` on the history `TN`: fill `ts` with the quadrature nodes
(ascending, as `sepkernel!` and `transition!` require), `om` with their weights,
and `EE`/`EB` with the per-node and per-sub-panel exponential rescalings that
keep the lower branch of `transition!` bounded. Both rescalings are read off the
affine node maps, so no `cumcr` call is needed.

Sub-panels are pinned to the epoch boundaries, so each node is an affine function
of the epoch parameters: nodes move smoothly with `TN` and can never migrate
between epochs.

!!! note "No stability guard"
    A finite epoch wide enough in coalescent units makes the `r` sweep in
    `fusedsweep!` gain rather than contract from bin to bin, and the iterate then
    climbs until it overflows to `NaN`. Reaching that needs a population size near
    the lower bound held across a very wide epoch, where the likelihood is already
    numerically zero, so it is recorded rather than guarded. The threshold, what
    actually diverges, and why a larger `nfin` is the cure are in app:tquad
    ("Stability") of the notes. However, likelihood optimization always starts from
    a valid point and should bounce back if NaN is ever hit.
"""
function timenodes!(ts::AbstractVector{<:Real}, om::AbstractVector{<:Real},
    EE::AbstractVector{<:Real}, EB::AbstractVector{<:Real},
    g::TimeGrid, TN::AbstractVector{<:Real}
)
    K = length(TN) ÷ 2
    @assert K == g.K "grid built for $(g.K) epochs, got $K"
    @assert length(ts) == ndt(g) "ts has length $(length(ts)), expected $(ndt(g))"
    @assert length(om) == ndt(g) "om has length $(length(om)), expected $(ndt(g))"
    @assert length(EE) == ndt(g) "EE has length $(length(EE)), expected $(ndt(g))"
    @assert length(EB) == npanels(g) "EB has length $(length(EB)), expected $(npanels(g))"

    j = 0
    p = 0
    @inbounds for k in 1:K-1
        a0 = getts(TN, k)
        b0 = getts(TN, k + 1)
        b0 > a0 || throw(ArgumentError(
            "timenodes!: epoch $k has non-positive width: T_$k = $a0, T_$(k+1) = $b0"
        ))
        Nk = getns(TN, k)
        Nk > 0 || throw(ArgumentError(
            "timenodes!: population size N_$k must be strictly positive, got $Nk"
        ))
        W = b0 - a0
        for q in 1:length(g.fedge)-1
            p += 1
            a = a0 + W * g.fedge[q]
            h = W * (g.fedge[q+1] - g.fedge[q]) / 2   # sub-panel half-width in t
            c = a + h
            EB[p] = exp(2 * h / (2 * Nk))
            for i in 1:g.msub
                j += 1
                t = c + h * g.zleg[i]
                ts[j] = t
                om[j] = g.wleg[i] * h
                EE[j] = exp((t - a) / (2 * Nk))
            end
        end
    end
    TK = getts(TN, K)
    NK = getns(TN, K)
    NK > 0 || throw(ArgumentError(
        "timenodes!: tail population size N_$K must be strictly positive, got $NK"
    ))
    twoNK = 2 * NK
    @inbounds for q in 1:length(g.uedge)-1
        p += 1
        ua = g.uedge[q]
        hu = (g.uedge[q+1] - ua) / 2        # sub-panel half-width in u
        cu = ua + hu
        EB[p] = exp(2 * hu)
        for i in 1:g.msub
            j += 1
            u = cu + hu * g.zleg[i]
            ts[j] = TK + twoNK * u
            om[j] = g.wleg[i] * hu * twoNK
            EE[j] = exp(u - ua)
        end
    end
    return nothing
end

struct IntegralArrays{T,R<:AbstractVector{<:Real},E<:AbstractVector{<:Real}}
    n_dt::Int
    nrs::Int
    nrscoarse::Int
    ys::DiffCache{Vector{T},Vector{T}}
    wcoarse::DiffCache{Vector{T},Vector{T}}
    grid::TimeGrid
    ts::DiffCache{Vector{T},Vector{T}}
    qs::DiffCache{Vector{T},Vector{T}}
    om::DiffCache{Vector{T},Vector{T}}
    EE::DiffCache{Vector{T},Vector{T}}
    EB::DiffCache{Vector{T},Vector{T}}
    Phi::DiffCache{Vector{T},Vector{T}}
    dgn::DiffCache{Vector{T},Vector{T}}
    Gc::DiffCache{Vector{T},Vector{T}}
    Ninv::DiffCache{Vector{T},Vector{T}}
    A::DiffCache{Vector{T},Vector{T}}
    Jf::DiffCache{Vector{T},Vector{T}}
    MJ::DiffCache{Vector{T},Vector{T}}
    J1::DiffCache{Vector{T},Vector{T}}
    edges::E
    rs::R
end

function IntegralArrays(grid::TimeGrid, nrscoarse::Int, 
    rs::AbstractVector{<:Real}, edges::AbstractVector{<:Real}, chunk, levels = 1
)
    n = ndt(grid)
    nrs = length(rs)
    dcvec(len = n) = DiffCache(zeros(Float64, len), chunk; levels)
    IntegralArrays(
        n, nrs, nrscoarse,
        DiffCache(zeros(Float64, nrs), chunk; levels),
        DiffCache(zeros(Float64, nrscoarse), chunk; levels),
        grid,
        dcvec(), dcvec(), dcvec(), dcvec(), dcvec(npanels(grid)),
        dcvec(), dcvec(), dcvec(), dcvec(),
        dcvec(), dcvec(), dcvec(), dcvec(),
        edges, rs
    )
end

"""
    getnpicard(mu, rho)

Number of Picard iterations (`transition!` applies) per bin that `fusedsweep!`
needs to keep its discretisation error below noise at the production binning.
Selected from `alpha = rho / (mu + rho)` alone, so it is fixed over a fit and
cannot make the objective discontinuous.

!!! warning "Validity bound"
    Calibrated for `alpha <= 0.8` only, i.e. `rho / mu <= 4`. Above that the
    returned count is NOT sufficient and callers must pass an explicit, larger
    `npicard`.

Calibration and the measured errors at the branch edges are in sec:accuracy of
the notes.
"""
function getnpicard(mu::Real, rho::Real)
    alpha = rho / (mu + rho)
    alpha <= 0.55 && return 2
    alpha <= 0.72 && return 3
    return 4
end

# Apply the semiseparable transition operator to `x`, writing the result into
# `out`. The upper branch runs backwards over sub-panels accumulating a suffix
# integral, the lower branch forwards accumulating a prefix integral referenced to
# each sub-panel's left edge (which is what `EE`/`EB` rescale), and the sub-panel
# holding the row contributes a partial integral through `g.Lpw`. That partial
# integral is what keeps the rule accurate across the kernel's moving corner; the
# diagonal atom `dgn` is exact. Derived in app:tquad ("The apply"), O(n * msub).
function transition!(out::AbstractVector{<:Real}, x::AbstractVector{<:Real},
    Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real}, Gc::AbstractVector{<:Real},
    Ninv::AbstractVector{<:Real}, EE::AbstractVector{<:Real}, EB::AbstractVector{<:Real},
    om::AbstractVector{<:Real}, g::TimeGrid
)
    T = eltype(out)
    m = g.msub
    npan = npanels(g)
    Lpw = g.Lpw

    sfx = zero(T)
    @inbounds for p in npan:-1:1
        s0 = (p - 1) * m
        Sp = zero(T)
        for i in 1:m
            Sp += x[s0+i] * om[s0+i]
        end
        for q in 1:m
            low = zero(T)
            for i in 1:m
                low += Lpw[q,i] * om[s0+i] * x[s0+i]
            end
            out[s0+q] = Phi[s0+q] * (sfx + Sp - low)
        end
        sfx += Sp
    end

    st = zero(T)
    @inbounds for p in 1:npan
        s0 = (p - 1) * m
        for q in 1:m
            low = zero(T)
            for i in 1:m
                low += Lpw[q,i] * om[s0+i] * EE[s0+i] * Gc[s0+i] * x[s0+i]
            end
            out[s0+q] += (st + low) * Ninv[s0+q] / EE[s0+q] + dgn[s0+q] * x[s0+q]
        end
        Sp = zero(T)
        for i in 1:m
            Sp += om[s0+i] * EE[s0+i] * Gc[s0+i] * x[s0+i]
        end
        st = (st + Sp) / EB[p]
    end
    return nothing
end

"""
    fusedsweep!(ys, ts, qs, om, EE, EB, Phi, dgn, Gc, Ninv, A, Jf, MJ, J1,
                grid, rs, edges, mu, rho, npicard, n_dt, nrs, TN)

One forward sweep in `r` over the Volterra form of the SMC' recursion for the
history `TN`, writing the expected number of segments at the lengths `rs` into
`ys`.

`edges` are histogram edges over integer segment lengths: bin `i` holds the
lengths `edges[i] … edges[i+1]-1` and is integrated over
`[edges[i] - 1/2, edges[i+1] - 1/2)`, so `rs[i]` is interior to every bin, unit
bins included, and one rule covers the whole grid.

The remaining arguments are the buffers `IntegralArrays` carries; `grid` is the
`TimeGrid` laid on `TN` by `timenodes!`. The sweep is sequential in `r` by
construction and is not threaded.

The scheme, the half-shift rule and the measured cost of each choice made here
are in sec:wbranch and app:slab of the notes.
"""
function fusedsweep!(ys::AbstractVector{<:Real},
    ts::AbstractVector{<:Real}, qs::AbstractVector{<:Real},
    om::AbstractVector{<:Real}, EE::AbstractVector{<:Real}, EB::AbstractVector{<:Real},
    Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real},
    Gc::AbstractVector{<:Real}, Ninv::AbstractVector{<:Real},
    A::AbstractVector{<:Real}, Jf::AbstractVector{<:Real},
    MJ::AbstractVector{<:Real}, J1::AbstractVector{<:Real},
    grid::TimeGrid,
    rs::AbstractVector{<:Real}, edges::AbstractVector{<:Real}, mu::Real, rho::Real,
    npicard::Int, n_dt::Int, nrs::Int,
    TN::AbstractVector{<:Real}
)
    @assert length(rs) == nrs
    @assert length(edges) == nrs + 1
    @assert npicard >= 1

    T = eltype(ys)
    rate = mu + rho
    alpha = rho / rate

    timenodes!(ts, om, EE, EB, grid, TN)
    for j in 1:n_dt
        qs[j] = pt(ts[j], TN)
    end
    sepkernel!(Phi, dgn, Gc, Ninv, ts, TN)

    fill!(A, zero(T))
    fill!(MJ, zero(T))

    # slab below the first edge: one rectangular step off the exact initial
    # condition (sec:wbranch, app:slab)
    w0 = edges[1] - 1//2
    if w0 > 0
        for j in 1:n_dt
            Jf[j] = rate * qs[j]
        end
        transition!(MJ, Jf, Phi, dgn, Gc, Ninv, EE, EB, om, grid)
        for j in 1:n_dt
            t = ts[j]
            A[j] = alpha * MJ[j] * (- expm1(-2rate * w0 * t)) / 2t
        end
    end

    scale = 2 * mu * TN[1] * (mu / rate)
    @inbounds for i in 1:nrs
        # w: full bin width; wi: from the shifted left edge to rs[i]
        w = edges[i+1] - edges[i]
        wi = rs[i] - edges[i] + 1//2
        for j in 1:n_dt
            J1[j] = rate * exp(-2rate * rs[i] * ts[j]) * qs[j]
        end
        for _ in 1:npicard
            for j in 1:n_dt
                t = ts[j]
                Jf[j] = J1[j] + A[j] * exp(-2rate * wi * t) +
                        alpha * MJ[j] * (- expm1(-2rate * wi * t)) / 2t
            end
            transition!(MJ, Jf, Phi, dgn, Gc, Ninv, EE, EB, om, grid)
        end
        s = zero(T)
        for j in 1:n_dt
            t = ts[j]
            # convolution part rebuilt from the MJ the last Picard apply
            # produced, one iterate fresher than the Jf above (sec:accuracy)
            jc = A[j] * exp(-2rate * wi * t) +
                 alpha * MJ[j] * (- expm1(-2rate * wi * t)) / 2t
            # terminal t integral of the convolution part; the 2t compensates
            # the factor the kernel carries
            s += jc * 2 * t * om[j]
            # roll the accumulator across the full bin width
            A[j] = exp(-2rate * w * t) * A[j] +
                   alpha * MJ[j] * (- expm1(-2rate * w * t)) / 2t
        end
        # order 1 comes from the analytic firstorder
        ys[i] = (firstorder(rs[i], rate, TN) + s) * scale
    end
    return nothing
end

"""
    fusedsweep!(bag::IntegralArrays, mu, rho, TN; npicard = 0)

Bag wrapper for the fused sweep, reading `rs` and `edges` off the bag.
`npicard = 0` selects the iteration count with [`getnpicard`](@ref); pass a
positive value to override it. Writes the density into `bag.ys` and leaves
`bag.wcoarse` untouched; `mldsmcp!` turns it into per-bin weights.
"""
function fusedsweep!(bag::IntegralArrays, mu::Real, rho::Real,
    TN::AbstractVector{<:Real}; npicard::Int = 0
)
    T = eltype(TN)
    np = npicard > 0 ? npicard : getnpicard(mu, rho)
    fusedsweep!(
        get_tmp(bag.ys, T),
        get_tmp(bag.ts, T),
        get_tmp(bag.qs, T),
        get_tmp(bag.om, T),
        get_tmp(bag.EE, T),
        get_tmp(bag.EB, T),
        get_tmp(bag.Phi, T),
        get_tmp(bag.dgn, T),
        get_tmp(bag.Gc, T),
        get_tmp(bag.Ninv, T),
        get_tmp(bag.A, T),
        get_tmp(bag.Jf, T),
        get_tmp(bag.MJ, T),
        get_tmp(bag.J1, T),
        bag.grid, bag.rs, bag.edges, mu, rho, np, bag.n_dt, bag.nrs, TN
    )
    return nothing
end

end