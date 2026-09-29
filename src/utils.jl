"""
    struct FitResult

A data structure to store the results of a fit.

See the introduction for how the model is 
parameterized [Data form of input and output](@ref).
Some methods are defined for this type to get the vector of parameters, std errors, 
model evidence, etc. See [`get_para`](@ref), [`sds`](@ref), [`evd`](@ref), 
[`pop_sizes`](@ref), [`durations`](@ref).
"""
struct FitResult
    nepochs::Int
    bin::Int
    mu::Float64
    rho::Float64
    para::Vector
    stderrors::Vector
    method::String
    converged::Bool
    lp::Float64
    logevd::Float64
    free::BitVector
    opt
end

function Base.show(io::IO, f::FitResult) 
    model = (f.nepochs == 1 ? "stationary" : "$(f.nepochs) epochs") *
            (f.bin > 1 ? " (binned $(f.bin))" : "")
    print(io, "Fit ", model, " ")
    print(io, f.method, " ")
    print(io, f.converged ? "●" : "○", " ")
    print(io, "[", @sprintf("%.1e",f.para[1]))
    for i in 2:length(f.para)
        print(io, ", ", @sprintf("%.1f",f.para[i]))
    end
    print(io, "] ", @sprintf("logL %.3f",f.lp), @sprintf(" | log-evidence %.3f",f.logevd))
end

"""
    pars(fit::FitResult)

Return the parameters of the fit.
"""
get_para(fit::FitResult) = copy(fit.para)

"""
    sds(fit::FitResult)

Return the standard deviations of the parameters of the fit.
"""
sds(fit::FitResult) = copy(fit.stderrors)

"""
    free(fit::FitResult)

Return a `BitVector` flagging which parameters of the fit were actually
estimated (`true`) versus held fixed (`false`), e.g. the `T`s when
the fit was obtained with [`fitNs!`](@ref).
"""
free(fit::FitResult) = copy(fit.free)

"""
    evd(fit::FitResult)

Return the log-evidence of the fit.
"""
evd(fit::FitResult) = fit.logevd

"""
    loglike(fit::FitResult)

Return the log-likelihood of the fit.
"""
loglike(fit::FitResult) = fit.lp

"""
    times(fit::FitResult)

Return the times of size changes.
"""
function times(fit::FitResult)
    ts = [Spectra.getts(fit.para, i) for i in 1:fit.nepochs]
    return ts
end

"""
    pop_sizes(fit::FitResult)

Return the fitted population sizes, from past to present.
"""
pop_sizes(fit::FitResult) = fit.para[2:2:end]

"""
    durations(fit::FitResult)

Return the fitted durations of the epochs.
"""
durations(fit::FitResult) = fit.para[3:2:end-1]

npar(fit::FitResult) = 2fit.nepochs

"""
    get_covar(fit::FitResult)
Return the covariance matrix of the parameters of the fit, computed as the
inverse of the log-likelihood Hessian at the optimum.
"""
function get_covar(fit::FitResult)
    hess = fit.opt.hess
    eigen_problem = eigen(hess)
    lambdas = eigen_problem.values
    lambdas = real.(lambdas)
    lambdas[lambdas .<= 0] .= eps()
    covar = eigen_problem.vectors *
        diagm(inv.(lambdas)) * eigen_problem.vectors'
    for i in eachindex(lambdas)
        for j in i:length(lambdas)
            covar[i,j] = (covar[i,j] + covar[j,i]) / 2
            covar[j,i] = covar[i,j]
        end
    end
    return covar
end

"""
    flags(fit::FitResult)

Return a named tuple of flags and diagnostics for the fit, including:
- `converged`: whether the optimization converged
- `convex`: whether the likelihood Hessian at the optimum
  is **strictly** positive definite.
- `opt_flag`: whether the return optimum is a local optimum,
  checked along the log-likelihood Hessian eigenvectors.
- `at_any_boundary`: whether any parameter is at its lower or upper bound
- `log_like`: the log-likelihood of the fit
- `log_evidence`: the log-evidence of the fit
- `optimizer_message`: the original message from the optimizer,
  which can be useful for diagnosing optimization issues.
"""
function flags(fit::FitResult)
    return (;
        converged = fit.converged,
        convex = fit.opt.convex_opt,
        opt_flag = fit.opt.optflag,
        fit.opt.at_any_boundary,
        log_like = fit.lp,
        log_evidence = fit.logevd,
        optimizer_message = fit.opt.optim_result.original
    )
end

function fraction(mu, rho, n)
    mu/(mu+rho) * (rho/(mu+rho))^(n-1)
end

mutable struct Deltas
    factors::Vector{Float64}
    state::Integer
end

function next!(d::Deltas)
    # assumes to be called from iteration over factors
    d.state += 1
    if d.state > length(d.factors)
        d.state = 1
    end
end

struct LBound <: AbstractVector{Float64}
    Ltot::Float64
    Nlow::Float64
    Tlow::Float64
    pars::Int
end
LBound(Ltot::Number,Nlow::Number,Tlow::Number,pars::Int) = LBound(
    Float64(Ltot), 
    Float64(Nlow), 
    Float64(Tlow),
    pars
)

Base.size(lb::LBound) = (lb.pars,)

function Base.getindex(lb::LBound, i::Int)
    if i == 1
        return lb.Ltot * 0.5
    elseif i%2 == 0
        return lb.Nlow
    else
        j = (lb.pars - i) ÷ 2 + 1
        return lb.Tlow
    end
end

struct UBound <: AbstractVector{Float64}
    Ltot::Float64
    Nupp::Float64
    Tupp::Float64
    pars::Int
end
UBound(Ltot::Number,Nupp::Number,Tupp::Number,pars::Int) = UBound(
    Float64(Ltot), 
    Float64(Nupp), 
    Float64(Tupp),
    pars
)

Base.size(ub::UBound) = (ub.pars,)

function Base.getindex(ub::UBound, i::Int)
    if i == 1
        return ub.Ltot * 1.001
    elseif i%2 == 0
        return ub.Nupp
    else
        return ub.Tupp
    end
end

mutable struct FitOptions
    nepochs::Int
    mu::Float64
    rho::Float64
    Ltot::Real
    init::Vector{Float64}
    perturb::BitVector
    delta::Deltas
    solver
    opt
    low::LBound
    upp::UBound
    prior::Vector{<:Distribution}
    force::Bool
    maxnts::Int
    naive::Bool
    onlyN::Bool
    msub::Int
    nfin::Int
    ntail::Int
    locut::Int
    diffcache
end

function Base.show(io::IO, fop::FitOptions)
    println(io, "FitOptions with:")
    println(io, "total genome length: ", fop.Ltot)
    println(io, "μ / bp / g: ", fop.mu)
    println(io, "ρ / bp / g: ", fop.rho)
    println(io, "N lower bound: ", fop.low.Nlow)
    println(io, "N upper bound: ", fop.upp.Nupp)
    println(io, "T lower bound: ", fop.low.Tlow)
    println(io, "T upper bound: ", fop.upp.Tupp)
    println(io, "solver: ", summary(fop.solver))
end

npar(fop::FitOptions) = 2fop.nepochs

"""
    FitOptions(Ltot, nhet, mu, rho; kwargs...)

Construct an an object of type FitOptions, requiring 
total genome length `Ltot` in base pairs, number of
heterozygous sites `nhet`, mutation rate `mu` and
recombination rate `rho` per base pair per generation.

## Optional Arguments
- `Tlow::Number=10`, `Tupp::Number=1e7`: The lower and upper bounds for the duration of epochs.
- `Nlow::Number=10`, `Nupp::Number=1e8`: The lower and upper bounds for the population sizes.
- `force::Bool=true`: if true try to fit further epochs even when no signal is found.
- `maxnts::Int=10`: The maximum number of new time splits to consider when adding a new epoch.
  Higher is greedier.
- `msub::Int=0`: Gauss-Legendre nodes per sub-panel in the time quadrature.
- `nfin::Int=0`: sub-panels per finite epoch.
- `ntail::Int=0`: sub-panels in the semi-infinite tail.
  Each of the three takes the `TimeGrid` default when zero.
- `locut::Int=1`: index of the first histogram bin to consider in the fit.
"""
function FitOptions(Ltot, nhet, mu, rho;
    Tlow = 10, Tupp = 1e7,
    Nlow = 10, Nupp = 1e8,
    nepochs::Int = 1,
    force::Bool = true,
    maxnts::Int = 10,
    naive::Bool = true,
    msub::Int = 0,
    nfin::Int = 0,
    ntail::Int = 0,
    locut::Int = 1
)
    N = 2nepochs
    init = zeros(N)
    # set bounds and prior for the parameters
    upp = UBound(Ltot,Nupp,Tupp,N)
    low = LBound(Ltot,Nlow,Tlow,N)
    prior = Uniform.(low,upp)
    perturb = falses(N)
    factors = [0.001, 0.01, 0.1, 0.5, 0.5, 0.9, 2] # mapreduce( i->fill(i, 10), vcat, [0.001, 0.01, 0.1, 0.5, 0.5, 0.9, 2] )
    delta = Deltas(factors, 0)

    dflt = Spectra.SMCpIntegrals.TIMEGRID_DEFAULTS
    iszero(msub)  && (msub  = dflt.msub)
    iszero(nfin)  && (nfin  = dflt.nfin)
    iszero(ntail) && (ntail = dflt.ntail)

    solver = LBFGS()
    maxiters = 30000
    maxtime = 60
    g_tol = 5e-8
    if nhet > 1e7
        maxiters = 80000
        maxtime = 180
        g_tol = 1e-5
    end

    return FitOptions(
        nepochs,
        mu,
        rho,
        Ltot,
        init,
        perturb,
        delta,
        solver,
        (; maxiters, maxtime, g_tol),
        low,
        upp,
        prior,
        force,
        maxnts,
        naive,
        false, # onlyN
        msub,
        nfin,
        ntail,
        locut,
        nothing
    )
end

function initialize!(fop::FitOptions, weights::AbstractVector{<:Integer})
    vol = sum(weights)
    @assert vol != 0 "Empty histogram!"
    N = 1/(4*fop.mu*(fop.Ltot/vol)) # can be rough estimate depending on binning
    n = npar(fop)
    fop.init[1] = fop.Ltot
    fop.init[2:end] .= N
    if n > 2
        nlin = 4 * fop.rho * N * fop.Ltot / n * 2
        grid = logrange(1, 1e7, 200)
        cum = 0
        i = n - 1
        t0 = 0
        for t in grid
            if i < 3
                break
            end
            l = cumulative_lineages(t, [fop.Ltot, N], fop.rho)
            if l - cum > nlin
                cum = l
                fop.init[i] = t - t0
                t0 = t
                i -= 2
            end
        end
    end
    setinit!(fop, fop.init)
    return nothing
end

import .Spectra.SMCpIntegrals: TIMEGRID_DEFAULTS

"""
    setinit!(fop::FitOptions, init::AbstractVector{<:Real})

Set the initial vector of parameters for the optimization which takes the `FitOptions` object `fop`.

Entries outside the prior support are truncated into it. When `fop` is set up
for an N-only optimization (see [`isonlyN`](@ref), [`fitNs!`](@ref)) only the
free entries are modified, see [`freemask`](@ref): the `T`s are held fixed by
the fit, so they are kept exactly as requested.
"""
function setinit!(fop::FitOptions, init::AbstractVector{<:Real})
    @assert length(init) == npar(fop) "Length of init vector must be equal to number of parameters"
    fop.init .= init
    mask = freemask(fop)
    for i in eachindex(fop.init)
        mask[i] || continue
        fop.init[i] <= fop.low[i] ? fop.init[i] = fop.low[i] * 1.001 : nothing
        fop.init[i] >= fop.upp[i] ? fop.init[i] = fop.upp[i] * 0.999 : nothing
    end
    if !isnaive(fop)
        fmin = TIMEGRID_DEFAULTS.fmin
        delta_max = 17
        frac = (1 − fmin^(1/fop.nfin))
        for i in 3:2:length(fop.init)-1
            delta = fop.init[i] / 2fop.init[i+1] * frac
            if delta > delta_max
                if mask[i]
                    fop.init[i] = 2fop.init[i+1] * delta_max * 0.99 / frac
                else
                    # T is held fixed, enlarge the following N instead
                    fop.init[i+1] = min(fop.init[i] * frac / (2delta_max * 0.99),
                        fop.upp[i+1] * 0.999)
                end
            end
        end
    end
    return nothing
end

function setnepochs!(fop::FitOptions, nepochs::Int)
    N = 2nepochs
    fop.nepochs = nepochs
    fop.init = zeros(N)
    fop.perturb = falses(N)
    L = fop.Ltot
    Nlow = fop.low.Nlow
    Nupp = fop.upp.Nupp
    Tlow = fop.low.Tlow
    Tupp = fop.upp.Tupp
    fop.low = LBound(L, Nlow, Tlow, N)
    fop.upp = UBound(L, Nupp, Tupp, N)
    fop.prior = Uniform.(fop.low, fop.upp)
    return nothing
end

function set_perturb!(fop::FitOptions, fit::FitResult)
    @assert npar(fop) == npar(fit)
    for i in eachindex(fop.perturb)
        fop.perturb[i] = fit.opt.at_lboundary[i] || 
            (fit.opt.at_uboundary[i] && i > 1) ||
            !fit.converged
    end
end

function reset_perturb!(fop::FitOptions)
    fop.perturb .= falses(npar(fop))
    fop.delta.state = 0
end

# struct PInit <: AbstractVector{Float64}
#     fop::FitOptions
# end

# Base.size(p::PInit) = (npar(p.fop),)

getdelta(fop::FitOptions) = fop.delta.factors[fop.delta.state]

# function Base.getindex(p::PInit, i::Int)
#     if !p.fop.perturb[i]
#         return p.fop.init[i]
#     else
#         dl = getdelta(p.fop)
#         low = p.fop.low[i]
#         upp = p.fop.upp[i]
#         if dl < 1
#             return rand(
#                 truncated(
#                     LogNormal(log(p.fop.init[i]), dl),
#                     low,
#                     upp
#                 )
#             )
#         else
#             return rand(Uniform(low, upp))
#         end
#     end
# end

# this can give an init that fires the NaN in the smc' branch
# it is currently used only for the Laplace, will need adaptation
# for future expansion
function getPinit(fop)
    pinit = similar(fop.init)
    dl = getdelta(fop)
    for i in 2:2:length(fop.init)
        if dl < 1
            T = rand(
                truncated(
                    LogNormal(log(fop.init[i-1]), dl),
                    fop.low[i-1],
                    fop.upp[i-1]
                )
            )
            N = rand(
                truncated(
                    LogNormal(log(fop.init[i]), dl),
                    fop.low[i],
                    fop.upp[i]
                )
            )
        else
            T = rand(Uniform(fop.low[i-1], fop.upp[i-1]))
            N = rand(Uniform(fop.low[i], fop.upp[i]))
        end
        if fop.perturb[i-1]
            N = fop.init[i] * T / fop.init[i-1]
            pinit[i-1] = T
            pinit[i] = N
        elseif fop.perturb[i]
            pinit[i-1] = fop.init[i-1]
            pinit[i] = N
        else
            pinit[i-1] = fop.init[i-1]
            pinit[i] = fop.init[i]
        end
    end
    return pinit
end

function isnaive(fop::FitOptions)
    return fop.naive
end

function setnaive!(fop::FitOptions, flag::Bool)
    fop.naive = flag
end

function isonlyN(fop::FitOptions)
    return fop.onlyN
end

function setonlyN!(fop::FitOptions, flag::Bool)
    fop.onlyN = flag
end

"""
    freemask(fop::FitOptions)

Return a `BitVector` of length `npar(fop)` flagging which entries of a `TN`
vector are estimated by the fit. All of them, unless `fop` is set up for an
N-only optimization (see [`isonlyN`](@ref), [`fitNs!`](@ref)): then only `L`
(index 1) and the `N`s (even indices) are free, i.e. `[true, true, false, true, ...]`,
while the `T`s (odd indices > 1) are held fixed.
"""
function freemask(fop::FitOptions)
    n = npar(fop)
    isonlyN(fop) || return trues(n)
    mask = falses(n)
    mask[1] = true
    mask[2:2:end] .= true
    return mask
end

"""
    setOptimOptions!(fop::FitOptions; kwargs...)

Set the options which are passed to `Optimization.solve`, see
[Optimization.jl](https://docs.sciml.ai/Optimization/stable/API/solve/#Common-Solver-Options-(Solve-Keyword-Arguments)).
and the specific `Optim.jl` section, which is the default optimizer. Defaults are:
- `solver`: The solver to use for the optimization, default is `LBFGS()`.
- `maxiters = 6000`
- `maxtime = 60` (in seconds)
- `g_tol = 5e-8`
If given more parameters, they are passed to the optimizer.
"""
function setOptimOptions!(fop::FitOptions;
    solver = fop.solver,
    maxiters = fop.opt.maxiters,
    maxtime = fop.opt.maxtime,
    g_tol = fop.opt.g_tol,
    kwargs...
)
    fop.opt = (; maxiters, maxtime, g_tol, kwargs...)
    fop.solver = solver
    return nothing
end