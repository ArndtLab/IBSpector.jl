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
- `convex_optimum`: whether the likelihood Hessian at the optimum
  is **strictly** positive definite.
- `ci_low`: whether the confidence interval of any parameter includes zero
- `at_any_boundary`: whether any parameter is at its lower or upper bound
- `log_like`: the log-likelihood of the fit
- `log_evidence`: the log-evidence of the fit
- `optimizer_message`: the original message from the optimizer,
  which can be useful for diagnosing optimization issues.
"""
function flags(fit::FitResult)
    return (;
        converged = fit.converged,
        convex_optimum = fit.opt.convex_opt,
        ci_low = any(fit.opt.ci_low .< 0),
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
    level::Float64
    force::Bool
    maxnts::Int
    naive::Bool
    order::Int
    ndt::Int
    locut::Int
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
- `level::Float64=0.95`: The confidence level for the confidence intervals on the parameters estimates.
- `force::Bool=true`: if true try to fit further epochs even when no signal is found.
- `maxnts::Int=5`: The maximum number of new time splits to consider when adding a new epoch.
  Higher is greedier.
- `order::Int=0`: maximum number of higher order SMC' corrections to account for
  (i.e. number of intermediate recombination events plus one). When zero, it
  is set automatically.
- `ndt::Int=0`: number of Legendre nodes to use for numerical integration.
  When zero, it is set automatically.
- `locut::Int=1`: index of the first histogram bin to consider in the fit.
"""
function FitOptions(Ltot, nhet, mu, rho;
    Tlow = 10, Tupp = 1e7,
    Nlow = 10, Nupp = 1e8,
    level = 0.95,
    nepochs::Int = 1,
    force::Bool = true,
    maxnts::Int = 5,
    naive::Bool = true,
    order::Int = 0,
    ndt::Int = 0,
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

    if iszero(ndt)
        if nhet > 1e7
            ndt = 1600
        else
            ndt = 800
        end
    end
    cutoff = 2e-5 # fraction of segments contributing to higher orders
    if iszero(order)
        o = findfirst(map(i->fraction(mu,rho,i),1:50) .< cutoff)
        isnothing(o) && (o = 50)
        order = o
    end

    solver = LBFGS()
    maxiters = 30000
    maxtime = 120
    g_tol = 5e-8
    if nhet > 1e7
        maxiters = 30000
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
        level,
        force,
        maxnts,
        naive,
        order,
        ndt,
        locut
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


"""
    setinit!(fop::FitOptions, init::AbstractVector{Float64})

Set the initial vector of parameters for the optimization which takes the `FitOptions` object `fop`.
"""
function setinit!(fop::FitOptions, init::AbstractVector{Float64})
    @assert length(init) == npar(fop)
    fop.init .= init
    for i in eachindex(fop.init)
        fop.init[i] <= fop.low[i] ? fop.init[i] = fop.low[i] * 1.001 : nothing
        fop.init[i] >= fop.upp[i] ? fop.init[i] = fop.upp[i] * 0.999 : nothing
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
            isinf(evd(fit)) ||
            !fit.converged
    end
end

function reset_perturb!(fop::FitOptions)
    fop.perturb .= falses(npar(fop))
    fop.delta.state = 0
end

struct PInit <: AbstractVector{Float64}
    fop::FitOptions
end

Base.size(p::PInit) = (npar(p.fop),)

getdelta(fop::FitOptions) = fop.delta.factors[fop.delta.state]

function Base.getindex(p::PInit, i::Int)
    if !p.fop.perturb[i]
        return p.fop.init[i]
    else
        dl = getdelta(p.fop)
        low = p.fop.low[i]
        upp = p.fop.upp[i]
        if dl < 1
            return rand(
                truncated(
                    LogNormal(log(p.fop.init[i]), dl),
                    low,
                    upp
                )
            )
        else
            return rand(Uniform(low, upp))
        end
    end
end

function isnaive(fop::FitOptions)
    return fop.naive
end

function setnaive!(fop::FitOptions, flag::Bool)
    fop.naive = flag
end

"""
    setOptimOptions!(fop::FitOptions; kwargs...)

Set the options which are passed to `Optimization.solve`, see
[Optimization.jl](https://docs.sciml.ai/Optimization/stable/API/solve/#Common-Solver-Options-(Solve-Keyword-Arguments)).
and the specific `Optim.jl` section, which is the default optimizer. Defaults are:
- `solver`: The solver to use for the optimization, default is `LBFGS()`.
- `maxiters = 30000`
- `maxtime = 120` (in seconds)
- `g_tol = 5e-8`
If given more parameters, they are passed to the optimizer.
"""
function setOptimOptions!(fop::FitOptions;
    solver = LBFGS(),
    maxiters = 30000,
    maxtime = 120,
    g_tol = 5e-8,
    kwargs...
)
    fop.opt = (; maxiters, maxtime, g_tol, kwargs...)
    fop.solver = solver
    return nothing
end