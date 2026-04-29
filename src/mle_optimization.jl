function getHessian(m::Turing.Optimisation.ModeResult; kwargs...)
    return Turing.Optimisation.StatsBase.informationmatrix(m; kwargs...)
end

# models

@model function model_epochs(edges::AbstractVector{<:Integer}, 
    counts::AbstractVector{<:Integer}, mu::Float64, locut::Int,
    TNdists::Vector{<:Distribution}
)
    TN ~ product_distribution(TNdists)
    a = 0.5
    last_hid_I = laplacekingmanint(edges[locut] - a, mu, TN)
    for i in locut:length(counts)
        @inbounds this_hid_I = laplacekingmanint(edges[i+1] - a, mu, TN)
        m = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
        if (m < 0) || isnan(m)
            # this happens when evaluating the model
            # after optimization, in the unconstrained
            # space, using Bijectors.
            # I could not find a mwe, (TODO: find one)
            # probably out of domain, apply a penalty
            m = 0
        end
        @inbounds counts[i] ~ Poisson(m)
    end
end

function llike(edges::AbstractVector{<:Integer}, 
    counts::AbstractVector{<:Integer}, mu::Float64, locut::Int,
    TN::AbstractVector{<:Real}
)
    ll = 0.
    a = 0.5
    last_hid_I = laplacekingmanint(edges[locut] - a, mu, TN)
    for i in locut:length(counts)
        @inbounds this_hid_I = laplacekingmanint(edges[i+1] - a, mu, TN)
        m = this_hid_I - last_hid_I
        @assert !isnan(m) TN
        @assert m>0 TN
        last_hid_I = this_hid_I
        ll += logpdf(Poisson(m), counts[i])
    end
    return ll
end

# unused
@model function model_corrected(edges::AbstractVector{<:Integer}, 
    counts::AbstractVector{<:Integer}, mu::Float64, rate::Float64, locut::Int,
    TNdists::Vector{<:Distribution}, corrections::AbstractVector{<:Real}
)
    TN ~ product_distribution(TNdists)
    a = 0.5
    last_hid_I = firstorderint(edges[locut] - a, rate, TN) * 2 * mu^2 * TN[1] / rate
    for i in locut:length(counts)
        @inbounds this_hid_I = firstorderint(edges[i+1] - a, rate, TN) * 2 * mu^2 * TN[1] / rate
        m = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
        if (m < 0) || isnan(m)
            # this happens when evaluating the model
            # after optimization, in the unconstrained
            # space, using Bijectors.
            # I could not find a mwe, (TODO: find one)
            # probably out of domain, apply a penalty
            m = 0
        end
        @inbounds counts[i] ~ Poisson(m + corrections[i])
    end
end

@model function modelsmcp!(dc::IntegralArrays, rs::AbstractVector{<:Real}, 
    edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer},
    mu::Float64, rho::Float64, locut::Int, TNdists::Vector{<:Distribution}
)
    TN ~ product_distribution(TNdists)
    mldsmcp!(dc, 1:dc.order, rs, edges, mu, rho, TN)
    m = get_tmp(dc.ys, eltype(TN))
    m .*= diff(edges)
    for i in locut:length(counts)
        if (m[i] < 0) || isnan(m[i])
            # this happens when evaluating the model
            # after optimization, in the unconstrained
            # space, using Bijectors.
            # I could not find a mwe, (TODO: find one)
            # probably out of domain, apply a penalty
            m[i] = 0
        end
        @inbounds counts[i] ~ Poisson(m[i])
    end
end

function llsmcp!(dc::IntegralArrays, rs::AbstractVector{<:Real}, 
    edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer},
    mu::Float64, rho::Float64, locut::Int, TN::AbstractVector{<:Real}
)
    mldsmcp!(dc, 1:dc.order, rs, edges, mu, rho, TN)
    ws = get_tmp(dc.ys, eltype(TN)) .* diff(edges)
    return llsmcp(ws, counts, locut)
end

function llsmcp(ws::AbstractVector{<:Real}, counts::AbstractVector{<:Integer},
    locut::Int
)
    ll = 0
    for i in locut:length(counts)
        if (ws[i] < 0) || isnan(ws[i])
            # this happens when evaluating the model
            # after optimization, in the unconstrained
            # space, using Bijectors.
            # I could not find a mwe, (TODO: find one)
            # probably out of domain, apply a penalty
            ws[i] = 0
        end
        @inbounds ll += logpdf(Poisson(ws[i]),counts[i])
    end
    return ll
end

# --- fitting

function fit_model_epochs!(options::FitOptions, h::Histogram{T,1,E};
    stats = true
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    fit_model_epochs!(options, h.edges[1], h.weights, Val(isnaive(options)); stats)
end


function fit_model_epochs!(
    options::FitOptions, edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, 
    ::Val{true};
    stats = true
)
    # get a good initial guess
    iszero(options.init) && initialize!(options, counts)
    pars_ = InitFromParams(VarNamedTuple(; TN = options.init))

    model = model_epochs(edges, counts, options.mu, options.locut, options.prior)
    logger = ConsoleLogger(stdout, Logging.Error)
    mle = with_logger(logger) do
        Turing.Optimisation.estimate_mode(
            model, MLE(), options.solver; initial_params=pars_, options.opt...
        )
    end
    return getFitResult(mle, options, counts, edges; stats)
end

# unused
function fit_model_epochs!(
    options::FitOptions, edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, 
    ::Val{false};
    stats = true
)

    # get a good initial guess
    iszero(options.init) && initialize!(options, counts)
    pars_ = InitFromParams(VarNamedTuple(; TN = options.init))

    # run the optimization
    rs = midpoints(edges)
    dc = IntegralArrays(options.order, options.ndt, length(rs), Val{length(options.init)}, 3)
    model = modelsmcp!(dc, rs, edges, counts, options.mu, options.rho, options.locut, options.prior)
    logger = ConsoleLogger(stdout, Logging.Error)
    mle = with_logger(logger) do
        Turing.Optimisation.estimate_mode(
            model, MLE(), options.solver; initial_params=pars_, options.opt...
        )
    end
    return getFitResult(mle, options, counts, edges; stats)
end

function getFitResult(mle, options::FitOptions, counts, edges; stats = true)
    para = mle.params[@varname(TN)]
    lp = mle.lp
    
    if stats
        hess = getHessian(mle)
    else
        hess = nothing
    end
    return getFitResult(hess, para, lp, mle.optim_result, options, counts, edges, stats)
end

function getFitResult(hess, para, lp, optim_result, options::FitOptions, counts, edges, stats)
    if stats
        eigen_problem = eigen(hess)
        lambdas = eigen_problem.values
    else
        eigen_problem = nothing
        lambdas = nothing
    end

    at_uboundary = map((x,u) -> (x>u/1.05), para, options.upp)
    at_lboundary = map((l,x) -> (x<l*1.05), options.low, para)
    stderrors = fill(Inf, length(para))
    zscore = fill(0.0, length(para))
    p = fill(1, length(para))
    q025 = fill(0.0, length(para))
    q975 = fill(0.0, length(para))
    q05 = fill(0.0, length(para))
    q95 = fill(0.0, length(para))
    q50 = fill(0.0, length(para))
    logevidence = -Inf
    marglike = 0
    convex_opt = false
    if stats && isreal(lambdas)
        lambdas = real.(lambdas)
        if all(lambdas .> 0)
            convex_opt = true
        end
        lambdas[lambdas .<= 0] .= eps()
        covar = eigen_problem.vectors *
            diagm(inv.(lambdas)) * eigen_problem.vectors'
        vars_ = diag(covar)
        stderrors = sqrt.(vars_)
        zscore = para ./ stderrors
        p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:right), zscore)
    
        # Confidence interval (CI)
        cut_vec = eigen(hess[2:end, 2:end]).vectors
        qs = slice(para, cut_vec, edges, counts, options)
        q025 .= qs[:,1]
        q975 .= qs[:,end]
        q05 .= qs[:,2]
        q95 .= qs[:,end-1]
        q50 .= qs[:,3]
    
        marglike = mvnormcdf(para, covar, options.low, options.upp)
        # assuming uniform prior and Taylor expansion of the log-like around the mode
        logevidence = lp + sum(log.(1.0 ./ (options.upp - options.low))) +
            log(marglike[1])
    end

    FitResult(
        options.nepochs,
        length(counts),
        options.mu,
        options.rho,
        q50,
        q025,
        q975,
        para,
        stderrors,
        summary(options.solver),
        Turing.Optimisation.SciMLBase.successful_retcode(optim_result),
        lp,
        logevidence,
        (;
            optim_result,
            at_any_boundary = any(at_uboundary) || any(at_lboundary), 
            at_uboundary, at_lboundary,
            options.low, options.upp, options.init,
            zscore, pvalues = p, q05, q95,
            convex_opt, marglike,
            hess)
    )
end

"""
    sample_model_epochs(options::FitOptions, h::Histogram{T,1,E}, fit::FitResult; nsamples = 10_000, naive = isnaive(options))

Sample `nsamples` from the posterior distribution of the parameters, starting
from initial point in MLE `fit` obtained from [`demoinfer`](@ref).

Requires the observed histogram `h` and the fit options `options`.
Return a `Chains` object from the `MCMCDiagnostics` module of `Turing`,
which contains the samples from the posterior distribution.
If `naive` is false, the sampling will be done using the SMC' likelihood, which is more accurate but
also more computationally intensive. If `naive` is true, the sampling will
be done using the closed-form integral likelihood, which requires to use a
modified histogram `h_mod` as output by [`demoinfer`](@ref).
"""
function sample_model_epochs(options::FitOptions, h::Histogram{T,1,E}, 
    fit::FitResult; nsamples::Int=10_000, naive = isnaive(options)
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    options_ = deepcopy(options)
    setnepochs!(options_, fit.nepochs)
    setnaive!(options_, naive)
    sample_model_epochs!(options_, fit, h.edges[1], h.weights, Val(isnaive(options_)); nsamples)
end

function sample_model_epochs!(
    options::FitOptions, fit::FitResult, edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer},
    ::Val{true};
    nsamples::Int=10_000
)
    setinit!(options, get_para(fit))

    model = model_epochs(edges, counts, options.mu, options.locut, options.prior)
    logger = ConsoleLogger(stdout, Logging.Error)
    
    init_ = InitFromParams(VarNamedTuple(; TN = options.init))
    chain = with_logger(logger) do
        sample(model, NUTS(), nsamples; initial_params=init_)
    end
    return chain
end

function sample_model_epochs!(
    options::FitOptions, fit::FitResult, edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer},
    ::Val{false};
    nsamples::Int=10_000
)
    setinit!(options, get_para(fit))
    covar = get_covar(fit)

    logger = ConsoleLogger(stdout, Logging.Error)
    
    init_ = InitFromParams(VarNamedTuple(; TN = options.init))
    rs = midpoints(edges)
    dc = IntegralArrays(options.order, options.ndt, length(rs), Val{length(options.init)}, 3)
    model = modelsmcp!(dc, rs, edges, counts, options.mu, options.rho, options.locut, options.prior)
    chain = with_logger(logger) do
        sample(model, MH(covar), nsamples; initial_params=init_)
    end
    return chain
end

# log-likelihood (and posterior) slices

function slice(TN::AbstractVector{<:Real}, eigenvec::AbstractMatrix{<:Real},
    edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, options::FitOptions;
    ngrid = 5_000, ps = [0.025, 0.05, 0.5, 0.95, 0.975]
)
    grid = zeros(2ngrid)
    qs = zeros(length(TN), length(ps))
    qs_ = zeros(length(TN), length(ps))
    for i in 1:length(TN)-1
        dir = eigenvec[:, i]
        pushfirst!(dir, 0.)
        # TN + λ * dir >= options.low
        # TN + λ * dir <= options.upp
        lmin = (options.low .- TN) ./ dir
        lmax = (options.upp .- TN) ./ dir
        for j in eachindex(TN)
            if lmin[j] > lmax[j]
                lmin[j], lmax[j] = lmax[j], lmin[j]
            end
        end
        lmin = maximum(lmin[2:end]) - 2/ngrid
        lmax = minimum(lmax[2:end]) + 2/ngrid
        lambdasp = logrange(1/ngrid, lmax, ngrid)
        lambdasn = logrange(abs(lmin), 1/ngrid, ngrid)
        grid .= 0.
        for j in 1:ngrid
            v = TN .- lambdasn[j] * dir
            @assert all(options.low/1.01 .<= v .<= options.upp*1.01) v
            ll = llike(edges, counts, options.mu, options.locut, v)
            grid[j] = ll
            v = TN .+ lambdasp[j] * dir
            @assert all(options.low/1.01 .<= v .<= options.upp*1.01) v
            ll = llike(edges, counts, options.mu, options.locut, v)
            grid[ngrid+j] = ll
        end
        grid .= grid .- maximum(grid)
        grid .= exp.(grid)
        trapz = (grid[1:end-1] .+ grid[2:end]) ./ 2
        for j in 2:ngrid
            trapz[j-1] *= -lambdasn[j] + lambdasn[j-1]
            trapz[ngrid+j-1] *= lambdasp[j] - lambdasp[j-1]
        end
        trapz[ngrid] *= lambdasp[1] - (-lambdasn[end])
        cum = cumsum(trapz)
        @assert !iszero(cum[end])
        cum ./= cum[end]
        for j in eachindex(ps)
            qi = findfirst(cum .>= ps[j])
            if qi <= ngrid
                qs_[:,j] .= -lambdasn[qi] * dir
            else
                qs_[:,j] .= lambdasp[qi-ngrid] * dir
            end
        end
        sort!(qs, dims=2)
        qs .+= qs_
    end
    qs .+= TN
    return qs
end