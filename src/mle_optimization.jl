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
    return getFitResult(mle, options, counts; stats)
end

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
    return getFitResult(mle, options, counts; stats)
end

function getFitResult(mle, options::FitOptions, counts; stats = true)
    para = mle.params[@varname(TN)]
    lp = mle.lp
    
    if stats
        hess = getHessian(mle)
    else
        hess = nothing
    end
    return getFitResult(hess, para, lp, mle.optim_result, options, counts, stats)
end

function getFitResult(hess, para, lp, optim_result, options::FitOptions, counts, stats)
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
    ci_low = fill(-Inf, length(para))
    ci_high = fill(Inf, length(para))
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
        q = Statistics.quantile(Distributions.Normal(), (1 + options.level) / 2)
        ci_low = para .- q .* stderrors
        ci_high = para .+ q .* stderrors
    
        marglike = mvnormcdf(para, covar, options.low, options.upp)
        # assuming uniform prior
        logevidence = lp + sum(log.(1.0 ./ (options.upp - options.low))) +
            log(marglike[1])
    end

    FitResult(
        options.nepochs,
        length(counts),
        options.mu,
        options.rho,
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
            zscore, pvalues = p, ci_low, ci_high,
            convex_opt, marglike,
            hess)
    )
end

"""
    sample_model_epochs!(options::FitOptions, h::Histogram{T,1,E}, init::AbstractVector{<:Real}; nsamples = 10_000, findmode = false)

Sample `nsamples` from the posterior distribution of the parameters, starting
from initial point `init`.

Requires the observed histogram `h` and the fit options `options`.
Return a `Chains` object from the `MCMCDiagnostics` module of `Turing`,
which contains the samples from the posterior distribution.
If `findmode` is true, the function will first find the mode of the
posterior distribution using optimization, and then use that as the
initial point for sampling. Otherwise, it will use the provided
`init` as the initial point for sampling.
"""
function sample_model_epochs!(options::FitOptions, h::Histogram{T,1,E}, 
    init::AbstractVector{<:Real}; nsamples::Int=10_000, findmode = false
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    @assert length(init)%2 == 0 "initial parameters should be of length 2*nepochs"
    setnepochs!(options, length(init)÷2)
    setinit!(options, init)
    sample_model_epochs!(options, h.edges[1], h.weights, Val(isnaive(options)); nsamples, findmode)
end

function sample_model_epochs!(
    options::FitOptions, edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer},
    ::Val{true};
    nsamples::Int=10_000, findmode = false
)
    # get a good initial guess
    iszero(options.init) && initialize!(options, counts)

    model = model_epochs(edges, counts, options.mu, options.locut, options.prior)
    logger = ConsoleLogger(stdout, Logging.Error)
    if findmode
        init_ = InitFromParams(VarNamedTuple(; TN = options.init))
        mle = with_logger(logger) do
            Turing.Optimisation.estimate_mode(
                model, MLE(), options.solver; initial_params=init_, options.opt...
            )
        end
        setinit!(options, mle.params[@varname(TN)])
    end
    init_ = InitFromParams(VarNamedTuple(; TN = options.init))
    chain = with_logger(logger) do
        sample(model, NUTS(), nsamples; initial_params=init_)
    end
    return chain
end