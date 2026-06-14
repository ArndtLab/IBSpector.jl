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
    @assert options.locut >= 1 "locut has to be at least 1"
    @assert options.locut <= length(h.weights) "locut cannot be greater than number of bins"
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
    return getFitResult(mle, options, edges, counts; stats)
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
    return getFitResult(mle, options, edges, counts; stats)
end

function getFitResult(mle, options::FitOptions, edges, counts; stats = true)
    para = mle.params[@varname(TN)]
    lp = mle.lp
    
    if stats
        hess = getHessian(mle)
    else
        hess = nothing
    end
    return getFitResult(hess, para, lp, mle.optim_result, options, edges, counts, stats)
end

function getFitResult(hess, para, lp, optim_result, options::FitOptions, edges, counts, stats)
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

        # assuming uniform prior on N and T and separability of the likelihood
        ci_low, ci_high, marglike = slice(para, eigen_problem.vectors, edges, counts, options)
        logevidence = lp + sum(log.(1.0 ./ (options.upp .- options.low))) + log(marglike)
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
            ci_low, ci_high,
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
        sample(model, NUTS(1000, 0.65; init_ϵ=0.1), nsamples; initial_params=init_)
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
    ngrid = 5_000
)
    ll_hat = llike(edges, counts, options.mu, options.locut, TN)
    ll_threshold = ll_hat - 2
    marglike = 1.0
    offset_low  = zeros(length(TN))
    offset_high = zeros(length(TN))
    v = similar(TN)
    global_lmax = maximum(options.upp .- options.low)
    lambdas = logrange(1/ngrid, global_lmax, ngrid)
    for i in 1:size(eigenvec, 2)
        dir = view(eigenvec, :, i)
        lambda_pos = global_lmax
        sum = eps()
        llp = ll_hat
        dx = 1/ngrid
        for j in 1:ngrid
            v .= TN .+ lambdas[j] * dir
            ll = llike(edges, counts, options.mu, options.locut,
                clamp.(v, options.low, options.upp)
            )
            if ll >= ll_threshold
                lambda_pos = lambdas[j]
            end
            if j > 1
                dx = lambdas[j] - lambdas[j-1]
            end
            if all(v .>= options.low) && all(v .<= options.upp)
                sum += (exp(ll - ll_hat) + exp(llp - ll_hat))/2 * dx
            end
            llp = ll
        end
        # negative direction
        lambda_neg = global_lmax
        llp = ll_hat
        dx = 1/ngrid
        for j in 1:ngrid
            v .= TN .- lambdas[j] * dir
            ll = llike(edges, counts, options.mu, options.locut,
                clamp.(v, options.low, options.upp)
            )
            if ll >= ll_threshold
                lambda_neg = lambdas[j]
            end
            if j > 1
                dx = lambdas[j] - lambdas[j-1]
            end
            if all(v .>= options.low) && all(v .<= options.upp)
                sum += (exp(ll - ll_hat) + exp(llp - ll_hat))/2 * dx
            end
            llp = ll
        end
        marglike *= sum
        pos_offset = lambda_pos * dir
        neg_offset = - lambda_neg * dir
        for j in eachindex(TN)
            if pos_offset[j] > 0
                offset_high[j] += pos_offset[j]
            else
                offset_low[j] += pos_offset[j]
            end
            if neg_offset[j] > 0
                offset_high[j] += neg_offset[j]
            else
                offset_low[j] += neg_offset[j]
            end
        end
    end
    # clamp final bounds to parameter space
    q_low  = clamp.(TN .+ offset_low,  options.low, options.upp)
    q_high = clamp.(TN .+ offset_high, options.low, options.upp)
    return q_low, q_high, marglike
end