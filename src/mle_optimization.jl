# the Hessian is taken w.r.t. the free parameters only, see `freemask`
function getHessian(edges::AbstractVector{<:Integer},
    counts::AbstractVector{<:Integer},
    options::FitOptions, TN::AbstractVector{<:Real}
)
    return getHessian(edges, counts, options, TN, freemask(options), Val(isnaive(options)))
end

function getHessian(edges::AbstractVector{<:Integer},
    counts::AbstractVector{<:Integer},
    options::FitOptions, TN::AbstractVector{<:Real}, mask::BitVector, ::Val{true}
)
    # information matrix is the negative Hessian of the log-likelihood
    return ForwardDiff.hessian(
        x -> -llike(edges, counts, options.mu, options.locut, merge_free(TN, x, mask)),
        TN[mask]
    )
end

function getHessian(edges::AbstractVector{<:Integer},
    counts::AbstractVector{<:Integer},
    options::FitOptions, TN::AbstractVector{<:Real}, mask::BitVector, ::Val{false}
)
    # information matrix is the negative Hessian of the log-likelihood
    return ForwardDiff.hessian(
        x -> -llsmcp!(options.diffcache, edges, counts,
            options.mu, options.rho, options.locut, merge_free(TN, x, mask)),
        TN[mask]
    )
end

"""
    merge_free(TNfixed::AbstractVector{<:Real}, x::AbstractVector{<:Real}, mask::BitVector)

Splice the free parameters `x` into a copy of the full `TN` vector `TNfixed`
at the positions flagged by `mask`, see [`freemask`](@ref). The other entries
keep the values in `TNfixed`.
"""
function merge_free(TNfixed::AbstractVector{<:Real}, x::AbstractVector{T},
    mask::BitVector
) where {T<:Real}
    TN = similar(TNfixed, T)
    TN .= TNfixed
    TN[mask] .= x
    return TN
end

# models
# `TN` holds only the free parameters, `TNdists` their priors,
# the full vector `TNfull` is rebuilt with `merge_free`

@model function model_epochs(edges::AbstractVector{<:Integer}, 
    counts::AbstractVector{<:Integer}, mu::Float64, locut::Int,
    TNdists::Vector{<:Distribution}, TNfixed::AbstractVector{<:Real}, mask::BitVector
)
    TN ~ product_distribution(TNdists)
    TNfull = merge_free(TNfixed, TN, mask)
    a = 0.5
    last_hid_I = laplacekingmanint(edges[locut] - a, mu, TNfull)
    for i in locut:length(counts)
        @inbounds this_hid_I = laplacekingmanint(edges[i+1] - a, mu, TNfull)
        m = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
        if (m < 0) || isnan(m)
            # this happens when evaluating the model
            # after optimization, in the unconstrained
            # space, using Bijectors.
            # I could not find a mwe, (TODO: find one)
            # probably out of domain, apply a penalty
            @addlogprob!(-Inf)
            return
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

@model function modelsmcp!(dc::IntegralArrays, edges::AbstractVector{<:Integer},
    counts::AbstractVector{<:Integer},
    mu::Float64, rho::Float64, locut::Int, TNdists::Vector{<:Distribution},
    TNfixed::AbstractVector{<:Real}, mask::BitVector
)
    TN ~ product_distribution(TNdists)
    TNfull = merge_free(TNfixed, TN, mask)
    mldsmcp!(dc, mu, rho, TNfull)
    map_fine_to_coarse!(dc, edges, eltype(TNfull))
    m = get_tmp(dc.wcoarse, eltype(TNfull))
    @assert length(m) == length(counts)
    for i in locut:length(counts)
        if (m[i] < 0) || isnan(m[i])
            # this happens when evaluating the model
            # after optimization, in the unconstrained
            # space, using Bijectors.
            # I could not find a mwe, (TODO: find one)
            # probably out of domain, apply a penalty
            # for this branch (smcp) it can also occur
            # for epochs with large cumulative coalescence
            # i.e. long and small Ne
            @addlogprob!(-Inf)
            return
        end
        @inbounds counts[i] ~ Poisson(m[i])
    end
end

function llsmcp!(dc::IntegralArrays, edges::AbstractVector{<:Integer},
    counts::AbstractVector{<:Integer},
    mu::Float64, rho::Float64, locut::Int, TN::AbstractVector{<:Real}
)
    mldsmcp!(dc, mu, rho, TN)
    map_fine_to_coarse!(dc, edges, eltype(TN))
    return llsmcp(get_tmp(dc.wcoarse, eltype(TN)), counts, locut)
end

function llsmcp(ws::AbstractVector{<:Real}, counts::AbstractVector{<:Integer},
    locut::Int
)
    @assert length(ws) == length(counts)
    ll = 0
    for i in locut:length(counts)
        if (ws[i] < 0) || isnan(ws[i])
            # this happens when evaluating the model
            # after optimization, in the unconstrained
            # space, using Bijectors.
            # I could not find a mwe, (TODO: find one)
            # probably out of domain, apply a penalty
            #
            # `-Inf * one(eltype(ws))`, not a bare `-Inf`: under ForwardDiff a
            # bare Float64 return makes the function type-unstable, and the
            # partials are extracted as ZERO rather than propagated. The
            # gradient is then exactly 0, which satisfies any `g_tol`, and
            # LBFGS reports `converged = true` at iteration 0 on a point it
            # could not evaluate. Preserving the type gives NaN partials
            # instead, so the convergence test can never fire here.
            return -Inf * one(eltype(ws))
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
    setonlyN!(options, false)
    fit_model_epochs!(options, h.edges[1], h.weights, Val(isnaive(options)); stats)
end

"""
    fitNs!(options::FitOptions, h::Histogram; stats = true)

Like [`fit_model_epochs!`](@ref), but only estimates the total genome length
`L` and the population sizes `N`, holding the epoch durations `T` fixed at
the values in `options.init` (set via `initialize!` if not already provided).
Set `options` for an N-only optimization, see [`freemask`](@ref), which
persists on `options` until the next call to [`fit_model_epochs!`](@ref).
"""
function fitNs!(options::FitOptions, h::Histogram{T,1,E}; stats = true
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    @assert options.locut >= 1 "locut has to be at least 1"
    @assert options.locut <= length(h.weights) "locut cannot be greater than number of bins"
    setonlyN!(options, true)
    fit_model_epochs!(options, h.edges[1], h.weights, Val(isnaive(options)); stats)
end


function fit_model_epochs!(
    options::FitOptions, edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, 
    ::Val{true};
    stats = true
)
    # get a good initial guess
    iszero(options.init) && initialize!(options, counts)
    mask = freemask(options)
    pars_ = InitFromParams(VarNamedTuple(; TN = options.init[mask]))

    model = model_epochs(edges, counts, options.mu, options.locut,
        options.prior[mask], options.init, mask)
    logger = ConsoleLogger(stdout, Logging.Error)
    mle = with_logger(logger) do
        Turing.Optimisation.estimate_mode(
            model, MLE(), options.solver; initial_params=pars_, options.opt...
        )
    end
    return getFitResult(mle, options, edges, counts; stats)
end

function fit_model_epochs!(
    options::FitOptions, edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, 
    ::Val{false};
    stats = true
)

    # get a good initial guess
    iszero(options.init) && initialize!(options, counts)
    mask = freemask(options)
    pars_ = InitFromParams(VarNamedTuple(; TN = options.init[mask]))

    # run the optimization
    @assert !isnothing(options.diffcache) "Diffcache is not initialized"
    model = modelsmcp!(options.diffcache, edges, counts,
        options.mu, options.rho, options.locut, options.prior[mask], options.init, mask)
    logger = ConsoleLogger(stdout, Logging.Error)
    mle = with_logger(logger) do
        Turing.Optimisation.estimate_mode(
            model, MLE(), options.solver; initial_params=pars_, options.opt...
        )
    end
    para = merge_free(options.init, mle.params[@varname(TN)], mask)
    mldsmcp!(options.diffcache, options.mu, options.rho, para)
    map_fine_to_coarse!(options.diffcache, edges, eltype(para))
    return getFitResult(mle, options, edges, counts; stats)
end

function getFitResult(mle, options::FitOptions, edges, counts; stats = true)
    para = merge_free(options.init, mle.params[@varname(TN)], freemask(options))
    lp = mle.lp
    
    if stats
        hess = getHessian(edges, counts, options, para)
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

    # fixed parameters are never at the boundary, have zero stderror
    # and a confidence interval collapsed to their value
    mask = freemask(options)
    at_uboundary = map((x,u) -> (x>u/1.05), para, options.upp) .& mask
    at_lboundary = map((l,x) -> (x<l*1.05), options.low, para) .& mask
    stderrors = ifelse.(mask, Inf, 0.0)
    ci_low = ifelse.(mask, -Inf, para)
    ci_high = ifelse.(mask, Inf, para)
    logevidence = -Inf
    marglike = 0
    convex_opt = false
    optflag = true
    if stats && isreal(lambdas)
        lambdas = real.(lambdas)
        if all(lambdas .> 0)
            convex_opt = true
        end
        lambdas[lambdas .<= 0] .= eps()
        covar = eigen_problem.vectors *
            diagm(inv.(lambdas)) * eigen_problem.vectors'
        vars_ = diag(covar)
        stderrors[mask] .= sqrt.(vars_)

        counts_ = copy(counts)
        if !isnaive(options)
            wn = integral_ws(edges, options.mu, para)
            w = get_tmp(options.diffcache.wcoarse, eltype(para))
            resid = (counts .- w) ./ sqrt.(w)
            wn .= wn .+ resid .* sqrt.(wn)
            counts_ .= round.(Int, max.(0, wn))
        end
        # assuming uniform prior on N and T and separability of the likelihood
        ci_low, ci_high, marglike, optflag = slice(para, eigen_problem.vectors, edges, counts_, options)
        logevidence = lp + sum(log.(1.0 ./ (options.upp .- options.low))[mask]) + log(marglike)
        if (!optflag && isnaive(options)) || isinf(logevidence)
            logevidence = -Inf
        end
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
        mask,
        (;
            optim_result,
            at_any_boundary = any(at_uboundary) || any(at_lboundary), 
            at_uboundary, at_lboundary,
            low = copy(options.low), upp = copy(options.upp), init = copy(options.init),
            ci_low, ci_high,
            convex_opt, marglike, optflag,
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
    setonlyN!(options_, false)
    sample_model_epochs!(options_, fit, h.edges[1], h.weights, Val(isnaive(options_)); nsamples)
end

"""
    sampleNs_posterior(options::FitOptions, h::Histogram{T,1,E}, fit::FitResult; nsamples = 10_000, naive = isnaive(options))

Like [`sample_model_epochs`](@ref), but only samples from the posterior of
the total genome length `L` and the population sizes `N`, holding the epoch
durations `T` fixed at the values in `fit`, see [`fitNs!`](@ref) for the
matching MLE optimization. The chain holds only the free parameters, in the
order of [`freemask`](@ref).
"""
function sampleNs_posterior(options::FitOptions, h::Histogram{T,1,E},
    fit::FitResult; nsamples::Int=10_000, naive = isnaive(options)
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    options_ = deepcopy(options)
    setnepochs!(options_, fit.nepochs)
    setnaive!(options_, naive)
    setonlyN!(options_, true)
    sample_model_epochs!(options_, fit, h.edges[1], h.weights, Val(isnaive(options_)); nsamples)
end

function sample_model_epochs!(
    options::FitOptions, fit::FitResult, edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer},
    ::Val{true};
    nsamples::Int=10_000
)
    setinit!(options, get_para(fit))
    mask = freemask(options)

    model = model_epochs(edges, counts, options.mu, options.locut,
        options.prior[mask], options.init, mask)
    logger = ConsoleLogger(stdout, Logging.Error)
    
    init_ = InitFromParams(VarNamedTuple(; TN = options.init[mask]))
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
    mask = freemask(options)

    logger = ConsoleLogger(stdout, Logging.Error)
    
    init_ = InitFromParams(VarNamedTuple(; TN = options.init[mask]))
    @assert !isnothing(options.diffcache) "Diffcache is not initialized"
    model = modelsmcp!(options.diffcache, edges, counts,
        options.mu, options.rho, options.locut, options.prior[mask], options.init, mask)
    chain = with_logger(logger) do
        sample(model, MH(covar), nsamples; initial_params=init_)
    end
    return chain
end

# log-likelihood (and posterior) slices
# `eigenvec` spans the free parameters only, see `freemask`,
# the fixed ones are neither moved nor bounded

function slice(TN::AbstractVector{<:Real}, eigenvec::AbstractMatrix{<:Real},
    edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, options::FitOptions;
    ngrid = 250
)
    ll_hat = llike(edges, counts, options.mu, options.locut, TN)
    ll_threshold = ll_hat - 2
    optflag = true
    marglike = 1.0
    offset_low  = zeros(length(TN))
    offset_high = zeros(length(TN))
    v = similar(TN)
    dir = zeros(length(TN))
    mask = freemask(options)
    low = ifelse.(mask, options.low, -Inf)
    upp = ifelse.(mask, options.upp, Inf)
    global_lmax = maximum((options.upp .- options.low)[mask])
    lambdas = logrange(1/ngrid, global_lmax, ngrid)
    for i in 1:size(eigenvec, 2)
        dir[mask] .= view(eigenvec, :, i)
        lambda_pos = global_lmax
        sum = eps()
        llp = ll_hat
        dx = 1/ngrid
        for j in 1:ngrid
            v .= TN .+ lambdas[j] * dir
            ll = llike(edges, counts, options.mu, options.locut,
                clamp.(v, low, upp)
            )
            if ll >= ll_threshold
                lambda_pos = lambdas[j]
            end
            if ll > ll_hat + 0.1 # 10% tolerance
                optflag = false
            end
            if j > 1
                dx = lambdas[j] - lambdas[j-1]
            end
            if all(v .> low) && all(v .< upp)
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
                clamp.(v, low, upp)
            )
            if ll >= ll_threshold
                lambda_neg = lambdas[j]
            end
            if ll > ll_hat + 0.1
                optflag = false
            end
            if j > 1
                dx = lambdas[j] - lambdas[j-1]
            end
            if all(v .> low) && all(v .< upp)
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
    q_low  = clamp.(TN .+ offset_low,  low, upp)
    q_high = clamp.(TN .+ offset_high, low, upp)
    return q_low, q_high, marglike, optflag
end