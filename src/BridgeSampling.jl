
module BridgeSampling
using Turing, StatsBase
using LinearAlgebra: dot
import Base: show

include("iterative_algorithm.jl")
include("proposal.jl")
include("transform.jl")
include("model_interface.jl")
include("formatting.jl")
include("error_estimate.jl")


export bridgesampling, value, error_estimate, bayes_factor, posterior_probabilities, show

struct LogMarginalLikelihood{TV}
    value::Float64
    niter::Int
    p_posterior::TV
    g_posterior::TV
    p_proposal::TV
    g_proposal::TV
end

""" 
    bridgesampling(samples, log_posterior, lb, ub; tol, maxiter, names)

    Estimates the log marginal likelihood with bridge sampling, given a matrix of posterior samples and an unnormalized log posterior function.

    # Arguments
    * `samples` : An ndims x nsamples matrix. The eltype can be either a scalar or an array of any dimension (e.g. multivariate parameters)
    * `log_posterior` : A function that takes one argument (one sample) and outputs the unnormalized log posterior probability (log joint)
    * `lb` and `ub` : respectively the lower and upper boundary of each dimension in the samples
    * `tol` : tolerance for the iterative scheme. Default = `1e-10`
    * `maxiter` : maximum number of iterations of the iterative scheme. Default = `1000`
    * `names` : A vector of names for each dimension in the samples. If specified, the input argument of `log_posterior` will be a `NamedTuple`. Default = `nothing`

    # Examples
    ```julia 
        # Bivariate standard normal distribution
        samples = rand(MvNormal(ones(2)), 10_000)

        log_posterior(x) = -0.5 * x' * x

        Bridge = bridgesampling(samples, log_posterior, [-Inf, -Inf], [Inf, Inf])
    ```

    # Sources 
        Quentin F. Gronau, Alexandra Sarafoglou, Dora Matzke, Alexander Ly, Udo Boehm, Maarten Marsman, David S. Leslie, Jonathan J. Forster, Eric-Jan Wagenmakers, Helen Steingroever,
        A tutorial on bridge sampling,
        Journal of Mathematical Psychology,
        Volume 81,
        2017,
        Pages 80-97,
        ISSN 0022-2496,
        https://doi.org/10.1016/j.jmp.2017.09.005.
"""
function bridgesampling(samples::AbstractMatrix, log_posterior::Function, lb, ub; tol=1e-10, maxiter=1_000, names=nothing)
    lb = informissing.(lb)
    ub = informissing.(ub)
    nd, ns = size(samples) 
    nd < ns || @warn "The number of samples ($ns) is lower than the number of dimensions ($nd). Make sure the samples matrix is ndim x nsamples."

    ## Split samples
    smp1 = samples[:,1:2:end]
    smp2 = samples[:,2:2:end]
    n₁ = size(smp1, 2)
    n₂ = size(smp2, 2)

    ## Transform samples
    trans_smp1 = transform(smp1, lb, ub)
    trans_smp2 = transform(smp2, lb, ub)
    wide_smp1, name_map = check_samples(trans_smp1, names) # widen the transformed samples to match with the MvNormal proposal
    wide_smp2, _ = check_samples(trans_smp2, names)
    
    ## Fit the proposal
    prop_dist = proposal(wide_smp2)

    ## Sample from the proposal
    prop_samples = rand(prop_dist, n₂)
    comp_prop_smp = format_samples(prop_samples, name_map) # Reformat for posterior evaluation
    comp_prop_smp = invtransform(comp_prop_smp, lb, ub) # Inverse transform to evaluate the posterior

    ## Get the (log) pdf for each sample for both distributions
    p_post, g_post, l₁ = get_pdf_samples(smp1, wide_smp1, log_posterior, prop_dist, lb, ub, names)
    p_prop, g_prop, l₂ = get_pdf_samples(comp_prop_smp, prop_samples, log_posterior, prop_dist, lb, ub, names)

    ## Iterative algorithm
    logml, i = iterative_algorithm(l₁, l₂, n₁, n₂; tol=tol, maxiter=maxiter)
    return LogMarginalLikelihood(logml, i, p_post, g_post, p_prop, g_prop)
end

function bridgesampling(chn::MCMCChains.Chains, mdl::Tm, trans=missing; kwargs...) where Tm <: AbstractMCMC.AbstractModel
    
    samples, param_names = extract_samples(chn, mdl)
    log_posterior, dist = log_posterior_func(mdl, param_names)
    if isa(trans, NamedTuple)
        trans_vec = [trans[pn] for pn in param_names]
    else
        trans_vec = trues(length(param_names))
    end
    #all(isa.(dist, Distribution{Univariate})) || throw("Multivariate distributions are not supported")
    return bridgesampling(samples, log_posterior, dist, trans_vec; names=nothing, kwargs...)
end

function bridgesampling(chn::MCMCChains.Chains, args...; kwargs...)
    samples = permutedims(Array(MCMCChains.get_sections(chn, :parameters)), [2,1])
    return bridgesampling(samples, args...; kwargs...)
end

informissing(x) = x
informissing(x::T) where T<:Real = isinf(x) ? missing : x 



value(LM::LogMarginalLikelihood, log::Bool=true) = log ? LM.value : exp(LM.value)

## Bayes factor
bayes_factor(LM1::LogMarginalLikelihood, LM2::LogMarginalLikelihood) = 
    exp(value(LM1) - value(LM2))

## Posterior model probabilities
function posterior_probabilities(LM::TV, prior_prob=fill(1/length(LM), length(LM))) where {T <: LogMarginalLikelihood, TV <: AbstractVector{T}}
    isprobvec(prior_prob) || throw("Prior probabilities must be a proability vector")
    ml = value.(LM, false)
    post_prob = ml .* prior_prob
    post_prob ./= sum(post_prob)
    return post_prob
end

## Show
function Base.show(io::IO, ::MIME"text/plain", LML::LogMarginalLikelihood)
    println(io, "Log-marginal likelihood ≈ $(LML.value) ($(LML.niter) iterations)")
    err = error_estimate(LML)
    println(io, "Error estimate : $(err.percent)%")
end

end 