using Turing, BridgeSampling

## Test on bivariate standard Normal distribution
samples = rand(MvNormal(ones(2)), 10_000)

log_posterior(x) = -0.5 * x' * x

Bridge = bridgesampling(samples, log_posterior, [-Inf, -Inf], [Inf, Inf])
Analytical = log(2*pi)

isapprox(value(Bridge), Analytical, atol = 1e-3)

## Hierarchical Normal example

# Generate data
true_μ = 0.0
true_τ² = 0.5
true_σ² = 1.0
n = 20
true_θ = rand(Normal(true_μ, sqrt(true_τ²)), n)
#y = rand.(Normal.(true_θ, sqrt(true_σ²)))
# Same values as in : https://cran.r-project.org/web/packages/bridgesampling/vignettes/bridgesampling_example_jags.html
y = vec([1.19365332  1.95745331 -0.72161754 -1.87380833 -1.16928239  0.51960853 -0.03610041  0.42508815  0.41119221 -0.81236980  0.72967357  3.48186722  2.31126381  2.00029422 -0.27643507  1.06882370 -0.95083599 -1.89651101  2.56019737  0.23703060])

# DynamicPPL models
@model H0(Y) = begin
    τ² ~ InverseGamma(1,1)
    θ ~ filldist(Normal(0.0, sqrt(τ²)), length(Y))
    Y ~ arraydist(Normal.(θ, 1.0))
end

@model H1(Y) = begin
    μ ~ Normal()
    τ² ~ InverseGamma(1,1)
    θ ~ filldist(Normal(μ, sqrt(τ²)), length(Y))
    Y ~ arraydist(Normal.(θ, 1.0))
end

M0 = H0(y)
M1 = H1(y)

# Sampling
chn0 = sample(M0, NUTS(), MCMCThreads(), 50_000, 3, burnin=2000)
chn1 = sample(M1, NUTS(), MCMCThreads(), 50_000, 3, burnin=2000)

# Estimate log marginal likelihood : directly with the DynamicPPL model (lower precision)
L0 = bridgesampling(chn0, M0)
L1 = bridgesampling(chn1, M1)

# Or with specified log posterior functions and boundaries
log_posterior_H0(x) = sum([logpdf(Normal(x.θ[i], 1.0), y[i]) for i = 1:n]) + 
                sum([logpdf(Normal(0.0, sqrt(x.τ²)), x.θ[i]) for i = 1:n]) + 
                logpdf(InverseGamma(1,1), x.τ²)

log_posterior_H1(x) = sum([logpdf(Normal(x.θ[i], 1.0), y[i]) for i = 1:n]) + 
                sum([logpdf(Normal(x.μ, sqrt(x.τ²)), x.θ[i]) for i = 1:n]) + 
                logpdf(InverseGamma(1,1), x.τ²) + 
                logpdf(Normal(), x.μ)

samplesH0, names = BridgeSampling.extract_samples(chn0, M0)
L0 = bridgesampling(samplesH0, log_posterior_H0, [0.0, -Inf], [Inf, Inf]; names=names)


samplesH1, names = BridgeSampling.extract_samples(chn1, M1)
L1 = bridgesampling(samplesH1, log_posterior_H1, [-Inf, 0.0, -Inf], [Inf, Inf, Inf]; names=names)

# Bayes factor
bayes_factor(L0, L1)

# Posterior model probabilities
posterior_probabilities([L0, L1])