# BridgeSampling.jl
[![Build status (Github Actions)](https://github.com/sqwayer/BridgeSampling.jl/workflows/CI/badge.svg)](https://github.com/sqwayer/BridgeSampling.jl/actions)
[![codecov.io](http://codecov.io/github/sqwayer/BridgeSampling.jl/coverage.svg?branch=main)](http://codecov.io/github/sqwayer/BridgeSampling.jl?branch=main)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://sqwayer.github.io/BridgeSampling.jl/dev/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://sqwayer.github.io/BridgeSampling.jl/dev/)

A Julia package for computing Normalizing Constants (aka Marginal Likelihoods), Bayes Factors and Posterior Model Probabilities.

This is a Julia version of the [R package](https://github.com/quentingronau/bridgesampling) `bridgesampling`, rewrote from scratch.

Quentin F. Gronau, Alexandra Sarafoglou, Dora Matzke, Alexander Ly, Udo Boehm, Maarten Marsman, David S. Leslie, Jonathan J. Forster, Eric-Jan Wagenmakers, Helen Steingroever, A tutorial on bridge sampling, Journal of Mathematical Psychology, Volume 81, 2017, Pages 80-97, ISSN 0022-2496, https://doi.org/10.1016/j.jmp.2017.09.005.

## Basic usage
The function `bridgesampling` estimates the log marginal likelihood with bridge sampling, given a matrix of posterior samples and an unnormalized log posterior function.
```julia
using BridgeSampling
 # Bivariate standard normal distribution
samples = rand(MvNormal(ones(2)), 10_000)
log_posterior(x) = -0.5 * x' * x
logML = bridgesampling(samples, log_posterior, [-Inf, -Inf], [Inf, Inf])
```

## Integration with Turing
The function can also be called with a `MCMCChain.Chains` object and a `DynamicPPL.Model` object and will automatically infer the log posterior function and the boundaries : 
```julia
L = bridgesampling(chn, M)
```
However this method is numerically less precise than defining the log-posterior explicitly.

## Error estimation
The function `error_estimate` computes an approximation of the relative mean squared error of the estimate and the coefficient of variation.

## Model comparison
```julia
using Turing
## Hierarchical Normal Model
# Generate data
true_μ = 0.0
true_τ² = 0.5
true_σ² = 1.0
n = 20
true_θ = rand(Normal(true_μ, sqrt(true_τ²)), n)
y = rand.(Normal.(true_θ, sqrt(true_σ²)))

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

# Sample
M0 = H0(y)
M1 = H1(y)
chn0 = sample(M0, NUTS(), MCMCThreads(), 50_000, 3, burnin=2000)
chn1 = sample(M1, NUTS(), MCMCThreads(), 50_000, 3, burnin=2000)

# Estimate log marginal likelihood : directly with the DynamicPPL model 
L0 = bridgesampling(chn0, M0)
L1 = bridgesampling(chn1, M1)

# Bayes factor
BF10 = bayes_factor(L1, L0)

# Posterior model probabilities
posterior_probabilities([L0, L1])
prior_prob = [0.4, 0.6]
posterior_probabilities([L0, L1], prior_prob)
```
