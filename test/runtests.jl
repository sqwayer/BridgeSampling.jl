using Test, BridgeSampling, Turing

## Test on bivariate standard Normal distribution
samples = rand(MvNormal([1.0 0.0; 0.0 1.0]), 10_000)

log_posterior(x) = -0.5 * x' * x

Bridge = bridgesampling(samples, log_posterior, [-Inf, -Inf], [Inf, Inf])
Analytical = log(2*pi)

@test isapprox(value(Bridge), Analytical, atol = 1e-3)