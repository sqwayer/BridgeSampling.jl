## Transformations
probit(x) = quantile(Normal(), x)

function transform!(trans_samples::AbstractMatrix, samples::AbstractMatrix, lb::AbstractVector, ub::AbstractVector) 
    nd, ns = size(samples)
    for j = 1:ns, i = 1:nd
        trans_samples[i,j] = transform(samples[i,j], lb[i], ub[i])
    end
    return trans_samples
end
transform(samples::AbstractMatrix, lb::AbstractVector, ub::AbstractVector) = transform!(similar(samples), samples, lb, ub)
transform(x, lb::Real, ub::Real) = probit((x - lb)/(ub - lb))
transform(x, ::Missing, ::Missing) = x
transform(x, lb::Real, ::Missing) = log(x - lb)
transform(x, ::Missing, ub::Real) = log(ub - x)

# With bijectors
transform(x, dist::Distribution, trans::Bool) = trans ? Bijectors.link(dist,x) : x

# Inverse transformations
function invtransform!(samples::AbstractMatrix, trans_samples::AbstractMatrix, lb::AbstractVector, ub::AbstractVector)
    nd, ns = size(samples)
    for j = 1:ns, i = 1:nd
        samples[i,j] = invtransform(trans_samples[i,j], lb[i], ub[i])
    end
    return samples
end
invtransform(trans_samples::AbstractMatrix, lb::AbstractVector, ub::AbstractVector) = invtransform!(similar(trans_samples), trans_samples, lb, ub)
invtransform(x, lb::Real, ub::Real) = lb + (ub - lb) * cdf(Normal(), x)
invtransform(x, ::Missing, ::Missing) = x
invtransform(x, lb::Real, ::Missing) = exp(x) + lb
invtransform(x, ::Missing, ub::Real) = ub - exp(x)

# With bijectors
invtransform(x, dist::Distribution, trans::Bool) = trans ? Bijectors.invlink(dist,x) : x

# Log jacobian contribution : x is the transformed variable
logjacobian_contribution(x, lb, ub) = log(ub-lb) + logpdf(Normal(), x)
logjacobian_contribution(x, ::Missing, ::Missing) = 0.0
logjacobian_contribution(x, lb, ::Missing) = x
logjacobian_contribution(x, ::Missing, ub) = x

# With bijectors
function logjacobian_contribution(x, dist::Distribution, trans::Bool)
    b =  Bijectors.bijector(dist)
    return trans ? Bijectors.logabsdetjac(b, Bijectors.invlink(dist, x)) : 0.0
end