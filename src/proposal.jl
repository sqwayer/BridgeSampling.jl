""" proposal(samples; transform=identity)
returns an MvNormal distribution and the log determinant of the jacobian matrix
"""

## MvNormal proposal
function proposal(samples::AbstractMatrix)
    μ = mean(samples, dims=2)
    Σ = cov(samples')
    return MvNormal(vec(μ), Σ)
end

## TODO : WRAP-3 proposal

