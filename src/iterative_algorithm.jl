## 
function get_pdf_samples(samples, trans_samples, log_posterior, prop_dist, lb, ub, names)
    n = size(samples, 2)
    n == size(trans_samples,2) || throw(DimensionMismatch("samples dimensions don't match"))

    g = zeros(n)
    p = zeros(n)

    for i in 1:n
        g_sample = @views(trans_samples[:,i])
        p_sample = @views(samples[:,i])
        logdetJ = sum(logjacobian_contribution.(transform.(p_sample, lb, ub), lb, ub)) 
        g[i] = logpdf(prop_dist, g_sample)
        p[i] = isnothing(names) ? log_posterior(p_sample) + logdetJ : log_posterior(NamedTuple{Tuple(names)}(p_sample)) + logdetJ
    end

    return p, g, p-g
end


## Iterative algo
function iterative_algorithm(l₁, l₂, n₁, n₂; tol, maxiter)

    lstar = median(l₁)
    r = 0.0
    s₁ = n₁ / (n₁ + n₂)
    s₂ = n₂ / (n₁ + n₂)

    logml = -Inf
    ϵ = 0.0 
    for i = 1:maxiter
        numterm = 0.0
        denomterm = 0.0
        for (l₁ⱼ, l₂ⱼ) in zip(l₁, l₂)
            numterm += exp(l₂ⱼ - lstar) / (s₁ * exp(l₂ⱼ - lstar) + s₂ * r)
            denomterm += 1 / (s₁ * exp(l₁ⱼ - lstar) + s₂ * r)
        end
        rnew = (1/n₂) * numterm / (denomterm / n₁)
        
        logmlnew = log(rnew) + lstar
        ϵ = abs((logmlnew - logml) / logmlnew)
        if ϵ < tol
            return logmlnew, i
        end
        logml = logmlnew
        r = rnew
    end
    @warn "Maximum number of iterations ($maxiter) reached before convergence under the tolerance level $tol"
    return logml, i
end