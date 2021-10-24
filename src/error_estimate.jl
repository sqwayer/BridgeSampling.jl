function error_estimate(LML)
    lm = value(LML)
    p_posterior = exp.(LML.p_posterior .- lm)
    p_proposal = exp.(LML.p_proposal .- lm)
    g_posterior = exp.(LML.g_posterior)
    g_proposal = exp.(LML.g_proposal)

    n1 = length(p_posterior)
    n2 = length(g_proposal)
    s1 = n1 / (n1+n2)
    s2 = n2 / (n1+n2)
    f1 = p_proposal ./ (s1 * p_proposal + s2 * g_proposal)
    f2 = g_posterior ./ (s1 * p_posterior + s2 * g_posterior)

    ρ_f2 = spectrum0(f2)
    
    re2 = 1/n2 * var( f1 ) / mean( f1 )^2
    re2 += ρ_f2/n1 * var( f2 ) / mean( f2 )^2
    
    return (rmse = re2, cv = sqrt(re2), percent = sqrt(re2)*100)
end

## Spectrum of the autoregressive model at frequency 0

function spectrum0(x)
    σ², B, order = fit_auto_regressive(x)
    return σ² / (1-sum(B))^2
end

function fit_auto_regressive(x)
    n = length(x)
    maxorder = Int(floor(10*log(10, n)))
    coeffs, varpred = levinson_fit(x, maxorder)
    order = argmin(vec(n * (log.(varpred') .+ 1) .+ 2 * ((1:maxorder) .+ 1)))

    b = -coeffs[order,1:order]
    σ² = varpred[order] 
    σ² *= n / (n - order - 1) 
    return σ², b, order
end

function levinson_fit(x::AbstractVector, p::Int)
    # Copyright 2012-2021 DSP.jl contributors : https://github.com/JuliaDSP/DSP.jl/blob/master/src/lpc.jl
    R_xx = autocov(x)
    a = zeros(p,p)
    prediction_err = zeros(1,p)

    # for m = 1
    a[1,1] = -R_xx[2]/R_xx[1]
    prediction_err[1] = R_xx[1]*(1-a[1,1]^2)

    # for m = 2,3,4,..p
    for m = 2:p
        a[m,m] = (-(R_xx[m+1] + dot(a[m-1,1:m-1],R_xx[m:-1:2]))/prediction_err[m-1])
        a[m,1:m-1] = a[m-1,1:m-1] + a[m,m] * a[m-1,m-1:-1:1]
        prediction_err[m] = prediction_err[m-1]*(1-a[m,m]^2)
    end

    # Return autocorrelation coefficients and error estimate
    a, prediction_err
end

