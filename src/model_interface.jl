# Interface with DynamicPPL

function log_posterior_func(mdl, paramnames)
    vi = DynamicPPL.VarInfo(mdl)
    vns = DynamicPPL.VarName.(paramnames)
    dist_vec = [DynamicPPL.getdist(vi, vn) for vn in vns]
    function log_posterior(paramvals)
        for i in eachindex(paramvals)
            DynamicPPL.setindex!(vi, paramvals[i], vns[i])
        end
        return DynamicPPL.logjoint(mdl, vi)
    end
    return log_posterior, dist_vec
end

function extract_samples(chn, mdl)
    vi = DynamicPPL.VarInfo(mdl)
    pnames = DynamicPPL.syms(vi)
    chn_params = get_params(chn)
    nd = length(pnames)
    ns = size(chn,1) * size(chn,3)
    samples = Matrix(undef, nd, ns)
    for i in 1:nd
        samples[i,:] = extract(chn_params[pnames[i]])
    end
    return samples, pnames
end

function extract(V::AbstractArray) # Univariate case
    return V[:]
end

function extract(V::Tuple) # Multivariate case
    S = hcat([vcat(vi...) for vi in V]...)
    ns, nd = size(S)
    return [S[i,:] for i = 1:ns]
end
