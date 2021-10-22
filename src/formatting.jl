## Helper functions to deal with multivariate parameters

# check_samples(samples, names) returns a widden matrix of scalar samples, and an associative array of parameters names and positions in the matrix
function check_samples(samples::TV, ::Nothing) where TV <: AbstractMatrix
    names = [Symbol("arg$i") for i = 1:size(samples, 1)]
    return check_samples(samples, names)
end

function check_samples(samples::TV, names::Tuple) where {T <: Real, TV <: AbstractMatrix{T}}
    return samples, NamedTuple{names}(eachindex(names))
end

function check_samples(samples, names)
    nn = length(names)
    nn == size(samples, 1) || throw(DimensionMismatch("$nn names for $(size(samples, 1)) parameters. Make sure the samples dimensions are nparams x nsamples"))
    ns = size(samples,2)
    nd = 0
    positions = Vector(undef, nn)
    for ni = 1:nn
        l = length(samples[ni, 1])
        positions[ni] = l > 1 ? (nd+1:nd+l) : nd+1
        nd += l
    end
    samples_wide = zeros(nd, ns)
    for i = 1:ns
        samples_wide[:,i] .= vcat(widden.(samples[:,i])...)
    end
    return samples_wide, NamedTuple{Tuple(names)}(positions)
end 

widden(x::Real) = x
widden(x::AbstractArray) = vec(x)

# format_samples(samples, name_map) returns a compact matrix of samples of various dimensions, given the map name_map
function format_samples(samples::TV, name_map::NamedTuple) where {T <: Real, TV <: AbstractMatrix{T}}
    nd = length(name_map)
    ns = size(samples, 2)
    samples_comp = Matrix(undef, nd, ns)
    for i = 1:ns
        for j = 1:nd
            samples_comp[j,i] = samples[name_map[j],i]
        end
    end
    return samples_comp
end
