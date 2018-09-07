# --------------- #
# Chebyshev Basis #
# --------------- #

struct Cheb <: BasisFamily end

mutable struct ChebParams{T<:Number} <: BasisParams
    n::Int
    a::T
    b::T

    function ChebParams{T}(n::Int, a::T, b::T) where T
        n <= 0 && error("n must be positive")
        a >= b && error("left endpoint (a) must be less than right end point (b)")
        new{T}(n, a, b)
    end
end

ChebParams(n::Int, a::T, b::T) where {T<:Number} = ChebParams{T}(n, a, b)
ChebParams(n::Int, a::T, b::T) where {T<:Integer} = ChebParams(n, Float64(a), Float64(b))

## BasisParams interface
# define these methods on the type, the instance version is defined over
# BasisParams
family(::Type{T}) where {T<:ChebParams} = Cheb
family_name(::Type{T}) where {T<:ChebParams} = "Cheb"
@generated Base.eltype(::Type{T}) where {T<:ChebParams} = T.parameters[1]

# methods that only make sense for instances
Base.min(cp::ChebParams) = cp.a
Base.max(cp::ChebParams) = cp.b
Base.length(cp::ChebParams) = cp.n

function Base.show(io::IO, p::ChebParams)
    m = string("Chebyshev interpoland parameters with ",
               "$(p.n) basis functions from $(p.a), $(p.b)")
    print(io, m)
end

function nodes(p::ChebParams{T}, ::Type{Val{0}}) where T
    s = (p.b-p.a) / 2  # 21
    m = (p.b+p.a) / 2  # 22
    half = convert(T, 1/2)
    k = π*(half:(p.n - half))  # 25
    m .- cos.(k ./ p.n) .* s  # 26
end

function nodes(p::ChebParams, ::Type{Val{1}})
    x = nodes(p, Val{0})
    aa = x[1]
    bb = x[end]
    c1 = (bb*p.a - aa*p.b)/(bb-aa)
    c2 = (p.b.-p.a)/(bb-aa)

    @inbounds @simd for ix in eachindex(x)
        x[ix] = c1 + c2 * x[ix]
    end
    x
end

function nodes(p::ChebParams{T}, ::Type{Val{2}}) where T
    s = (p.b-p.a) / 2  # 21
    m = (p.b+p.a) / 2  # 22
    k = pi*(zero(T):(p.n - one(T)))  # 33
    m .- cos.(k ./ (p.n-1)) .* s  # 34
end

# chebnode.m -- DONE
nodes(p::ChebParams, nodetype::Integer=0) = nodes(p, Val{min(2, nodetype)})

function derivative_op(p::ChebParams, x, order=1)
    n, a, b = p.n, p.a, p.b
    if order > 0
        # TODO: figure out some caching mechanism that avoids globals
        D = Array{SparseMatrixCSC{basis_eltype(p, x),Int64}}(max(2, order)) # 49
        i = repmat(1:n', 1, n)
        j = i'  # 50

        # 51
        inds = find((rem.(i + j, 2) .== 1) .& (j .> i))
        r, c = similar(inds), similar(inds)
        for ix in 1:length(inds)
            r[ix], c[ix] = ind2sub((n, n), inds[ix])
        end

        d = sparse(r, c, (4/(b-a)) * (vec(j[1, c])-1), n-1, n)  # 52
        d[1, :] ./= 2  # 53
        D[1] = d  # 54
        for ii in 2:max(2, order)
            D[ii] = d[1:n-ii, 1:n-ii+1] * D[ii-1]  # 56
        end
    else
        D = Array{SparseMatrixCSC{basis_eltype(p, x),Int64}}(abs(order))  # 64
        nn = n - order  # 65
        z = (0.25 * (b - a)) ./(1:nn)  # 66
        d = sparse(vcat(1:nn, 1:nn-2), vcat(1:nn, 3:nn), vcat(z, -z[1:nn-2]),
                   nn, nn)  # 67
        d[1, 1] *= 2  # 68
        d0 = ((-1).^(0:nn-1)') .* sum(d, 1)  # 69
        D[1] = sparse(vcat(d0[1:n]', d[1:n, 1:n]))  # 70
        for ii=-2:-1:order
            ind = 1:n-ii-1
            D[-ii] = sparse(vcat(d0[ind]', d[ind, ind]) * D[-ii-1])
        end
    end
    D, ChebParams(n-order, a, b)
end

function evalbase(p::ChebParams, x::AbstractArray=nodes(p, 1), order::Int=0, nodetype::Int=1)
    n, a, b = p.n, p.a, p.b
    minorder = min(0, order)  # 30

    # compute 0-order basis
    local bas::Matrix{basis_eltype(p,x)}  # stupid type stability...
    if nodetype == 0
        temp = ((n-0.5):-1:0.5)''  # 41
        bas = cos.((pi./n).*temp.*(0:(n-1-minorder))')  # 42
    else
        bas = evalbasex(ChebParams(n-minorder, a, b), x)  # 44
    end

    if order != 0
        D = derivative_op(p, x, order)[1]
        B = view(bas, :, 1:n-order)*D[abs(order)]
    else
        B = bas
    end

    return B
end

function evalbase(p::ChebParams, x::AbstractArray, order::AbstractVector{Int}, nodetype::Int=1)
    n, a, b = p.n, p.a, p.b
    minorder = min(0, minimum(order))  # 30
    maxorder = maximum(order)

    # compute 0-order basis
    # if nodetype == 0
    #     temp = ((n-0.5):-1:0.5)''  # 41
    #     bas = convert(Matrix{basis_eltype(p, x)},
    #         cos.((pi./n).*temp.*(0:(n-1-minorder))')
    #     )  # 42
    # else
        bas = evalbasex(ChebParams(n-minorder, a, b), x)  # 44
    # end

    B = Array{Matrix{basis_eltype(p, x)}}(length(order))
    if maxorder > 0 D = derivative_op(p, x, maxorder)[1] end
    if minorder < 0 I = derivative_op(p, x, minorder)[1] end

    for ii in 1:length(order)
        if order[ii] == 0
            B[ii] = bas[:, 1:n]
        elseif order[ii] > 0
            B[ii] = view(bas, :, 1:n-order[ii]) * D[order[ii]]
        else
            B[ii] = view(bas, :, 1:n-order[ii]) * I[-order[ii]]
        end
    end

    return B
end

_unscale(p::ChebParams, x::T) where {T<:Number} = (2/(p.b-p.a)) * (x-(p.a+p.b)/2)

function evalbasex!(out::AbstractMatrix, z::AbstractArray{T},
                    p::ChebParams, x::AbstractArray{T}) where T<:Number
    if size(out) != (size(x, 1), p.n)
        throw(DimensionMismatch("out must be (size(x, 1), p.n)"))
    end

    if size(z) != size(x)
        throw(DimensionMismatch("z must be same size as x"))
    end

    # Note: for julia 0.6+ we can do z .= _unscale.(p, x)
    z .= _unscale.([p], x)
    m = length(z)

    @inbounds out[:, 1] = 1.0
    @inbounds out[:, 2] = z

    scale!(z, 2.0)

    @inbounds for j in 3:p.n
        @simd for i in 1:m
            out[i, j] = z[i] * out[i, j-1] - out[i, j-2]
        end
    end
    out
end

function evalbasex!(out::AbstractMatrix, p::ChebParams, x::AbstractArray)
    z = similar(x)
    evalbasex!(out, z, p, x)
end

function evalbasex(p::ChebParams, x::AbstractArray)
    evalbasex!(Array{basis_eltype(p, x)}(size(x, 1), p.n), p, x)
end
