# ------------------- #
# BasisMatrix Type #
# ------------------- #

abstract type AbstractBasisMatrixRep end
const ABSR = AbstractBasisMatrixRep

struct Tensor <: ABSR end
struct Direct <: ABSR end
struct Expanded <: ABSR end

struct BasisMatrix{BST<:ABSR, TM<:AbstractMatrix}
    order::Matrix{Int}
    vals::Matrix{TM}
end

Base.eltype(bm::BasisMatrix{BST,TM}) where {BST, TM} = eltype(TM)

Base.show(io::IO, b::BasisMatrix{BST}) where {BST} =
    print(io, "BasisMatrix{$BST} of order $(b.order)")

Base.ndims(bs::BasisMatrix) = size(bs.order, 2)

# not the same if either type parameter is different
function ==(::BasisMatrix{BST1}, ::BasisMatrix{BST2}) where {BST1<:ABSR,BST2<:ABSR}
    false
end

function ==(::BasisMatrix{BST,TM1},
            ::BasisMatrix{BST,TM2}) where {BST<:ABSR,TM1<:AbstractMatrix,TM2<:AbstractMatrix}
    false
end

# if type parameters are the same, then it is the same if all fields are the
# same
function ==(b1::BasisMatrix{BST,TM},
            b2::BasisMatrix{BST,TM}) where {BST<:ABSR,TM<:AbstractMatrix}
    b1.order == b2.order && b1.vals == b2.vals
end

# -------------- #
# Internal Tools #
# -------------- #

@inline function _checkx(N, x::AbstractMatrix)
    size(x, 2) != N && error("Basis is $N dimensional, x must have $N columns")
    x
end

@inline function _checkx(N, x::AbstractVector{T}) where T
    # if we have a 1d basis, we can evaluate at each point
    if N == 1
        return x
    end

    # If Basis is > 1d, one evaluation point and reshape to (1,N) if possible...
    if length(x) == N
        return reshape(x, 1, N)
    end

    # ... or throw an error
    error("Basis is $N dimensional, x must have $N elements")
end

@inline function _checkx(N, x::TensorX)
    # for BasisMatrix{Tensor} family. Need one vector per dimension
    if length(x) == N
        return x
    end

    # otherwise throw an error
    error("Basis is $N dimensional, need one Vector per dimension")
end

# common checks for all convert methods
function check_convert(bs::BasisMatrix, order::Matrix)
    d = ndims(bs)
    numbas, d1 = size(order)

    d1 != d && error("ORDER incompatible with basis functions")  # 35-37

    # 39-41
    if any(minimum(order, dims=1) .< bs.order)
        error("Order of derivative operators exceeds basis")
    end
    return d, numbas, d1
end

"""
Do common transformations to all constructor of `BasisMatrix`

##### Arguments

- `N::Int`: The number of dimensions in the corresponding `Basis`
- `x::AbstractArray`: The points for which the `BasisMatrix` should be
constructed
- `order::Array{Int}`: The order of evaluation for each dimension of the basis

##### Returns

- `m::Int`: the total number of derivative order basis functions to compute.
This will be the number of rows in the matrix form of `order`
- `order::Matrix{Int}`: A `m × N` matrix that, for each of the `m` desired
specifications, gives the derivative order along all `N` dimensions
- `minorder::Matrix{Int}`: A `1 × N` matrix specifying the minimum desired
derivative order along each dimension
- `numbases::Matrix{Int}`: A `1 × N` matrix specifying the total number of
distinct derivative orders along each dimension
- `x::AbstractArray`: The properly transformed points at which to evaluate
the basis

"""
function check_basis_structure(N::Int, x, order)
    order = _check_order(N, order)

    # initialize basis structure (66-74)
    m = size(order, 1)  # by this time order is a matrix
    if m > 1
        minorder = minimum(order, dims=1)
        numbases = (maximum(order, dims=1) .- minorder) .+ 1
    else
        minorder = order + zeros(Int, 1, N)
        numbases = fill(1, 1, N)
    end

    x = _checkx(N, x)

    return m, order, minorder, numbases, x
end


# --------------- #
# convert methods #
# --------------- #

# no-op. Don't worry about the order argument.
Base.convert(::Type{T}, bs::BasisMatrix{T}, order=bs.order) where {T<:ABSR} = bs

# funbconv from direct to expanded
function Base.convert(::Type{Expanded}, bs::BasisMatrix{Direct,TM},
                  order=fill(0, 1, size(bs.order, 2))) where TM
    d, numbas, d1 = check_convert(bs, order)

    vals = Array{TM}(undef, numbas, 1)

    for i=1:numbas
        vals[i] = bs.vals[order[i, d] - bs.order[d]+1, d]  # 63
        for j=d-1:-1:1
            vals[i] = row_kron(vals[i],
                               bs.vals[order[i, j] - bs.order[j]+1, j])  #65
        end
    end
    BasisMatrix{Expanded,TM}(order, vals)
end

# funbconv from tensor to expanded
function Base.convert(::Type{Expanded}, bs::BasisMatrix{Tensor,TM},
                  order=fill(0, 1, size(bs.order, 2))) where TM
    d, numbas, d1 = check_convert(bs, order)

    vals = Array{TM}(undef, numbas, 1)

    for i=1:numbas  # 54
        vals[i] = bs.vals[order[i, d] - bs.order[d]+1, d]  # 55
        for j=d-1:-1:1  # 56
            # 57
            vals[i] = kron(vals[i], bs.vals[order[i, j] - bs.order[j]+1, j])
        end
    end

    BasisMatrix{Expanded,TM}(order, vals)
end

# funbconv from tensor to direct
# HACK: there is probably a more efficient way to do this, but since I don't
#       plan on doing it much, this will do for now. The basic point is that
#       we need to expand the rows of each element of `vals` so that all of
#       them have prod([size(v, 1) for v in bs.vals])) rows.
function Base.convert(::Type{Direct}, bs::BasisMatrix{Tensor,TM},
                      order=fill(0, 1, size(bs.order, 2))) where TM
    d, numbas, d1 = check_convert(bs, order)
    vals = Array{TM}(undef, numbas, d)
    raw_ind = Array{Vector{Int}}(undef, d)

    for j=1:d
        for i=1:size(bs.vals, 1)
            if !(isempty(bs.vals[i, j]))  # 84
                raw_ind[j] = collect(1:size(bs.vals[i, j], 1))  # 85
                break
            end
        end
    end

    ind = gridmake(raw_ind...)  # 90

    for j=1:d, i=1:numbas
        if !(isempty(bs.vals[i, j]))  # 93
            vals[i, j] = bs.vals[i, j][ind[:, j], :]  # 94
        end
    end

    BasisMatrix{Direct,TM}(order, vals)
end

# ------------ #
# Constructors #
# ------------ #

# method to construct BasisMatrix in direct or expanded form based on
# a matrix of `x` values  -- funbasex
function BasisMatrix(::Type{T2}, basis::Basis{N,BF}, ::Direct,
                     x::AbstractArray=nodes(basis)[1], order=0) where {N,BF,T2}
    m, order, minorder, numbases, x = check_basis_structure(N, x, order)
    # 76-77
    out_order = minorder
    out_format = Direct()
    val_type = bmat_type(T2, basis, x)
    vals = Array{val_type}(undef, maximum(numbases), N)

    # now do direct form, will convert to expanded later if needed
    for j=1:N
        # 126-130
        if (m > 1)
            orderj = unique(order[:, j])
        else
            orderj = order[1, j]
        end

        #131-135
        if length(orderj) == 1
            vals[1, j] = evalbase(T2, basis.params[j], x[:, j], orderj[1])
        else
            vals[orderj-minorder[j]+1, j] =
                evalbase(T2, basis.params[j], x[:, j], orderj)
        end
    end

    # construct Direct Format
    BasisMatrix{Direct,val_type}(out_order, vals)
end

function BasisMatrix(::Type{T2}, basis::Basis, ::Expanded,
                     x::AbstractArray=nodes(basis)[1], order=0) where T2  # funbasex
    # create direct form, then convert to expanded
    bsd = BasisMatrix(T2, basis, Direct(), x, order)
    convert(Expanded, bsd, bsd.order)
end

function BasisMatrix(::Type{T2}, basis::Basis{N,BT}, ::Tensor,
                     x::TensorX=nodes(basis)[2], order=0) where {N,BT,T2}

    m, order, minorder, numbases, x = check_basis_structure(N, x, order)
    out_order = minorder
    out_format = Tensor()
    val_type = bmat_type(T2, basis, x[1])  # TODO: reduce(promot_type, eltype.(x))
    vals = Array{val_type}(undef, maximum(numbases), N)

    # construct tensor base
    for j in 1:N
        # 113-117
        if (m > 1)
            orderj = unique(order[:, j])
        else
            orderj = order[1, j]
        end

        #118-122
        if length(orderj) == 1
            vals[1, j] = evalbase(T2, basis.params[j], x[j], orderj[1])
        else
            vals[orderj-minorder[j]+1, j] = evalbase(T2, basis.params[j], x[j], orderj)
        end
    end

    BasisMatrix{Tensor,val_type}(out_order, vals)
end

# When the user doesn't supply a ABSR, we pick one for them.
# for x::AbstractMatrix we pick direct
# for x::TensorX we pick Tensor
function BasisMatrix(::Type{T2}, basis::Basis, x::AbstractArray, order=0) where T2
    BasisMatrix(T2, basis, Direct(), x, order)
end

function BasisMatrix(::Type{T2}, basis::Basis, x::TensorX, order=0) where T2
    BasisMatrix(T2, basis, Tensor(), x, order)
end


# method to allow passing types instead of instances of ABSR
function BasisMatrix(::Type{T2}, basis, ::Type{BST},
                     x::Union{AbstractArray,TensorX}, order=0) where {BST<:ABSR,T2}
    BasisMatrix(T2, basis, BST(), x, order)
end

function BasisMatrix(basis, ::Type{BST},
                     x::Union{AbstractArray,TensorX}, order=0) where BST<:ABSR
    BasisMatrix(basis, BST(), x, order)
end

# method without vals eltypes
function BasisMatrix(basis::Basis, tbm::TBM,
                     x::Union{AbstractArray,TensorX}, order=0) where TBM<:ABSR
    BasisMatrix(Nothing, basis, tbm, x, order)
end

function BasisMatrix(basis::Basis, x::Union{AbstractArray,TensorX}, order=0)
    BasisMatrix(Nothing, basis, x, order)
end

# method without x
function BasisMatrix(basis::Basis, tbm::Union{Type{TBM},TBM}) where TBM<:ABSR
    BasisMatrix(Nothing, basis, tbm)
end
