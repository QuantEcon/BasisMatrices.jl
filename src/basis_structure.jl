# ------------ #
# BMOrder Type #
# ------------ #

struct BMOrder
    dims::Vector{Int}
    order::Matrix{Int}
end

function ==(bmo1::BMOrder, bmo2::BMOrder)
    bmo1.order == bmo2.order && bmo1.dims == bmo2.dims
end

function Base.convert(::Type{BMOrder}, order::Matrix{Int})
    BMOrder(ones(size(order, 2)), order)
end

function _dims_to_colspans(dims::Vector{Int})
    # column ranges for each entry in `bm.vals`
    cols = Array{typeof(1:2)}(length(dims))
    cols[1] = 1:dims[1]
    for i in 2:length(cols)
        start = last(cols[i-1]+1)
        cols[i] = (start):(start-1+dims[i])
    end
    cols
end

_dims_to_colspans(bmo::BMOrder) = _dims_to_colspans(bmo.dims)

Base.size(bmo::BMOrder, i::Int) = size(bmo.order, i)

# ---------------- #
# BasisMatrix Type #
# ---------------- #

abstract type AbstractBasisMatrixRep end
const ABSR = AbstractBasisMatrixRep

struct Tensor <: ABSR end
struct Direct <: ABSR end
struct Expanded <: ABSR end

mutable struct BasisMatrix{BST<:ABSR, TM<:AbstractMatrix}
    order::BMOrder
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
        minorder = minimum(order, 1)
        numbases = (maximum(order, 1) - minorder) + 1
    else
        minorder = order + zeros(Int, 1, N)
        numbases = fill(1, 1, N)
    end

    x = _checkx(N, x)

    return m, order, minorder, numbases, x
end

function _unique_rows(mat::AbstractMatrix{T}) where {T}
    out = Vector{T}[]
    for row in 1:size(mat, 1)
        if mat[row, :] in out
            continue
        end
        push!(out, mat[row, :])
    end

    # sort so we can leverage that fact in _extract_inds later
    # TODO: maybe consider this later...
    # sort!(collect(out), order=Base.Order.Lexicographic)
    out
end

# --------------- #
# convert methods #
# --------------- #

function Base.convert(
        ::Type{T}, bs::BasisMatrix{T,TM}, _order=fill(0, 1, size(bs.order, 2))
    ) where {T,TM}
    order = _check_order(size(bs.order.order, 2), _order)

    # unflip the inds because I don't want to do kroneckers, I just want to
    # extract up the basis matrices as they are
    inds = flipdim(_extract_inds(bs, order), 2)

    nrow = size(order, 1)
    ncol = size(bs.vals, 2)

    vals = Array{TM}(nrow, ncol)
    for row in 1:nrow
        for col in 1:ncol
            vals[row, col] = deepcopy(bs.vals[inds[row, col]])
        end
    end

    bm_order = BMOrder(bs.order.dims, order)
    BasisMatrix{T,TM}(order, vals)
end

function _to_expanded(bs::BasisMatrix{T,TM}, _order, reducer::Function) where {T,TM}
    order = _check_order(size(bs.order.order, 2), _order)
    inds = _extract_inds(bs, order)

    nrow = size(inds, 1)
    vals = Array{TM}(nrow, 1)

    for row in 1:nrow
        vals[row] = reduce(reducer, bs.vals[inds[row, :]])
    end

    bm_order = BMOrder(bs.order.dims, order)
    BasisMatrix{Expanded,TM}(order, vals)
end


# funbconv from direct to expanded
function Base.convert(::Type{Expanded}, bs::BasisMatrix{Direct}, order=0)
    _to_expanded(bs, order, row_kron)
end

# funbconv from tensor to expanded
function Base.convert(::Type{Expanded}, bs::BasisMatrix{Tensor}, order=0)
    _to_expanded(bs, order, kron)
end

# funbconv from tensor to direct
# HACK: there is probably a more efficient way to do this, but since I don't
#       plan on doing it much, this will do for now. The basic point is that
#       we need to expand the rows of each element of `vals` so that all of
#       them have prod([size(v, 1) for v in bs.vals])) rows.
function Base.convert(
        ::Type{Direct}, bs::BasisMatrix{Tensor,TM},
        _order=fill(0, 1, size(bs.order, 2))
    ) where TM
    order = _check_order(size(bs.order.order, 2), _order)
    numbas = size(order, 1)

    # unflip the inds because I don't want to do kroneckers, I just want to
    # expand basis matrices in place
    inds = flipdim(_extract_inds(bs, order), 2)

    N = size(bs.vals, 2)
    vals = Array{TM}(size(inds))

    for row in 1:size(inds, 1)
        expansion_inds = gridmake(([size(x, 1) for x in bs.vals[inds[row, :]]]...))
        for col in 1:N
            vals[row, col] = bs.vals[row, col][expansion_inds[:, col], :]
        end
    end

    bm_order = BMOrder(bs.order.dims, order)
    BasisMatrix{Direct,TM}(bm_order, vals)
end

# ------------ #
# Constructors #
# ------------ #

# method to construct BasisMatrix in direct or expanded form based on
# a matrix of `x` values  -- funbasex
function BasisMatrix(
        ::Type{T2}, basis::Basis{N,BF}, ::Direct,
        _x::AbstractArray=nodes(basis)[1], _order=0
    ) where {N,BF,T2}
    m, order, minorder, numbases, x = check_basis_structure(N, _x, _order)
    Np = length(basis.params)

    val_type = bmat_type(T2, basis, x)
    vals = Array{val_type}(maximum(numbases), Np)

    order_dims = collect(ndims.(basis.params))
    colspans = _dims_to_colspans(order_dims)
    order_vals = fill(typemax(Int), size(vals, 1), N)

    for (i_params, params) in enumerate(basis.params)
        cols = colspans[i_params]
        orders_p = _unique_rows(order[:, cols])

        if length(cols) == 1
            _orders_1d = vcat(orders_p...)::Vector{Int}
            rows = 1:length(_orders_1d)
            vals[rows, i_params] = evalbase(T2, params, x[:, cols[1]], _orders_1d)
            order_vals[rows, cols[1]] = _orders_1d
        else  # multi-dim params
            for (i, ord) in enumerate(orders_p)
                vals[i, i_params] = evalbase(T2, params, x[:, cols], ord)
                order_vals[i, cols] = ord
            end
        end
    end

    bm_order = BMOrder(order_dims, order_vals)
    return BasisMatrix{Direct,val_type}(bm_order, vals)
end

function BasisMatrix(::Type{T2}, basis::Basis, ::Expanded,
                     x::AbstractArray=nodes(basis)[1], order=0) where T2  # funbasex
    # create direct form, then convert to expanded
    bsd = BasisMatrix(T2, basis, Direct(), x, order)
    convert(Expanded, bsd, bsd.order.order)
end

function BasisMatrix(
        ::Type{T2}, basis::Basis{N,BT}, ::Tensor,
        _x::TensorX=nodes(basis)[2], _order=0
    ) where {N,BT,T2}

    m, order, minorder, numbases, x = check_basis_structure(N, _x, _order)
    Np = length(basis.params)

    val_type = bmat_type(T2, basis, x[1])
    vals = Array{val_type}(maximum(numbases), Np)

    order_dims = collect(ndims.(basis.params))
    colspans = _dims_to_colspans(order_dims)
    order_vals = fill(typemax(Int), size(vals, 1), N)

    for (i_params, params) in enumerate(basis.params)
        cols = colspans[i_params]
        orders_p = _unique_rows(order[:, cols])

        if length(cols) == 1
            _orders_1d = vcat(orders_p...)::Vector{Int}
            rows = 1:length(_orders_1d)
            vals[rows, i_params] = evalbase(T2, params, x[i_params], _orders_1d)
            order_vals[rows, cols[1]] = _orders_1d
        else  # multi-dim params
            for (i, ord) in enumerate(orders_p)
                vals[i, i_params] = evalbase(T2, params, x[i_params], ord)
                order_vals[i, cols] = ord
            end
        end

    end

    bm_order = BMOrder(order_dims, order_vals)
    return BasisMatrix{Tensor,val_type}(bm_order, vals)
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
    BasisMatrix(Void, basis, tbm, x, order)
end

function BasisMatrix(basis::Basis, x::Union{AbstractArray,TensorX}, order=0)
    BasisMatrix(Void, basis, x, order)
end

# method without x
function BasisMatrix(basis::Basis, tbm::Union{Type{TBM},TBM}) where TBM<:ABSR
    BasisMatrix(Void, basis, tbm)
end
