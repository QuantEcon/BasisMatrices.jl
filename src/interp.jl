# ---------------- #
# Fitting routines #
# ---------------- #

# Tensor representation, single function method; calls function that computes coefficients below below
function get_coefs(basis::Basis, bs::BasisMatrix{Tensor}, y::Vector{T}) where T
    _get_coefs_deep(basis, bs, y)[:,1]
end

# Tensor representation, multiple function method; calls function that computes coefficients below
function get_coefs(basis::Basis, bs::BasisMatrix{Tensor}, y::Matrix{T}) where T
    _get_coefs_deep(basis, bs, y)
end

function _get_coefs_deep(basis::Basis, bs::BasisMatrix{Tensor}, y)
    if any(bs.order[1, :] .!= 0)
        error("invalid basis structure - first elements must be order 0")
    end
    to_kron = bs.vals[1, :]  # 68
    ckronxi(to_kron, y, ndims(basis):-1:1)  # 66
end

# convert to expanded and call the method below
function get_coefs(basis::Basis, bs::BasisMatrix{Direct}, y)
    get_coefs(basis, convert(Expanded, bs), y)
end

get_coefs(basis::Basis, bs::BasisMatrix{Expanded}, y) = bs.vals[1] \ y

# common checks to be run at the top of each funfit
function check_funfit(basis::Basis, x, y)
    m = size(y, 1)
    length(basis) > m && error("Can't be more basis funcs than points in y")
    return m
end

# get_coefs(::Basis, ::BasisMatrix, ::Array) does almost all the work for
# these methods
function funfitxy(basis::Basis, bs::BasisMatrix, y)
    check_funfit(basis, bs, y)
    c = get_coefs(basis, bs, y)
    c, bs
end

# use tensor form
function funfitxy(basis::Basis, x::TensorX, y)
    m = check_funfit(basis, x, y)

    bs = BasisMatrix(basis, Tensor(), x, 0)
    c = get_coefs(basis, bs, y)
    c, bs
end

function funfitxy(basis::Basis, x, y)
    # check input sizes
    m = check_funfit(basis, x, y)

    # additional check
    size(x, 1) != m && error("x and y are incompatible")

    bs = BasisMatrix(basis, Direct(), x, 0)
    c = get_coefs(basis, bs, y)
    c, bs
end

function funfitf(basis::Basis, f::Function, args...)
    X, xn = nodes(basis)
    y = f(X, args...)
    funfitxy(basis, xn, y)[1]
end

function Base.:\(b::Basis, y::AbstractArray)
    x123 = nodes(b)[2]
    funfitxy(b, x123, y)[1]
end

Base.:\(b::Basis, f::Function) = funfitf(b, f)

# ---------- #
# Evaluation #
# ---------- #

function _funeval(c, bs::BasisMatrix{Tensor}, order::AbstractMatrix{Int})  # funeval1
    kk, d = size(order)  # 95

    # 98 reverse the order of evaluation: B(d) × B(d-1) × ⋯ × B(1)
    order = flipdim(order .+ (size(bs.vals, 1)*(0:d-1)' - bs.order+1), 2)

    # 99
    nx = prod([size(bs.vals[1, j], 1) for j=1:d])

    _T = promote_type(eltype(c), eltype(bs))
    f = Array{_T,3}(nx, size(c, 2), kk)  # 100

    for i=1:kk
        f[:, :, i] = ckronx(bs.vals, c, order[i, :])  # 102
    end
    f
end

function _funeval(c, bs::BasisMatrix{Direct}, order::AbstractMatrix{Int})  # funeval2
    kk, d = size(order)  # 95
    # 114 reverse the order of evaluation: B(d)xB(d-1)x...xB(1)
    order = flipdim(order .+ (size(bs.vals, 1)*(0:d-1)' - bs.order+1), 2)

    _T = promote_type(eltype(c), eltype(bs))
    f = Array{_T,3}(size(bs.vals[1], 1), size(c, 2), kk)  # 116

    for i in 1:kk
        f[:, :, i] = cdprodx(bs.vals, c, order[i, :])  # 118
    end
    f
end

function _funeval(c, bs::BasisMatrix{Expanded}, order::AbstractMatrix{Int})  # funeval3
    nx = size(bs.vals[1], 1)
    kk = size(order, 1)

    _T = promote_type(eltype(c), eltype(bs))
    f = Array{_T,3}(nx, size(c, 2), kk)
    for i=1:kk
        this_order = order[i, :]
        ind = findfirst(x->bs.order[x, :] == this_order, 1:kk)
        if ind == 0
            msg = string("Requested order $(this_order) not in BasisMatrix ",
                         "with order $(bs.order)")
            error(msg)
        end
        f[:, :, i] = bs.vals[ind]*c  # 154
    end

    f
end

# 1d basis + x::Number + c::Mat => 1 point, many func ==> out 1d
funeval(c::AbstractMatrix, basis::Basis{1}, x::Real, order=0) =
    vec(funeval(c, basis, fill(x, 1, 1), order))

# 1d basis + x::Number + c::Vec => 1 point, 1 func ==> out scalar
funeval(c::AbstractVector, basis::Basis{1}, x::Real, order=0) =
    funeval(c, basis, fill(x, 1, 1), order)[1]

# 1d basis + x::Vec + c::Mat => manypoints, many func ==> out 2d
funeval(c::AbstractMatrix, basis::Basis{1}, x::AbstractVector{T}, order=0) where {T<:Number} =
    funeval(c, basis, x[:, :], order)

# 1d basis + x::Vec + c::Vec => manypoints, 1 func ==> out 1d
funeval(c::AbstractVector, basis::Basis{1}, x::AbstractVector{T}, order=0) where {T<:Number} =
    vec(funeval(c, basis, reshape(x, length(x), 1), order))

# N(>1)d basis + x::Vec + c::Vec ==> 1 point, 1 func ==> out scalar
funeval(c::AbstractVector, basis::Basis{N}, x::AbstractVector{T}, order=0) where {N,T<:Number} =
    funeval(c, basis, reshape(x, 1, N), order)[1]

# N(>1)d basis + x::Vec + c::Mat ==> 1 point, many func ==> out vec
funeval(c::AbstractMatrix, basis::Basis{N}, x::AbstractVector{T}, order=0) where {N,T<:Number} =
    vec(funeval(c, basis, reshape(x, 1, N), order))

function funeval(c, basis::Basis{N}, x::TensorX, order::Int=0) where N
    # check inputs
    size(x, 1) == N ||  error("x must have d=$N elements")

    if order != 0
        msg = string("passing order as integer only allowed for $(order=0).",
                     " Try calling the version where `order` is a matrix")
        error(msg)
    end

    _order = fill(0, 1, N)
    bs = BasisMatrix(SplineSparse, basis, Tensor(), x, _order)  # 67
    funeval(c, bs, _order)
end

function funeval(c, basis::Basis{N}, x::TensorX, _order::AbstractMatrix) where N
    # check inputs
    size(x, 1) == N ||  error("x must have d=$N elements")
    order = _check_order(N, _order)

    # construct tensor form
    bs = BasisMatrix(SparseMatrixCSC, basis, Tensor(), x, order)  # 67

    # pass to specialized method below
    return funeval(c, bs, order)
end

function funeval(c, basis::Basis{N}, x::AbstractMatrix, order::Int=0) where N
    # check inputs
    @boundscheck size(x, 2) == N || error("x must have d=$(N) columns")

    if order != 0
        msg = string("passing order as integer only allowed for $(order=0).",
                     " Try calling the version where `order` is a matrix")
        error(msg)
    end

    _order = fill(0, 1, N)
    bs = BasisMatrix(SplineSparse, basis, Direct(), x, _order)  # 67
    _out = funeval(c, bs, _order)

    # we only had one order, so we want to collapse the third dimension of _out
    return _out[:, :, 1]
end

function funeval(c, basis::Basis{N}, x::AbstractMatrix, _order::AbstractMatrix) where N
    # check that inputs are conformable
    @boundscheck size(x, 2) == N || error("x must have d=$(N) columns")  # 62
    order = _check_order(N, _order)

    # construct BasisMatrix in Direct for
    bs = BasisMatrix(SplineSparse, basis, Direct(), x, order)  # 67

    # pass of to specialized method below
    funeval(c, bs, order)
end

function funeval(c::AbstractVector, bs::BasisMatrix, order::AbstractMatrix{Int})
    _funeval(c, bs, order)[:, 1, :]
end

function funeval(c::AbstractMatrix, bs::BasisMatrix, order::AbstractMatrix{Int})
    _funeval(c, bs, order)
end

# default method
function funeval(c::AbstractVector, bs::BasisMatrix,
                 order::Vector{Int}=fill(0, size(bs.order, 2)))
    _funeval(c, bs, reshape(order, 1, length(order)))[:, 1, 1]
end

function funeval(c::AbstractMatrix, bs::BasisMatrix,
                 order::Vector{Int}=fill(1, size(bs.order, 2)))
    _funeval(c, bs, reshape(order, 1, length(order)))[:, :, 1]
end

# ------------------------------ #
# Convenience `Interpoland` type #
# ------------------------------ #

mutable struct Interpoland{TB<:Basis,TC<:AbstractArray,TBM<:BasisMatrix{Tensor}}
    basis::TB  # the basis -- can't change
    coefs::TC  # coefficients -- might change
    bmat::TBM  # BasisMatrix at nodes of `b` -- can't change
end

function Interpoland(basis::Basis, bs::BasisMatrix{Tensor}, y::AbstractArray)
    c = get_coefs(basis, bs, y)
    Interpoland(basis, c, bs)
end

# compute Tensor form and hand off to method above
function Interpoland(basis::Basis, y::AbstractArray)
    bs = BasisMatrix(basis, Tensor())
    Interpoland(basis, bs, y)
end

"""
Construct an Interpoland from a function.

The function must have the signature `f(::AbstractMatrix)::AbstractArray`
where each column of the input matrix is a vector of values along a single
dimension
"""
function Interpoland(basis::Basis, f::Function)
    x, xd = nodes(basis)
    y = f(x)
    bs = BasisMatrix(basis, Tensor(), xd)
    Interpoland(basis, bs, y)
end

Interpoland(p::BasisParams, f::Function) = Interpoland(Basis(p), f)

# let funeval take care of order and such. This just exists to make it so the
# user doesn't have to keep track of the coefficient vector
(itp::Interpoland)(x, order=0) = funeval(itp.coefs, itp.basis, x, order)

# now, given a new vector of `y` data we construct a new coefficient vector
function update_coefs!(interp::Interpoland, y::AbstractArray)
    # leverage the BasisMatrix we kept around
    c = funfitxy(interp.basis, interp.bmat, y)[1]
    copy!(interp.coefs, c)  # update c inplace b/c Interpoland is immutable
end

# similar for a function -- just hand off to above
update_coefs!(interp::Interpoland, f::Function) =
    update_coefs!(interp, f(nodes(interp.basis)[1]))

# alias update_coefs! to fit!
fit!(interp::Interpoland, y::AbstractArray) = update_coefs!(interp, y)
fit!(interp::Interpoland, f::Function) = update_coefs!(interp, f)

Base.show(io::IO, ::Interpoland{T,N,BST}) where {T,N,BST<:ABSR} =
    print(io, "$N dimensional interpoland")
