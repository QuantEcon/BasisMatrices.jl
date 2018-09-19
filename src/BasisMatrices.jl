module BasisMatrices

# TODO: still need to write fund, minterp

#=
Note that each subtype of `BT<:BasisFamily` (with associated `PT<:BasisParam`)
will define the following constructor methods:

```julia
# basis constructors
Basis(::BT, args...)
Basis(::PT)

# node constructor
nodes(::PT)
```

=#

import Base: ==, *, \
using Base.Cartesian

using QuantEcon: gridmake, gridmake!, ckron, fix, fix!

using Combinatorics: with_replacement_combinations
using Base.Iterators: product

# types
export BasisFamily, Cheb, Lin, Spline, Basis, Smolyak,
       BasisParams, ChebParams, LinParams, SplineParams, SmolyakParams,
       AbstractBasisMatrixRep, Tensor, Expanded, Direct,
       BasisMatrix, Interpoland, SplineSparse, RowKron

# functions
export nodes, get_coefs, funfitxy, funfitf, funeval, evalbase,
       derivative_op, row_kron, evaluate, fit!, update_coefs!,
       complete_polynomial, complete_polynomial!, n_complete

#re-exports
export gridmake, gridmake!, ckron

# stdlib
using SparseArrays, LinearAlgebra, Statistics

abstract type BasisFamily end
abstract type BasisParams end
const IntSorV = Union{Int, AbstractVector{Int}}
const TensorX = Union{Tuple{Vararg{AbstractVector}},AbstractVector{<:AbstractVector}}

include("util.jl")
include("spline_sparse.jl")

# include the families

# BasisParams interface
SparseArrays.issparse(::Type{T}) where {T<:BasisParams} = false
Base.ndims(::BasisParams) = 1
for f in [:family, :family_name, :(SparseArrays.issparse), :(Base.eltype)]
    @eval $(f)(::T) where {T<:BasisParams} = $(f)(T)
end
include("cheb.jl")
include("lin.jl")
include("spline.jl")
include("complete.jl")
include("smolyak.jl")

evalbase(p::BasisParams, x::Number, args...) = evalbase(p, [x], args...)

# now some more interface methods that only make sense once we have defined
# the subtypes
basis_eltype(::TP, x) where {TP<:BasisParams} = promote_type(eltype(TP), eltype(x))
basis_eltype(::Type{TP}, x) where {TP<:BasisParams} = promote_type(eltype(TP), eltype(x))
"""
    basis_eltype(p::Union{BasisParams,Type{<:BasisParams}, x)

Return the eltype of the Basis matrix that would be obtained by calling
`evalbase(p, x)`
"""
basis_eltype

# give the type of the `vals` field based on the family type parameter of the
# corresponding basis. `Spline` and `Lin` use sparse, `Cheb` uses dense
# a hybrid must fall back to a generic AbstractMatrix{Float64}
# the default is just a dense matrix
# because there is only one dense version, we will start with the sparse
# case and overload for Cheb
bmat_type(::Type{TP}, x) where {TP<:BasisParams} = SparseMatrixCSC{basis_eltype(TP, x),Int}
bmat_type(::Type{T2}, ::Type{TP}, x) where {TP<:BasisParams,T2} = bmat_type(TP, x)
function bmat_type(::Type{T2}, ::Type{TP}, x) where {TP<:BasisParams,T2<:SplineSparse}
    SplineSparse{basis_eltype(TP, x),Int}
end

bmat_type(::Type{T}, x) where {T<:ChebParams} = Matrix{basis_eltype(T, x)}
function bmat_type(::Type{T2}, ::Type{TP}, x) where {TP<:ChebParams,T2<:SplineSparse}
    bmat_type(TP, x)
end

# version where there isn't an x passed
bmat_type(::Type{TP}) where {TP<:BasisParams} = bmat_type(TP, one(eltype(TP)))
function bmat_type(::Type{T2}, ::Type{TP}) where {TP<:BasisParams,T2}
    bmat_type(T2, TP, one(eltype(TP)))
end

# add methods to instances
bmat_type(::T) where {T<:BasisParams} = bmat_type(T)
bmat_type(ss::Type{T2}, ::TF) where {TF<:BasisParams,T2} = bmat_type(T2, TF)
bmat_type(ss::Type{T2}, ::TF, x) where {TF<:BasisParams,T2} = bmat_type(T2, TF, x)

# default method for evalbase with extra type hint is to just ignore the extra
# type hint
evalbase(::Type{T}, bp::BasisParams, x, order) where {T} = evalbase(bp, x, order)


# include other
include("basis.jl")
include("basis_structure.jl")
include("interp.jl")


# deprecations
@deprecate BasisStructure BasisMatrix

end # module
