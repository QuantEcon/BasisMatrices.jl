__precompile__()

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

using QuantEcon: gridmake, gridmake!, ckron, fix, fix!

using Compat
using Combinatorics: with_replacement_combinations
using Iterators: product

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

abstract BasisFamily
abstract BasisParams
typealias TensorX Union{Tuple{Vararg{AbstractVector}},AbstractVector{TypeVar(:TV,AbstractVector)}}
typealias IntSorV Union{Int, AbstractVector{Int}}

include("util.jl")
include("spline_sparse.jl")

# include the families

# BasisParams interface
Base.issparse{T<:BasisParams}(::Type{T}) = false
Base.ndims(::BasisParams) = 1
for f in [:family, :family_name, :(Base.issparse), :(Base.eltype)]
    @eval $(f){T<:BasisParams}(::T) = $(f)(T)
end
include("cheb.jl")
include("lin.jl")
include("spline.jl")
include("complete.jl")
include("smolyak.jl")

# now some more interface methods that only make sense once we have defined
# the subtypes

# give the type of the `vals` field based on the family type parameter of the
# corresponding basis. `Spline` and `Lin` use sparse, `Cheb` uses dense
# a hybrid must fall back to a generic AbstractMatrix{Float64}
# the default is just a dense matrix
bmat_type{TP<:BasisParams}(::Type{TP}) = Matrix{eltype(TP)}
bmat_type{TP<:BasisParams,T2}(::Type{T2}, ::Type{TP}) = Matrix{eltype(TP)}

# default to SparseMatrixCSC
bmat_type{T}(::Union{Type{LinParams{T}},Type{SplineParams{T}}}) = SparseMatrixCSC{eltype(T),Int}
bmat_type{T,T2}(::Type{T2}, ::Union{Type{LinParams{T}},Type{SplineParams{T}}}) = SparseMatrixCSC{eltype(T),Int}

# specialize for SplineSparse
bmat_type{T,T2<:SplineSparse}(::Type{T2}, ::Union{Type{LinParams{T}},Type{SplineParams{T}}}) =
    SplineSparse{eltype(T),Int}

# add methods to instances
bmat_type{T<:BasisParams}(::T) = bmat_type(T)
bmat_type{TF<:BasisParams,T2}(ss::Type{T2}, ::TF) = bmat_type(T2, TF)

# default method for evalbase with extra type hint is to just ignore the extra
# type hint
evalbase{T}(::Type{T}, bp::BasisParams, x, order) = evalbase(bp, x, order)


# include other
include("basis.jl")
include("basis_structure.jl")
include("interp.jl")


# deprecations
@deprecate BasisStructure BasisMatrix

end # module
