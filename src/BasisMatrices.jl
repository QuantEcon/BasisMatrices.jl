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
    @eval $(f)(t::BasisParams) = $(f)(typeof(t))
end
include("cheb.jl")
include("lin.jl")
include("spline.jl")
include("complete.jl")
include("smolyak.jl")


# include other
include("basis.jl")
include("basis_structure.jl")
include("interp.jl")


# deprecations
@deprecate BasisStructure BasisMatrix

end # module
