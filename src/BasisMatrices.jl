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

# types
export BasisFamily, Cheb, Lin, Spline, Basis,
       BasisParams, ChebParams, LinParams, SplineParams,
       AbstractBasisMatrixRep, Tensor, Expanded, Direct,
       BasisMatrix, Interpoland, SplineSparse, RowKron

# functions
export nodes, get_coefs, funfitxy, funfitf, funeval, evalbase,
       derivative_op, row_kron, evaluate, fit!, update_coefs!,
       complete_polynomial, complete_polynomial!, n_complete

include("util.jl")
include("spline_sparse.jl")
include("basis.jl")
include("basis_structure.jl")
include("interp.jl")

# include the rest of the Julian API
include("cheb.jl")
include("spline.jl")
include("lin.jl")

# include comlpete
include("complete.jl")


# deprecations
@deprecate BasisStructure BasisMatrix

end # module
