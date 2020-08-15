# BasisMatrices

[![Build Status](https://travis-ci.org/QuantEcon/BasisMatrices.jl.svg?branch=master)](https://travis-ci.org/QuantEcon/BasisMatrices.jl) [![codecov.io](http://codecov.io/github/QuantEcon/BasisMatrices.jl/coverage.svg?branch=master)](http://codecov.io/github/QuantEcon/BasisMatrices.jl?branch=master) [![Coverage Status](https://coveralls.io/repos/QuantEcon/BasisMatrices.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/QuantEcon/BasisMatrices.jl?branch=master)


Portions of this library are inspired by the [CompEcon Matlab toolbox](http://www4.ncsu.edu/~pfackler/compecon/toolbox.html) by Paul Fackler and Mario Miranda. The original Matlab code was written to accompany the publication

> Miranda, Mario J., and Paul L. Fackler. Applied computational economics and finance. MIT press, 2004.

The portions of this package that are based on their code have been licensed with their permission.

## Quick (and incomplete intro)


### Matlab-esque interface

For an API similar to the original [CompEcon Matlab package](http://www4.ncsu.edu/~pfackler/compecon/toolbox.html) by Miranda and Fackler, please see the [CompEcon.jl](https://github.com/QuantEcon/CompEcon.jl) package.

## Example

Here's an example of how to use the Julia-based API to set up multi-dimensional basis matrix and work with it.

```julia
using LinearAlgebra: lu
linspace(a,b,n) = range(a; stop=b, length=n)
ygrid0 = linspace(-4, 4, 10)
agrid0 = linspace(0.0.^0.4, 100.0.^0.4, 25).^(1/0.4)

# method one, using the Basis constructor multiple times
basis = Basis(SplineParams(agrid0, 0, 3),  # cubic spline
              SplineParams(ygrid0, 0, 1))  # linear

# method two, constructing separately, then calling `Basis` with the two
a_basis = Basis(SplineParams(agrid0, 0, 3))
y_basis = Basis(SplineParams(ygrid0, 0, 1))
basis = Basis(a_basis, y_basis)

# Construct state vector (matrix). Note that for higher order splines points
# are added to the input vector, so let's extract the actual grid points used
# from the second argument
S, (agrid, ygrid) = nodes(basis)

# construct basis matrix and its lu-factorization for very fast inversion
# NOTE: I am doing this in a round-about way. I could have just done
#       Φ = BasisMatrix(basis), but doing it this way gives me the direct
#       representation so I get Φ_y without repeating any calculations
Φ_direct = BasisMatrix(basis, Direct(), S, [0 0])
Φ_y = Φ_direct.vals[2]
Φ = convert(Expanded, Φ_direct, [0 0]).vals[1]
lu_Φ = lu(Φ)
```

## Basic Overview of Julian API

This section provides a sketch of the type based Julian API.

### Theoretical Foundation

To understand the Julian API and type system, we first need to understand the fundamental theory behind the interpolation scheme implemented here. Interpolation in BasisMatrices is built around three key concepts:

1. An functional `Basis`: for each dimension, the basis specifies
    - family of basis function (B spline, Chebyshev polynomials, ect.)
    - domain (bounds)
    - interpolation nodes (grid on domain)
2. A `BasisMatrix`:
    - Represents the evaluation of basis functions at the interpolation nodes
    - Constructed one dimension at a time, then combined with tensor product
3. A coefficient vector: used to map from domain of the `Basis` into real line

### Core types

Functionality implemented around 5 core types (or type families) that relate closely to the theoretical concepts outlined above.

#### Representing the `Basis`

The first two groups of type are helper types used to facilitate construction of the `Basis`. They are the `BasisFamily` and the `BasisParams` types:

First is the `BasisFamily`:

```julia
abstract type BasisFamily end
struct Cheb <: BasisFamily end
struct Lin <: BasisFamily end
struct Spline <: BasisFamily end

abstract type BasisParams end
mutable struct ChebParams <: BasisParams
    n::Int
    a::Float64
    b::Float64
end

mutable struct SplineParams <: BasisParams
    breaks::Vector{Float64}
    evennum::Int
    k::Int
end

mutable struct LinParams <: BasisParams
    breaks::Vector{Float64}
    evennum::Int
end
```

`BasisFamily` is an abstract type, whose subtypes are singletons that specify the class of functions in the basis.

`BasisParams` is an abstract type, whose subtypes are type types that hold all information needed to construct the Basis of a particular class

Then we have the central `Basis` type:

```julia
mutable struct Basis{N,TP<:Tuple}
    params::TP     # params to construct basis
end
```

The `params` field of this type contains an `N` element tuple, where each
element is a subtype of `BasisParams`. This represents an `N` dimensional
basis.

The `Basis` has support for the following methods:

- A whole slew of constructors
- `getindex(b::Basis, i::Int)`: which extracts the univariate `Basis` along the `i`th dimension
- `ndims`: The number of dimensions
- `length`: the product of the `n` field
- `size(b::Basis, i::Int)`: The `i`th element of the `n` field (number of basis functions in dimension `i`)
- `size(b::Basis)`: `b.n` as a tuple instead of a vector (similar to `size(a::Array)`)
- `==`: test two basis for equality
- `nodes(b::Basis)->(Matrix, Vector{Vector{Float64}})`: the interpolation nodes. the first element is the tensor product of all dimensions, second element is a vector of vectors, where the `i`th element contains the nodes along dimension `i`.
- `min` and `max` gives an `N` element tuple of the lower and upper bounds of the domain in each dimension.

#### `BasisMatrix` representation

Next we turn to representing the `BasisMatrix`, which is responsible for keeping track of the basis functions evaluated at the interpolation nodes. To keep track of this representation, we have another family of helper types:

```julia
abstract type AbstractBasisMatrixRep end
const ABSR = AbstractBasisMatrixRep

struct Tensor <: ABSR end
struct Direct <: ABSR end
struct Expanded <: ABSR end
```

`AbstractBasisMatrixRep` is an abstract types, whose subtypes are singletons that specify how the basis matrices are stored. To understand how they are different, we need to see the structure of the `BasisMatrix` type:

```julia
mutable struct BasisMatrix{BST<:ABSR, TM<:AbstractMatrix}
    order::Matrix{Int}
    vals::Matrix{TM}
end
```

The `order` field keeps track of what order of derivative or integral the arrays inside `vals` correspond to.


The content inside `vals` will vary based on the type Parameter `BST<:AbstractBasisMatrixRep`:

1. for `BST==Tensor` `vals` will store the evaluation of the basis functions at each of the integration nodes independently. Thus, the `[d, i]` element will be the derivative order `d` Basis at the interpolation nodes along the `i`th dimension (each column is a basis function, each row is an interpolation node). This is the most compact and memory efficient representation
2. For `BST==Direct` `vals` will expand along the first dimension (rows) of the array so that for each `i`, the `[d, i]` element will have `length(basis)` rows and `basis.n[i]` (modulo loss or addition of basis functions from derivative/integral operators.)
3. For `BST==Expanded` `vals` will be expanded along both the rows and the columns and will contain a single `Matrix` for each desired derivative order. This format is the least memory efficient, but simplest conceptually for thinking about how to compute a coefficient vector (if `y` is `f(x)` then `coefs[d] = b.vals[d] \ y`)

See the file
[demo/basis_mat_formats.jl](https://github.com/QuantEcon/BasisMatrices.jl/blob/master/demo/basis_mat_formats.jl)
for a more detailed description of the basis matrix formats.

#### Convenience `Interpoland` type

Finally the convenient `Interpoland` type:

```julia
mutable struct Interpoland{TB<:Basis,TC<:AbstractArray,TBM<:BasisMatrix{Tensor}}
    basis::TB  # the basis -- can't change
    coefs::TC  # coefficients -- might change
    bmat::TBM  # BasisMatrix at nodes of `b` -- can't change
end
```

The main purpose of this type is to keep track of the coefficient vector and the `Basis` so the user doesn't have to carry both of them around. It also holds a `BasisMatrix` for the evaluation of the basis matrices at the interpolation nodes. This means that if the coefficient vector needs to be updated, this `BasisMatrix` will not be re-computed.
