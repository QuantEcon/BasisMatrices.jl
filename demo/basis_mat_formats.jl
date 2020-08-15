using BasisMatrices

# construct 2d basis
basis = Basis(LinParams(15, -2, 2), LinParams(10, -1, 3))
S, (x, y) = nodes(basis)

# define function to approximate
func(x, y) = exp(-2*x) * sin(y)
f = func.(S[:, 1], S[:, 2])

### getting coefficient vector

## tensor form
bmt = BasisMatrix(basis, Tensor(), [x, y])

# bmt.vals[1] is all basis funcs for first dim, evaluated at nodes for first dim
# similar for 2
@assert bmt.vals[1] == evalbase(basis.params[1], x, 0)
@assert bmt.vals[2] == evalbase(basis.params[2], y, 0)

# to get coefficients we need to do kron(bmt.vals[2], bmt.vals[1]) \ f
c1 = kron(bmt.vals[2], bmt.vals[1]) \ f

# calling funfitxy with the BasisMatrix instance will use optimized version of
# that operation that never constructs full kronecker product
c2 = funfitxy(basis, bmt, f)[1]
@assert maximum(abs, c1 - c2) < 1e-14

## Direct form
bmd = BasisMatrix(basis, Direct(), S)

# bmd.vals[1] will be have size (length(x)*length(y), length(x)) and
# bmd.vals[2] will be have size (length(x)*length(y), length(y))
# Each column represents the evaluation at one basis function.

# The reason there are length(x)*length(y) rows in both entries of bmd.vals is
# that an expansion down the columns (i.e. at the evaluation points) has
# already been done. The ordering of the rows follows column major conventions,
# which means that the first dimension varies fastest. To see this more clearly
# look at the form of the matrix `S`.

# Now to get coefficients we will need to compute row_kron(bm.vals[2], bm.vals[2]) \ f
c3 = row_kron(bmd.vals[2], bmd.vals[1]) \ f

@assert c3 ≈ c2

# calling funfitxy with the BasisMatrix instance will use optimized version of
# that operation that never constructs row wise kronecker product
c4 = funfitxy(basis, bmd, f)[1]

@assert c4 ≈ c2

## Expanded form
bme = BasisMatrix(basis, Expanded(), S)

# to get coefficients we just need to do bme.vals[1] \ f
c5 = bme.vals[1] \ f

@assert c5 ≈ c2

# the reason this worked is that Expanded form computes the fully expanded
# version of the basis matrix. This means that the expansion down columns and
# across rows has already been done.

# the funfitxy method for the Expanded BasisMatrix does exactly this operation
c6 = funfitxy(basis, bme, f)[1]

@assert c6 ≈ c2


#=

Compare/contrast formats:

- Tensor is the most efficient in terms of memory, but you are limited to
evaluation at a full cartesian product of the individual vectors you pass in
- Direct is the middle in terms of efficiency, but doesn't suffer from the
flexibility issue we saw with Expanded: you can evaluate at arbitrary points in
the 2d space simply by passing in the points as rows to the matrix S. However,
if we do want the cartesian product of 1d vectors, we are better off using
Tensor because the Direct form will repeat calculations for the expanded grid
points (to see it, look at how the second column of S repeats the first
gridpoint of y a total of length(x) times. When you build the Direct form, we
don't assume any structure about the rows of S, so we will repeat the
computations needed to evaluate the BasisMatrices along the second dimension
length(x) times. The same holds for the first dimension (each evaluation is
repeated length(y) times), but it is easier to explain using the second.)
- Expanded is the least efficient in terms of memory and CPU time, but is the
easiest to reason about when thinking about how to construct a coefficient
vector. It also has the benefit of specifying arbitrary evaluation points as
rows of the matrix argument.

=#

### Conversion

# you can only convert "outword" in terms of expanding matrices. This means
# you can convert from Tensor to either Direct or Expanded, or you can convert
# direct to Expanded. No other combinations are possible.

## Tensor -> Direct
# conversion from Tensor -> Direct amounts to expanding the columns of the
# tensor form such that elements are repeated in column major order (i.e.
# the first column varies fastest)
Φd1 = (
    repeat(bmt.vals[1], length(y)),
    hcat([repeat(bmt.vals[2][:, i], inner=length(x)) for i in 1:length(y)]...)
    )

@assert bmd.vals[1] ≈ Φd1[1]
@assert bmd.vals[2] ≈ Φd1[2]

## Tensor -> expanded
# we've already seen this one as just kron(bmt.vals[2], bmt.vals[1]) (think
# about how we computed the c1 and c5)
Φe1 = kron(bmt.vals[2], bmt.vals[1])
@assert Φe1 ≈ bme.vals[1]

## Direct -> Expanded
# We've also seen that this is row_kron(bmd.vals[2], bmd.vals[1]) (check c3 and
# c5)
Φe2 = row_kron(bmd.vals[2], bmd.vals[1])
@assert Φe2 ≈ bme.vals[1]

# The julia methods `convert(Format, bm, [order])` do these operations for you,
# but return a fully formed BasisMatrix object instead of just a plain Julia
# AbstractMatrix subtype
Φd2 = convert(Direct, bmt)
@assert bmd.vals[1] ≈ Φd2.vals[1]
@assert bmd.vals[2] ≈ Φd2.vals[2]

Φe3 = convert(Expanded, bmt)
@assert Φe3.vals[1] ≈ bme.vals[1]

Φe4 = convert(Expanded, bmd)
@assert Φe4.vals[1] ≈ bme.vals[1]

### Evaluation off nodes

# the last stop on our tour is how we might consider evaluating the
# interpolated function at points different from the interpolation nodes.
# As an example, suppose we want to evaluate at all the original y points, but
# new x points. We will want the full combination of all these points
x2 = collect(range(-1.5; stop=1.5, length=40))

# for evaluation of our approximation, let's evaluate func at these points
f2 = vec(func.(x2, y'))

## Using Tensor form
# Let's try using tensor form first. Remember that bmt.vals[2] already contains
# the evaluation of all  the basis functions in the 2nd dimension at all the
# points in y. This means that we only need to evaluate a new basis matrix for
# the x2 vector
Φ1_x2 = evalbase(basis.params[1], x2, 0)

# then to evaluate our interpolated function we do...
ft1 = kron(bmt.vals[2], Φ1_x2)*c1

# the approximation actually doesn't fit func extremely well, so we will have a
# relatively loose sense of success in our interpolation
@assert maximum(abs, ft1 - f2) < 1.0

# if we didn't want to form the full `kron` we could have construct a
# BasisMatrix instance by hand (NOTE: to do this we had to do a relatively ugly
# thing and allocate vals first, then populate it and we had to pass type
# params to the BasisMatrix constructor. We can get around this, we just need
# to be more clever in how we accept arguments to BasisMatrix).
bmt2_vals = Array{typeof(Φ1_x2)}(undef,1, 2)
bmt2_vals[1] = Φ1_x2; bmt2_vals[2] = bmt.vals[2]
bmt2 = BasisMatrix{Tensor,typeof(Φ1_x2)}([0 0], bmt2_vals)
ft2 = funeval(c1, bmt2)

@assert ft1 ≈ ft2

## Using Direct form
# to use direct form, we first need to construct the expanded grid on x2 and y
S2 = BasisMatrices.gridmake(x2, y)
bmd2 = BasisMatrix(basis, Direct(), S2)

# now we can evaluate using row_kron(bmd2.vals[2], bmd2.vals[1]) * c
fd1 = row_kron(bmd2.vals[2], bmd2.vals[1]) * c1
@assert fd1 ≈ ft1

# we could also use the funeval method which would help us avoid building that
# full row_kron matrix
fd2 = funeval(c1, bmd2)
@assert fd2 ≈ ft1


## Using Expanded form
# we can also use S2 to construct the Expanded basis matrix
bme2 = BasisMatrix(basis, Expanded(), S2)

# evaluation now is juse bme2.vals[2] * c
fe1 = bme2.vals[1]*c1

@assert fe1 ≈ ft1

# Here the funeval version is identical to the operation from above
fe2 = funeval(c1, bme2)
@assert fe2 ≈ ft1
