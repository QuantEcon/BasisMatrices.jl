module BasisMatricesTests

using BasisMatrices
using Test, LinearAlgebra, SparseArrays, Statistics

tests = ["types.jl", "basis.jl", "util.jl", "spline.jl", "interp.jl",
         "cheb.jl", "lin.jl", "basis_structure.jl", "complete.jl",
         "spline_sparse.jl", "smol.jl"]

if length(ARGS) > 0
    tests = ARGS
end

end_jl(s) = endswith(s, ".jl") ? s : s * ".jl"

for t in tests
    printstyled("* $t\n", color=:green)
    include(end_jl(t))
end


end  # module
