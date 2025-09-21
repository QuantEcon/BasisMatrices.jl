# BasisMatrices.jl

BasisMatrices.jl is a Julia package for computational economics focused on function approximation and interpolation using basis matrices. It provides tools for working with Chebyshev polynomials, B-splines, linear interpolation, and Smolyak sparse grids.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap, build, and test the repository:
- Julia is already installed (use `julia --version` to check)
- `cd /home/runner/work/BasisMatrices.jl/BasisMatrices.jl`
- `julia -e "using Pkg; Pkg.activate(\".\")"` -- activate the package environment
- `julia -e "using Pkg; Pkg.activate(\".\"); Pkg.instantiate()"` -- install dependencies (takes ~2 seconds)
- `julia -e "using Pkg; Pkg.activate(\".\"); Pkg.test()"` -- run full test suite (takes ~115 seconds). **NEVER CANCEL.** Set timeout to 180+ seconds.

### Load and test package functionality:
- ALWAYS run the bootstrapping steps first
- Load package: `julia -e "using Pkg; Pkg.activate(\".\"); using BasisMatrices"` -- takes ~2 seconds
- Test basic functionality:
  ```julia
  julia -e "
  using Pkg; Pkg.activate(\".\")
  using BasisMatrices
  
  # Test Chebyshev basis
  basis = Basis(ChebParams(5, -1, 1))
  println(\"Created Chebyshev basis with 5 nodes\")
  
  # Test 2D basis with splines and linear
  basis2d = Basis(LinParams(15, -2, 2), LinParams(10, -1, 3))
  S, (x, y) = nodes(basis2d)
  println(\"Created 2D basis with \$(size(S, 1)) nodes\")
  "
  ```

### Run demos and examples:
- Demo files are in `demo/` directory
- `cd demo && julia --project=.. -e "include(\"basis_mat_formats.jl\")"` -- comprehensive demo (takes ~16 seconds)
- Examples notebook: `demo/examples.ipynb` (Jupyter notebook with Julia kernel)

## Validation

### Manual validation scenarios:
- **ALWAYS** test basic functionality after making changes:
  1. Load the package successfully
  2. Create different types of basis functions (Chebyshev, Spline, Linear)
  3. Generate nodes and construct basis matrices
  4. Test interpolation functionality
- Run the `basis_mat_formats.jl` demo to ensure core functionality works
- **NEVER** skip validation because it takes time - functionality validation is critical

### Test execution:
- Individual test files can be run: `julia -e "using Pkg; Pkg.activate(\".\"); include(\"test/types.jl\")"`
- Full test suite: `julia -e "using Pkg; Pkg.activate(\".\"); Pkg.test()"` -- **NEVER CANCEL.** Always wait ~115 seconds for completion.
- Tests are comprehensive and cover all core functionality (7900+ individual assertions)

## Common tasks

### Repository structure:
```
.
├── .github/workflows/    # CI configuration (GitHub Actions)
├── .travis.yml           # Legacy Travis CI (still active)
├── LICENSE.md           # MIT license
├── Project.toml         # Package metadata and dependencies
├── README.md            # Package documentation
├── demo/                # Educational examples and demos
├── src/                 # Source code
│   ├── BasisMatrices.jl # Main module file
│   ├── basis.jl         # Core Basis type and methods
│   ├── cheb.jl          # Chebyshev polynomial basis
│   ├── spline.jl        # B-spline basis functions
│   ├── lin.jl           # Linear interpolation
│   ├── smolyak.jl       # Smolyak sparse grids
│   ├── interp.jl        # Interpolation methods
│   └── util.jl          # Utility functions
└── test/                # Test suite
    ├── runtests.jl      # Main test runner
    └── *.jl             # Individual test files
```

### Key Julia package commands:
- `Pkg.activate(".")` -- activate local package environment
- `Pkg.instantiate()` -- install/update dependencies
- `Pkg.test()` -- run test suite
- `Pkg.status()` -- show package status and dependencies

### Project.toml dependencies:
```toml
[deps]
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
QuantEcon = "fcd29c91-0bd7-5a09-975d-7ac3f643a60c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
```

### Julia version compatibility:
- Supports Julia 1.6+ (see Project.toml)
- Current CI tests against Julia 1.10 and 1.11 on Ubuntu, Windows, macOS

### Core functionality overview:
- **Basis types**: Chebyshev (`ChebParams`), Splines (`SplineParams`), Linear (`LinParams`), Smolyak (`SmolyakParams`)
- **BasisMatrix representations**: Tensor (memory efficient), Direct (flexible), Expanded (conceptually simple)
- **Key functions**: `nodes()`, `evalbase()`, `funfitxy()`, `funeval()`, `complete_polynomial()`
- **Interpolation**: Function approximation, evaluation at arbitrary points, derivative computation

### Troubleshooting:
- Method definition warnings in `util.jl:310` are expected
- If tests fail with "IOError: write: broken pipe", it's usually due to output truncation, not a real failure

### CI and workflows:
- GitHub Actions workflows in `.github/workflows/`
- Runs on push to master and PRs
- Tests Julia 1.x (latest) and LTS on Ubuntu, Windows, macOS
- Includes code coverage reporting to Codecov
- CompatHelper runs daily to check dependency updates
- TagBot handles automatic tagging for releases

## Important Notes

### Timing expectations:
- **Package instantiation**: ~2 seconds (after first time)
- **Package loading**: ~2 seconds
- **Full test suite**: ~115 seconds -- **NEVER CANCEL.** Set timeout to 180+ seconds minimum.
- **Demo execution**: ~16 seconds for `basis_mat_formats.jl`

### Julia ecosystem specifics:
- Uses `Pkg` package manager (not npm/pip equivalent)
- `Project.toml` defines package metadata and dependencies
- `Manifest.toml` locks specific dependency versions
- Activate package environment with `Pkg.activate(".")` before any operations

### No build system required:
- Julia packages typically don't require compilation/build steps
- Just activate environment and instantiate dependencies
- No Makefile, npm scripts, or similar build tools needed

### Related packages:
- [CompEcon.jl](https://github.com/QuantEcon/CompEcon.jl) provides Matlab-compatible API
- [QuantEcon.jl](https://github.com/QuantEcon/QuantEcon.jl) provides broader computational economics tools