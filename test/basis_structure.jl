# Tests conversion among basis-structure representations

@testset "Test Basis Structure Representations" begin
    mb = Basis(SplineParams(linspace(-3, 3, 9), 0, 1),
               ChebParams(7, 1e-4, 10.0),
               LinParams(linspace(-4, 2, 11), 0))

    # construct evaluation points
    X, x123 = nodes(mb)

    # construct expanded, direct, and tensor basis-structure representation
    Φ_expanded = BasisMatrix(mb, Expanded(), X)
    Φ_direct = BasisMatrix(mb, Direct(), X)
    Φ_tensor = BasisMatrix(mb, Tensor(), x123)

    # construct expanded, direct, and tensor basis-structure representation with 1D basis
    Φ_expanded_1d = BasisMatrix(mb[1], Expanded(), X[:,1])
    Φ_direct_1d = BasisMatrix(mb[1], Direct(), X[:,1])

    @testset "test standard Base methods" begin
        # test == and ndims, multiD
        for Φ in (Φ_expanded, Φ_direct, Φ_tensor)
            @test  Φ  ==  Φ
            @test ndims(Φ) == 3
        end

        @test Φ_expanded != Φ_direct
        @test Φ_expanded != Φ_tensor
        @test Φ_tensor != Φ_direct
        @test Φ_expanded_1d != Φ_expanded

        for Φ in (Φ_direct, Φ_tensor)
            @test ( Φ_expanded  ==  Φ ) == false
        end

        # test == and ndims, 1D
        for Φ in (Φ_expanded_1d, Φ_direct_1d)
            @test  Φ  ==  Φ
            @test ndims(Φ) == 1
        end

    end

    @testset "test convert methods" begin
        @test  Φ_direct  ==  convert(Direct, Φ_tensor)
        @test  Φ_expanded  ==  convert(Expanded, Φ_direct)
        @test  Φ_expanded  ==  convert(Expanded, Φ_tensor)
        @test ==(Φ_expanded,
                 convert(Expanded, convert(Direct, Φ_tensor)))


        # test the no-op covert method
        @test convert(Expanded, Φ_expanded) == Φ_expanded
        @test convert(Direct, Φ_direct) == Φ_direct
        @test convert(Tensor, Φ_tensor) == Φ_tensor

        # test that we can't do expanded -> (direct|tensor) or direct -> tensor
        @test_throws MethodError convert(Direct, Φ_expanded)
        @test_throws MethodError convert(Tensor, Φ_expanded)
        @test_throws MethodError convert(Tensor, Φ_direct)
    end

    @testset "test internal tools" begin
        ## test bmat_type
        for (TP, TM) in [(SplineParams{Vector{Float64}}, SparseMatrixCSC{Float64,Int}),
                         (SplineParams{Vector{Float32}}, SparseMatrixCSC{Float32,Int}),
                         (LinParams{AbstractVector{Float16}}, SparseMatrixCSC{Float16,Int}),
                         (ChebParams{Complex{Float64}}, Matrix{Complex{Float64}})]
            @test TM == @inferred BasisMatrices.bmat_type(TP)
        end

        @test BasisMatrices.bmat_type(SplineSparse, mb) == AbstractMatrix{Float64}
        @test BasisMatrices.bmat_type(mb) == AbstractMatrix{Float64}

        @test BasisMatrices.bmat_type(SplineSparse, mb[1]) == SplineSparse{Float64,Int}
        @test BasisMatrices.bmat_type(mb[1]) == SparseMatrixCSC{Float64,Int}

        @test BasisMatrices.bmat_type(SplineSparse, mb[2]) == Matrix{Float64}
        @test BasisMatrices.bmat_type(mb[2]) == Matrix{Float64}

        @test BasisMatrices.bmat_type(SplineSparse, mb[3]) == SplineSparse{Float64,Int}
        @test BasisMatrices.bmat_type(mb[3]) == SparseMatrixCSC{Float64,Int}

        ## test _checkx
        # create test data
        xm = rand(1, 2)
        xv = rand(2)
        xvv = [rand(2) for i=1:2]

        @test BasisMatrices._checkx(2, xm) == xm
        @test BasisMatrices._checkx(2, xv) == reshape(xv, 1, 2)
        @test BasisMatrices._checkx(2, xvv) == xvv
        @test BasisMatrices._checkx(1, xv) == xv

        @test_throws ErrorException BasisMatrices._checkx(2, rand(1, 3))
        @test_throws ErrorException BasisMatrices._checkx(2, rand(3))
        @test_throws ErrorException BasisMatrices._checkx(2, [rand(2) for i=1:3])

        ## test check_convert
        for Φ in (Φ_expanded, Φ_direct, Φ_tensor)
            @test BasisMatrices.check_convert(Φ, zeros(1, 3)) == (3, 1, 3)
            @test_throws ErrorException BasisMatrices.check_convert(Φ, zeros(1, 2))
            @test_throws ErrorException BasisMatrices.check_convert(Φ, fill(-1, 1, 3))
        end

        ## test check_basis_structure
        @testset "test check_basis_structure" begin
            order1 = zeros(Int, 1, 2)
            out1 = BasisMatrices.check_basis_structure(2, xm, order1)
            @test out1 == (1, order1, order1, [1 1], xm)

            # test when order::Int
            @test BasisMatrices.check_basis_structure(2, xm, 0) == out1

            # test the reshape(order, 1, N) branch (isa(order, Vector))
            @test BasisMatrices.check_basis_structure(2, xm, [0, 0]) == out1

            # check N=1 --> order = fill(order, 1, 1) branch
            out_1d = BasisMatrices.check_basis_structure(1, xv, 0)
            order_1d_out = fill(0, 1, 1)
            @test out_1d == (1, order_1d_out, order_1d_out, fill(1, 1, 1), xv)

            # check m > 1 branch
            order_m2 = [0 0; 1 1; 0 -1]
            out_m2 = BasisMatrices.check_basis_structure(2, xm, order_m2)
            @test out_m2 == (3, order_m2, [0 -1], [2 3], xm)


            order2 = zeros(Int, 1, 5)  # should throw error
            @test_throws ErrorException BasisMatrices.check_basis_structure(2, X,
                                                                       order2)
        end


    end

    @testset "constructors" begin
        # test constructor methods with ABSR types instead of instances
        @test BasisMatrix(mb, Expanded, X) == Φ_expanded
        @test BasisMatrix(mb, Direct, X) == Φ_direct
        @test BasisMatrix(mb, Tensor, x123) == Φ_tensor
        @test BasisMatrix(mb[1], Expanded, X[:,1]) == Φ_expanded_1d
        @test BasisMatrix(mb[1], Direct, X[:,1]) == Φ_direct_1d

        # test extra type methods
        @test BasisMatrix(Void, mb, Expanded, X) == Φ_expanded
        @test BasisMatrix(Void, mb, Direct, X) == Φ_direct
        @test BasisMatrix(Void, mb, Tensor, x123) == Φ_tensor
        @test BasisMatrix(Void, mb[1], Expanded, X[:,1]) == Φ_expanded_1d
        @test BasisMatrix(Void, mb[1], Direct, X[:,1]) == Φ_direct_1d

        ## Test SplineSparseMethods
        @test BasisMatrix(SplineSparse, mb, Expanded, X) == Φ_expanded

        # this one will be different -- need to break it out
        bmd = @inferred BasisMatrix(SplineSparse, mb, Direct, X)
        @test isa(bmd.vals[1], SplineSparse{Float64,Int})
        @test bmd.vals[1] == Φ_direct.vals[1]  # this works even though they have different types

        @test isa(bmd.vals[2], Matrix{Float64})
        @test bmd.vals[2] == Φ_direct.vals[2]

        @test isa(bmd.vals[3], SplineSparse{Float64,Int})
        @test bmd.vals[3] == Φ_direct.vals[3]

        bmt = @inferred BasisMatrix(SplineSparse, mb, Tensor, x123)
        @test isa(bmt.vals[1], SplineSparse{Float64,Int})
        @test bmt.vals[1] == Φ_tensor.vals[1]  # this works even though they have different types

        @test isa(bmt.vals[2], Matrix{Float64})
        @test bmt.vals[2] == Φ_tensor.vals[2]

        @test isa(bmt.vals[3], SplineSparse{Float64,Int})
        @test bmt.vals[3] == Φ_tensor.vals[3]

    end

    @testset "Test from PR #25" begin
        basisμ = Basis(Cheb, 20, 0.0, 1.0)
        basisσ = Basis(Cheb, 20, 0.0, 1.0)
        basis = Basis(basisμ, basisσ)
        S, (μs, σs) = nodes(basis)
        bs = BasisMatrix(basis, Expanded(), S, [0 2])
        @test isa(bs, BasisMatrix{Expanded}) == true
    end

    # call show (which calls writemime) so we can get 100% coverage
    @testset "Printing" begin
        iob = IOBuffer()
        show(iob, Φ_tensor)
    end

end  # testset
