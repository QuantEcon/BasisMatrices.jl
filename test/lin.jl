@testset "Test Linear Basis" begin

    bas1 = BasisMatrices.Basis(LinParams([0.0, 4.0], 5))

    @testset "test type" begin
        bas2 = BasisMatrices.Basis(LinParams([0.0, 2.0, 4.0], 5))
        @test  bas1.params[1].breaks  ==  [0.0, 1.0, 2.0, 3.0, 4.0]
        @test  bas2.params[1].evennum  ==  3
        @test_throws ErrorException BasisMatrices.Basis(LinParams([0.0], 5))
        @test_throws ErrorException BasisMatrices.Basis(LinParams([0.0, 1.5, 4.0], 5))
    end

    @testset "test constructor" begin

       valbas_lin = @inferred BasisMatrices.evalbase(bas1.params[1])
       valbas_spl = @inferred BasisMatrices.evalbase(BasisMatrices.SplineParams([0.0, 1.0, 2.0, 3.0, 4.0], 0, 1))

       @test  valbas_lin  ==  valbas_spl

    end

    @testset "test derivative" begin

    	#test derivative of basis indirectly, by fitting exponential function
    	bas2 = Basis(LinParams([0, 1.0], 1000000))
    	coeffs = funfitf(bas2, exp)
    	points = rand(10000)

    	d1 = evalbase(SparseMatrixCSC, bas2.params[1], points, 1) * coeffs
        @test ≈(d1, exp.(points), atol=1e-7)

    end
    @test BasisMatrices.family(LinParams) == Lin
    @test BasisMatrices.family_name(LinParams) == "Lin"
    @test issparse(LinParams)

    @test BasisMatrices.family(bas1.params[1]) == Lin
    @test BasisMatrices.family_name(bas1.params[1]) == "Lin"
    @test issparse(bas1.params[1])

    @test_throws ErrorException evalbase(SplineSparse, bas1.params[1], rand(4), 1)


    @testset "non Float64 eltypes" begin
        p = LinParams(Float32[0.1, 0.2, 0.3, 0.4, 0.5])
        @test Float32 == eltype(@inferred BasisMatrices.evalbase(p, Float32[0.25, 0.35]))
        @test Float32 == eltype(@inferred BasisMatrices.evalbase(p, Float32[0.25, 0.35], 1))
        @test Float64 == eltype(@inferred BasisMatrices.evalbase(p, [0.25, 0.35]))
    end

end
