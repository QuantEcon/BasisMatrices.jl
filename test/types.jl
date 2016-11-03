sp = SplineParams(linspace(0, 5, 10), 0, 3)
cp = ChebParams(6, 2.0, 5.0)
lp = LinParams(linspace(0, 5, 10), 0)

@testset "Test subtype structure" begin
    @testset "Basis families" begin
        for T in [Cheb, Lin, Spline]
            @test T <: BasisFamily
        end
    end

    @testset "params" begin
        for T in [ChebParams, SplineParams, LinParams]
            @test T <: BasisParams
        end
    end

    @testset "basis structure representations" begin
        for T in [Tensor, Direct, Expanded]
            @test T <: AbstractBasisMatrixRep
        end
    end
end

@testset "test _param method" begin
    for (T, TP) in [(Cheb, ChebParams), (Lin, LinParams), (Spline, SplineParams)]
        @test BasisMatrices._param(T) == TP
        @test BasisMatrices._param(T()) == TP
    end
end

@testset "test more Param methods" begin

    @testset "Test extra outer constructors" begin
        @test sp == SplineParams(10, 0, 5, 3)
        @test lp == LinParams(10, 0, 5)
    end

    @testset "test equality of params" begin
        @test (  ==(sp, lp) ) == false
        @test (  ==(sp, cp) ) == false
        @test (  ==(cp, lp) ) == false

        @test   ==(sp, sp)
        @test   ==(cp, cp)
        @test   ==(lp, lp)
    end
end
