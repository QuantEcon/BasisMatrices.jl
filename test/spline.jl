@testset "Test Spline Basis Evaluation" begin

    # constuct specific case that we can compute by hand

    n = 5
    a = 0
    b = 1
    params = SplineParams(linspace(a,b,n),0,1)

    x = rand(10000)

    basestr = @inferred BasisMatrices.evalbase(params, x)
    base = full(basestr)

    # vector of basis functions

    function linBspline(u)
        if u >= 0 && u < 0.25
            return [1-4*u,4*u,0,0,0]

        elseif u >= 0.25 && u < 0.5
            return [0,2*(1 - 2*u),4*u-1,0,0]

        elseif u >= 0.5 && u < 0.75
            return [0,0,3 - 4*u,2*(2*u - 1),0]

        else
            return [0,0,0,4*(1 - u),4*u-3]
        end
    end

    manualbase = Array{Float64}(length(x), n)

    for i in 1:length(x)
        manualbase[i, :] = linBspline(x[i])
    end

    @testset "test evalbase with linear B spline" begin
        @test all(manualbase .== base)
    end

    # test stuff that isn't implemented
    @test_throws ErrorException evalbase(SplineSparse, params, nodes(params), 1)
    @test_throws ErrorException evalbase(SplineSparse, params, nodes(params), -1)
    @test_throws ErrorException evalbase(params, nodes(params), -1)

    @test BasisMatrices.family(SplineParams) == Spline
    @test BasisMatrices.family_name(SplineParams) == "Spline"
    @test issparse(SplineParams)

    @test BasisMatrices.family(params) == Spline
    @test BasisMatrices.family_name(params) == "Spline"
    @test issparse(params)


    @testset "non Float64 eltypes" begin
        p = SplineParams(Float32[0.1, 0.2, 0.3, 0.4, 0.5], 0, 3)
        @test Float32 == eltype(@inferred BasisMatrices.evalbase(p, Float32[0.25, 0.35]))
        @test Float32 == eltype(@inferred BasisMatrices.evalbase(p, Float32[0.25, 0.35], 1))
        @test Float64 == eltype(@inferred BasisMatrices.evalbase(p, [0.25, 0.35]))
    end
end
