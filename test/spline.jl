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

    @testset "augbreaks" begin
        breaks = cumsum(rand(10))

        for k in 1:10
            for minorder in 0:k-1
                n_rep = k - minorder
                augbreaks = vcat(fill(breaks[1], n_rep), breaks, fill(breaks[end], n_rep))
                ab = Augbreaks(breaks, k, minorder)
                @test ab == augbreaks
            end
        end

    end
end
