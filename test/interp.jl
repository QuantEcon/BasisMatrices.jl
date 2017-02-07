cases = ["B-splines","Cheb","Lin"]

# construct basis
holder = (BasisMatrices.Basis(BasisMatrices.SplineParams(15,-1,1,1),BasisMatrices.SplineParams(20,-5,2,3)),
    BasisMatrices.Basis(BasisMatrices.ChebParams(15,-1,1),BasisMatrices.ChebParams(20,-5,2)),
    BasisMatrices.Basis(BasisMatrices.LinParams(15,-1,1),BasisMatrices.LinParams(20,-5,2)))

@testset for (i, case) in enumerate(cases)
    basis = holder[i]

    # get nodes
    X, x12 = BasisMatrices.nodes(basis)

    # function to interpolate
    f(x1, x2) = cos.(x1) ./ exp.(x2)
    f(X::Matrix) = f(X[:, 1], X[:, 2])

    # function at nodes
    y = f(X)

    # benchmark coefficients
    c_direct, bs_direct = BasisMatrices.funfitxy(basis, X, y)

    @testset "test funfitxy for tensor and direct agree on coefs" begin
        c_tensor, bs_tensor = BasisMatrices.funfitxy(basis, x12, y)
        @test maximum(abs, c_tensor -  c_direct) <=  1e-12
    end

    @testset "test funfitf" begin
        c = BasisMatrices.funfitf(basis,f)
        @test maximum(abs, c -  c_direct) <=  1e-12
    end

    @testset "test funeval methods" begin
        # single point
        sp = BasisMatrices.funeval(c_direct,basis,X[5:5,:])[1]
        @test maximum(abs, sp -  y[5]) <= 1e-12

        # multiple points using tensor directly
        mp = BasisMatrices.funeval(c_direct,basis,x12)
        @test maximum(abs, mp -  y) <=  1e-12

        # multiple points using direct
        mp = BasisMatrices.funeval(c_direct,basis,X)
        @test maximum(abs, mp -  y) <=  1e-12

        # multiple points giving basis in direct form
        mpd = BasisMatrices.funeval(c_direct,bs_direct)
        @test maximum(abs, mpd -  y) <=  1e-12

        # multiple points giving basis in expanded form
        Phiexp = Base.convert(BasisMatrices.Expanded,bs_direct)
        mpe = BasisMatrices.funeval(c_direct,Phiexp)
        @test maximum(abs, mpe -  y) <=  1e-12

    end

    @testset "test interpoland methods" begin
        # (Basis,BasisMatrix,..)
        intp1 = BasisMatrices.Interpoland(basis, y)
        @test maximum(abs, intp1(X) - y) <= 1e-12

        # (Basis,Array,..)
        intp2 = BasisMatrices.Interpoland(basis, y)
        @test maximum(abs, intp2(X) - y) <= 1e-12

        # (BasisParams,Function)
        intp3 = BasisMatrices.Interpoland(basis, f)
        @test maximum(abs, intp3(X) - y) <= 1e-12
    end

    @testset "Printing" begin
        iob = IOBuffer()
        show(iob, BasisMatrices.Interpoland(basis, y))
    end

    # TODO: call show on an interpoland instance to get coverage for writemime

end # testset
