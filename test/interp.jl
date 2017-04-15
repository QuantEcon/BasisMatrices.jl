cases = ["B-splines","Cheb","Lin"]

# construct basis
holder = (
    Basis(SplineParams(25, -1, 1, 1),SplineParams(20, -1, 2, 3)),
    Basis(ChebParams(25, -1, 1),ChebParams(20, -1, 2)),
    Basis(LinParams(25, -1, 1),LinParams(20, -1, 2))
)

@testset for (i, case) in enumerate(cases)
    basis = holder[i]

    # get nodes
    X, x12 = nodes(basis)

    # function to interpolate
    f(x1, x2) = cos.(x2) ./ exp.(x1)
    f(X::Matrix) = f.(X[:, 1], X[:, 2])

    # function at nodes
    y = f(X)
    yprime2 = - sin.(X[:, 2]) ./ exp.(X[:, 1])

    # benchmark coefficients
    c_direct, bs_direct = funfitxy(basis, X, y)

    # this order isn't included bs_direct
    @test_throws ErrorException funeval(c_direct, convert(Expanded, bs_direct), [0 1])

    @testset "test funfitxy for tensor and direct agree on coefs" begin
        c_tensor, bs_tensor = funfitxy(basis, x12, y)
        @test maximum(abs, c_tensor -  c_direct) <=  1e-12
    end

    @testset "test funfitf" begin
        c = funfitf(basis, f)
        @test maximum(abs, c -  c_direct) <=  1e-12
    end

    @testset "test funeval methods" begin
        # single point
        sp = @inferred funeval(c_direct, basis, X[5:5, :])[1]
        @test maximum(abs, sp -  y[5]) <= 1e-12

        # multiple points using tensor directly
        mp = @inferred funeval(c_direct, basis, x12)
        @test maximum(abs, mp -  y) <=  1e-12

        # multiple points using direct
        mp = @inferred funeval(c_direct, basis, X)
        @test maximum(abs, mp -  y) <=  1e-12

        # multiple points giving basis in direct form
        mpd = @inferred funeval(c_direct, bs_direct)
        @test maximum(abs, mpd -  y) <=  1e-12

        # multiple points giving basis in expanded form
        Phiexp = Base.convert(Expanded, bs_direct)
        mpe = @inferred funeval(c_direct, Phiexp)
        @test maximum(abs, mpe -  y) <=  1e-12

        # order != 0. Note for Spline err is 7e-5. For Cheb 9e-14 and Lin 2e-1
        der1 = @inferred funeval(c_direct, basis, X, [0 1])
        @test maximum(abs, der1 - yprime2) <= 2e-1
        der2 = @inferred funeval(c_direct, basis, x12, [0 1])
        @test maximum(abs, der1 - yprime2) <= 2e-1
        @test maximum(abs, der1 - der2) <= 1e-12

    end

    @testset "other funeval methods" begin
        b1 = basis[1]
        y1 = sin.(x12[1])
        c = funfitxy(b1, x12[1], y1)[1]
        for (i, x) in enumerate(x12[1])
            @test ≈(y1[i], @inferred(funeval(c, b1, x)), atol=1e-14)
        end

        for (i, x) in enumerate(x12[1])
            @test ≈([y1[i], y1[i]], @inferred(funeval([c c], b1, x)), atol=1e-14)
        end

        @test ≈(y1, @inferred(funeval(c, b1, x12[1])), atol=1e-14)
        @test ≈([y1 y1], @inferred(funeval([c c], b1, x12[1])), atol=1e-14)

    end

    @testset "test interpoland methods" begin
        # (Basis, BasisMatrix, Array)
        intp1 = Interpoland(basis, BasisMatrix(basis, Tensor()), y)
        @test maximum(abs, intp1(X) - y) <= 1e-12

        # (Basis, Array,..)
        intp2 = Interpoland(basis, y)
        @test maximum(abs, intp2(X) - y) <= 1e-12

        # (BasisParams, Function)
        intp3 = Interpoland(basis, f)
        @test maximum(abs, intp3(X) - y) <= 1e-12
    end

    @testset "Printing" begin
        iob = IOBuffer()
        show(iob, Interpoland(basis, y))
    end

    # errors
    @test_throws ErrorException funeval(c_direct, basis, x12, 1)
    @test_throws ErrorException funeval(c_direct, basis, X, 1)

end # testset
