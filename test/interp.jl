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
        if case != "Lin"
            der1 = @inferred funeval(c_direct, basis, X, [0 1])
            @test maximum(abs, der1 - yprime2) <= 2e-1
            der2 = @inferred funeval(c_direct, basis, x12, [0 1])
            @test maximum(abs, der1 - yprime2) <= 2e-1
            @test maximum(abs, der1 - der2) <= 1e-12
        end

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

    _order = fill(0, 1, ndims(basis))
    @test BasisMatrices._extract_inds(BasisMatrix(basis, Direct()), _order) == [2 1]

end # testset

@testset "multi dim params" begin
    sp = SmolyakParams(Val{2}, 3, zeros(2), ones(2))
    pc = ChebParams(10, -4, 4)
    order = [0 0 0 0 0 0; 1 0 0 0 0 1; 1 0 0 0 0 0]
    bm = BasisMatrix(Basis(pc, sp, sp, pc), Direct(), fill(0.4, 1, 6), order)

    @test size(bm.vals) == (2, 4)
    @test bm.vals[1, 1] ==  BasisMatrices.evalbase(pc, [0.4], 0)
    @test bm.vals[1, 2] ==  BasisMatrices.evalbase(sp, [0.4 0.4], 0)
    @test bm.vals[1, 3] ==  BasisMatrices.evalbase(sp, [0.4 0.4], 0)
    @test bm.vals[1, 4] ==  BasisMatrices.evalbase(pc, [0.4], 0)

    @test bm.vals[2, 1] ==  BasisMatrices.evalbase(pc, [0.4], 1)
    @test_throws UndefRefError bm.vals[2, 2]
    @test_throws UndefRefError bm.vals[2, 3]
    @test bm.vals[2, 4] ==  BasisMatrices.evalbase(pc, [0.4], 1)


    row1 = [7 5 3 1]
    row2 = [7 5 3 2]
    row3 = [8 5 3 1]
    row4 = [8 5 3 2]
    ord1 = [0 0 0 0 0 0]
    ord2 = [1 0 0 0 0 0]
    ord3 = [0 0 0 0 0 1]
    ord4 = [1 0 0 0 0 1]

    for (want, arg) in zip(
            [row1, row2, row3, row4, [row1; row2], [row1; row3], [row1; row4], [row1; row2; row3], [row1; row2; row4], [row1; row2; row3; row4]],
            [ord1, ord2, ord3, ord4, [ord1; ord2], [ord1; ord3], [ord1; ord4], [ord1; ord2; ord3], [ord1; ord2; ord4], [ord1; ord2; ord3; ord4]]
        )
        @test BasisMatrices._extract_inds(bm, arg) == want
    end
    # TODO: _extract_inds, BasisMatrix, convert, funeval, cdprodx, ckronx
end  # testset
