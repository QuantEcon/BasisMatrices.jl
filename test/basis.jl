@testset "Test Basis" begin
    bt = [Spline(), Cheb(), Lin()]
    n = [9, 7, 11]
    a = [-3, 1e-4, -4]
    b = [3, 10.0, 2]
    params = (
        SplineParams(range(-3, stop=3, length=9), 0, 1),
        ChebParams(7, 1e-4, 10.0),
        LinParams(range(-4, stop=2, length=11), 0)
    )

    # use univariate constructors
    b1, b2, b3 = map(Basis, params)

    # directly construct multivariate basis using (default) inner constructor
    b_all = Basis{3,typeof(params)}(params)

    # should be equivalent to Basis(b1, b1)
    b_spline3d = Basis(Spline(), [n[1], n[1]], [a[1], a[1]], [b[1], b[1]], [1, 1])

    # should be equivalent to Basis(b2, b2)
    b_cheb3d = Basis(Cheb(), [n[2], n[2]], [a[2], a[2]], [b[2], b[2]])

    # should be equivalent to Basis(b3, b3)
    b_lin3d = Basis(Lin(), [n[3], n[3]], [a[3], a[3]], [b[3], b[3]])

    @testset "constructors" begin
        @test map(x->isa(x, Basis{1}), (b1, b2, b3)) == (true, true, true)
        @test isa(b_all, Basis{3}) == true

        # use convenience outer constructor to not supply type parameter when
        # constructing multi-varaite basis
        @test b_all == @inferred Basis(params)

        # use Basis(params) constructor
        @test b1 == @inferred Basis(params[1])
        @test b2 == @inferred Basis(params[2])
        @test b3 == @inferred Basis(params[3])

        # test Basis(b1, b2, b3) format
        @test b_all == @inferred Basis(b1, b2, b3)

        # test Basis(params1, params2, params3)
        @test b_all == @inferred Basis(params...)

        # test fundefn type methods
        @test b_spline3d == @inferred Basis(b1, b1)
        @test b_cheb3d == @inferred Basis(b2, b2)
        @test b_lin3d == @inferred Basis(b3, b3)

        # test that basis of different dimensions are not equal
        @test (  ==(b_spline3d, b1) ) == false
        @test (  ==(b_cheb3d, b2) ) == false
        @test (  ==(b_lin3d, b3) ) == false
    end

    @testset "getindex and combining preserves basis" begin
        @test b_all[1] == b1
        @test b_all[2] == b2
        @test b_all[3] == b3
        @test Basis(b_all[1], b_all[2], b_all[3]) == b_all
        for bas in [b1, b2, b3]
            @test bas[1] == bas
            @test_throws ErrorException bas[2]
        end
    end

    @testset "ndims, length, min, max, and size" begin
        for (i, bas) in enumerate((b1, b2, b3))
            @test ndims(bas) == 1
            @test_throws ErrorException size(bas, 2)
            @test min(bas) == (a[i],)
            @test max(bas) == (b[i],)
        end

        @test ndims(b_all) == 3
        @test length(b_all) == prod(n)
        @test size(b_all) == tuple(n...)
        @test size(b_all, 1) == n[1]
        @test min(b_all) == tuple(a...)
        @test max(b_all) == tuple(b...)

    end

    @testset "test nodes" begin
        # extract nodes from individual basis
        n1, n2, n3 = nodes.(params)
        @test (n1, (n1,)) == @inferred nodes(b1)
        @test (n2, (n2,)) == @inferred nodes(b2)
        @test (n3, (n3,)) == @inferred nodes(b3)

        # extract product nodes as well as nodes along both dimensions using
        # the 3d basis
        n_all = gridmake(n1, n2, n3)
        @test (n_all, (n1, n2, n3)) == @inferred nodes(b_all)

        # test the nodes from basis 1 are what we expect, are the same as the
        # corresponding nodes from the 3d basis and have the correct length
        @test n1 == collect(range(-3, stop=3, length=9))
        @test length(n1) == n[1]

        # test that the nodes from basis 2 are the same as corresponding nodes
        # from 3d basis and have correct length
        @test length(n2) == n[2]

        # test that the nodes from basis 3 are the same as corresponding nodes
        # from 3d basis and have correct length
        @test n3 == collect(range(-4, stop=2, length=11))
        @test length(n3) == n[3]

        # verify that the nodes from combined 3d basis is correct size
        @test size(n_all) == (length(n1)*length(n2)*length(n3), 3)
    end

    # call show (which calls writemime) so I can get 100% coverage :)
    @testset "Printing" begin
        iob = IOBuffer()
        show(iob, ChebParams(10, -1, 1))
        show(iob, SplineParams(10, -1, 1))
        show(iob, LinParams(10, -1, 1))
        show(iob, b_all)
    end

    @testset "issue #36" begin
        for p_func in (LinParams, SplineParams, ChebParams)
            p = p_func(5, -1, 1)
            @test evalbase(p, 0.4) == evalbase(p, [0.4])
        end
    end
end
