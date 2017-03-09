@testset "test BasisMatrices.lookup" begin
    table1 = [1.0, 4.0]
    table2 = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]

    x = [0.5, 1.0, 1.5, 4.0, 5.5]
    x2 = [0.5, 2.0]
    @test BasisMatrices.lookup(table1, x, 0) == [0, 1, 1, 2, 2]
    @test BasisMatrices.lookup(table1, x, 1) == [1, 1, 1, 2, 2]
    @test BasisMatrices.lookup(table1, x, 2) == [0, 1, 1, 1, 1]
    @test BasisMatrices.lookup(table1, x, 3) == [1, 1, 1, 1, 1]

    @test BasisMatrices.lookup(table2, x, 0) == [0, 3, 3, 12, 12]
    @test BasisMatrices.lookup(table2, x, 1) == [3, 3, 3, 12, 12]
    @test BasisMatrices.lookup(table2, x, 2) == [0, 3, 3, 8, 8]
    @test BasisMatrices.lookup(table2, x, 3) == [3, 3, 3, 8, 8]


    @test BasisMatrices.lookup([1.0], x2, 0) == [0, 1]
    @test BasisMatrices.lookup([1.0], x2, 1) == [1, 1]
    @test BasisMatrices.lookup([1.0], x2, 2) == [0, 1]
    @test BasisMatrices.lookup([1.0], x2, 3) == [1, 1]

    # test scalar version of lookup
    x2 = collect(linspace(-2.0, 4.0, 10))
    @test [BasisMatrices.lookup(x2, -3.0, i) for i=0:3] == [0, 1, 0, 1]
    @test [BasisMatrices.lookup(x2, 5.0, i) for i=0:3] == [10, 10, 9, 9]
    @test [BasisMatrices.lookup(x2, i, 0) for i=x2] == collect(0:length(x2)-1)
    @test [BasisMatrices.lookup(x2, i, 1) for i=x2] == [1; 1:length(x2)-1]
end

@testset "test BasisMatrices._check_order" begin
    # check (::Int, ::Int) method
    @test BasisMatrices._check_order(10, 0) == fill(0, 1, 10)
    @test BasisMatrices._check_order(1, 0) == fill(0, 1, 1)

    # check (::Int, ::Vector) method
    ov = [0, 0]
    @test BasisMatrices._check_order(2, ov) == reshape(ov, 1, 2)
    @test_throws DimensionMismatch BasisMatrices._check_order(3, ov)

    # check (::Int, ::Matrix) method
    om = [0 0]
    @test BasisMatrices._check_order(2, om) == om
    @test BasisMatrices._check_order(1, om) == om'
    @test_throws DimensionMismatch BasisMatrices._check_order(3, ov)
end

@testset "test BasisMatrices.ckronx" begin
    # will test by constructing an interpoland, then evaluating at the nodes
    # and verifying that we get back our original function
    basis = Basis(Basis(Spline(), 13, -1.0, 1.0, 3),
                  Basis(Spline(), 18, -5.0, 3.0, 3))
    X, x12 = nodes(basis);

    # make up a funciton and evaluate at the nodes
    f(x1, x2) = cos.(x1) ./ exp.(x2)
    f(X::Matrix) = f(X[:, 1], X[:, 2])
    y = f(X)

    # fit the interpoland in Tensor form (tensor b/c using x12)
    c, bs = funfitxy(basis, x12, y);

    # verify that we are actually interpolating -- all heavy lifting in funeval
    # is done by ckronx so this is effectively testing that we wrote that
    # function properly
    @test maximum(abs, funeval(c, bs, [0 0]) - y) <= 1e-13
end

@testset "test row_kron" begin
    h = ["a" "b"; "c" "d"]
    z = ["1" "2" "3"; "4" "5" "6"]
    want = ["a1" "a2" "a3" "b1" "b2" "b3"; "c4" "c5" "c6" "d4" "d5" "d6"]
    @test row_kron(h, z) == want

    # now test on some bigger matrices
    a = randn(400, 3)
    b = randn(400, 5)
    out = row_kron(a, b)
    @test size(out) == (400, 15)

    rows_good = true
    for row=1:400
        rows_good &= out[row, :] == kron(a[row, :], b[row, :])
    end
    @test rows_good == true


    for i in 1:100
        A = sprandn(30, 5, 0.3)
        B = sprandn(30, 3, 0.3)

        want = row_kron(A, B)

        @test maximum(abs, want - row_kron(full(A), B)) == 0
        @test maximum(abs, want - row_kron(A, full(B))) == 0
        @test maximum(abs, want - row_kron(full(A), full(B))) == 0
    end
end

@testset "cdprodx" begin
    for nrow in 3:3:100
        for nb in 2:5
            b = [sprand(nrow, rand(5:13), 0.3) for _ in 1:nb]
            c = rand(prod([size(_, 2) for _ in b]))

            full_b = reduce(row_kron, b)
            want = full_b * c
            have = BasisMatrices.cdprodx(b, c)

            @test maximum(abs, want - have) < 1e-12

            # test RowKron object
            rk = RowKron(b...)
            @test maximum(abs, want - rk*c) < 1e-12

            # now test transpose
            c2 = rand(size(rk, 1))
            want2 = full_b'c2
            @test maximum(abs, want2 - Base.At_mul_B(rk, c2)) < 1e-12
        end
    end
end
