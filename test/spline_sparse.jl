@testset "Test SplineSparse" begin

    # vals should be length 4
    @test_throws ErrorException SplineSparse{Float64,Int,1,2}(2, rand(3), [1, 1])

    for i in 1:10
        @test BasisMatrices.n_chunks(SplineSparse{Float64,Int64,i,i}) == i
        @test BasisMatrices.chunk_len(SplineSparse{Float64,Int64,i,i}) == i
    end

    @testset "full" begin
        s = SplineSparse(1, 2, 3, 1:4, 1:2)
        want = [1 2 0; 0 3 4]
        @test full(s) == want

        s = SplineSparse(2, 2, 6, 1:8, [1, 5, 2, 4])
        want = [1 2 0 0 3 4
                0 5 6 7 8 0]

        @test full(s) == want
    end

    @testset "findnz" begin
        s = SplineSparse(1, 2, 3, 1:4, 1:2)
        want = ([1, 1, 2, 2], [1, 2, 2, 3], [1, 2, 3, 4])
        @test findnz(s) == want

        s = SplineSparse(2, 2, 6, 1:8, [1, 5, 2, 4])
        want = (
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 2, 5, 6, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6, 7, 8]
        )
        @test findnz(s) == want
    end

    @testset "row_kron" begin
        s1 = SplineSparse(1,2, 3, rand(1:10, 12), rand(1:2, 6))
        s2 = SplineSparse(1,2, 4, rand(1:10, 12), rand(1:3, 6))
        want = row_kron(full(s1), full(s2))
        s12 = row_kron(s1, s2)

        @test full(s12) == want
        @test full(row_kron(s12, s1)) == row_kron(full(s12), full(s1))
        @test full(row_kron(s12, s12)) == row_kron(full(s12), full(s12))

        Base.zero(::Type{String}) = ""

        s1 = SplineSparse(1,2, 2, ["a", "b", "c", "d"], [1, 1])
        s2 = SplineSparse(1,3, 3, map(string, 1:6), [1, 1])

        want = ["a1" "a2" "a3" "b1" "b2" "b3"
                "c4" "c5" "c6" "d4" "d5" "d6"]
        have = row_kron(s1, s2)

        @test full(have) == want

    end

    @testset "getindex" begin
        s = SplineSparse(1, 2, 10, rand(12), rand(1:9, 6))
        full_s = full(s)

        for r in 1:size(s, 1)
            for c in 1:size(s, 1)
                @test s[r, c] == full_s[r, c]
            end
        end
    end

    @testset "*" begin
        s1 =  SplineSparse(1, 2, 3, 1:4, [1, 2])
        s2 =  SplineSparse(1, 3, 4, 1:6, [2, 1])
        s12 = SplineSparse(2, 3, 12, [1, 2, 3, 2, 4, 6, 12, 15, 18, 16, 20, 24],
                           [2, 6, 5, 9])

        x1 = rand(3)
        x2 = rand(4)
        x12 = rand(12)

        x1m = rand(3, 3)
        x2m = rand(4, 3)
        x12m = rand(12, 3)

        @test s1*x1 == full(s1)*x1
        @test s2*x2 == full(s2)*x2
        @test s12*x12 == full(s12)*x12

        @test s1*x1m == full(s1)*x1m
        @test s2*x2m == full(s2)*x2m
        @test s12*x12m == full(s12)*x12m

        # RowKron specialization
        rk = RowKron(s1, s2)

        @test rk * x12 ≈ full(s12)*x12
        @test rk * x12m ≈ full(s12)*x12m

        # mutating...
        out12 = zeros(2)
        out12m = zeros(2, 3)

        A_mul_B!(out12, rk, x12)
        @test out12 ≈ full(s12) * x12

        A_mul_B!(out12m, rk, x12m)
        @test out12m ≈ full(s12) * x12m

        # matrix evaluation of RowKron

        # errors with RowKron
        rk2 = RowKron(s1, s2, s12)
        @test_throws ArgumentError A_mul_B!(zeros(size(rk2,1)), rk2, rand(size(rk2, 2)))
        @test_throws DimensionMismatch A_mul_B!(zeros(size(rk,1)+1), rk, rand(size(rk, 2)))


    end

end
