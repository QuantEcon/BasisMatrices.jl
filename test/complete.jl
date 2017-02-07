@testset "Testing logic using strings" begin
    Base.one{T<:AbstractString}(::Type{T}) = "one "
    z = ["x ", "y ", "z "]

    want_1 = ["one " "one x " "one y " "one z "]
    @test complete_polynomial(reshape(z, 1, length(z)), 1) == want_1
    # Test vector methods to make sure generates same thing as matrix methods
    @test complete_polynomial(z, 1) == want_1[:]

    want_2 = ["one " "one x " "one x x " "one x y " "one x z " "one y " "one y y " "one y z " "one z " "one z z "]
    @test complete_polynomial(reshape(z, 1, length(z)), 2) == want_2
    # Test vector methods to make sure generates same thing as matrix methods
    @test complete_polynomial(z, 2) == want_2[:]

end

@testset "testing numerical routintes" begin
    z = rand(100, 10)
    the_ones = ones(size(z, 1))
    the_zeros = zeros(size(z, 1))
    for d in 1:5
        n_comp = n_complete(size(z, 2), d)

        # Test basis matrices
        buff_d = Array(Float64, size(z, 1), n_comp)
        out_d = complete_polynomial(z, d)
        complete_polynomial!(z, d, buff_d)
        @test size(out_d, 1) == size(z, 1)
        @test size(out_d, 2) == n_comp
        @test all(isapprox.(out_d[:, 1], the_ones))
        @test all(isapprox.(out_d[:, 2], z[:, 1]))
        @test d == 1 ? true : all(isapprox.(out_d[:, 3], z[:, 1].*z[:, 1]))
        @test all(isapprox.(out_d[:, end], z[:, end].^d))
        @test all(isapprox.(out_d, buff_d))

        # Test derivatives
        buff_der_d = Array(Float64, size(z, 1), n_comp)
        out_der_d = complete_polynomial(z, d, 1)
        complete_polynomial!(z, d, 1, buff_der_d)
        @test size(out_der_d, 1) == size(z, 1)
        @test size(out_der_d, 2) == n_comp
        @test all(isapprox.(out_der_d[:, 1], the_zeros))
        @test all(isapprox.(out_der_d[:, 2], the_ones))
        @test d ==1 ? true : all(isapprox.(out_der_d[:, 3], 2.*z[:, 1]))
        @test d <= 2 ? true : all(isapprox.(out_der_d[:, 4], 3.*z[:, 1].*z[:, 1]))
        @test all(isapprox.(out_der_d[:, end], the_zeros))
        @test all(isapprox.(out_der_d, buff_der_d))

    end

    z = rand(10, 3)
    z2 = z[1, :]
    all_zero = zeros(10)
    all_ones = ones(10)
    out_2 = complete_polynomial(z, 2)
    # need to test columns size(z, 2) + 2:end
    @test all(isapprox.(out_2[:, 1], all_ones))
    @test all(isapprox.(out_2[:, 2], z[:, 1]))
    @test all(isapprox.(out_2[:, 3], z[:, 1].*z[:, 1]))
    @test all(isapprox.(out_2[:, 4], z[:, 1].*z[:, 2]))
    @test all(isapprox.(out_2[:, 5], z[:, 1].*z[:, 3]))
    @test all(isapprox.(out_2[:, 6], z[:, 2]))
    @test all(isapprox.(out_2[:, 7], z[:, 2].*z[:, 2]))
    @test all(isapprox.(out_2[:, 8], z[:, 2].*z[:, 3]))
    @test all(isapprox.(out_2[:, 9], z[:, 3]))
    @test all(isapprox.(out_2[:, 10], z[:, 3].*z[:, 3]))
    # Test vector methods to make sure generates same thing as matrix methods
    @test all(isapprox.(out_2[1, :], complete_polynomial(z2, 2)))

    out_der_2 = complete_polynomial(z, 2, 2)
    # need to test columns size(z, 2) + 2:end
    @test all(isapprox.(out_der_2[:, 1], all_zero))
    @test all(isapprox.(out_der_2[:, 2], all_zero))
    @test all(isapprox.(out_der_2[:, 3], all_zero))
    @test all(isapprox.(out_der_2[:, 4], z[:, 1]))
    @test all(isapprox.(out_der_2[:, 5], all_zero))
    @test all(isapprox.(out_der_2[:, 6], all_ones))
    @test all(isapprox.(out_der_2[:, 7], 2 .* z[:, 2]))
    @test all(isapprox.(out_der_2[:, 8], z[:, 3]))
    @test all(isapprox.(out_der_2[:, 9], all_zero))
    @test all(isapprox.(out_der_2[:, 10], all_zero))
    # Test vector methods to make sure generates same thing as matrix methods
    @test all(isapprox.(out_2[1, :], complete_polynomial(z2, 2)))

end
