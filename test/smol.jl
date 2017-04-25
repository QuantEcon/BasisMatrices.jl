@testset "Smolyak"

    # build some SmolyakParams objects to test with
    p1 = SmolyakParams(3, 3)
    p2 = SmolyakParams(2, [3, 2])
    p3 = SmolyakParams(2, 3, [-3.0, -4.0], [1.5, 2.5])
    p4 = SmolyakParams(2, [3, 4], [-3.0, -4.0], [1.5, 2.5])
    p5 = SmolyakParams(2, 3, Float32[-3.0, -4.0], Float32[1.5, 2.5])

    # check error paths for constructors
    @test_throws ErrorException SmolyakParams(1, 3)
    @test_throws ErrorException SmolyakParams(2, [3, 2, 3])
    @test_throws ErrorException SmolyakParams(2, [3, 0])
    @test_throws ErrorException SmolyakParams(2, 0)

    @test Float64 == @inferred eltype(p1)
    @test Float64 == @inferred eltype(p2)
    @test Float64 == @inferred eltype(p3)
    @test Float64 == @inferred eltype(p4)
    @test Float32 == @inferred eltype(p5)

    @test 3 == @inferred ndims(p1)
    @test 2 == @inferred ndims(p2)
    @test 2 == @inferred ndims(p3)
    @test 2 == @inferred ndims(p4)
    @test 2 == @inferred ndims(p5)

    @test [-1.0, -1.0, -1.0] == @inferred min(p1)
    @test [-1.0, -1.0] == @inferred min(p2)
    @test [-3.0, -4.0] == @inferred min(p3)
    @test [-3.0, -4.0] == @inferred min(p4)
    @test Float32[-3.0, -4.0] == @inferred min(p5)

    @test [1.0, 1.0, 1.0] == @inferred max(p1)
    @test [1.0, 1.0] == @inferred max(p2)
    @test [1.5, 2.5] == @inferred max(p3)
    @test [1.5, 2.5] == @inferred max(p4)
    @test Float32[1.5, 2.5] == @inferred max(p5)

    for p in [p1, p2, p3, p4, p5]
        @test BasisMatrices.Smolyak == @inferred BasisMatrices.family(p)
        @test "Smolyak" == @inferred BasisMatrices.family_name(p)

        @test begin
            my_nodes = @inferred(nodes(p))
            min(p) == vec(minimum(my_nodes, 1))
            max(p) == vec(maximum(my_nodes, 1))
        end
    end

    # make the show code run
    io = IOBuffer()
    show(io, p1)

    #= TODO

    ## these are in src/smolyak.jl

    - smol_inds
    - poly_inds
    - build_grid
    - build_B!
    - build_B
    - dom2cube!
    - dom2cube
    - cube2dom!
    - cube2dom

    ## These are in src/smol_util.jl

    - Permuter
    - start(::Permuter)
    - done(::Permuter)
    - next(::Permuter)
    - cartprod
    - m_i
    - cheby2n
    - cheby2n!
    - s_n
    - a_chain
    - phi_chain
    - smol_inds
    - poly_inds
    - build_grid
    - build_B!
    - build_B
    - dom2cube!
    - dom2cube
    - cube2dom!
    - cube2dom

    =#



end
