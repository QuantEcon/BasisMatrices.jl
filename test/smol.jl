@testset "Smolyak" begin

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
            min(p) == vec(minimum(my_nodes, dims=1))
            max(p) == vec(maximum(my_nodes, dims=1))
        end
    end

    @testset "Permuter" begin
        # make sure we get the right number of elements
        for i in 2:10
            p = BasisMatrices.Permuter(collect(1:i))
            len = 0
            for x in p; len += 1; end
            @test len == factorial(i)
        end

        # test elements with longer set
        p = BasisMatrices.Permuter([1, 2 ,3])
        want = [
            [1, 2, 3],
            [1, 3, 2],
            [2, 1, 3],
            [2, 3, 1],
            [3, 1, 2],
            [3, 2, 1]
        ]
        have = Vector{Int}[]
        for x in p; push!(have, x); end
        for x in want
            @test x in have
        end
    end

    @testset "cartprod" begin
        # cartprod returns the same thing as gridmake, but is optimized for the
        # use case we have in the smolyak code. When I replaced it with gridmake
        # in phi_inds I got a 4x slowdown
        # So, we keep cartprod alive, but we can easily test using gridmake

        for n in 2:6
            arrs = [rand(Int, 5) for x in 1:n]
            gm = BasisMatrices.gridmake(arrs...)
            cp = BasisMatrices.cartprod(arrs)
            @test gm == cp
        end
    end

    @testset "m_i" begin
        for i in -10:1:-1
            @test_throws ErrorException BasisMatrices.m_i(i)
        end
        @test 0 == @inferred BasisMatrices.m_i(0)
        @test 1 == @inferred BasisMatrices.m_i(1)
        for i in 2:10
            @test 2^(i-1)+1 == @inferred BasisMatrices.m_i(i)
        end
    end

    @testset "cheby2n" begin
        #=
        cheby2n(x, n) is equivalent to evalbase(ChebParams(n+1, -1, 1), x), but
        is special cased to assume the domain is -1, 1 and is faster in the
        smolyak code. See the example below

        julia> function with_cheb_params(x, n)
                   p = BasisMatrices.ChebParams(n+1, -1, 1)
                   BasisMatrices.evalbase(p, x)
               end
        option1 (generic function with 1 method)

        julia> x = [-0.7, 0.1, 0.7];

        julia> with_cheb_params(x, 3)
        3×4 Array{Float64,2}:
         1.0  -0.7  -0.02   0.728
         1.0   0.1  -0.98  -0.296
         1.0   0.7  -0.02  -0.728

         julia> BasisMatrices.cheby2n(x, 3)
         3×4 Array{Float64,2}:
          1.0  -0.7  -0.02   0.728
          1.0   0.1  -0.98  -0.296
          1.0   0.7  -0.02  -0.728

        julia> @benchmark with_cheb_params($x, 3)
        BenchmarkTools.Trial:
          memory estimate:  496 bytes
          allocs estimate:  7
          --------------
          minimum time:     196.485 ns (0.00% GC)
          median time:      212.165 ns (0.00% GC)
          mean time:        290.248 ns (24.16% GC)
          maximum time:     8.343 μs (93.95% GC)
          --------------
          samples:          10000
          evals/sample:     606

        julia> @benchmark BasisMatrices.cheby2n($x, 3)
        BenchmarkTools.Trial:
          memory estimate:  176 bytes
          allocs estimate:  1
          --------------
          minimum time:     50.185 ns (0.00% GC)
          median time:      52.861 ns (0.00% GC)
          mean time:        60.466 ns (6.60% GC)
          maximum time:     821.639 ns (87.49% GC)
          --------------
          samples:          10000
          evals/sample:     979


        So, we keep this version, but can test against the more general version
        in ChebParams
        =#

        x = 2 .* (rand(10) .- 0.5)
        for n in 2:20
            p = BasisMatrices.ChebParams(n+1, -1.0, 1.0)
            want = BasisMatrices.evalbase(p, x)
            have = BasisMatrices.cheby2n(x, n)
            @test have == want

            # test mutating version
            fill!(have, 0.0)
            BasisMatrices.cheby2n!(have, x, n, 1)
            @test have == want

            # now we test the more genral version of cheby2n that evaluates higher
            # dimensional arrays
            have2 = BasisMatrices.cheby2n(hcat(x, x), n)
            @test size(have2) == (length(x), 2, n+1)
            for i in 1:2
                @test have2[:, i, :] == want
            end
        end
    end

    @testset "s_n" begin
        # example from the paper
        @test [0.0] ≈ @inferred BasisMatrices.s_n(1)
        @test [1, 0.0, -1] ≈ @inferred BasisMatrices.s_n(2)
        @test [1, 1/sqrt(2), 0.0, -1/sqrt(2), -1] ≈ @inferred BasisMatrices.s_n(3)
        want_4 = [
            1,
            sqrt(2+sqrt(2))/2,  1/sqrt(2), sqrt(2-sqrt(2))/2,
            0.0,
            -sqrt(2-sqrt(2))/2, -1/sqrt(2), -sqrt(2+sqrt(2))/2,
            -1.0]
        @test want_4 ≈ @inferred BasisMatrices.s_n(4)
    end

    @testset "a_chain" begin
        want = Dict(1=>[0.0],2=>[-1.0, 1.0])
        @test want == @inferred BasisMatrices.a_chain(1)
        @test want == @inferred BasisMatrices.a_chain(2)

        for n in 3:6
            # test that we get the n elements that are all subsets of s_n
            a = BasisMatrices.a_chain(n)
            sn = BasisMatrices.s_n(n)
            @test length(a) == n
            len = 0
            for (i, el) in a
                len += length(el)
                @test el ⊆ sn
            end
            @test len == length(sn)

            # test that sets are all disjoint
            for i1 in 1:n
                vec1 = a[i1]
                for i2 in 1:n
                    i1 == i2 && continue

                    # make sure no elements from a_i2 are in a_i1
                    for el2 in a[i2]
                        @test !(el2 ∈ vec1)
                    end
                end
            end
        end
    end

    @testset "phi_chain" begin
        want = Dict(1=>1:1, 2=>2:3)
        @test want == @inferred BasisMatrices.phi_chain(1)
        @test want == @inferred BasisMatrices.phi_chain(2)

        for n in 3:10
            phi = BasisMatrices.phi_chain(n)
            # test that we get the n elements that are all subsets of s_n
            sn = BasisMatrices.s_n(n)
            @test length(phi) == n
            @test maximum(phi[n]) == length(sn)
        end

        phi = BasisMatrices.phi_chain(10)
        @test length(phi[1]) == 1
        @test length(phi[2]) == 2
        for i in 3:10
            @test length(phi[i]) == 2^(i-2)
        end

        have = vcat([phi[i] for i in 1:10]...)
        want = collect(1:last(phi[10]))
        @test have == want
    end

    @testset "smol_inds" begin
        # do a couple test cases by hand...
        inds = BasisMatrices.smol_inds(2, 2)
        want = [
            [1, 1], [2, 1], [1, 2], [3, 1], [1, 3], [2, 2]
        ]
        for i in want
            @test i in inds
        end
        @test length(inds) == length(want)

        inds = BasisMatrices.smol_inds(3, 2)
        want = [
            [1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2],
            [1, 2, 2], [2, 1, 2], [2, 2, 1],
            [1, 1, 3], [1, 3, 1], [3, 1, 1]
        ]
        for i in want
            @test i in inds
        end
        @test length(inds) == length(want)

        inds = BasisMatrices.smol_inds(2, 3)
        want = [
            [1, 1], [2, 1], [1, 2],
            [3, 1], [1, 3],
            [4, 1], [1, 4],
            [2, 3], [3, 2],
            [2, 2]
        ]
        for i in want
            @test i in inds
        end
        @test length(inds) == length(want)

        # then make sure other cases satisfy constraint.
        for d in 2:10
            for mu in 1:3
                inds = BasisMatrices.smol_inds(d, mu)
                for el in inds
                    sum_el = sum(el)
                    @test sum_el >= d
                    @test sum_el <= d + mu
                end
            end
        end

        @testset "anisotropic case" begin
            @test_throws ErrorException BasisMatrices.smol_inds(2, [1,])
            @test_throws ErrorException BasisMatrices.smol_inds(2, [2,])
            @test_throws ErrorException BasisMatrices.smol_inds(2, [2, 3, 2])

            # do a couple test cases by hand...
            inds = BasisMatrices.smol_inds(2, [2, 2])
            want = [
                [1, 1], [2, 1], [1, 2], [3, 1], [1, 3], [2, 2]
            ]
            for i in want
                @test i in inds
            end
            @test length(inds) == length(want)

            inds = BasisMatrices.smol_inds(2, [2, 3])
            want = [
                [1, 1], [2, 1], [1, 2], [3, 1], [1, 3], [2, 2],
                [1, 4], [2, 3], [3, 2]
            ]
            for i in want
                @test i in inds
            end
            @test length(inds) == length(want)

            inds = BasisMatrices.smol_inds(3, [2, 2, 2])
            want = [
                [1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2],
                [1, 2, 2], [2, 1, 2], [2, 2, 1],
                [1, 1, 3], [1, 3, 1], [3, 1, 1]
            ]
            for i in want
                @test i in inds
            end
            @test length(inds) == length(want)

            inds = BasisMatrices.smol_inds(3, [2, 3, 2])
            want = [
                [1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2],
                [1, 2, 2], [2, 1, 2], [2, 2, 1],
                [1, 1, 3], [1, 3, 1], [3, 1, 1],
                [1, 2, 3], [1, 3, 2],
                [2, 1, 3], [2, 3, 1],
                [3, 1, 2], [3, 2, 1],
                [2, 2, 2], [1, 4, 1]
            ]
            for i in want
                @test i in inds
            end
            @test length(inds) == length(want)

            inds = BasisMatrices.smol_inds(2, [3, 3])
            want = [
                [1, 1], [2, 1], [1, 2],
                [3, 1], [1, 3],
                [4, 1], [1, 4],
                [2, 3], [3, 2],
                [2, 2]
            ]
            for i in want
                @test i in inds
            end
            @test length(inds) == length(want)

            for d in 2:10
                mu = fill(2, d)
                for mu2 in 1:3
                    mu[d] = mu2
                    inds = BasisMatrices.smol_inds(d, mu)
                    for el in inds
                        sum_el = sum(el)
                        @test sum_el >= d
                        @test sum_el <= d + maximum(mu)
                        @test all(el .<= mu .+ 1)
                    end
                end
            end

        end
    end

    @testset "poly_inds" begin
        # TODO: come up with better tests
        pinds  = @inferred BasisMatrices.poly_inds(3, 2)
        @test minimum(pinds) == 1
        @test maximum(pinds) == 5
        for d in 2:10
            grid = BasisMatrices.build_grid(d, 2)
            pinds = @inferred BasisMatrices.poly_inds(d, 2)
            @test size(pinds, 2) == d
            @test size(pinds, 1) == size(grid, 1)
        end
    end

    @testset "build_grid" begin
        # TODO: come up with better tests
        grid = @inferred BasisMatrices.build_grid(3, 2)
        @test minimum(grid) ≈ -1.0
        @test maximum(grid) ≈ 1.0
        for d in 2:10
            grid = @inferred BasisMatrices.build_grid(d, 2)
            @test size(grid, 2) == d
        end
    end

    @testset "build_B" begin
        # TODO: come up with better tests
        x = range(-1, stop=1, length=5)
        for d in 2:5
            for mu in 1:3
                X = BasisMatrices.cartprod([x for foobar in 1:d])
                b_inds = BasisMatrices.poly_inds(d, mu)
                B = @inferred BasisMatrices.build_B(d, mu, X, b_inds)
                @test size(B) == (size(X, 1), size(b_inds, 1))

                # test mutating version
                out = similar(B)
                BasisMatrices.build_B!(out, d, mu, X, b_inds)
                @test out ≈ B
            end
        end
    end

    @testset "dom2cube and cube2dom" begin
        pts = reshape(range(-1, stop=1, length=10), 10, 1)
        lb = [-3.0]
        ub = [4.0]
        dom = @inferred BasisMatrices.cube2dom(pts, lb, ub)
        @test minimum(dom) ≈ lb[1]
        @test maximum(dom) ≈ ub[1]
        cub = @inferred BasisMatrices.dom2cube(dom, lb, ub)
        @test minimum(cub) ≈ -1.0
        @test maximum(cub) ≈ 1.0

        # test round tripping
        @test maximum(abs, cub - pts) < 1e-15

        # test mutating version
        out = similar(pts)
        BasisMatrices.cube2dom!(out, pts, lb, ub)
        @test out ≈ dom

        BasisMatrices.dom2cube!(out, dom, lb, ub)
        @test out ≈ cub
    end

    # make the show code run
    io = IOBuffer()
    show(io, p1)

    @testset "SmolyakParams wrapper for smol_util.jl methods" begin
        for p in (p1, p2, p3, p4, p5)
            # nodes
            X = @inferred BasisMatrices.nodes(p)
            @test minimum(X, dims=1) ≈ p.a'
            @test maximum(X, dims=1) ≈ p.b'

            # dom2cube
            cube = @inferred BasisMatrices.dom2cube(X, p)
            @test cube ≈ BasisMatrices.build_grid(p.d, p.mu)
            out = similar(cube)
            BasisMatrices.dom2cube!(out, X, p)
            @test cube ≈ out

            # cube2dom
            dom = @inferred BasisMatrices.cube2dom(cube, p)
            @test dom ≈ X
            BasisMatrices.cube2dom!(out, cube, p)
            @test X ≈ out

            # smol_inds, poly_inds
            @test BasisMatrices.smol_inds(p.d, p.mu) == @inferred BasisMatrices.smol_inds(p)
            @test BasisMatrices.poly_inds(p.d, p.mu) == @inferred BasisMatrices.poly_inds(p)

            # build_B
            want = BasisMatrices.build_B(p.d, p.mu, cube, p.pinds)
            @test want ≈ @inferred BasisMatrices.build_B(p, cube)
            out = similar(want)
            BasisMatrices.build_B!(out, p, cube)
            @test want ≈ out

            # @test BasisMatrices.build_grid(p.d, p.mu) ≈ @inferred BasisMatrices.nodes(p)
        end
    end

    @testset "evalbase" begin
        for p in (p1, p2, p3, p4, p5)
            X = BasisMatrices.nodes(p)
            cube = BasisMatrices.dom2cube(X, p)
            want = BasisMatrices.build_B(p.d, p.mu, cube, p.pinds)
            @test want ≈ @inferred BasisMatrices.evalbase(p, X)

            cube2 = 2 .* (rand(50, p.d) .- 0.5)
            want2 = BasisMatrices.build_B(p.d, p.mu, cube2, p.pinds)
            X2 = BasisMatrices.cube2dom(cube2, p)
            @test want2 ≈ @inferred BasisMatrices.evalbase(p, X2)

        end
    end
end
