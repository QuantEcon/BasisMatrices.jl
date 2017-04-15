params = BasisMatrices.ChebParams(7, 1e-4, 10.0)

nod = BasisMatrices.nodes(params)
nod_end = BasisMatrices.nodes(params, 1) # extended nodes to endpoints
nod_l = BasisMatrices.nodes(params, 3) # Lobatto nodes

function manualeval(x, a, b, n)

    z = 2*(x-a)/(b-a)-1
    m = length(x)

    bas = Array{Float64}(m, n)
    bas[:,1:2] = [ones(m, 1) z]

    for i in 3:n

        bas[:, i] .= 2.0.*z .* bas[:, i-1] .- bas[:, i-2]

    end

    return bas

end

mB = manualeval(nod, params.a[1], params.b[1], params.n[1])

#=
function manualderiv1(a,b,n)

    x = BasisMatrices.nodes(BasisMatrices.ChebParams(n-1,a,b))
    z = (2*(x-a)/(b-a)-1)''

    D = [zeros(length(z),1) ones(length(z),1) 4*z 12*z.^2-3 32*z.^3-16*z 80*z.^4-60*z.^2+5 192*z.^5-192*z.^3+36*z]

    return D

end

function manualint1(a,b,n)

    x = BasisMatrices.nodes(BasisMatrices.ChebParams(n+1,a,b))
    z = (2*(x-a)/(b-a)-1)''

    I = [z .5*z.^2 2/3*z.^3-z z.^4-3/2*z.^2 8/5*z.^5-8/3*z.^3+z 16/6*z.^6-5*z.^4+2.5*z.^2 32/7*z.^7-48/5*z.^5+6*z.^3-z]

    return I

end

D = manualderiv1(nod,params.a[1],params.b[1])
I = manualint1(nod,params.a[1],params.b[1])
=#

@testset "test Cheb" begin

    @testset "test nodes" begin
        @test (nod[1] == params.a[1],nod[end] == params.b[1]) == (false, false)
        @test ≈([nod_end[1],nod_end[end]], [params.a[1],params.b[1]], atol=1e-15)
        @test ≈([nod_l[1],nod_l[end]], [params.a[1],params.b[1]], atol=1e-15)
    end

    @testset "test derivative/integral" begin
        deriv, params_d = BasisMatrices.derivative_op(params,1)
        int, params_i = BasisMatrices.derivative_op(params,-1)

        @test params_d.n  ==  params.n[1]-1
        @test params_i.n  ==  params.n[1]+1
        @test ≈([params_d.a,params_d.b], [params.a[1],params.b[1]], atol=1e-15)
        @test ≈([params_i.a,params_i.b], [params.a[1],params.b[1]], atol=1e-15)
    end

    @testset "test evalbase" begin
        B = @inferred BasisMatrices.evalbase(params, nod, 0, 0)
        @test ≈(B, mB, atol=1e-14)
    end

    @test BasisMatrices.family(ChebParams) == Cheb
    @test BasisMatrices.family_name(ChebParams) == "Cheb"
    @test !issparse(ChebParams)

    @test BasisMatrices.family(params) == Cheb
    @test BasisMatrices.family_name(params) == "Cheb"
    @test !issparse(params)

    @test_throws DimensionMismatch BasisMatrices.evalbasex!(rand(2, 7), rand(4), params, rand(4))
    @test_throws DimensionMismatch BasisMatrices.evalbasex!(rand(4, 6), rand(4), params, rand(4))

end  # testset
