# Unrelated to the actual grid, just stuff we use

struct Permuter{T<:AbstractVector}
    a::T
    len::Int
    # sort so we can deal with repeated elements
    Permuter{T}(a::T) where {T} = new{T}(sort(a), length(a))
end
Permuter(a::T) where {T<:AbstractVector} = Permuter{T}(a)

Base.start(p::Permuter) = p.len
Base.done(p::Permuter, i::Int) = i == 0
function Base.next(p::Permuter, i::Int)
    while true
        i -= 1

        if p.a[i] < p.a[i+1]
            j = p.len

            while p.a[i] >= p.a[j]
                j -= 1
            end

            p.a[i], p.a[j] = p.a[j], p.a[i]  # swap(p.a[j], p.a[i])
            t = p.a[(i + 1):end]
            reverse!(t)
            p.a[(i + 1):end] = t

            return copy(p.a), p.len
        end

        if i == 1
            reverse!(p.a)
            return copy(p.a), 0
        end
    end
end

function cartprod(arrs, out=Array{eltype(arrs[1])}(
                                prod([length(a) for a in arrs]),
                                length(arrs))
                                )
    sz = Int[length(a) for a in arrs]
    k = 1
    for v in product(arrs...)
        i = 1
        for el in v
            out[k, i] = el
            i += 1
        end
        k += 1
    end

    return out
end

# Building the actual grid

function m_i(i::Int)
    i < 0 && error("DomainError: i must be positive")
    i == 0 ? 0 : i == 1 ? 1 : 2^(i - 1) + 1
end

function cheby2n(x::AbstractArray{T}, n::Int, kind::Int=1) where T<:Number
    out = Array{T}(size(x)..., n+1)
    cheby2n!(out, x, n, kind)
end

"""
Computes first `n` Chebychev polynomials of the `kind` kind
evaluated at each point in `x` and places them in `out`. The trailing dimension
of `out` indexes the chebyshev polynomials. All inner dimensions correspond to
points in `x`.
"""
function cheby2n!(out::AbstractArray{T}, x::AbstractArray{T,N},
                  n::Int, kind::Int=1) where {T<:Number,N}
    if size(out) != tuple(size(x)..., n+1)
        error("out must have dimensions $(tuple(size(x)..., n+1))")
    end

    R = CartesianRange(size(x))
    # fill first element with ones
    @inbounds @simd for I in R
        out[I, 1] = one(T)
        out[I, 2] = kind*x[I]
    end

    @inbounds for i in 3:n+1
        @simd for I in R
            out[I, i] = 2x[I] * out[I, i - 1] - out[I, i - 2]
        end
    end
    out
end

"""
Finds the set `S_n` , which is the `n`th Smolyak set of Chebychev extrema
"""
function s_n(n::Int)
    n < 1 && error("DomainError: n must be positive")

    if n == 1
        return [0.0]
    end

    m = m_i(n)
    j = 1:m
    pts = cos.(pi .* (j .- 1.0) ./ (m .- 1.0))
    @inbounds @simd for i in eachindex(pts)
        pts[i] = abs(pts[i]) < 1e-12 ? 0.0 : pts[i]
    end

    pts
end

doc"""
Finds all of the unidimensional disjoint sets of Chebychev extrema that are
used to construct the grid.  It improves on past algorithms by noting  that
$A_{n} = S_{n}$ [evens] except for $A_1= \{0\}$  and $A_2 = \{-1, 1\}$.
Additionally, $A_{n} = A_{n+1}$ [odds] This prevents the calculation of these
nodes repeatedly. Thus we only need to calculate biggest of the S_n's to build
the sequence of $A_n$ 's

See section 3.2 of the paper...
"""
function a_chain(n::Int)
    sn = s_n(n)
    a = Dict{Int,Vector{Float64}}()
    sizehint!(a, n)

    # These are constant and don't follow the pattern.
    a[1] = [0.0]
    a[2] = [-1.0, 1.0]

    for i in n:-1:3
        a[i] = sn[2:2:end]
        sn = sn[1:2:end]
    end

    a
end



doc"""
For each number in 1 to `n`, compute the Smolyak indices for the corresponding
basis functions. This is the $n$ in $\phi_n$. The output is A dictionary whose
keys are the Smolyak index `n` and values are ranges containing all basis
polynomial subscripts for that Smolyak index
"""
function phi_chain(n::Int)
    max_ind = m_i(n)
    phi = Dict{Int, UnitRange{Int64}}()
    phi[1] = 1:1
    phi[2] = 2:3
    low_ind = 4  # current lower index

    for i = 3:n
        high_ind = m_i(i)
        phi[i] = low_ind:high_ind
        low_ind = high_ind + 1
    end

    phi
end

## ---------------------- ##
#- Construction Utilities -#
## ---------------------- ##

doc"""
    smol_inds(d::Int, mu::Int)

Finds all of the indices that satisfy the requirement that $d \leq \sum_{i=1}^d
\leq d + \mu$.
"""
function smol_inds(d::Int, mu::Int)

    p_vals = 1:(mu+1)

    # PERF: size_hint here if it is slow
    poss_inds = Vector{Int}[]

    for el in with_replacement_combinations(p_vals, d)
        if d < sum(el) <= d + mu
            push!(poss_inds, el)
        end
    end

    # PERF: size_hint here if it is slow
    true_inds = Vector{Int}[ones(Int, d)]  # we will always have (1, 1, ...,  1)
    for val in poss_inds
        for el in Permuter(val)
            push!(true_inds, el)
        end
    end

    return true_inds
end

doc"""
    smol_inds(d::Int, mu::AbstractVector{Int})

Finds all of the indices that satisfy the requirement that $d \leq \sum_{i=1}^d
\leq d + \mu_i$.

This is the anisotropic version of the method that allows mu to vary for each
dimension
"""
function smol_inds(d::Int, mu::AbstractVector{Int})
    # Compute indices needed for anisotropic smolyak grid given number of
    # dimensions d and a vector of mu parameters mu

    length(mu) != d &&  error("ValueError: mu must have d elements.")

    mu_max = maximum(mu)
    mup1 = mu + 1

    p_vals = 1:(mu_max+1)

    poss_inds = Vector{Int}[]

    for el in with_replacement_combinations(p_vals, d)
        if d < sum(el) <= d + mu_max
            push!(poss_inds, el)
        end
    end

    true_inds = Vector{Int}[ones(Int64, d)]
    for val in poss_inds
        for el in Permuter(val)
            if all(el .<= mup1)
                push!(true_inds, el)
            end
        end
    end

    return true_inds
end

"""
Build indices specifying all the Cartesian products of Chebychev polynomials
needed to build Smolyak polynomial
"""
function poly_inds(d::Int, mu::IntSorV, inds::Vector{Vector{Int}}=smol_inds(d, mu))::Matrix{Int}
    phi_n = phi_chain(maximum(mu) + 1)
    vcat([cartprod([phi_n[i] for i in el]) for el in inds]...)
end

"""
Use disjoint Smolyak sets to construct Smolyak grid of degree `d` and density
parameter `mu`
"""
function build_grid(d::Int, mu::IntSorV, inds::Vector{Vector{Int}}=smol_inds(d, mu))::Matrix{Float64}
    An = a_chain(maximum(mu) + 1)  # use maximum in case mu is Vector
    vcat([cartprod([An[i] for i in el]) for el in inds]...)
end


"""
Compute the matrix `B(pts)` from equation 22 in JMMV 2013. This is the basis
matrix
"""
function build_B!(out::AbstractMatrix{T}, d::Int, mu::IntSorV,
                  pts::Matrix{Float64}, b_inds::Matrix{Int64}) where T
    # check dimensions
    npolys = size(b_inds, 1)
    npts = size(pts, 1)
    size(out) == (npts, npolys) || error("Out should be size $((npts, npolys))")

    # fill out with ones so tensor product below works
    fill!(out, one(T))

    # compute all the chebyshev polynomials we'll need
    Ts = cheby2n(pts, m_i(maximum(mu) + 1))

    @inbounds for ind in 1:npolys, k in 1:d
        b = b_inds[ind, k]
        for i in 1:npts
            out[i, ind] *= Ts[i, k, b]
        end
    end

    return out
end

function build_B(d::Int, mu::IntSorV, pts::Matrix{Float64}, b_inds::Matrix{Int64})
    build_B!(Array{Float64}(size(pts, 1), size(b_inds, 1)), d, mu, pts, b_inds)
end

function dom2cube!(out::AbstractMatrix, pts::AbstractMatrix,
                   lb::AbstractVector, ub::AbstractVector)
    d = length(lb)
    n = size(pts, 1)

    size(out) == (n, d) || error("out should be $((n, d))")

    @inbounds for i_d in 1:d
        center = lb[i_d] + (ub[i_d] - lb[i_d])/2
        radius = (ub[i_d] - lb[i_d])/2
        @simd for i_n in 1:n
            out[i_n, i_d] = (pts[i_n, i_d] - center)/radius
        end
    end

    out
end

function cube2dom!(out::AbstractMatrix, pts::AbstractMatrix,
                   lb::AbstractVector, ub::AbstractVector)
    d = length(lb)
    n = size(pts, 1)

    size(out) == (n, d) || error("out should be $((n, d))")

    @inbounds for i_d in 1:d
        center = lb[i_d] + (ub[i_d] - lb[i_d])/2
        radius = (ub[i_d] - lb[i_d])/2
        @simd for i_n in 1:n
            out[i_n, i_d] = center + pts[i_n, i_d]*radius
        end
    end

    out
end

for f in [:dom2cube!, :cube2dom!]
    no_bang = Symbol(string(f)[1:end-1])
    @eval $(no_bang)(pts::AbstractMatrix{T}, lb::AbstractVector, ub::AbstractVector) where {T} =
        $(f)(Array{T}(size(pts, 1), length(lb)), pts, lb, ub)
end
