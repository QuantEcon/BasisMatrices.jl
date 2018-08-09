# --------------- #
# Hierarchical Basis #
# --------------- #

struct Hierarc <: BasisFamily end

mutable struct HierarcParams{T<:Number} <: BasisParams
    a::T
    b::T
    l::Int # refinement level
    h::Float64  # step size in univariate grid
    nodes::Vector{Float64}
    n::Int  

    function HierarcParams{T}(a::T, b::T, l::Int) where T
        l <= 1 && error("refinement level must be >= 1")
        a >= b && error("left endpoint (a) must be less than right end point (b)")
        h = 1.0 / (2.0^l)
        nodes = Float64[i * h for i in 0:2^l]
        new{T}( a, b, l, h, nodes, length(nodes))
    end
end

HierarcParams(a::T, b::T, l::Int) where {T<:Number}  = HierarcParams{T}( a, b, l)
HierarcParams(a::T, b::T, l::Int) where {T<:Integer} = HierarcParams(Float64(a), Float64(b),l)

## BasisParams interface
# define these methods on the type, the instance version is defined over
# BasisParams
family(::Type{T}) where {T<:HierarcParams} = Hierarc
family_name(::Type{T}) where {T<:HierarcParams} = "Hierarc"
@generated Base.eltype(::Type{T}) where {T<:HierarcParams} = T.parameters[1]

# methods that only make sense for instances
Base.min(cp::HierarcParams) = cp.a
Base.max(cp::HierarcParams) = cp.b
nodes(p::HierarcParams) = p.nodes

function Base.show(io::IO, p::HierarcParams)
    m = string("Hierarchical Basis parameters ",
               "from $(p.a) to $(p.b) \nrefinement level: $(p.l)\nnodes:$(p.n)")
    print(io, m)
end

"""
    box2dom(p::HierarcParams)

map box ``z\in [0,1]`` to domain ``x\in [a,b]``
"""
function box2dom(p::HierarcParams)
    p.nodes .* (p.b-p.a) + p.a
end

"""
    dom2box(x::AbstractArray,p::HierarcParams)

map domain ``x\in [a,b]`` to box ``z\in [0,1]``
"""
function dom2box(x::AbstractArray,p::HierarcParams)
    (x .- p.a) ./ (p.b - p.a)
end

function hat(x::Float64)
    y = 1.0 - abs(x)
    return y < 0 ? 0.0 : y
end
function hat(x::Vector{Float64})
    y = 1.0 .- abs.(x)
    clamp!(y,0.0,1.0)
    return y
end


function evalbase(p::HierarcParams,x::AbstractArray=nodes(p))
    sparse(hcat([BasisMatrices.hat((x .- i*p.h) ./ p.h) for i in 0:2^p.l]...))
end


function test()

    p = BasisMatrices.HierarcParams(-3,2,4)
    BasisMatrices.evalbase(p,[0.0,0.01,0.02])

end



