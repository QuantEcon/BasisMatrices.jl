#=

Chase Coleman (ccoleman@stern.nyu.edu)
Spencer Lyon (spencer.lyon@stern.nyu.edu)

References
----------

Judd, Kenneth L, Lilia Maliar, Serguei Maliar, and Rafael Valero. 2013.
    "Smolyak Method for Solving Dynamic Economic Models: Lagrange
    Interpolation, Anisotropic Grid and Adaptive Domain".

Krueger, Dirk, and Felix Kubler. 2004. "Computing Equilibrium in OLG
    Models with Stochastic Production." Journal of Economic Dynamics and
    Control 28 (7) (April): 1411-1436.

=#

# include helper file
include("smol_util.jl")

struct Smolyak <: BasisFamily end

struct SmolyakParams{T,Tmu<:IntSorV,d} <: BasisParams
    mu::Tmu
    a::Vector{T}
    b::Vector{T}

    # internal fields
    inds::Vector{Vector{Int}}  # Smolyak indices
    pinds::Matrix{Int64}  # Polynomial indices

    function SmolyakParams{T,Tmu,d}(
            mu::Tmu, a::Vector{T}, b::Vector{T}
        ) where {T,Tmu,d}
        d < 2 && error("You passed d = $d. d must be greater than 1")
        if length(mu) > 1
            # working on building an anisotropic grid
            if length(mu) != d
                error("mu must have d elements. It has $(length(mu))")
            end

            if any(mu .< 1)
                error("All elements of mu must be greater than 1 (recieved $mu)")
            end
        else
            mu < 1 && error("You passed mu = $mu. mu must be greater than 1")
        end

        inds = smol_inds(d, mu)
        pinds = poly_inds(d, mu, inds)

        new{T,Tmu,d}(mu, a, b, inds, pinds)
    end
end

function SmolyakParams(
        ::Union{Val{d},Type{Val{d}}}, mu::Tmu, a::Vector{T}=fill(-1.0, d),
        b::Vector{T}=fill(1.0, d)
    ) where {T,Tmu,d}
    SmolyakParams{T,Tmu,d}(mu, a, b)
end

# add methods to helper routines from other files
smol_inds(sp::SmolyakParams) = sp.inds

function poly_inds(
        sp::SmolyakParams{T,Tmu,d}, inds::Vector{Vector{Int}}=sp.inds
    ) where {T,Tmu,d}
    poly_inds(d, sp.mu, inds)
end

function build_grid(
        sp::SmolyakParams{T,Tmu,d}, inds::Vector{Vector{Int}}=sp.inds
    ) where {T,Tmu,d}
    build_grid(d, sp.mu, inds)
end

function build_B!(
        out::AbstractMatrix{T}, sp::SmolyakParams{Tp,Tmu,d},
        pts::Matrix{Float64}, b_inds::Matrix{Int64}=sp.pinds
    ) where {T,Tp,Tmu,d}
    build_B!(out, d, sp.mu, pts, b_inds)
end

function build_B(sp::SmolyakParams, pts::Matrix{Float64},
                 b_inds::Matrix{Int64}=sp.pinds)
    build_B!(Array{Float64}(size(pts, 1), size(b_inds, 1)), sp, pts, b_inds)
end

function dom2cube!(out::AbstractMatrix, pts::AbstractMatrix,
                   sp::SmolyakParams)
    dom2cube!(out, pts, sp.a, sp.b)
end
dom2cube(pts::AbstractMatrix, sp::SmolyakParams) = dom2cube(pts, sp.a, sp.b)

function cube2dom!(out::AbstractMatrix, pts::AbstractMatrix,
                   sp::SmolyakParams)
    cube2dom!(out, pts, sp.a, sp.b)
end
cube2dom(pts::AbstractMatrix, sp::SmolyakParams) = cube2dom(pts, sp.a, sp.b)

## BasisParams interface
# define these methods on the type, the instance version is defined over
# BasisParams
family(::Type{T}) where {T<:SmolyakParams} = Smolyak
family_name(::Type{T}) where {T<:SmolyakParams} = "Smolyak"
Base.eltype(::Type{SmolyakParams{T,Tmu,d}}) where {T,Tmu,d} = T

# methods that only make sense for instances
Base.min(p::SmolyakParams) = p.a
Base.max(p::SmolyakParams) = p.b
Base.ndims(::Union{SmolyakParams{T,Tmu,d},Type{SmolyakParams{T,Tmu,d}}}) where {T,Tmu,d} = d

function Base.show(io::IO, p::SmolyakParams{T,Tmu,d}) where {T,Tmu,d}
    m = string("Smolyak interpolation parameters in $(d) dimensions",
               " from $(p.a) × $(p.b)")
    print(io, m)
end

# TODO: fix this
function Base.length(sp::SmolyakParams{T,Tmu,d})::Int where {T,Tmu,d}
    mu = sp.mu
    mu == 1 ? 2d - 1 :
    mu == 2 ? Int(1 + 4d + 4d*(d-1)/2 ) :
    mu == 3 ? Int(1 + 8d + 12d*(d-1)/2 + 8d*(d-1)*(d-2)/6) :
    error("We only know the number of grid points for mu ∈ [1, 2, 3]")
end

function nodes(sp::SmolyakParams)
    dom_grid = build_grid(sp)
    cube2dom!(dom_grid, dom_grid, sp)
end

function evalbase(sp::SmolyakParams, x::AbstractVector, order=0)
    evalbase(sp, reshape(x, 1, :), order)
end

function evalbase(sp::SmolyakParams, x::AbstractArray, order=0)
    for _o in order
        if _o != 0
        m = string(
            "Derivatives for Smolyak are not implemented yet...",
            "\nPlease open an issue on GitHub if you would like use this feature"
        )
        error(m)
        end
    end
    cube_pts = dom2cube(x, sp)
    build_B(sp, cube_pts, sp.pinds)
end
