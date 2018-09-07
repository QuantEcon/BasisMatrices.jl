for (T, TP) in [(Cheb, ChebParams), (Lin, LinParams), (Spline, SplineParams)]
    @eval _param(::$(T)) = $TP
    @eval _param(::Type{$(T)}) = $TP
end

# params of same type are equal if all their fields are equal
==(p1::T, p2::T) where {T<:BasisParams} =
    all(map(nm->getfield(p1, nm) == getfield(p2, nm), fieldnames(p1)))::Bool

# Bases of different dimension can't be equal
==(::T1, ::T2) where {T1<:BasisParams,T2<:BasisParams} = false

# ---------- #
# Basis Type #
# ---------- #

mutable struct Basis{N,TP<:Tuple}
    params::TP     # params to construct basis
end

Base.min(b::Basis) = min.(b.params)
Base.max(b::Basis) = max.(b.params)
Base.ndims(::Basis{N}) where {N} = N
Base.ndims(::Type{Basis{N,TP}}) where {N,TP} = N

_get_TP(::Union{Basis{N,TP},Type{Basis{N,TP}}}) where {N,TP} = TP

function Base.show(io::IO, b::Basis{N}) where N
    m = """
    $N dimensional Basis on the hypercube formed by $(min(b)) × $(max(b)).
    Basis families are $(join(string.(family_name.(b.params)), " × "))
    """
    print(io, m)
end

Basis(params::BasisParams...) = _Basis(params)
Basis(params::Tuple) = _Basis(params)

# hack to make method above type stable
@generated function _Basis(params)
    N = sum(ndims, params.parameters)
    quote
        Basis{$N,$params}(params)
    end
end

# combining basis -- fundef-esque method
Basis(bs::Basis...) = _Basis2(bs)

# Another hack to make the above type stable
@generated function _Basis2(bs)
    N = sum(ndims, bs.parameters)

    # tup_of_tups will be a tuple where each element is a tuple of types
    # we want to concatenate them and end up with with a single tuple of types
    # to do that we first put them in a vector, then splat that vector into a
    # Tuple. Note that we don't use `tuple` because that will create a tuple
    # obejct, not the `Tuple` type.
    tup_of_tups = map(_get_TP, bs.parameters)
    basis_types = []
    for x in tup_of_tups
        push!(basis_types, x.parameters...)
    end
    TP = Tuple{basis_types...}
    quote
        new_params = []
        for x in bs
            push!(new_params, x.params...)
        end
        Basis{$N,$TP}(tuple(new_params...))
    end
end

# fundefn type method
Basis(bt::BasisFamily, n::Int, a, b) = Basis(_param(bt)(n, a, b))

Basis(::Type{T}, n::Int, a, b) where {T<:BasisFamily} = Basis(T(), n, a, b)

Basis(bt::T, n::Vector, a::Vector, b::Vector) where {T<:BasisFamily} =
    Basis(map(_param(T), n, a, b)...)

# special method for Spline that adds `k` argument
Basis(::Spline, n::Int, a, b, k) = Basis(SplineParams(n, a, b, k))
Basis(::Spline, n::Vector, a::Vector, b::Vector, k::Vector=ones(Int, length(n))) =
    Basis(map(SplineParams, n, a, b, k)...)::Basis{length(n)}

# ----------------- #
# Basis API methods #
# ----------------- #

# separating Basis -- just re construct it from the nth set of params
function Base.getindex(basis::Basis{N}, n::Int) where N
    n < 0 || n > N && error("n must be between 1 and $N")
    Basis(basis.params[n])::Basis{1}
end

_all_sparse(b::Basis{N,TP}) where {N,TP} = all(issparse, TP.parameters)

# other AbstractArray like methods for Basis
Base.length(b::Basis) = prod(length, b.params)
Base.size(b::Basis, i::Int) = length(b[i])  # uses method on previous line
Base.size(b::Basis{N}) where {N} = map(length, b.params)

# Bases of different dimension can't be equal
==(::Basis{N}, ::Basis{M}) where {N,M} = false

# basis are equal if all fields of the basis are equal
==(b1::Basis{N}, b2::Basis{N}) where {N} =
    all(map(nm->getfield(b1, nm) == getfield(b2, nm), fieldnames(b1)))::Bool

function nodes(b::Basis{1})
    x = nodes(b.params[1])
    (x, (x,))
end

function nodes(b::Basis)  # funnode method
    xcoord = nodes.(b.params)
    x = gridmake(xcoord...)
    return x, xcoord
end

@generated function bmat_type(::Type{TO}, bm::Basis{N,TP}, x=1.0) where {N,TP,TO}
    if N == 1
        out = bmat_type(TO, TP.parameters[1], x)
    else
        out = bmat_type(TO, TP.parameters[1], x)
        for this_TP in TP.parameters[2:end]
            this_out = bmat_type(TO, this_TP, x)
            if this_out != out
                out = AbstractMatrix{promote_type(eltype(out), eltype(this_out))}
            end
        end
    end
    return :($out)
end

bmat_type(b::Basis) = bmat_type(Void, b)
bmat_type(b::Basis, x) = bmat_type(Void, b, x)
