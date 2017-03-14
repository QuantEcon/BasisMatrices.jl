for (T, TP) in [(Cheb, ChebParams), (Lin, LinParams), (Spline, SplineParams)]
    @eval _param(::$(T)) = $TP
    @eval _param(::Type{$(T)}) = $TP
end

# params of same type are equal if all their fields are equal
=={T<:BasisParams}(p1::T, p2::T) =
    all(map(nm->getfield(p1, nm) == getfield(p2, nm), fieldnames(p1)))::Bool

# Bases of different dimension can't be equal
=={T1<:BasisParams,T2<:BasisParams}(::T1, ::T2) = false

# ---------- #
# Basis Type #
# ---------- #

type Basis{N,TP<:Tuple}
    params::TP     # params to construct basis
end

Base.min(b::Basis) = min.(b.params)
Base.max(b::Basis) = max.(b.params)
Base.ndims{N}(::Basis{N}) = N
Base.ndims{N,TP}(::Type{Basis{N,TP}}) = N

_get_TP{N,TP}(::Basis{N,TP}) = TP
_get_TP{N,TP}(::Type{Basis{N,TP}}) = TP

function Base.show{N}(io::IO, b::Basis{N})
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
    N = length(params.parameters)
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
    TP = Tuple{vcat([[x.parameters...] for x in tup_of_tups]...)...}
    quote
        Basis{$N,$TP}(tuple(vcat([[x.params...] for x in bs]...)...))
    end
end

# fundefn type method
Basis(bt::BasisFamily, n::Int, a, b) = Basis(_param(bt)(n, a, b))

Basis{T<:BasisFamily}(::Type{T}, n::Int, a, b) = Basis(T(), n, a, b)

Basis{T<:BasisFamily}(bt::T, n::Vector, a::Vector, b::Vector) =
    Basis(map(_param(T), n, a, b)...)

# special method for Spline that adds `k` argument
Basis(::Spline, n::Int, a, b, k) = Basis(SplineParams(n, a, b, k))
Basis(::Spline, n::Vector, a::Vector, b::Vector, k::Vector=ones(Int, length(n))) =
    Basis(map(SplineParams, n, a, b, k)...)::Basis{length(n)}

# ----------------- #
# Basis API methods #
# ----------------- #

# separating Basis -- just re construct it from the nth set of params
function Base.getindex{N}(basis::Basis{N}, n::Int)
    n < 0 || n > N && error("n must be between 1 and $N")
    Basis(basis.params[n])::Basis{1}
end

_all_sparse{N,TP}(b::Basis{N,TP}) = all(issparse, TP.parameters)

# other AbstractArray like methods for Basis
Base.length(b::Basis) = prod(length, b.params)
Base.size(b::Basis, i::Int) = length(b[i])  # uses method on previous line
Base.size{N}(b::Basis{N}) = map(length, b.params)

# Bases of different dimension can't be equal
=={N,M}(::Basis{N}, ::Basis{M}) = false

# basis are equal if all fields of the basis are equal
=={N}(b1::Basis{N}, b2::Basis{N}) =
    all(map(nm->getfield(b1, nm) == getfield(b2, nm), fieldnames(b1)))::Bool

function nodes(b::Basis{1})
    x = nodes(b.params[1])
    (x, [x])
end

function nodes(b::Basis)  # funnode method
    xcoord = nodes.(b.params)
    x = gridmake(xcoord...)
    return x, xcoord
end

@generated function bmat_type{N,TP,TO}(::Type{TO}, bm::Basis{N,TP})
    if N == 1
        out = bmat_type(TO, TP.parameters[1])
    else
        out = bmat_type(TO, TP.parameters[1])
        for this_TP in TP.parameters[2:end]
            this_out = bmat_type(TO, this_TP)
            if this_out != out
                out = AbstractMatrix{promote_type(eltype(out), eltype(this_out))}
            end
        end
    end
    return :($out)
end

bmat_type(b::Basis) = bmat_type(Void, bm)
