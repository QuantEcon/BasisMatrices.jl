struct SplineSparse{T,I,n_chunks,chunk_len} <: AbstractSparseMatrix{T,I}
    ncol::Int
    vals::Vector{T}
    cols::Vector{I}

    function SplineSparse{T,I,n_chunks,chunk_len}(
            col, vals, cols
        ) where {T,I,n_chunks,chunk_len}
        if length(cols)*chunk_len != length(vals)
            error("vals and cols not conformable")
        end
        new{T,I,n_chunks,chunk_len}(col, vals, cols)
    end
end

function SplineSparse(N::Int, L::Int, ncol::Int, vals::AbstractVector{T},
                      cols::AbstractVector{I}) where {T,I}
    SplineSparse{T,I,N,L}(ncol, vals, cols)
end

n_chunks(::Type{SplineSparse{T,I,N,L}}) where {T,I,N,L} = N
chunk_len(::Type{SplineSparse{T,I,N,L}}) where {T,I,N,L} = L
Base.eltype(::Type{SplineSparse{T,I,N,L}}) where {T,I,N,L} = T
ind_type(::Type{SplineSparse{T,I,N,L}}) where {T,I,N,L} = I
Base.copy(x::SplineSparse{T,I,N,L}) where {T,I,N,L} = SplineSparse{T,I,N,L}(x.ncol, x.vals, x.cols)

for f in [:n_chunks, :chunk_len, :(Base.eltype), :ind_type]
    @eval $(f)(s::SplineSparse) = $(f)(typeof(s))
end

@inline val_ix(s::SplineSparse{T,I,N,L}, row, chunk, n) where {T,I,N,L} =
    N*L*(row-1) + L*(chunk-1) + n

@inline col_ix(s::SplineSparse{T,I,N}, row, chunk) where {T,I,N} =
    N*(row-1) + chunk

function Base.Array(s::SplineSparse{T,I,N,L}) where {T,I,N,L}
    nrow = _nrows(s)
    out = zeros(T, nrow, s.ncol)

    for row in 1:nrow
        for chunk in 1:N
            first_col = s.cols[col_ix(s, row, chunk)]
            @simd for n in 1:L
                out[row, first_col+n-1] = s.vals[val_ix(s, row, chunk, n)]
            end
        end
    end

    out
end

function SparseArrays.findnz(s::SplineSparse{T,I,N,L}) where {T,I,N,L}
    nrow = _nrows(s)
    rows = repeat(collect(I, 1:nrow), inner=[N*L])
    cols = Array{I}(undef, length(s.vals))

    for row in 1:nrow
        for chunk in 1:N
            first_col = s.cols[col_ix(s, row, chunk)]
            for n in 1:L
                cols[val_ix(s, row, chunk, n)] = first_col+n-1
            end
        end
    end

    rows, cols, s.vals
end

function Base.convert(::Type{SparseMatrixCSC}, s::SplineSparse)
    I, J, V = findnz(s)
    sparse(I, J, V, _nrows(s), s.ncol)
end

SparseArrays.sparse(s::SplineSparse) = convert(SparseMatrixCSC, s)

function Base.getindex(s::SplineSparse{T,I,N,L}, row::Integer, cols::Integer) where {T,I,N,L}
    for chunk in 1:N
        first_col = s.cols[col_ix(s, row, chunk)]

        if cols < first_col || cols > (first_col + L-1)
            continue
        end

        n = cols-first_col+1
        return s.vals[val_ix(s, row, chunk, n)]

    end

    zero(T)
end

_nrows(s::SplineSparse) = Int(length(s.vals) / (n_chunks(s)*chunk_len(s)))
Base.size(s::SplineSparse) = (_nrows(s), s.ncol)
Base.size(s::SplineSparse, i::Integer) = i == 1 ? _nrows(s) :
                                         i == 2 ? s.ncol :
                                         1

@generated function row_kron(
        s1::SplineSparse{T1,I1,N1,len1}, s2::SplineSparse{T2,I2,N2,len2}
    ) where {T1,I1,N1,len1,T2,I2,N2,len2}

    # new number of chunks the length chunks times the number of chunks in
    # first matrix times number of chunks in second matrix
    N = len1*N1*N2

    # new chunk length is the chunk length from the second matrix
    len = len2

    # T and I are  easy...
    T = promote_type(T1, T2)
    I = promote_type(I1, I2)

    quote

    nrow = _nrows(s1)
    _nrows(s2) == nrow || error("s1 and s2 must have same number of rows")

    cols = Array{$I}(undef, $N*nrow)
    vals = Array{$T}(undef, $N*$len*nrow)

    ix = 0
    c_ix = 0
    @inbounds for row in 1:nrow
        for chunk1 in 1:N1
            first_col1 = s1.cols[col_ix(s1, row, chunk1)]

            for n1 in 1:len1
                v1 = s1.vals[val_ix(s1, row, chunk1, n1)]

                for chunk2 in 1:N2
                    first_col2 = s2.cols[col_ix(s2, row, chunk2)]

                    cols[c_ix+=1] = (first_col1+n1-2)*s2.ncol+first_col2

                    @simd for n2 in 1:len2
                        vals[ix+=1] = v1*s2.vals[val_ix(s2, row, chunk2, n2)]
                    end
                end
            end
        end
    end

    SplineSparse{$T,$I,$N,$len}(s1.ncol*s2.ncol, vals, cols)
    end
end

function LinearAlgebra.mul!(out::AbstractVector{Tout},
                       s::SplineSparse{T,I,N,L},
                       v::AbstractVector) where {T,I,N,L,Tout}
    @inbounds for row in eachindex(out)
        val = zero(Tout)
        for chunk in 1:N
            first_col = s.cols[col_ix(s, row, chunk)]

            @simd for n in 1:L
                val += s.vals[val_ix(s, row, chunk, n)] * v[first_col+(n-1)]
            end
        end

        out[row] = val
    end
    out
end

function *(s::SplineSparse{T}, v::AbstractVector{T2}) where {T,T2}
    size(s, 2) == size(v, 1) || throw(DimensionMismatch())

    out_T = promote_type(T, T2)
    out = Array{out_T}(undef, size(s, 1))
    mul!(out, s, v)
end

# TODO: define method A_mul_B!(ss::SplineSparse, csc::SparseMatrixCSC)
# TODO: define method A_mul_B!(ss::SplineSparse, ss2::SplineSparse)

function *(s::SplineSparse{T,I,N,L}, m::AbstractMatrix{T2}) where {T,I,N,L,T2}
    size(s, 2) == size(m, 1) || throw(DimensionMismatch())

    out_T = promote_type(T, T2)
    out = zeros(out_T, size(s, 1), size(m, 2))

    @inbounds for row in 1:size(s,1)
        for chunk in 1:N
            first_col = s.cols[col_ix(s, row, chunk)]

            for n in 1:L
                s_val = s.vals[val_ix(s, row, chunk, n)]
                s_col = first_col+(n-1)

                for m_col in 1:size(m, 2)
                    out[row, m_col] += s_val * m[s_col, m_col]
                end

            end
        end
    end

    out

end

# TODO: implement me for real to avoid conversion to CSC
function \(s::SplineSparse, x::Union{AbstractVector,AbstractMatrix})
    sparse(s) \ x
end

function tensor_prod(t::Type{T}, syms, inds, lens, add_index) where T<:AbstractArray
    if length(syms) == 0
        subs = [:($(Symbol("i_", j)) + $(inds[j])) for j in length(inds):-1:1]
        out = Expr(:ref, :c, subs...)
        T <: AbstractMatrix && push!(out.args, :col)
        return out
    else
        exprs = []
        for i in 1:lens[1]
            e = Expr(
                :call,
                :(*),
                Symbol(syms[1], "_", i),
                tensor_prod(
                    t,
                    syms[2:end],
                    cat(inds, [i-1], dims=1),
                    lens[2:end],
                    add_index
                )
            )
            push!(exprs, e)
        end
        return Expr(:call, :(+), exprs...)
    end
end

_ncol(x::AbstractArray) = size(x, 2)

shape_c_expr(::Type{T}) where {T<:AbstractVector} = :(reshape(_c, reverse(_ncol.(rk.B))))
shape_c_expr(::Type{T}) where {T<:AbstractMatrix} = :(reshape(_c, reverse(_ncol.(rk.B))..., size(_c, 2)))

const RKSS = RowKron{<:Tuple{Vararg{<:SplineSparse}}}

# if we have all `SplineSparse`s we can special case out = rk*c
@generated function LinearAlgebra.mul!(out::StridedVecOrMat,
                                  rk::RKSS,
                                  _c::StridedVecOrMat)
    N = length(rk.parameters[1].parameters)
    first_v = Expr(:(=), Symbol("v_", N), :(val_ix(rk.B[$N], row, 1, 1)))
    Ls = Int[chunk_len(i) for i in rk.parameters[1].parameters]

    unpack_Bs_args = []
    for i in 1:N
        L = Ls[i]
        iv_sym = Symbol("iv_", i)
        new_ex = [Expr(
            :(=),
            Symbol("v", i, "_", j),
            :(rk.B[$i].vals[$iv_sym + $(j-1)])
        ) for j in 1:L]
        push!(unpack_Bs_args, new_ex...)
    end
    unpack_Bs = Expr(:block, unpack_Bs_args...)

    syms = [Symbol("v", i) for i in 1:N]
    prod = tensor_prod(_c, syms, [], Ls, false)

    code = quote
        for B in rk.B
            n_chunks(B) == 1 && continue
            msg = "Only supported for combining univariate bases for now..."
            throw(ArgumentError(msg))
        end

        Base.@boundscheck begin
            if size(out, 1) != size(rk, 1)
                msg = "out should have $(size(rk, 1)) columns"
                throw(DimensionMismatch(msg))
            end
        end

        c = $(shape_c_expr(_c))

        @inbounds for col in 1:size(out, 2)
            for row in 1:size(out, 1)
                @nexprs $N j -> i_j = rk.B[j].cols[col_ix(rk.B[j], row, 1)]
                @nexprs $N j -> iv_j = val_ix(rk.B[j], row, 1, 1)
                $(unpack_Bs)
                out[row, col] = $(prod)
            end
        end
    end

    code

end
