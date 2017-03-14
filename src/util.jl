# ---------------- #
# Helper functions #
# ---------------- #
import Base: *

# ckronx.m -- DONE
function ckronx{TM<:AbstractMatrix}(b::AbstractMatrix{TM}, c::Array,
                                    ind::AbstractArray{Int}=1:length(b))
    d = length(ind)  # 26
    n = Array{Int}(d)  # 27
    for i=1:d  # 28
        n[i] = size(b[ind[i]], 2)
    end

    if prod(n) != size(c, 1)  # 29-31
        m = "b and c are not conformable (b suggests size(c, 1) should be $(prod(n)))"
        error(m)
    end

    z = c'  # 32
    mm = 1  # 33
    for i=1:d
        m = Int(length(z) / n[i])  # 35
        z = reshape(z, m, n[i])  # 36
        z = b[ind[i]] * z'  # 37
        mm = mm * size(z, 1)  # 38
    end
    reshape(z, mm, size(c, 2))  # 40
end

# dprod.m  - DONE
function row_kron{S,T}(A::AbstractMatrix{S}, B::AbstractMatrix{T})
    out = _allocate_row_kron_out(A, B)
    row_kron!(out, A, B)
end

function row_kron!(out::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    # get input dimensions
    nobsa, na = size(A)
    nobsb, nb = size(B)

    nobsa != nobsb && error("A and B must have same number of rows")

    # fill in each element. To do this we make sure we access each array
    # consistent with its column major memory layout.
    @inbounds for ia=1:na, ib=1:nb, t=1:nobsa
        out[t, nb*(ia-1) + ib] = A[t, ia] * B[t, ib]
    end
    out
end

function row_kron!(out::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC)
    colptr = out.colptr
    rowval = out.rowval
    nzval = out.nzval

    na = size(A, 2)
    nb = size(B, 2)
    av = A.nzval
    bv = B.nzval

    colptr[1] = 1
    ck = 1  # running total for colptr
    rv_ix = 0  # which entry of rowval and nzval should be filled
    col_ix = 1 # which entry of colptr should be filled

    @inbounds for i in 1:na  # columns of a
        start_a = A.colptr[i]
        len_a = A.colptr[i+1] - start_a
        b_end = 1
        for j in 2:nb+1  # columns of b
            ix_b = b_end
            b_end = B.colptr[j]

            ix_a = 0

            while ix_a < len_a && ix_b < b_end
                if A.rowval[start_a+ix_a] == B.rowval[ix_b]
                    # found one!
                    nzval[rv_ix+=1] = av[start_a + ix_a] * bv[ix_b]
                    rowval[rv_ix] = A.rowval[start_a+ix_a]
                    ix_a += 1
                    ck += 1
                elseif A.rowval[start_a+ix_a] < B.rowval[ix_b]
                    ix_a += 1
                else
                    ix_b += 1
                end
            end

            colptr[col_ix+=1] = ck
        end

    end

    out
end

function row_kron!{T}(out::SparseMatrixCSC, A::AbstractMatrix{T}, B::SparseMatrixCSC)
    colptr = out.colptr
    rowval = out.rowval
    nzval = out.nzval
    my_zero = zero(T)

    na = size(A, 2)
    nb = size(B, 2)
    br = B.rowval
    bv = B.nzval

    colptr[1] = 1
    ck = 1  # running total for colptr
    rv_ix = 0  # which entry of rowval and nzval should be filled
    c_ix = 1 # which entry of colptr should be filled

    @inbounds for i in 1:na  # columns of a
        ix_b = 1
        for j in 2:nb+1  # columns of b
            b_end = B.colptr[j]

            while ix_b < b_end  # rows
                row = br[ix_b]
                if A[row, i] != my_zero
                    nzval[rv_ix+=1] = A[row, i] * bv[ix_b]
                    rowval[rv_ix] = row
                    ck += 1
                end
                ix_b += 1
            end
            colptr[c_ix+=1] = ck
        end
    end
    out
end


function row_kron!{T}(out::SparseMatrixCSC, A::SparseMatrixCSC, B::AbstractMatrix{T})
    colptr = out.colptr
    rowval = out.rowval
    nzval = out.nzval
    my_zero = zero(T)

    na = size(A, 2)
    nb = size(B, 2)
    ar = A.rowval
    av = A.nzval

    colptr[1] = 1
    ck = 1  # running total for colptr
    rv_ix = 0  # which entry of rowval and nzval should be filled
    c_ix = 1 # which entry of colptr should be filled

    @inbounds for i in 1:na  # columns of a
        for j in 1:nb  # columns of b
            for ptr in A.colptr[i]:(A.colptr[i+1]-1)  # rows
                row = ar[ptr]
                if B[row, j] != my_zero
                    nzval[rv_ix+=1] = av[ptr] * B[row, j]
                    rowval[rv_ix] = row
                    ck += 1
                end
            end
            colptr[c_ix+=1] = ck
        end
    end
    out
end

# ckronxi.m -- DONE
ckronxi{T<:Number}(b::Matrix{T}, c, ind=1:length(b)) = b \ c  # 23

function ckronxi(b::Array, c, ind=1:length(b))
    d = length(ind)  # 25
    n = Int[size(b[ind[i]], 2) for i=1:d]  #26-27
    prod(n) != size(c, 1) && error("b and c are not conformable")  # 28-30

    z = c'  # 31
    mm = 1  # 32
    for i=1:d  # 33
        m = round(Int, prod(size(z)) / n[i])  # 34
        z = reshape(z, m, n[i])  # 35
        z = b[ind[i]] \ z'  # 36
        mm *= size(z, 1)  # 37
    end  # 38
    reshape(z, mm, size(c, 2))  # 39
end

immutable RowKron{T<:Tuple}
    B::T
end

function RowKron(B::AbstractMatrix...)
    nrow = map(x -> size(x, 1), B)
    if any(x -> x != nrow[1], nrow)
        msg = "All matrices must have the same number of rows"
        throw(DimensionMismatch(msg))
    end

    if any(x -> !(isa(x, AbstractMatrix)), B)
        msg = "All arguments to RowKron must be matrices"
        throw(ArgumentError(msg))
    end

    RowKron(B)
end

Base.length(rk::RowKron) = length(rk.B)
@generated Base.eltype{T}(::RowKron{T}) =
    reduce(promote_type, map(eltype, T.parameters))

Base.issparse(rk::RowKron) = map(issparse, rk.B)

function Base.size(rk::RowKron, i::Int)
    if i == 1
        size(rk.B[1], 1)
    elseif i == 2
        prod(map(_ -> size(_, 2), rk.B))
    else
        1
    end
end

Base.size(rk::RowKron) = (size(rk, 1), size(rk, 2))

sizes(rk::RowKron, i::Integer) = collect(map(_ -> size(_, i), rk.B))

for (f, op, transp) in ((:A_mul_B!, :identity, false),
                        (:Ac_mul_B!, :ctranspose, true),
                        (:At_mul_B!, :transpose, true))

    # code to do type checks for input args
    checks = transp ? quote
        size(out, 1) == size(rk, 2) || throw(DimensionMismatch())
        size(out, 2) == size(c, 2) || throw(DimensionMismatch())
        size(c, 1) == size(rk, 1) || throw(DimensionMismatch())
        fill!(out, 0.0)
    end : quote
        size(out, 1) == size(rk, 1) || throw(DimensionMismatch())
        size(out, 2) == size(c, 2) || throw(DimensionMismatch())
        size(c, 1) == size(rk, 2) || throw(DimensionMismatch())
        fill!(out, 0.0)
    end

    # the outer loop has a different range depending on transpose
    outer_loop_range = transp ? :(1:size(out, 1)) : :(1:size(c, 1))

    # How we fill `out` is also different...
    fill_out = transp ? quote
        if _last_sparse
            for ptr in (Bend.colptr[ccB]):(Bend.colptr[ccB+1]-1)
                _r = Bend.rowval[ptr]
                _v = $(op)(Bend.nzval[ptr])
                out[i] += d[_r, end] * _v * c[_r]
            end
        else
            for _r in 1:nrow
                full_B = d[_r, end] * Bend[_r, ccB]
                out[i] += full_B * c[_r]
            end
        end
    end : quote
        if _last_sparse
            for ptr in (Bend.colptr[ccB]):(Bend.colptr[ccB+1]-1)
                _r = Bend.rowval[ptr]
                _v = $(op)(Bend.nzval[ptr])
                full_B = d[_r, end] * _v

                for _c in 1:ccol
                    out[_r, _c] += full_B * c[i, _c]
                end
            end
        else
            for _r in 1:nrow
                full_B = d[_r, end] * Bend[_r, ccB]
                for _c in 1:ccol
                    out[_r, _c] += full_B * c[i, _c]
                end
            end
        end
    end

    @eval begin
    function Base.$(f)(out::StridedVecOrMat, rk::RowKron, c::StridedVecOrMat)
        $(checks)

        _is_sparse = map(x -> isa(x, SparseMatrixCSC), rk.B)
        _last_sparse = _is_sparse[end]

        nrow = size(rk, 1)
        Nb = length(rk)
        d = Array{eltype(rk)}(nrow, Nb)
        d[:, 1] = one(eltype(rk))

        n_col_B = sizes(rk, 2)
        cur_col_B = copy(n_col_B) + 1

        # simplify notation
        ccol = size(c, 2)
        Bend = rk.B[end]

        for i in $(outer_loop_range)  # loop over rows of out
            if cur_col_B[end] > n_col_B[end]
                cur_col_B[end] = 1
                j = Nb
                while j > 1
                    j -= 1
                    cur_col_B[j] += 1
                    cur_col_B[j] <= n_col_B[j] && break
                    cur_col_B[j] = 1
                end

                for col_d in j+1:Nb
                    _j = col_d - 1
                    _col_j = cur_col_B[_j]
                    this_B = rk.B[_j]

                    if _is_sparse[_j]
                        d[:, col_d] = 0.0

                        # fill in non-zero rows
                        for ptr in this_B.colptr[_col_j]:(this_B.colptr[_col_j+1]-1)
                            _r = this_B.rowval[ptr]
                            _v = $(op)(this_B.nzval[ptr])
                            d[_r, col_d] = d[_r, col_d-1] * _v
                        end

                    else
                        @simd for row_d in 1:nrow
                            d[row_d, col_d] = d[row_d, col_d-1] * this_B[row_d, _col_j]
                        end
                    end
                end

            end

            # now we use d[:, end] .* B[end][:, cur_col_B[end]] .* c[i]
            ccB = cur_col_B[end]

            $(fill_out)

            cur_col_B[end] += 1
        end
        out
    end
    end  # @eval begin
end

function *(rk::RowKron, c::StridedVector)
    out = zeros(promote_type(eltype(rk), eltype(c)), size(rk, 1))
    A_mul_B!(out, rk, c)
    out
end

function *(rk::RowKron, c::StridedMatrix)
    out = zeros(promote_type(eltype(rk), eltype(c)), size(rk, 1))
    A_mul_B!(out, rk, c)
    out
end

function Base.At_mul_B(rk::RowKron, c::StridedVector)
    out = zeros(promote_type(eltype(rk), eltype(c)), size(rk, 2))
    At_mul_B!(out, rk, c)
    out
end

function Base.At_mul_B(rk::RowKron, c::StridedMatrix)
    out = zeros(promote_type(eltype(rk), eltype(c)), size(rk, 2), size(c, 2))
    At_mul_B!(out, rk, c)
    out
end

# cdprodx.m -- DONE
cdprodx{T<:Number}(b::Matrix{T}, c, ind=1:prod(size(b))) = b*c  # 39

function cdprodx(b::Array, c::StridedVecOrMat,
                 ind::AbstractArray{Int}=1:prod(size(b)))
    _check_cdprodx(b, c, ind)
    rk = RowKron(b[ind]...)
    rk*c
end

# nodeunif.m -- DONE
function nodeunif(n::Int, a::Int, b::Int)
    x = linspace(a, b, n)
    return x, x
end

function nodeunif(n::Array, a::Array, b::Array)
    d = length(n)
    xcoord = Array{Any}(d)
    for k=1:d
        xcoord[k] = linspace(a[k], b[k], n[k])
    end
    return gridmake(xcoord...), xcoord
end

function lookup(table::Vector, x::Real, p::Int=0)
    ind = searchsortedfirst(table, x) - 1
    m = length(table)

    if ind == m && p >= 2
        tosub = 1
        @inbounds for i in m-1:-1:1
            if table[i] == table[end]
                tosub += 1
            else
                break
            end
        end
        return m - tosub
    end

    if ind == 0 && (p == 1 || p == 3)
        ix = 0
        @inbounds for i in 1:m
            if table[1] == table[i]
                ix += 1
            else
                break
            end
        end
        return ix
    end
    ind
end

# lookup.c -- DONE
function lookup(table::AbstractVector, x::AbstractVector, p::Int=0)
    n = length(table)
    m = length(x)
    out = fill(42, m)

    # lower enbound adjustment
    numfirst = 1
    t1 = table[1]
    for i=2:n
        if table[i] == t1
            numfirst += 1
        else
            break
        end
    end

    # upper endpoint adjustment
    tn = table[n]
    if p >= 2
        n -= 1
        for i=n:-1:1
            if table[i] == tn
                n -= 1
            else
                break
            end
        end
    end

    n1 = n - 1
    n2 = n - 1

    if n - numfirst < 1  # only one unique value in table
        if p == 1 || p == 3
            for i=1:m
                out[i] = numfirst
            end
        else
            for i=1:m
                if table[1] <= x[i]
                    out[i] = numfirst
                else
                    out[i] = 0
                end
            end
        end
        return out
    end

    jlo = 1

    for i=1:m
        inc = 1
        xi = x[i]
        if xi >= table[jlo]
            jhi = jlo + 1
            while jhi <= n && xi >= table[jhi]
                jlo = jhi
                jhi += inc
                if jhi > n
                    jhi = n+1
                end
            end
        else
            jhi = jlo
            jlo -= 1
            while jlo > 0 && xi < table[jlo]
                jhi = jlo
                jlo -= inc
                if jlo < 1
                    jlo = 0
                    break
                else
                    inc += inc
                end
            end
        end

        while jhi - jlo > 1
            j = (jhi + jlo) >> 1
            if xi >= table[j]
                jlo = j
            else
                jhi = j
            end
        end

        out[i] = jlo

        if jlo < 1
            jlo = 1
            if p == 1 || p == 3
                out[i] = numfirst
            end
        end

        if jlo == n1
            jlo = n2
        end
    end

    out
end


# utility to expand the order input if needed
# used in basis_structure.jl and interp.jl
_check_order(N::Int, order::Int) = fill(order, 1, N)
_check_order(N::Int, order::Vector) = reshape(order, 1, N)

function _check_order(N::Int, order::Matrix)
    if size(order, 2) == N
        return order
    end

    if size(order, 1) == N
        m = size(order, 2)
        return reshape(order, m, N)
    end

    error("invalid order argument. Should have $N columns")
end

function _check_cdprodx(b::Array, c, ind::AbstractArray{Int})
    _ind_min, _ind_max = extrema(ind)
    @assert _ind_min > 0 && _ind_max <= length(b) "ind not conformable"
end

function _nnz_per_row{T}(A::Matrix{T})
    counts = zeros(Int, size(A, 1))
    my_zero = zero(T)
    @inbounds for col in 1:size(A, 2), row in 1:size(A, 1)
        if A[row, col] != my_zero
            counts[row] += 1
        end
    end
    counts
end

function _nnz_per_row{T}(A::SparseMatrixCSC{T})
    counts = zeros(Int, size(A, 1))
    @inbounds for col in 1:size(A, 2), ptr in A.colptr[col]:(A.colptr[col+1]-1)
        counts[A.rowval[ptr]] += 1
    end
    counts
end

function _row_kron_sparse_out_nnz(A, B)
    # get number of non-zeros in out
    acounts = _nnz_per_row(A)
    bcounts = _nnz_per_row(B)
    k = 0
    @inbounds @simd for _r in 1:size(A, 1)
        k += acounts[_r]*bcounts[_r]
    end
    k
end

function _allocate_row_kron_out{T,S}(::Type{SparseMatrixCSC},
                                     A::AbstractMatrix{T},
                                     B::AbstractMatrix{S})
    nobsa, na = size(A)
    nobsb, nb = size(B)
    k = _row_kron_sparse_out_nnz(A, B)
    SparseMatrixCSC(nobsa, na*nb,
        ones(Int, na*nb+1),          # colptr
        Array{Int}(k),                # rowval
        Array{promote_type(S,T)}(k)  # nzval
    )
end

function _allocate_row_kron_out{T,S}(::Type{Matrix},
                                     A::AbstractMatrix{T},
                                     B::AbstractMatrix{S})
    nobsa, na = size(A)
    nobsb, nb = size(B)
    Array{promote_type(S,T)}(nobsa, na*nb)
end

_allocate_row_kron_out(A::SparseMatrixCSC, B::SparseMatrixCSC) =
    _allocate_row_kron_out(SparseMatrixCSC, A, B)

_allocate_row_kron_out(A::AbstractMatrix, B::SparseMatrixCSC) =
    _allocate_row_kron_out(SparseMatrixCSC, A, B)

_allocate_row_kron_out(A::SparseMatrixCSC, B::AbstractMatrix) =
    _allocate_row_kron_out(SparseMatrixCSC, A, B)

_allocate_row_kron_out(A::AbstractMatrix, B::AbstractMatrix) =
    _allocate_row_kron_out(Matrix, A, B)
