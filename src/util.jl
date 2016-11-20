# ---------------- #
# Helper functions #
# ---------------- #

# ckronx.m -- DONE
function ckronx{TM<:AbstractMatrix}(b::AbstractMatrix{TM}, c::Array,
                                    ind::AbstractArray{Int}=1:length(b))
    d = length(ind)  # 26
    n = Array(Int, d)  # 27
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
function row_kron!(A::AbstractMatrix, B::AbstractMatrix, out::AbstractMatrix)
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

function row_kron{S,T}(A::AbstractMatrix{S}, B::AbstractMatrix{T})
    nobsa, na = size(A)
    nobsb, nb = size(B)
    out = Array(promote_type(S, T), nobsa, na*nb)
    row_kron!(A, B, out)
    out
end

function row_kron{S,T}(A::SparseMatrixCSC{S}, B::SparseMatrixCSC{T})
    nobsa, na = size(A)
    nobsb, nb = size(B)

    # get number of non-zeros in out
    acounts = zeros(Int, nobsa)
    @inbounds for col in 1:na, ptr in A.colptr[col]:(A.colptr[col+1]-1)
        acounts[A.rowval[ptr]] += 1
    end

    bcounts = zeros(Int, nobsa)
    @inbounds for col in 1:nb, ptr in B.colptr[col]:(B.colptr[col+1]-1)
        bcounts[B.rowval[ptr]] += 1
    end

    k = 0
    @inbounds @simd for _r in 1:nobsa
        k += acounts[_r]*bcounts[_r]
    end

    av = A.nzval
    bv = B.nzval

    colptr = zeros(Int, na*nb+1)
    rowval = zeros(Int, k)
    nzval = zeros(promote_type(S, T), k)

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

    SparseMatrixCSC(nobsa, na*nb, colptr, rowval, nzval)
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

# cdprodx.m -- DONE
cdprodx{T<:Number}(b::Matrix{T}, c, ind=1:prod(size(b))) = b*c  # 39

# cdprodx.c
function cdprodx(b::Array, c, ind::AbstractVector{Int}=1:prod(size(b)))
    # input verification
    nrow = size(b[1], 1)
    for _b in b[2:end]
        @assert size(_b, 1) == nrow "Must have same # of rows"
    end

    crow = size(c, 1)
    ccol = size(c, 2)
    ncol = *([size(_b, 2) for _b in b]...)
    @assert ncol == crow "nrows in c must be product of cols in b"

    _ind_min, _ind_max = extrema(ind)
    @assert _ind_min > 0 && _ind_max <= length(b) "ind not conformable"

    # allocate temporary and ouput arrays
    Nb = length(ind)
    out = zeros(Float64, nrow, ccol)

    # put B in desired order
    B = [b[_] for _ in ind]
    _is_sparse = [issparse(_) for _ in B]
    _last_sparse = _is_sparse[end]
    d = Array(Float64, nrow, Nb)
    d[:, 1] = 1.0

    n_col_B = [size(_, 2) for _ in B]
    cur_col_B = copy(n_col_B) + 1

    Bend = B[end]

    @inbounds for i in 1:crow  # loop over rows of c
        if cur_col_B[end] > n_col_B[end]
            cur_col_B[end] = 1
            j = Nb
            while j > 1
                j -= 1
                cur_col_B[j] += 1

                if cur_col_B[j] <= n_col_B[j]
                    break
                else
                    cur_col_B[j] = 1
                end
            end

            for col_d in j+1:Nb
                _j = col_d - 1
                _col_j = cur_col_B[_j]
                this_B = B[_j]

                if _is_sparse[_j]
                    d[:, col_d] = 0.0

                    # fill in non-zero rows
                    for ptr in this_B.colptr[_col_j]:(this_B.colptr[_col_j+1]-1)
                        _r = this_B.rowval[ptr]
                        _v = this_B.nzval[ptr]
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
        if _last_sparse
            if ccol == 1
                ci = c[i]
                for ptr in Bend.colptr[ccB]:(Bend.colptr[ccB+1]-1)
                    _r = Bend.rowval[ptr]
                    _v = Bend.nzval[ptr]
                    out[_r] += d[_r, end] * _v * ci
                end
            else
                for ptr in (Bend.colptr[ccB]):(Bend.colptr[ccB+1]-1)
                    _r = Bend.rowval[ptr]
                    _v = Bend.nzval[ptr]
                    full_B = d[_r, end] * _v

                    for _c in 1:ccol
                        out[_r, _c] += full_B * c[i, _c]
                    end
                end
            end
        else
            if ccol == 1
                ci = c[i]
                @simd for _r in 1:nrow
                    out[_r] += d[_r, end] * Bend[_r, ccB] * ci
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
        cur_col_B[end] += 1
    end
    out

end

# cckronx.m -- DONE
cckronx{T<:Number}(b::Matrix{T}, c, ind=1:prod(size(b))) = b * c  # 23

function cckronx(b::Array, c, ind=1:prod(size(b)))
    d = length(ind)  # 25
    n = Int[size(b[ind[i]], 2) for i=1:d]  #26-27
    prod(n) != size(c, 1) && error("b and c are not conformable")  # 28-30

    z = c'  # 31
    mm = 1  # 32
    for i=1:d  # 33
        m = prod(size(z)) / n[i]  # 34
        z = reshape(z, m, n[i])  # 35
        z = b[ind[i]] \ z'  # 36
        mm = mm*size(z, 1)  # 37
    end  # 38
    reshape(z, mm, size(c, 2))  # 39
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

function squeeze_trail(x::Array)
    sz = size(x)
    squeezers = Int[]
    n = length(sz)
    for i=n:-1:1
        if sz[i] == 1
            push!(squeezers, i)
        else
            break
        end
    end
    squeeze(x, tuple(squeezers...))
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
