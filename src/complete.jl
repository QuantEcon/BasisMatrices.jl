# --------------------------------------------------------- #
# Stuff to construct basis matrices of complete polynomials #
# --------------------------------------------------------- #

using Base.Cartesian: @nloops, @nexprs

"""
Construct basis matrix for complete polynomial of degree `d`, given
input data `z`. `z` is assumed to be the degree 1 realization of each
variable. For example, if variables are `q`, `r`, and `s`, then `z`
should be `z = [q r s]`

Output is a basis matrix. In our example, with `d` set to 2 we would have

TODO: update docstring to properly give order of terms

```julia
out = [ones(size(z,1)) q r s q.*r q.*s r.*s q.^2 q.*r q.*s r.^2 r.*s s.^2]
```
"""
:complete_polynomial

immutable Degree{N} end

function n_complete(n::Int, D::Int)
    out = 1
    for d=1:D
        tmp = 1
        for j=0:d-1
            tmp *= (n+j)
        end
        out += div(tmp, factorial(d))
    end
    out
end

#
# Vector functions
#
@generated function complete_polynomial!{N}(z::Vector, d::Degree{N},
                                            out::Vector)
    complete_polynomial_impl!(z, d, out)
end

function complete_polynomial_impl!{T,N}(z::Type{Vector{T}}, ::Type{Degree{N}},
                                        out::Type{Vector{T}})
    big_temp = Expr(:(=), Symbol("tmp_$(N+1)"), one(T))
    quote
        nvar = length(z)
        if length(out) != (n_complete(nvar, $N))
            error("z, out not compatible")
        end

        # reset first column to ones
        out[1] = one($T)

        ix = 1
        $big_temp
        @nloops($N, # number of loops
                i,  # counter
                d->((d == $N ? 1 : i_{d+1}) : nvar),  # ranges
                d->(d == 1 ?
                    (begin
                         ix += 1
                         out[ix] = tmp_{d+1}*z[i_d]
                     end) :
                    (begin
                        ix += 1
                        tmp_d = tmp_{d+1}*z[i_d]
                        out[ix] = tmp_d
                    end)),  # preexpr
                Expr(:block, :nothing)  # bodyexpr
                )
        out
    end
end

function complete_polynomial{T}(z::Vector{T}, d::Int)
    nvar = length(z)
    out = Array(T, n_complete(nvar, d))
    complete_polynomial!(z, Degree{d}(), out)::Vector{T}
end

function complete_polynomial!{T}(z::Vector{T}, d::Int, out::Vector{T})
    complete_polynomial!(z, Degree{d}(), out)::Vector{T}
end

#
# Matrix functions
#
@generated function complete_polynomial!{N}(z::Matrix, d::Degree{N},
                                            out::Matrix)
    complete_polynomial_impl!(z, d, out)
end

function complete_polynomial_impl!{T,N}(z::Type{Matrix{T}}, ::Type{Degree{N}},
                                        out::Type{Matrix{T}})
    quote
        nobs, nvar = size(z)
        if size(out) != (nobs, n_complete(nvar, $N))
            error("z, out not compatible")
        end

        # reset first column to ones
        @inbounds for i=1:nobs
            out[i, 1] = one($T)
        end

        ix = 1
        @nloops($N, # number of loops
                i,  # counter
                d->((d == $N ? 1 : i_{d+1}) : nvar),  # ranges
                d->((begin
                        ix += 1
                        @inbounds @simd for r=1:nobs
                            tmp = one($T)
                            @nexprs $N-d+1 j->(tmp *= z[r, i_{$N-j+1}])
                            out[r, ix]=tmp
                        end
                    end)),  # preexpr
                Expr(:block, :nothing)  # bodyexpr
                )
        out
    end
end

function complete_polynomial{T}(z::Matrix{T}, d::Int)
    nobs, nvar = size(z)
    out = Array(T, nobs, n_complete(nvar, d))
    complete_polynomial!(z, Degree{d}(), out)::Matrix{T}
end

function complete_polynomial!{T}(z::Matrix{T}, d::Int, out::Matrix{T})
    complete_polynomial!(z, Degree{d}(), out)::Matrix{T}
end

