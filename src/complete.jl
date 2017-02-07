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

```julia
out = [ones(size(z,1)) q q.^2 q.*r q.*s r r.^2 r.*s s s.^2]
```

TODO:
  * Currently a bit more code repetition than is desireable. It would be
    nice to cut down on the repetition between the `complete_polynomial`
    functions
  * Mutating arguments are currently last... Change them to first
  * Current algorithm for computing derivatives is kind of slow -- Is
    there any obvious ways to improve this?

"""
:complete_polynomial

immutable Degree{N} end
immutable Derivative{D} end

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
# Generating basis functions
#
@generated function complete_polynomial!{N,T,Ndim}(z::Array{T,Ndim}, d::Degree{N},
                                                   out::Array{T,Ndim})
    complete_polynomial_impl!(z, d, out)
end

function complete_polynomial!{T, Ndim}(z::Array{T,Ndim}, d::Int, out::Array{T,Ndim})
    complete_polynomial!(z, Degree{d}(), out)

    return out
end


#
# Vector versions for generating basis functions
#
function complete_polynomial_impl!{T,N}(z::Type{Vector{T}}, ::Type{Degree{N}},
                                        out::Type{Vector{T}})
    outer_temp = Expr(:(=), Symbol("tmp_$(N+1)"), one(T))
    quote
        nvar = length(z)
        if length(out) != (n_complete(nvar, $N))
            error("z, out not compatible")
        end

        # reset first column to ones
        out[1] = one($T)

        ix = 1
        $outer_temp
        @nloops($N, # number of loops
                i,  # counter
                d->((d == $N ? 1 : i_{d+1}) : nvar),  # ranges
                d->((begin
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
    complete_polynomial!(z, Degree{d}(), out)

    return out
end

#
# Matrix versions for generating basis functions
#
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
    complete_polynomial!(z, Degree{d}(), out)

    return out
end

#
# Generating 1st derivative of basis functions
#
@generated function complete_polynomial!{N,D,T,Ndim}(z::Array{T,Ndim}, d::Degree{N},
                                                     der::Derivative{D}, out::Array{T,Ndim})
    complete_polynomial_impl!(z, d, der, out)
end

function complete_polynomial!{T, Ndim}(z::Array{T,Ndim}, d::Int, der::Int, out::Array{T,Ndim})
    complete_polynomial!(z, Degree{d}(), Derivative{der}(), out)::Array{T,Ndim}

    return out
end

#
# Vector versions for generating first derivative of basis functions
#
function complete_polynomial_impl!{T,N,D}(z::Type{Vector{T}}, ::Type{Degree{N}},
                                          ::Type{Derivative{D}}, out::Type{Vector{T}})
    notD_top = Expr(:(=), Symbol("notD_$(N+1)"), one(T))
    coeff_top = Expr(:(=), Symbol("coeff_$(N+1)"), zero(T))
    quote
        nvar = length(z)
        if length(out) != (n_complete(nvar, $N))
            error("z, out not compatible")
        end

        # reset first element to zero
        out[1] = zero($T)

        ix = 1
        $notD_top
        $coeff_top
        @nloops($N, # number of loops
                i,  # counter
                d->((d == $N ? 1 : i_{d+1}) : nvar),  # ranges
                d->((begin
                        ix += 1
                        # Depending on what i_d is, update variables
                        if i_d == D
                            coeff_d = coeff_{d+1} + 1
                            notD_d = notD_{d+1}
                        else
                            coeff_d = coeff_{d+1}
                            notD_d = notD_{d+1}*z[i_d]
                        end

                        out[ix] = coeff_d * z[D]^(coeff_d-1) * notD_d
                    end)),  # preexpr
                Expr(:block, :nothing)  # bodyexpr
                )
        out
    end
end

function complete_polynomial{T}(z::Vector{T}, d::Int, der::Int)
    nvar = length(z)
    out = Array(T, n_complete(nvar, d))
    complete_polynomial!(z, Degree{d}(), Derivative{der}(), out)::Vector{T}

    return out
end

#
# Matrix versions for generating first derivative of basis functions
#
function complete_polynomial_impl!{T,N,D}(z::Type{Matrix{T}}, ::Type{Degree{N}},
                                          ::Type{Derivative{D}}, out::Type{Matrix{T}})
    coeff_top = Expr(:(=), Symbol("coeff_$(N+1)"), zero(T))
    quote
        nobs, nvar = size(z)
        if size(out) != (nobs, n_complete(nvar, $N))
            error("z, out not compatible")
        end

        # reset first element to zero
        @inbounds @simd for r=1:nobs
            out[r, 1] = zero($T)
        end

        ix = 1
        $coeff_top
        @nloops($N, # number of loops
                i,  # counter
                d->((d == $N ? 1 : i_{d+1}) : nvar),  # ranges
                d->((begin
                        ix += 1

                        # Depending on what i_d is, update variables
                        coeff_d = ifelse(i_d == D, coeff_{d+1} + 1, coeff_{d+1})

                        @inbounds @simd for r=1:nobs
                            tmp = one($T)
                            @nexprs $N-d+1 j->(tmp *= ifelse(i_{$N-j+1} != D, z[r, i_{$N-j+1}], one($T)))
                            out[r, ix] = coeff_d * tmp * z[r, D]^(coeff_d-1)
                        end
                    end)),  # preexpr
                Expr(:block, :nothing)  # bodyexpr
                )
        out
    end
end

function complete_polynomial{T}(z::Matrix{T}, d::Int, der::Int)
    nobs, nvar = size(z)
    out = Array(T, nobs, n_complete(nvar, d))
    complete_polynomial!(z, Degree{d}(), Derivative{der}(), out)::Matrix{T}

    return out
end

