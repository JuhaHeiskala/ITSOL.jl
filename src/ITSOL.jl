# MIT License
# 
# Copyright (c) 2021 Juha Tapio Heiskala
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

module ITSOL

using SparseArrays
import ITSOL_2_jll
import ZITSOL_1_jll

const libitsol2 = ITSOL_2_jll.libITSOL_2
const libzitsol1 = ZITSOL_1_jll.libZITSOL_1

# Sparse matrix data structures used by ITSOL libraries (ITSOL_2, ZITSOL_1)
mutable struct SpaFmt{T}
    n::Int32
    nzcount::Ptr{Cint}
    ja::Ptr{Ptr{Cint}}
    ma::Ptr{Ptr{T}}

    ja_ptrs::Array{Ptr{Cint},1}
    ma_ptrs::Array{Ptr{T},1}
    
    nzc_data::Array{Cint,1}
    ja_data::Array{Cint,1}
end

mutable struct ILUfac{T}
    n::Int32
    L::Ptr{Cvoid}
    D::Ptr{T}
    U::Ptr{Cvoid}
    work::Ptr{Cint}
end

mutable struct IluSpar{T}
    n::Cint
    C::Ptr{Cvoid}
    L::Ptr{Cvoid}
    U::Ptr{Cvoid}

    rperm::Ptr{Cint}
    perm::Ptr{Cint}
    perm2::Ptr{Cint}

    D1::Ptr{T}
    D2::Ptr{T}
    wk::Ptr{T}
end

# ILU0 solver

function ilu0(A::SparseMatrixCSC{T}, milu) where T <: Number
    itsolA = make_SpaFmt(A)
    iLU = ILUfac{T}(A.n, C_NULL, C_NULL, C_NULL, C_NULL)

    if T <: Real
        f = ccall((:itsol_pc_ilukC, libitsol2), Int32, (Int32, Ref{SpaFmt}, Ref{ILUfac}, Cint, Ptr{Cvoid}),
                  Cint(0), itsolA, iLU, Cint(milu), C_NULL)
    else
        f = ccall((:zilukC, libzitsol1), Int32, (Int32, Ref{SpaFmt}, Ref{ILUfac}, Cint, Ptr{Cvoid}),
                  Cint(0), itsolA, iLU, Cint(milu), C_NULL)
    end

    L = _make_jl_sparse(T, iLU.L)
    U = _make_jl_sparse(T, iLU.U)
    
    D = Array{T, 1}(undef, iLU.n)
    unsafe_copyto!(pointer(D), iLU.D, iLU.n)
                   
    return (L, D, U)
end

# ILUC solver

function iluc(A::SparseMatrixCSC{T}, droptol, milu) where T <: Number
    itsolA = make_SpaFmt(A)

    inputILU = ILUfac{T}(A.n, C_NULL, C_NULL, C_NULL, C_NULL)

    if T<:Real
        f = ccall((:itsol_CSClumC, libitsol2), Int32, (Ref{SpaFmt}, Ref{ILUfac}, Cint), itsolA, inputILU, Cint(0))
    
        iLU = ILUfac{T}(A.n, C_NULL, C_NULL, C_NULL, C_NULL)

        f = ccall((:itsol_pc_ilutc, libitsol2), Cint, (Ref{ILUfac}, Ref{ILUfac}, Cint, Cdouble, Cint, Cint, Ptr{Cvoid}),
                  inputILU, iLU, Cint(A.n), Cdouble(droptol), Cint(5), Cint(milu), C_NULL)
    else
        f = ccall((:zCSClumC, libzitsol1), Int32, (Ref{SpaFmt}, Ref{ILUfac}, Cint), itsolA, inputILU, Cint(0))
    
        iLU = ILUfac{T}(A.n, C_NULL, C_NULL, C_NULL, C_NULL)

        f = ccall((:zilutc, libzitsol1), Cint, (Ref{ILUfac}, Ref{ILUfac}, Cint, Cdouble, Cint, Cint, Ptr{Cvoid}),
                  inputILU, iLU, Cint(A.n), Cdouble(droptol), Cint(5), Cint(milu), C_NULL)
    end
    
    L = _make_jl_sparse(T, iLU.L)
    U = _make_jl_sparse(T, iLU.U)
    
    D = Array{T, 1}(undef, iLU.n)
    unsafe_copyto!(pointer(D), iLU.D, iLU.n)
                   
    return (L, D, U)
end

# ILUtp solver

function ilutp(A::SparseMatrixCSC{T}, droptol, thresh, milu, udiag) where T<:Number
    itsolA = make_SpaFmt(A)
    iLU = IluSpar{T}(A.n, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                  C_NULL, C_NULL, C_NULL, C_NULL)

    droptolA = zeros(Cdouble, 7)
    droptolA[6] = droptol
    droptolA[7] = droptol

    lfilA = zeros(Cint, 7)
    lfilA[6] = A.n
    lfilA[7] = A.n

    if T<:Real
        f = ccall((:itsol_pc_ilutpC, libitsol2), Int32,
                  (Ref{SpaFmt}, Ptr{Cdouble}, Ptr{Cint}, Cdouble, Cint, Ref{IluSpar}, Cint),
                  itsolA, pointer(droptolA), pointer(lfilA), thresh, Cint(A.n), iLU, Cint(udiag))
    else
        f = ccall((:zilutpC, libzitsol1), Int32,
                  (Ref{SpaFmt}, Ptr{Cdouble}, Ptr{Cint}, Cdouble, Cint, Ref{IluSpar}, Cint),
                  itsolA, pointer(droptolA), pointer(lfilA), thresh, Cint(A.n), iLU, Cint(udiag))
    end
    
    L = _make_jl_sparse(T, iLU.L)
    U = _make_jl_sparse(T, iLU.U)
    
    P = Array{Cint, 1}(undef, iLU.n)
    unsafe_copyto!(pointer(P), iLU.perm2, iLU.n)
    
    return (L,U,P)
end


# convert from Julia SparseArray to ITSOL sparse array
function make_SpaFmt(A::SparseMatrixCSC{T}) where T <: Number
    
    nz_data = Array{Cint,1}(undef, A.n)
    ja_data = Array{Cint,1}(undef, nnz(A))   

    ja_ptrs = Array{Ptr{Cint},1}(undef, A.n)
    ma_ptrs = Array{Ptr{T},1}(undef, A.n)
    
    cum_nnz = 0
    for c_ix = 1:A.n
        col_nnz = Cint(length(nzrange(A,c_ix)))
        nz_data[c_ix] = col_nnz

        for r_ix = A.colptr[c_ix]:(A.colptr[c_ix]+col_nnz-1)
            ja_data[r_ix]  = Cint(A.rowval[r_ix]-1)
        end
        ja_ptrs[c_ix] = pointer(ja_data) + cum_nnz*sizeof(Cint)
        ma_ptrs[c_ix] = pointer(A.nzval) + cum_nnz*sizeof(T)
        cum_nnz += col_nnz
    end

    itsolA = SpaFmt{T}(Cint(A.n), pointer(nz_data), pointer(ja_ptrs), pointer(ma_ptrs), 
                    ja_ptrs, ma_ptrs, nz_data, ja_data)
        
    return itsolA
end

# convert from ITSOL SparseArray to Julia sparse array
function  _make_jl_sparse(T, A::Ptr{Cvoid})
    n = unsafe_load(Ptr{Cint}(A))

    row_nz_cnt = Array{Int32,1}(undef, n+1)
    row_nz_cnt[1] = Int32(1)
    unsafe_copyto!(pointer(row_nz_cnt)+sizeof(Int32), unsafe_load(Ptr{Ptr{Cint}}(A+8)), n)
    colptr = cumsum(row_nz_cnt)

    n_nz = colptr[end]-1
    
    rowval = Vector{Int32}(undef, n_nz)
    nzval = Vector{T}(undef, n_nz)

    r_ix_array_ptr = unsafe_load(Ptr{Ptr{Cvoid}}(A+16))
    val_array_ptr = unsafe_load(Ptr{Ptr{Cvoid}}(A+24))
    for c = 1:n
        unsafe_copyto!(pointer(rowval)+(colptr[c]-1)*sizeof(Cint),
                       Ptr{Cint}(unsafe_load(Ptr{Ptr{Cvoid}}(r_ix_array_ptr), c)), row_nz_cnt[c+1])
        unsafe_copyto!(pointer(nzval)+(colptr[c]-1)*sizeof(T),
                       Ptr{T}(unsafe_load(Ptr{Ptr{Cvoid}}(val_array_ptr), c)), row_nz_cnt[c+1])
        
    end
    jlA = SparseMatrixCSC{T,Int64}(n, n, Int64.(colptr), (Int64.(rowval)).+Int64(1), nzval)
    return jlA    
end

end # module

