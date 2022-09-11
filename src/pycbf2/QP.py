from cffi import FFI
import numba as nb
import numpy as np
import pkg_resources as pkg

_ffi = FFI()

_ffi.cdef(
    "int qp_solve(double* solution,"
    " double* H,"
    " double* _f,"
    " double* A, "
    "double* lb,"
    " double* ub,"
    " size_t n_vars,"
    " size_t n_constraints);"
)

lib = _ffi.dlopen(pkg.resource_filename(__name__, "lib/libQP.dylib"))

_qpsolve = lib.qp_solve


@nb.njit
def qpsolve(H, f, A, lb, ub):

    solution = np.empty_like(f)

    _qpsolve(
        _ffi.from_buffer(solution),
        _ffi.from_buffer(H.flatten()),
        _ffi.from_buffer(f),
        _ffi.from_buffer(A),
        _ffi.from_buffer(lb),
        _ffi.from_buffer(ub),
        H.shape[0],
        A.shape[0],
    )

    return solution
