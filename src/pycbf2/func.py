import numba
import numpy as np
from numba import float64, prange


@numba.njit(float64[:, :](float64[:]), cache=True, parallel=True)
def skew(v: numba.float64[:]) -> numba.float64[:, :]:
    return np.array(((0, -v[2], v[1]), (v[2], 0, -v[0]), (-v[1], v[0], 0)))


@numba.njit(float64[:, :, :](float64[:, :]), cache=True, parallel=True)
def skew3(m: numba.float64[:, :]) -> numba.float64[:, :, :]:
    n_cols = m.shape[1]
    ret: numba.float64[:, :, :] = np.zeros((n_cols, 3, 3), dtype=numba.float64)
    for i in prange(n_cols):
        ret[i] = skew(m.T[i])
    return ret


@numba.njit(
    float64[:, :](
        numba.types.UniTuple(numba.types.UniTuple(numba.float64[:, :], 2), 2)
    ),
    cache=True,
    parallel=True,
)
def block_4x4(block: numba.float64[:, :]) -> numba.float64[:, :]:
    return np.concatenate(
        (
            np.concatenate((block[0][0], block[0][1]), axis=1),
            np.concatenate((block[1][0], block[1][1]), axis=1),
        )
    )


@numba.njit(float64[:, :](float64[:], float64), cache=True, parallel=True)
def axang_rotmat(axis, angle):
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        (
            (aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)),
            (2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)),
            (2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc),
        )
    )


@numba.njit(
    numba.float64[:, :](numba.float64[:, :, :], numba.float64[:]), fastmath=True
)
def t_v_p(A, B):
    C = np.zeros((A.shape[1], A.shape[0]))
    for i in range(A.shape[0]):
        C[:, i] = A[i] @ B
    return C


@numba.njit(
    numba.float64[:, :, :](numba.float64[:, :, :], numba.float64[:, :]), fastmath=True
)
def t_m_p(A, B):
    C = np.zeros((A.shape[0], A.shape[1], B.shape[1]))
    for i in range(A.shape[0]):
        C[i] = A[i] @ B
    return C


@numba.njit(
    numba.float64[:, :, :](numba.float64[:, :], numba.float64[:, :, :]), fastmath=True
)
def m_t_p(B, A):
    C = np.zeros((A.shape[0], B.shape[0], A.shape[2]))
    for i in range(A.shape[0]):
        C[i] = B @ A[i]
    return C


def sparse(size, coordinate):
    zero = np.zeros(size)
    zero[coordinate[0]][coordinate[1]] = 1
    return zero
