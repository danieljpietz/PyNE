import numba
import numpy as np
from numba import float64


@numba.njit(float64[:, :](float64[:]))
def skew(v: numba.float64[:]) -> numba.float64[:, :]:
    return np.array(((0, -v[2], v[1]), (v[2], 0, -v[0]), (-v[1], v[0], 0)))


@numba.njit(
    float64[:, :](numba.types.UniTuple(numba.types.UniTuple(numba.float64[:, :], 2), 2))
)
def block_4x4(block: numba.float64[:, :]) -> numba.float64[:, :]:
    return np.concatenate(
        (
            np.concatenate((block[0][0], block[0][1]), axis=1),
            np.concatenate((block[1][0], block[1][1]), axis=1),
        )
    )


@numba.njit(float64[:, :](float64[:], float64))
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


def sparse(size, coordinate):
    zero = np.zeros(size)
    zero[coordinate[0]][coordinate[1]] = 1
    return zero
