
import numpy as np

def skew(v):
    return np.array(((0, -v[2], v[1]), (v[2], 0, -v[0]), (-v[1], v[0], 0)))


def skew3(m):
    n_cols = m.shape[1]
    ret = np.zeros((n_cols, 3, 3), dtype=float)
    for i in range(n_cols):
        ret[i] = skew(m.T[i])
    return ret



def block_4x4(block):
    return np.concatenate(
        (
            np.concatenate((block[0][0], block[0][1]), axis=1),
            np.concatenate((block[1][0], block[1][1]), axis=1),
        )
    )


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



def t_v_p(A, B):
    C = np.zeros((A.shape[1], A.shape[0]))
    for i in range(A.shape[0]):
        C[:, i] = A[i] @ B
    return C



def t_m_p(A, B):
    return A @ B



def m_t_p(B, A):
    return B @ A


def sparse(size, coordinate):
    zero = np.zeros(size)
    zero[coordinate[0]][coordinate[1]] = 1
    return zero
