import numpy as np


def rotx(theta, lib=np):
    theta = theta.squeeze()
    return np.array(
        [
            [1, 0, 0],
            [0, lib.cos(theta), -lib.sin(theta)],
            [0, lib.sin(theta), lib.cos(theta)],
        ]
    )


def roty(theta, lib=np):
    theta = theta.squeeze()
    return np.array(
        [
            [lib.cos(theta), 0, lib.sin(theta)],
            [0, 1, 0],
            [-lib.sin(theta), 0, lib.cos(theta)],
        ]
    )


def rotz(theta, lib=np):
    theta = theta.squeeze()
    return np.array(
        [
            [lib.cos(theta), -lib.sin(theta), 0],
            [lib.sin(theta), lib.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def skew(v):
    v = np.reshape(v, (3)).astype(float)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def skew3(m):
    return np.array([skew(v.squeeze()) for v in m.transpose()])


def sparse(size, coordinate):
    zero = np.zeros(size)
    zero[coordinate[0]][coordinate[1]] = 1
    return zero


def quadprog_solve_qp(P, q, G, h, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return np.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


