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
    v = v.squeeze()
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def skew3(m):
    return np.array([skew(v) for v in m.transpose()])


def sparse(size, coordinate):
    zero = np.zeros(size)
    zero[coordinate[0]][coordinate[1]] = 1
    return zero
