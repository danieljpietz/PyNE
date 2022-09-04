import numba
from .type import nbLink
from .NEAlgorithm import system_dynamics, system_forces
import numpy as np
import numba as nb
import pandas as pd


def ne_compile(system: nbLink):
    system_compiled = system.compile()
    return system_compiled


@nb.njit
def eval(ne_system, x, xdot):
    dof, system, n_forces, forces, forces_map = ne_system
    (
        H,
        d,
    ) = system_dynamics(dof, system, x, xdot)
    F = system_forces(dof, n_forces, forces, forces_map)
    return np.concatenate((xdot, np.linalg.solve(H, F - d)))


@nb.njit
def step(ne_system, x, xdot, h):
    dof = ne_system[0]
    k1 = eval(ne_system, x, xdot)
    k2 = eval(ne_system, x + h * (k1[:dof] / 2), xdot + h * (k1[dof:] / 2))
    k3 = eval(ne_system, x + h * (k2[:dof] / 2), xdot + h * (k2[dof:] / 2))
    k4 = eval(ne_system, x + h * k3[:dof], xdot + h * k3[dof:])
    return (h / 6) * (k1 + 2 * (k2 + k3) + k4)


def simulate(ne_system, x, xdot, h, t):
    dof: numba.int64 = ne_system[0]
    x = np.reshape(x, dof)
    xdot = np.reshape(xdot, dof)
    t, results = _simulate(dof, ne_system, x, xdot, h, t)
    df = pd.DataFrame(
        results, columns=[f"{s}{i}" for s in ["x", "xdot"] for i in range(dof)]
    )
    df.insert(0, "t", t)
    return df


@nb.njit
def _simulate(dof, ne_system, x, xdot, h, t):
    tRange = np.arange(t[0], t[1], h)
    result = np.zeros((len(tRange), 2 * dof))
    result[0, :] = np.concatenate((x, xdot))
    for i in range(1, len(tRange)):
        x = result[i - 1, :dof]
        xdot = result[i - 1, dof:]
        result[i, :] = result[i - 1, :] + step(ne_system, x, xdot, h)
    return tRange, result
