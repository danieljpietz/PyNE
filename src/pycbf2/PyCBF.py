from .type import nbLink
from .NEAlgorithm import system_dynamics, system_forces
from .func import t_v_p
import numpy as np
import pandas as pd
from .cbf import cbf_eval

def ne_compile(system: nbLink):
    system_compiled = system.compile()
    return system_compiled


def eval(ne_system, x, xdot):
    dof, system, n_forces, forces, d_forces, forces_map, _cbf = ne_system
    H, d, d_H, d_d = system_dynamics(dof, system, x, xdot)
    F, d_F = system_forces(dof, n_forces, forces, d_forces, forces_map)

    hInv = np.linalg.inv(H)

    forces = F - d
    d_forces = d_F - d_d


    xddot = hInv @ forces
    d_xddot = hInv @ (d_forces - t_v_p(d_H, xddot))

    cbf_funcs, clf_funcs, _urefs, _input_matrix = _cbf

    X = np.concatenate((x, xdot))
    XD = np.concatenate((X, xddot))

    _barrier, _dot_barrier, _barrier_gradient, _barrier_hessian, = cbf_funcs
    cbf = (_barrier[0](X), _dot_barrier[0](XD), _barrier_gradient[0](X), _barrier_hessian[0](X))

    _lyapanov, _dot_lyapanov, _lyapanov_gradient, _lyapanov_hessian, = clf_funcs
    clf = (_lyapanov[0](X), _dot_lyapanov[0](XD), _lyapanov_gradient[0](X), _lyapanov_hessian[0](X))

    uref = _urefs(x, xdot)
    input_matrix = _input_matrix(x, xdot)

    f = np.concatenate((xdot, xddot))
    g = np.concatenate((np.zeros((dof, len(uref))), hInv @ input_matrix))

    d_f = np.concatenate((np.concatenate((np.zeros((dof, dof)), np.eye(dof)), axis=1), d_xddot))

    sys_packet = (dof, f, d_f, g)

    control_forces = cbf_eval(
        sys_packet, cbf, clf, uref, None, None
    )

    forces_with_control = forces + input_matrix @ control_forces

    return np.concatenate((xdot, np.linalg.solve(H, forces_with_control))), cbf[0], control_forces



def step(ne_system, x, xdot, h):
    dof = ne_system[0]
    k1, cbf1, cf1 = eval(ne_system, x, xdot)
    k2, cbf2, cf2 = eval(ne_system, x + h * (k1[:dof] / 2), xdot + h * (k1[dof:] / 2))
    k3, cbf3, cf3 = eval(ne_system, x + h * (k2[:dof] / 2), xdot + h * (k2[dof:] / 2))
    k4, cbf4, cf4 = eval(ne_system, x + h * k3[:dof], xdot + h * k3[dof:])

    return (h / 6) * (k1 + 2 * (k2 + k3) + k4), (1 / 6) * (cbf1 + 2 * (cbf2 + cbf3) + cbf4), (1 / 6) * (cf1 + 2 * (cf2 + cf3) + cf4)



def _simulate(dof, ne_system, x, xdot, h, t, ulen):
    tRange = np.arange(t[0], t[1], h)
    result = np.zeros((len(tRange), 2 * dof))
    control_forces = np.zeros((len(tRange), ulen))
    cbf = np.zeros((len(tRange)))
    result[0, :] = np.concatenate((x, xdot))
    for i in range(1, len(tRange)):
        x = result[i - 1, :dof]
        xdot = result[i - 1, dof:]
        xddot, _cbf, _cf = step(ne_system, x, xdot, h)
        result[i, :] = result[i - 1, :] + xddot
        cbf[i] = _cbf
        control_forces[i] = _cf
    cbf[0] = cbf[1]
    control_forces[0] = control_forces[1]
    return tRange, result, cbf, control_forces


def simulate(ne_system, x, xdot, h, t):
    dof = ne_system[0]
    x = np.reshape(x, dof)
    xdot = np.reshape(xdot, dof)
    ulen = len(ne_system[-1][2](x, xdot))
    t, results, cbf, control = _simulate(dof, ne_system, x, xdot, h, t, ulen)
    df = pd.DataFrame(
        results, columns=[f"{s}{i}" for s in ["x", "xdot"] for i in range(dof)]
    )
    df.insert(0, "cbf", cbf)
    df.insert(0, "t", t)
    for i in range(ulen):
        df[f'u_{i}'] = control[:, i]
    return df
