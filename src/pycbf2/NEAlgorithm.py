import numpy as np
from numba import njit, int64, float64, void
from .type import nbLink, _nbLink
from .func import block_4x4, skew, axang_rotmat

nb_instancetype = nbLink.class_type.instance_type
_nb_instancetype = _nbLink.class_type.instance_type


@njit(void(_nb_instancetype, _nb_instancetype, float64[:], float64[:]))
def recursive_kinematics(
    parent: _nbLink, child: _nbLink, gamma: float64[:], dotgamma: float64[:]
):
    child.x = gamma[child.index]
    child.xdot = dotgamma[child.index]

    if child.link_type:
        child.rotation_local = child.rotation_offset @ axang_rotmat(
            child.axis, gamma[child.index]
        )

    child.rotation_global = parent.rotation_global @ child.rotation_local

    child.angular_velocity = child.IHat @ dotgamma

    child.linear_velocity = child.ITilde @ dotgamma

    child.JNPrime = block_4x4(
        (
            (child.rotation_local.transpose(), np.zeros((3, 3))),
            (-parent.rotation_global @ skew(child.position), np.eye(3)),
        )
    )

    child.jacobian = child.JNPrime @ parent.jacobian + np.concatenate(
        (child.IHat, parent.rotation_global @ child.ITilde)
    )

    child.angular_velocity = child.jacobian[0:3] @ dotgamma

    child.dotJNPrime = block_4x4(
        (
            (
                -skew(child.angular_velocity) @ child.rotation_local.transpose(),
                np.zeros((3, 3)),
            ),
            (
                parent.rotation_global
                @ (
                    skew(parent.angular_velocity) @ skew(child.position)
                    + skew(child.linear_velocity)
                ),
                np.zeros((3, 3)),
            ),
        )
    )

    child.dotJacobian = (
        child.dotJNPrime @ parent.jacobian
        + child.JNPrime @ parent.dotJacobian
        + np.concatenate(
            (
                np.zeros((3, child.dof)),
                parent.rotation_global @ skew(parent.angular_velocity) @ child.ITilde,
            )
        )
    )


@njit(void(_nb_instancetype, float64[:]))
def link_dynamics(link: _nbLink, dotgamma: float64):
    M_LINK_corner = skew(link.GAMMA) @ link.rotation_global.transpose()
    link.M = block_4x4(
        (
            (link.inertia_tensor, M_LINK_corner),
            (
                M_LINK_corner.transpose(),
                np.array(((link.mass, 0, 0), (0, link.mass, 0), (0, 0, link.mass))),
            ),
        )
    )

    link.H = link.jacobian.transpose() @ link.M @ link.jacobian

    d_link_star = np.concatenate(
        (
            np.cross(
                link.angular_velocity, link.inertia_tensor @ link.angular_velocity
            ),
            link.rotation_global
            @ np.cross(
                link.angular_velocity, np.cross(link.angular_velocity, link.GAMMA)
            ),
        )
    )

    link.d = (
        link.jacobian.transpose() @ link.M @ link.dotJacobian @ dotgamma
        + link.jacobian.transpose() @ d_link_star
    )


@njit
def system_dynamics(
    dof: int64,
    system,
    gamma,
    d_gamma,
):
    H = np.zeros((dof, dof), dtype=float)
    d = np.zeros(dof, dtype=float)
    for link in system[1:]:
        recursive_kinematics(link.parent.properties, link.properties, gamma, d_gamma)
        link_dynamics(link.properties, d_gamma)
        H += link.properties.H
        d += link.properties.d
    return H, d


@njit
def system_forces(dof, n_forces, forces, links):
    F = np.zeros(dof, dtype=float)
    for i in range(n_forces):
        link = links[i]

        F += forces[i](link.properties)

    return F
