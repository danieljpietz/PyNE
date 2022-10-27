import numpy as np
from .type import _nbLink
from .func import block_4x4, skew, axang_rotmat
from .CBFAlgorithm import differential_kinematics, differential_dynamics


def recursive_kinematics(
    parent: _nbLink, child: _nbLink, gamma, dotgamma
):
    child.gamma = gamma
    child.dotgamma = dotgamma

    child.x = gamma[child.index]
    child.xdot = dotgamma[child.index]

    if child.link_type:
        child.rotation_local = child.rotation_offset @ axang_rotmat(
            child.axis, gamma[child.index]
        )

    child.rotation_global = parent.rotation_global @ child.rotation_local

    child.angular_velocity_local = child.IHat @ dotgamma

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

    child.angular_velocity_global = child.jacobian[0:3] @ dotgamma

    child.dotJNPrime = block_4x4(
        (
            (
                -skew(child.angular_velocity_local) @ child.rotation_local.transpose(),
                np.zeros((3, 3)),
            ),
            (
                -parent.rotation_global
                @ (
                    skew(parent.angular_velocity_global) @ skew(child.position)
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
                parent.rotation_global
                @ skew(parent.angular_velocity_global)
                @ child.ITilde,
            )
        )
    )



def link_dynamics(link: _nbLink, dotgamma):
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
                link.angular_velocity_global,
                link.inertia_tensor @ link.angular_velocity_global,
            ),
            link.rotation_global
            @ np.cross(
                link.angular_velocity_global,
                np.cross(link.angular_velocity_global, link.GAMMA),
            ),
        )
    )

    link.d = (
        link.jacobian.transpose() @ (link.M @ link.dotJacobian @ dotgamma
        + d_link_star)
    )


def system_dynamics(
    dof,
    system,
    gamma,
    d_gamma,
):
    H = np.zeros((dof, dof), dtype=float)
    d = np.zeros(dof, dtype=float)

    d_H = np.zeros((dof, dof, dof), dtype=float)
    d_d = np.zeros((dof, 2 * dof), dtype=float)

    for link in system[1:]:
        recursive_kinematics(link.parent.properties, link.properties, gamma, d_gamma)
        link_dynamics(link.properties, d_gamma)

        H += link.properties.H
        d += link.properties.d

        differential_kinematics(link.parent.properties, link.properties)
        differential_dynamics(link.properties, d_gamma)

        d_H += link.properties.d_H
        d_d += link.properties.d_d

    return H, d, np.concatenate((d_H, np.zeros((dof, dof, dof), dtype=float))), d_d


def system_forces(dof, n_forces, forces, d_forces, links):
    F = np.zeros(dof, dtype=float)
    d_F = np.zeros((dof, 2 * dof), dtype=float)
    for i in range(n_forces):
        F += forces[i](links[i].properties)
        d_F += d_forces[i](links[i].properties)
    return F, d_F
