from .type import _nbLink
from .func import *

def differential_kinematics(parent: _nbLink, child: _nbLink):

    dof = child.dof

    child.d_rotation_local = t_m_p(skew3(child.IHat), child.rotation_local)
    d_skew_position_local = skew3(child.ITilde)
    d_angular_velocity_local = child.IHat
    d_skew_angular_velocity_local = np.concatenate(
        (np.zeros((dof, 3, 3)), skew3(d_angular_velocity_local))
    )

    d_skew_angular_velocity_local_t = d_skew_angular_velocity_local.transpose((0, 2, 1))

    child.d_rotation_global = t_m_p(
        parent.d_rotation_global, child.rotation_local
    ) + m_t_p(parent.rotation_global, child.d_rotation_local)

    child.d_angular_velocity_global[:, :dof] = (
        parent.d_angular_velocity_global[:, :dof]
        + t_v_p(parent.d_rotation_global, child.angular_velocity_local)
        + parent.rotation_global @ d_angular_velocity_local
    )

    child.d_angular_velocity_global_skewed = skew3(
        child.d_angular_velocity_global[:, :dof]
    )

    d_JBStar = t_m_p(parent.d_rotation_global, child.position_local_skewed) + m_t_p(
        parent.rotation_global, child.d_skew_position_local
    )

    d_JBPrime = np.concatenate(
        (
            np.concatenate(
                (child.d_rotation_global.transpose((0, 2, 1)), np.zeros((dof, 3, 3))),
                axis=2,
            ),
            np.concatenate((d_JBStar, np.zeros((dof, 3, 3))), axis=2),
        ),
        axis=1,
    )

    d_JBVec = np.concatenate(
        (
            (np.zeros((dof, 3, dof))),
            t_m_p(child.d_rotation_global, child.ITilde),
        ),
        axis=1,
    )

    child.d_jacobian = (
        t_m_p(d_JBPrime, parent.jacobian)
        + m_t_p(child.JNPrime, parent.d_jacobian[:dof])
        + d_JBVec
    )

    ## Derivatives w.r.t. xdot

    d_rotation_local_t = np.concatenate(
        (child.d_rotation_global.transpose((0, 2, 1)), np.zeros((dof, 3, 3)))
    )
    d_skew_position_local = np.concatenate(
        (d_skew_position_local.transpose((0, 2, 1)), np.zeros((dof, 3, 3)))
    )

    d_linear_velocity_skewed = np.concatenate(
        (np.zeros((dof, 3, 3)), skew3(child.ITilde))
    )

    parent_d_rotation_global_full = np.concatenate(
        (parent.d_rotation_global, np.zeros((dof, 3, 3)))
    )
    parent_d_jacobian_full = np.concatenate(
        (parent.d_jacobian, np.zeros((dof, 6, dof)))
    )
    d_JBPrime = np.concatenate((d_JBPrime, np.zeros((dof, 6, 6))))

    d_dotJBPrime11 = t_m_p(
        d_skew_angular_velocity_local_t, child.rotation_global.T
    ) + m_t_p(skew(child.angular_velocity_local).T, d_rotation_local_t)

    d_dotJBPrime21Inner = (
        np.concatenate(
            (
                t_m_p(
                    child.d_angular_velocity_global_skewed, child.position_local_skewed
                ),
                np.zeros((dof, 3, 3)),
            )
        )
        + m_t_p(skew(parent.angular_velocity_global), d_skew_position_local)
        + d_linear_velocity_skewed
    )

    dotJBPrime21Inner = skew(child.angular_velocity_local).T @ child.rotation_global.T

    d_dotJBPrime21 = -(
        t_m_p(parent_d_rotation_global_full, dotJBPrime21Inner)
        + m_t_p(parent.rotation_global, d_dotJBPrime21Inner)
    )

    d_dotJBPrime = np.concatenate(
        (
            np.concatenate((d_dotJBPrime11, np.zeros((2 * dof, 3, 3))), axis=2),
            np.concatenate((d_dotJBPrime21, np.zeros((2 * dof, 3, 3))), axis=2),
        ),
        axis=1,
    )

    d_dotJBStar = np.concatenate(
        (
            (np.zeros((2 * dof, 3, dof))),
            t_m_p(
                t_m_p(parent_d_rotation_global_full, skew(child.angular_velocity_local))
                + m_t_p(parent.rotation_global, d_skew_angular_velocity_local),
                child.ITilde,
            ),
        ),
        axis=1,
    )

    child.d_dotJacobian = (
        t_m_p(d_dotJBPrime, parent.jacobian)
        + m_t_p(child.dotJNPrime, parent_d_jacobian_full)
        + t_m_p(d_JBPrime, parent.dotJacobian)
        + m_t_p(child.JNPrime, parent.d_dotJacobian)
        + d_dotJBStar
    )


#@nb.njit(void(_nb_instancetype, float64[:]), fastmath=True, cache=True)
def differential_dynamics(link: _nbLink, dotgamma):

    dof = link.dof

    d_m_corner_elem = m_t_p(
        skew(link.GAMMA), link.d_rotation_global.transpose((0, 2, 1))
    )

    dM = np.concatenate(
        (
            np.concatenate((np.zeros((dof, 3, 3)), d_m_corner_elem), axis=2),
            np.concatenate(
                (d_m_corner_elem.transpose((0, 2, 1)), np.zeros((dof, 3, 3))), axis=2
            ),
        ),
        axis=1,
    )

    jtm = link.jacobian.T @ link.M

    d_jtm = t_m_p(link.d_jacobian.transpose((0, 2, 1)), link.M) + m_t_p(
        link.jacobian.T, dM
    )

    link.d_H = t_m_p(d_jtm, link.jacobian) + m_t_p(
        (link.jacobian.T @ link.M), link.d_jacobian
    )

    inertia_omega_product = link.inertia_tensor @ link.angular_velocity_global
    d_inertia_omega_product = link.inertia_tensor @ link.d_angular_velocity_global

    d_skew_omega_global = skew3(link.d_angular_velocity_global)
    skew_omega_global = skew(link.angular_velocity_global)

    d_Jacobian = np.concatenate((link.d_jacobian, np.zeros((dof, 6, dof))))
    d_jtm = np.concatenate((d_jtm, np.zeros((dof, dof, 6))))
    d_rotation_global = np.concatenate((link.d_rotation_global, np.zeros((dof, 3, 3))))

    ang_vel_squared = skew(link.angular_velocity_global) @ skew(
        link.angular_velocity_global
    )

    dPrime = np.concatenate(
        (
            (
                skew(link.angular_velocity_global)
                @ link.inertia_tensor
                @ link.angular_velocity_global
            ),
            (link.rotation_global @ (ang_vel_squared) @ link.GAMMA),
        )
    )

    d_dPrime1 = (
        np.concatenate(
            (
                t_v_p(link.d_angular_velocity_global_skewed, inertia_omega_product),
                np.zeros((3, dof)),
            ),
            axis=1,
        )
        + skew_omega_global @ d_inertia_omega_product
    )

    d_dPrime2 = t_v_p(
        (
            t_m_p(d_rotation_global, ang_vel_squared)
            + m_t_p(2 * link.rotation_global @ skew_omega_global, d_skew_omega_global)
        ),
        link.GAMMA,
    )

    d_dPrime = np.concatenate((d_dPrime1, d_dPrime2))

    d_dotgamma = np.concatenate((np.zeros((dof, dof)), np.eye(dof)), axis=1)

    dot_jacob_dot_x_product = link.dotJacobian @ dotgamma
    d_dot_jacob_dot_product = t_v_p(link.d_dotJacobian, dotgamma) + (
        link.dotJacobian @ d_dotgamma
    )

    link.d_d = (
        t_v_p(d_jtm, dot_jacob_dot_x_product)
        + (jtm @ d_dot_jacob_dot_product)
        + t_v_p(d_Jacobian.transpose((0, 2, 1)), dPrime)
        + link.jacobian.T @ d_dPrime
    )
