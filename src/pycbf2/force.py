import numpy as np
from .func import skew, m_t_p, t_m_p, t_v_p
from .type import _nbLink


def neforce(func):
    return (func)


def d_neforce(func):
    return (func)


def jointforce(func):
    return (func)


def d_jointforce(func):
    return (func)



def my_decorator(func):
    return func


class Force:
    def __init__(self, callback, d_callback):
        self.callback = callback
        self.d_callback = d_callback

    def _jit(self):
        return self.callback, self.d_callback


class JointForce(Force):
    def __init__(self, callback, d_callback):
        super(JointForce, self).__init__(
            self._joint_force_wrapper(callback), self._d_joint_force_wrapper(d_callback)
        )

    @staticmethod
    def _joint_force_wrapper(_callback):
        @neforce
        def _joint_force(link):
            retVal = np.zeros(link.dof)
            retVal[link.index] = _callback(link)
            return retVal

        return _joint_force

    @staticmethod
    def _d_joint_force_wrapper(_callback):
        @d_neforce
        def _d_joint_force(link):
            retVal = np.zeros((link.dof, 2 * link.dof))
            retVal[link.index, :] = _callback(link)
            return retVal

        return _d_joint_force


class Friction(JointForce):
    def __init__(self, coeff):
        super(Friction, self).__init__(
            self._friction_cb_factory(coeff), self._d_friction_cb_factory(coeff)
        )

    @staticmethod
    def _friction_cb_factory(coeff):
        @jointforce
        def _friction_callback(link):
            return -link.xdot * coeff

        return _friction_callback

    @staticmethod
    def _d_friction_cb_factory(coeff):
        @d_jointforce
        def _d_friction_callback(link):
            re = np.zeros(2 * link.dof)
            re[link.index + link.dof] = -coeff
            return re

        return _d_friction_callback


class POSForce(Force):
    def __init__(self, pos, vec):
        super(POSForce, self).__init__(
            self._force_cb_factory(np.reshape(pos, 3), np.reshape(vec, 3))
        )

    @staticmethod
    def _force_cb_factory(pos, vec):
        @neforce
        def _force_callback(link):
            return link.jacobian.transpose() @ np.concatenate(
                (skew(pos) @ link.rotation_global.transpose() @ vec, link.mass * vec)
            )

        return _force_callback


class Gravity(Force):
    def __init__(self, vec):
        super(Gravity, self).__init__(
            self._grav_cb_factory(np.reshape(vec, 3)),
            self._d_grav_cb_factory(np.reshape(vec, 3)),
        )

    @staticmethod
    def _grav_cb_factory(vec):
        @neforce
        def _grav_callback(link: _nbLink):
            return link.jacobian.transpose() @ np.concatenate(
                (
                    skew(link.GAMMA) @ link.rotation_global.transpose() @ vec,
                    link.mass * vec,
                )
            )

        return _grav_callback

    @staticmethod
    def _d_grav_cb_factory(_vec):
        @d_neforce
        def _d_grav_callback(link: _nbLink):

            vec = np.zeros(3)
            vec[0] = _vec[0]
            vec[1] = _vec[1]
            vec[2] = _vec[2]

            temp = np.concatenate(
                (
                    skew(link.GAMMA) @ link.rotation_global.transpose() @ vec,
                    link.mass * vec,
                )
            )

            d_temp = np.concatenate(
                (
                    skew(link.GAMMA)
                    @ t_v_p(link.d_rotation_global.transpose(0, 2, 1), vec),
                    np.zeros((3, link.dof)),
                )
            )

            res = (
                t_v_p(link.d_jacobian.transpose(0, 2, 1), temp)
                + link.jacobian.T @ d_temp
            )

            return -np.concatenate((res, np.zeros((link.dof, link.dof))), axis=1)

        return _d_grav_callback
