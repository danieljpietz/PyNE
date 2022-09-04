import numba as nb
import numpy as np
from .func import skew
from .type import _nbLink

force_function_spec = nb.float64[:](_nbLink.class_type.instance_type)
joint_force_function_spec = nb.float64(_nbLink.class_type.instance_type)


def neforce(func):
    return nb.njit(force_function_spec)(func)


def jointforce(func):
    return nb.njit(joint_force_function_spec)(func)


class Force:
    def __init__(self, callback):
        self.callback = callback

    def _jit(self):
        return self.callback


class JointForce(Force):
    def __init__(self, callback):
        super(JointForce, self).__init__(self._joint_force_wrapper(callback))

    @staticmethod
    def _joint_force_wrapper(_callback):
        @neforce
        def _joint_force(link):
            retVal = np.zeros(link.dof)
            retVal[link.index] = _callback(link)
            return retVal

        return _joint_force


class Friction(JointForce):
    def __init__(self, coeff):
        super(Friction, self).__init__(self._friction_cb_factory(coeff))

    @staticmethod
    def _friction_cb_factory(coeff):
        @jointforce
        def _friction_callback(link):
            return -link.xdot * coeff

        return _friction_callback


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
        super(Gravity, self).__init__(self._grav_cb_factory(np.reshape(vec, 3)))

    @staticmethod
    def _grav_cb_factory(vec):
        @nb.njit(force_function_spec)
        def _grav_callback(link):
            return link.jacobian.transpose() @ np.concatenate(
                (
                    skew(link.COM) @ link.rotation_global.transpose() @ vec,
                    link.mass * vec,
                )
            )

        return _grav_callback
