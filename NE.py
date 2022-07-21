import pandas as pd
from functions import *
import matplotlib.pyplot as plt
import threading

class _Link(object):
    def __init__(self, mass, COM, inertia):
        self.mass = mass
        self.COM = np.reshape(COM, (3, 1))
        self.inertia = inertia
        self.first_mass_moment = mass * self.COM
        self.first_mass_moment_skew = skew(self.first_mass_moment)
        self.mass_ident = np.diag(3 * [mass])
        self.children = []
        self.forces = []

    def add_child(self, child):
        self.children.append(child)

    def add_force(self, force):
        force.add_link(self)
        self.forces.append(force)

    def update(self, CBF=False):
        self.kinematics()

        self.H = np.zeros((self.dof, self.dof))
        self.d_H = np.zeros(((self.dof, self.dof, self.dof)))

        self.d = np.zeros((self.dof, 1))
        self.d_d = np.zeros((2*self.dof, self.dof, 1))

        self.F = np.zeros((len(self.x), 1))

        for child in self.children:
            child.update(CBF)
            self.H += child.H.astype(float)
            self.d += child.d.astype(float)
            if CBF:
                self.d_H += child.d_H.astype(float)
                self.d_d += child.d_d.astype(float)
            self.F += child.F.astype(float)

    def dynamics(self):

        self.m_corner_elem = self.first_mass_moment_skew @ self.rotation_global.transpose()

        self.M = np.block(
            [
                [self.inertia, self.m_corner_elem],
                [self.m_corner_elem.transpose(), self.mass_ident],
            ]
        )

        self.jtm = self.Jacobian.transpose() @ self.M

        self.H = self.jtm @ self.Jacobian

        self.omega_global_squared = self.omega_global_skewed @ self.omega_global_skewed

        self.dPrime = np.block(
            [
                [self.omega_global_skewed @ self.inertia @ self.omega_global],
                [self.mass * self.rotation_global @ self.omega_global_squared @ self.COM],
            ]
        )

        self.dot_jacob_dot_x_product = self.JacobianDot @ self.xdot
        self.d = self.jtm @ self.dot_jacob_dot_x_product + self.Jacobian.transpose() @ self.dPrime

        F = np.zeros((6, 1))
        fJoint = np.zeros((len(self.d), 1))

        for f in self.forces:
            if isinstance(f, JointForce):
                fJoint += f.get()
            else:
                pos, vec = f.get()
                F += np.block([[skew(pos) @ self.rotation_global.transpose() @ vec],
                               [self.mass * vec]]).astype(float)

        self.F = self.Jacobian.transpose() @ F + fJoint

    def differential_dynamics(self):
        d_m_corner_elem = (
                self.first_mass_moment_skew @ self.d_rotation_global.transpose((0, 2, 1))
        )

        dM = np.block(
            [
                [np.zeros((self.dof, 3, 3)), d_m_corner_elem],
                [d_m_corner_elem.transpose((0, 2, 1)), np.zeros((self.dof, 3, 3))],
            ]
        )

        d_jtm = (
                self.d_Jacobian.transpose((0, 2, 1)) @ self.M
                + self.Jacobian.transpose() @ dM
        )

        self.d_H = d_jtm @ self.Jacobian + self.jtm @ self.d_Jacobian

        d_omega_global = np.concatenate(
            [np.zeros((3, self.dof)), self.Jacobian[0:3]], 1
        )

        inertia_omega_product = self.inertia @ self.omega_global
        d_inertia_omega_product = self.inertia @ d_omega_global
        d_inertia_omega_product = d_inertia_omega_product[np.newaxis, ...].swapaxes(
            2, 0
        )

        d_skew_omega_global = skew3(d_omega_global)
        skew_omega_global = skew(self.omega_global)

        d_Jacobian = np.concatenate(
            [self.d_Jacobian, np.zeros((self.dof, 6, self.dof))]
        )
        d_jtm = np.concatenate([d_jtm, np.zeros((self.dof, self.dof, 6))])
        d_rotation_global = np.concatenate(
            [self.d_rotation_global, np.zeros((self.dof, 3, 3))]
        )
        dPrime = np.block(
            [
                [skew(self.omega_global) @ self.inertia @ self.omega_global],
                [self.rotation_global @ self.omega_global_squared @ self.first_mass_moment],
            ]
        )

        d_dPrime1 = (
                d_skew_omega_global @ inertia_omega_product
                + skew_omega_global @ d_inertia_omega_product
        )
        d_dPrime2 = (
                            d_rotation_global @ self.omega_global_squared
                            + 2 * self.rotation_global @ skew_omega_global @ d_skew_omega_global
                    ) @ self.first_mass_moment

        d_dPrime = np.block([[d_dPrime1], [d_dPrime2]])

        dot_jacob_dot_x_product = self.JacobianDot @ self.xdot
        d_dot_jacob_dot_product = self.JacobianDot @ self.xdot + (
                    self.JacobianDot @ np.block([[np.zeros((self.dof, self.dof))], [np.eye(self.dof)]]).transpose())[
            np.newaxis, ...].swapaxes(2, 0)

        self.d = self.jtm @ dot_jacob_dot_x_product + self.Jacobian.transpose() @ self.dPrime
        self.d_d = (
                d_jtm @ dot_jacob_dot_x_product
                + self.jtm @ d_dot_jacob_dot_product
                + d_Jacobian.transpose((0, 2, 1)) @ dPrime
                + self.Jacobian.transpose() @ d_dPrime
        )


class NESystem(_Link):
    def __init__(self):
        super(NESystem, self).__init__(mass=0, COM=np.array([[0], [0], [0]]), inertia=np.eye(0))
        self.position = np.array([[0], [0], [0]])
        self.rotation_global = np.eye(3)
        self.dof = len(self.x)
        self.Jacobian = np.zeros((6, self.dof))
        self.d_Jacobian = np.zeros((self.dof, 6, self.dof))
        self.JacobianDot = np.zeros((6, self.dof))
        self.d_JacobianDot = np.zeros((2 * self.dof, 6, self.dof))
        self.d_rotation_global = np.zeros((3, 3, self.dof))
        self.d_omega_global = np.zeros((3, self.dof))
        self.d_skew_omega_global = skew3(self.d_omega_global)
        self.skew_omega_global = skew(np.zeros(3))

    def __init_subclass__(cls, **kwargs):
        cls.___init___ = cls.__init__

        def __init_wrapper__(self):
            pass

        cls.__init__ = __init_wrapper__

    def kinematics(self):
        self.x = np.reshape(self.x, (len(self.x), 1))
        self.xdot = np.reshape(self.xdot, (len(self.xdot), 1))
        temp = self.Jacobian @ self.xdot
        self.omega_global = temp[0:3]
        self.omega_global_skewed = skew(self.omega_global)
        self.velocity = temp[4:6]

    def eval(self, x, xdot):
        self.x = np.reshape(x, (len(x), 1)).astype(float)
        self.xdot = np.reshape(xdot, (len(x), 1)).astype(float)
        self.___init___(x, xdot)
        self.update()
        return np.concatenate([xdot, np.linalg.solve(self.H, self.F - self.d)])

    def __call__(self, x, xdot):
        return self.eval(x, xdot)


class Link(_Link):
    def __init__(self, parent, mass, COM, inertia, position, rotation, RMap=None, PMap=None):
        super(Link, self).__init__(mass, COM, inertia)
        self.dof = len(parent.x)
        if RMap is None:
            self.ITilde = np.reshape(PMap, (3, self.dof))
            self.IHat = np.zeros((self.ITilde.shape))
        elif PMap is None:
            self.IHat = np.reshape(RMap, (3, self.dof))
            self.ITilde = np.zeros((self.IHat.shape))
        self.position = np.reshape(position, (3, 1))
        self.rotation = np.reshape(rotation, (3, 3))
        self.parent = parent
        self.parent.add_child(self)

        self.x = np.reshape(parent.x, (self.dof, 1))
        self.xdot = np.reshape(parent.xdot, (self.dof, 1))

    def update(self, CBF):

        self.kinematics()
        self.dynamics()
        
        if CBF:
            self.differential_kinematics()
            self.differential_dynamics()

        for child in self.children:
            child.update(CBF)
            self.H += child.H.astype(float)
            self.d += child.d.astype(float)

            if CBF:
                self.d_H += child.d_H.astype(float)
                self.d_d += child.d_d.astype(float)

            self.F += child.F.astype(float)

    def kinematics(self):
        #
        # We are going to use a lot of skewed vectors and transposed matrices. We can save a little bit of time by
        # precomputing these values
        #

        self.rotation_global = (self.parent.rotation_global @ self.rotation)

        self.omega_local = self.IHat @ self.xdot
        self.omega_global = self.parent.omega_global + self.rotation @ self.omega_local
        self.velocity = self.ITilde @ self.xdot

        self.rotation_local_transpose = self.rotation.transpose()
        self.position_local_skewed = skew(self.position)
        self.velocity_local_skewed = skew(self.velocity)

        self.omega_local_skewed = skew(self.omega_local)
        self.omega_global_skewed = skew(self.omega_global)

        self.JacobianPrime = np.block(
            [
                [self.rotation_local_transpose, np.zeros((3, 3))],
                [
                    self.parent.rotation_global @ self.position_local_skewed.transpose(),
                    np.eye(3),
                ],
            ]
        )

        self.Jacobian = self.JacobianPrime @ self.parent.Jacobian + np.block(
            [[self.IHat], [self.parent.rotation_global @ self.ITilde]]
        )

        self.JacobianStar = np.block(
            [
                [
                    self.omega_local_skewed @ self.parent.rotation_global,
                    np.zeros((3, 3)),
                ],
                [
                    -self.parent.rotation_global
                    @ (
                            self.parent.omega_global_skewed @ self.position_local_skewed
                            + self.velocity_local_skewed
                    ),
                    np.zeros((3, 3)),
                ],
            ]
        )

        self.dotJBPrime11 = self.omega_local_skewed.transpose() @ self.rotation.transpose()

        self.dotJBPrime21Inner = (
                self.parent.omega_global_skewed @ self.position_local_skewed + self.velocity_local_skewed
        )

        self.dotJBPrime21 = -(self.parent.rotation_global @ self.dotJBPrime21Inner)

        self.dotJBPrime = np.block(
            [[self.dotJBPrime11, np.zeros((3, 3))], [self.dotJBPrime21, np.zeros((3, 3))]]
        )

        self.dotJBStar = np.block(
            [
                [np.zeros((3, self.dof))],
                [self.parent.rotation_global @ self.omega_local_skewed @ self.ITilde],
            ]
        )

        self.JacobianDot = (
                self.dotJBPrime @ self.parent.Jacobian + self.JacobianPrime @ self.parent.JacobianDot + self.dotJBStar
        )

        self.JacobianDot = (
                self.JacobianStar @ self.parent.Jacobian
                + self.JacobianPrime @ self.parent.JacobianDot
                + np.block(
            [
                [np.zeros((3, self.dof))],
                [
                    self.parent.rotation_global
                    @ self.omega_local_skewed
                    @ self.ITilde
                ],
            ]
        )
        )

    def differential_kinematics(self):

        parent = self.parent

        self.d_rotation = skew3(self.IHat) @ self.rotation
        self.d_rotation_global = (
                parent.d_rotation_global @ self.rotation
                + parent.rotation_global @ self.d_rotation
        )

        self.d_skew_position_local = skew3(self.ITilde)

        self.d_velocity_local_skewed = np.block(
            [
                [[np.zeros((self.dof, 3, 3))]],
                [[skew3(self.ITilde)]],
            ]
        )

        self.d_omega_local = self.IHat
        d_skew_omega_local = np.block(
            [[[np.zeros((self.dof, 3, 3))]], [[skew3(self.d_omega_local)]]]
        )

        d_skew_omega_local_t = d_skew_omega_local.transpose((0, 2, 1))

        self.d_omega_global = (
                parent.d_omega_global[:, :self.dof]
                + np.sum(parent.d_rotation_global @ self.omega_local, axis=0)
                + parent.rotation_global @ self.d_omega_local
        )

        self.d_omega_global = np.block([np.zeros((3, self.dof)), self.d_omega_global])
        self.d_skew_omega_global = skew3(self.d_omega_global)

        d_JBStar = (
                parent.d_rotation_global @ self.position_local_skewed
                + parent.rotation_global @ self.d_skew_position_local
        )

        d_JBPrime = np.block(
            [
                [self.d_rotation.transpose((0, 2, 1)), np.zeros((self.dof, 3, 3))],
                [d_JBStar, np.zeros((self.dof, 3, 3))],
            ]
        )

        d_JBVec = np.block(
            [
                [np.zeros((self.dof, 3, self.dof))],
                [parent.d_rotation_global @ self.ITilde],
            ]
        )

        self.d_Jacobian = (
                d_JBPrime @ parent.Jacobian + self.JacobianPrime @ parent.d_Jacobian[:self.dof] + d_JBVec
        )

        ## Below this line is the full derivatives that depend on xdot as well as x

        d_rotation_local_t = np.block(
            [[[self.d_rotation.transpose((0, 2, 1))]], [[np.zeros((self.dof, 3, 3))]]]
        )

        d_skew_position_local = np.block(
            [[[self.d_skew_position_local]], [[np.zeros((self.dof, 3, 3))]]]
        )
        parent_d_rotation_global_full = np.block(
            [[[parent.d_rotation_global]], [[np.zeros((self.dof, 3, 3))]]]
        )
        parent_d_jacobian_full = np.block(
            [[[parent.d_Jacobian]], [[np.zeros((self.dof, 6, self.dof))]]]
        )

        d_JBPrime = np.block([[[np.zeros((self.dof, 6, 6))]], [[d_JBPrime]]])

        d_dotJBPrime11 = (
                d_skew_omega_local_t @ self.rotation.transpose()
                + self.omega_local_skewed.transpose() @ d_rotation_local_t
        )

        d_dotJBPrime21Inner = (
                self.d_skew_omega_global @ self.position_local_skewed
                + parent.omega_global_skewed @ d_skew_position_local
                + self.velocity_local_skewed
        )

        d_dotJBPrime21 = -(
                parent_d_rotation_global_full @ self.dotJBPrime21Inner
                + parent.rotation_global @ d_dotJBPrime21Inner
        )

        d_dotJBPrime = np.block(
            [
                [d_dotJBPrime11, np.zeros((2 * self.dof, 3, 3))],
                [d_dotJBPrime21, np.zeros((2 * self.dof, 3, 3))],
            ]
        )

        d_dotJBStar = np.block(
            [
                [np.zeros((2 * self.dof, 3, self.dof))],
                [
                    (
                            parent_d_rotation_global_full @ self.omega_local_skewed
                            + parent.rotation_global @ d_skew_omega_local
                    )
                    @ self.ITilde
                ],
            ]
        )

        self.d_JacobianDot = (
                d_dotJBPrime @ parent.Jacobian
                + self.dotJBPrime @ parent_d_jacobian_full
                + d_JBPrime @ parent.JacobianDot
                + self.JacobianPrime @ parent.d_JacobianDot
                + d_dotJBStar
        )

class Force(object):

    def __init__(self, vector, location):
        self.link = None
        self.vector = np.reshape(vector, (3, 1))
        self.location = location

    def add_link(self, link):
        self.link = link


class COMForce(Force):
    def __init__(self, vector):
        super(COMForce, self).__init__(vector, None)

    def get(self):
        return self.link.COM, self.vector * self.link.mass


class JointForce(Force):
    def __init__(self, x=None):
        self.x = x
        self.M = None

    def add_link(self, link):
        super(JointForce, self).add_link(link)
        M = self.link.IHat + self.link.ITilde
        self.M = M[M.nonzero()[0]]

    def get(self):
        return self.M @ self.x


class Control1DOF(JointForce):
    def __init__(self, x):
        super(Control1DOF, self).__init__(x)

    def add_link(self, link):
        super(Control1DOF, self).add_link(link)
        self.x = np.reshape(link.dof * [self.x], (link.dof, 1))


class Gravity(COMForce):
    def __init__(self, vector):
        super(Gravity, self).__init__(vector)


class Friction(JointForce):
    pass


class ViscousFriction(Friction):

    def __init__(self, coefficient):
        self.link = None
        self.coefficient = coefficient

    def get(self):
        self.x = -self.coefficient * self.link.xdot
        return super(ViscousFriction, self).get()


class Solver(object):
    class Solution(object):
        def __init__(self, t, x, xdot):
            self.t = t
            self.x = x
            self.xdot = xdot

        def x_df(self):
            pd.DataFrame({**{'time': self.t}, **{key: values for key, values in
                                                 zip([f"x{i}" for i in range(len(self.x[0]))], self.x.transpose())},
                          **{key: values for key, values in
                             zip([f"dx{i}" for i in range(len(self.x[0]))], self.xdot.transpose())}}
                         )

        def df(self):
            return pd.DataFrame({**{'time': self.t.flatten()}, **{key: values for key, values in
                                                                  zip([f"x{i}" for i in range(len(self.x[0]))],
                                                                      self.x.transpose())},
                                 **{key: values for key, values in
                                    zip([f"dx{i}" for i in range(len(self.x[0]))], self.xdot.transpose())}}
                                )

    def __init__(self, system):
        self.system = system
        self.tX = []
        self.tXDot = []

    def solve(self, x0, dx0, tRange, tStep):
        timestamp = np.arange(*tRange, tStep)
        self.tX = np.zeros((len(timestamp), len(x0)))
        self.tXDot = np.zeros((len(timestamp), len(x0)))
        self.tX[0] = x0
        self.tXDot[0] = dx0
        for i, t in enumerate(timestamp[1:]):
            (self.tX[i + 1], self.tXDot[i + 1]) = (x.flatten().astype(float) for x in
                                                   self.step(tStep, self.tX[i], self.tXDot[i]))

        return self.Solution(timestamp, self.tX, self.tXDot)

    def dsolve(self, x0, dx0, tRange, tError=0.1, n_default=1000, ):
        timestamp = [min(tRange)]
        self.tX = [np.reshape(x0, (len(x0), 1))]
        self.tXDot = [np.reshape(dx0, (len(x0), 1))]

        tStepDefault = (max(tRange) - min(tRange)) / n_default
        tStep = tStepDefault

        t = tRange[0]

        while t < tRange[1]:
            step = self.step(tStep, self.tX[-1], self.tXDot[-1])
            self.tX.append(self.tX[-1] + step[0])
            self.tXDot.append(self.tXDot[-1] + step[1])
            stepDivisor = np.linalg.norm(np.concatenate([*step]), 1)
            if stepDivisor != 0:
                tStep = max(0.25 * tStep / np.linalg.norm(np.concatenate([*step]), 1), 5 * tStepDefault)
            else:
                tStep = tStepDefault
            t = t + tStep
            timestamp.append(t)

        return self.Solution(np.reshape(timestamp, (len(timestamp), 1)),
                             np.reshape(self.tX, (len(self.tX), len(x0))),
                             np.reshape(self.tXDot, (len(self.tX), len(x0))))

    def step(self):
        raise

    pass


class RungeKutta(Solver):
    def __init__(self, system):
        super(RungeKutta, self).__init__(system)

    def step(self, h, x=None, xdot=None):
        if x is None and xdot is None:
            (x, xdot) = (self.system.x, self.system.xdot)
        x = np.reshape(x, (len(x), 1)).astype(float)
        xdot = np.reshape(xdot, (len(x), 1)).astype(float)
        k1 = self.system.eval(x, xdot)
        k2 = self.system.eval(x + h * (k1[:len(x)] / 2), xdot + h * (k1[len(x):] / 2))
        k3 = self.system.eval(x + h * (k2[:len(x)] / 2), xdot + h * (k2[len(x):] / 2))
        k4 = self.system.eval(x + h * k3[:len(x)], xdot + h * k3[len(x):])
        step = (h / 6) * (k1 + 2 * (k2 + k3) + k4)

        return (step[:len(x)], step[len(x):])

    pass
