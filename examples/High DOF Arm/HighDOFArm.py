import pycbf2
from pycbf2 import PyCBF, force, cbf, Link, LinkType
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


def axang_rotmat(axis, angle):
    a = sym.cos(angle / 2.0)
    b, c, d = -axis * sym.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        (
            (aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)),
            (2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)),
            (2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc),
        )
    )


def cyl_z(h, M, R):
    return M * np.diag(
        [
            (1 / 12) * h**2 + (1 / 4) * R**2,
            (1 / 12) * h**2 + (1 / 4) * R**2,
            1 / 2 * R**2,
        ]
    )


def cyl_y(h, M, R):
    return M * np.diag(
        [
            (1 / 12) * h**2 + (1 / 4) * R**2,
            1 / 2 * R**2,
            (1 / 12) * h**2 + (1 / 4) * R**2,
        ]
    )


def cyl_x(h, M, R):
    return M * np.diag(
        [
            1 / 2 * R**2,
            (1 / 12) * h**2 + (1 / 4) * R**2,
            (1 / 12) * h**2 + (1 / 4) * R**2,
        ]
    )


def _cbf(x):

    X = [1, 0, 0]
    Y = [0, 1, 0]
    Z = [0, 0, 1]

    r1 = [0, 0, 0]
    r2 = ([0, 0, 1],)
    r3 = ([0.5, 0.5, 0],)
    r4 = ([1, 0, 0],)
    r5 = [0, -0.5, 0]
    r6 = [1, 0, 0]
    r7 = [0, 0.5, 0]

    axes = [Z, Y, X, Y, X, Y, X]
    positions = [r1, r2, r3, r4, r5, r6, r7]

    end_effector_position = np.zeros((3, 1), dtype=float)
    rotmat_last = np.eye(3, dtype=float)

    for i in range(1, 7):
        rotmat_last = rotmat_last @ (axang_rotmat(np.array(axes[i - 1]), x[i - 1]))
        end_effector_position += (
            rotmat_last @ np.reshape(positions[i], (3, 1))
        ).astype(float)

    return end_effector_position[2] - 0.5


class Arm(pycbf2.NESystem):
    def __init__(self):
        super(Arm, self).__init__()

        X = [1, 0, 0]
        Y = [0, 1, 0]
        Z = [0, 0, 1]

        r1 = [0, 0, 0]
        r2 = ([0, 0, 1],)
        r3 = ([0.5, 0.5, 0],)
        r4 = ([1, 0, 0],)
        r5 = [0, -0.5, 0]
        r6 = [1, 0, 0]
        r7 = [0, 0.5, 0]

        m = 1

        COMS = [
            [0, 0, 0],
            [0.25, 0.25, 0],
            [0.5, 0, 0],
            [0, -0.25, 0],
            [0.5, 0, 0],
            [0, 0.25, 0],
            [0, 0, 0],
        ]

        JJ_X = cyl_x(1, m, 0.25)
        JJ_Y = cyl_y(1, m, 0.25)
        JJ_Z = cyl_z(1, m, 0.25)

        inertias = [
            10 * np.reshape(mat, (3, 3))
            for mat in [JJ_Z, JJ_Y, JJ_X, JJ_Y, JJ_X, JJ_Y, JJ_X]
        ]

        axes = [Z, Y, X, Y, X, Y, X]
        positions = [r1, r2, r3, r4, r5, r6, r7]

        self.links = []

        for i in range(7):
            link = Link(
                parent=self.links[i - 1] if i else self,
                mass=m,
                center_of_mass=COMS[i],
                inertia_tensor=inertias[i],
                index=i,
                axis=axes[i],
                link_type=LinkType.rotational,
                rotation_local=np.eye(3),
                position=positions[i],
            )

            link.add_force(force.Gravity([0, 0, 0]), force.Friction(1))
            self.links.append(link)

        t, x, xdot = self.cbf_vars()

        class Controller(cbf.ControlFunc):
            def __init__(self):
                self.cbf = 1
                end_effector_position = sym.Matrix(np.zeros(3))
                rotmat_last = sym.Matrix(np.eye(3))
                for i in range(1, 7):
                    rotmat_last = rotmat_last @ sym.simplify(
                        axang_rotmat(np.array(axes[i - 1]), x[i - 1])
                    )
                    end_effector_position += rotmat_last @ np.reshape(
                        positions[i], (3, 1)
                    )
                    end_effector_position = sym.simplify(end_effector_position)

                    # self.cbf *= (end_effector_position[2] - 0.5)

                self.cbf = end_effector_position[2] - 0.5
                self.clf = 0
                self.t = 0

            def uref(self, x, xdot):
                if self.t < 3:
                    GOAL = -np.pi / 2 * np.ones(7)
                elif self.t < 6:
                    GOAL = -np.pi * np.ones(7)#np.zeros(7)
                else:
                    GOAL = np.array([-np.pi, -np.pi, -2*np.pi, -2 * np.pi, -np.pi, -np.pi, -np.pi])

                #GOAL = np.array([-np.pi, -np.pi, 2 * np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 0])
                self.t += (1/60)/4

                return 10 * (GOAL - x - 0.5 * xdot)

            def input_matrix(self, x, xdot):
                return np.diag([2, 2, 1, 2, 1, 1, 1])

        self.controller = Controller()


import time as tm

s = Arm().compile()
t1 = tm.time()
df = PyCBF.simulate(s, np.zeros(7), np.zeros(7), (1/60), (0, 25))
t2 = tm.time()
df.plot(x="t", y=[f"x{i}" for i in range(7)])
plt.title("7 DOF Manipulator Natural Response")
plt.show()
df.plot(x="t", y=[f"u_{i}" for i in range(7)])
plt.title("7 DOF Manipulator CBF Control Input")
plt.show()
df.plot(x="t", y="cbf")
plt.title("7 DOF Manipulator CBF Value ")
plt.show()
print(t2 - t1)

df.to_csv("SawyerCBFOutput.csv")
