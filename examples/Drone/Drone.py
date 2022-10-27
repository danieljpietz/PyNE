import pycbf2
from pycbf2 import PyCBF, force, cbf, Link, LinkType
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


class Drone(pycbf2.NESystem):
    def __init__(self):
        super(Drone, self).__init__()

        self.drone_x = Link(
            parent=self,
            mass=0,
            center_of_mass=[0, 0, 0],
            inertia_tensor=np.eye(3),
            index=0,
            axis=[1, 0, 0],
            link_type=LinkType.prismatic,
            rotation_local=np.eye(3),
            position=[0, 0, 0],
        )

        self.drone_y = Link(
            parent=self.drone_x,
            mass=1,
            center_of_mass=[0, 0, 0],
            inertia_tensor=np.eye(3),
            index=1,
            axis=[0, 1, 0],
            link_type=LinkType.prismatic,
            rotation_local=np.eye(3),
            position=[0, 0, 0],
        )

        self.drone_rotation = Link(
            parent=self.drone_y,
            mass=0,
            center_of_mass=[0, 0, 0],
            inertia_tensor=np.eye(3),
            index=2,
            axis=[0, 0, 1],
            link_type=LinkType.rotational,
            rotation_local=np.eye(3),
            position=[0, 0, 0],
        )

        self.drone_y.add_force(force.Gravity([0, -9.81, 0]))

        t, x, xdot = self.cbf_vars()

        class Controller(cbf.ControlFunc):
            def __init__(self):
                self.cbf = ((sym.sin(x[0]) + 0.5) - x[1]) * (
                    x[1] - (sym.sin(x[0]) - 0.5)
                )
                self.clf = 0  # (x[1] - 10)**2 + (x[0] - 10)**2
                self.error = 0

            def input_matrix(self, x, xdot):
                theta = x[2]
                return np.array(
                    [
                        [-np.sin(theta), -np.sin(theta)],
                        [np.cos(theta), np.cos(theta)],
                        [-1, 1],
                    ]
                )

            def uref(self, x, xdot):
                xGoal = 5 * np.pi / 2
                yGoal = np.sin(x[0])
                xdotGoal = (xGoal - x[0]) - 1.5 * xdot[0]
                height_forces = (
                    np.array([9.81, 9.81]) / (2 * np.cos(x[2]))
                    + (yGoal - np.array([x[1], x[1]]))
                    - 3 * xdot[1]
                )
                lat_forces = np.zeros(2)
                lat_forces[1] = np.arctan(xdot[0] - xdotGoal) - 5 * x[2] - 5 * xdot[2]
                lat_forces[0] = -lat_forces[1]
                return height_forces + lat_forces

        self.controller = Controller()


import time as tm

s = Drone().compile()
t1 = tm.time()
df = PyCBF.simulate(s, [0, 0, 0], [0, 0, 0], 0.05, (0, 20))
t2 = tm.time()
print(f"First sim took {t2 - t1} seconds")
df.plot(x="t", y=["x0", "x1", "x2"])
plt.show()
df.plot(x="t", y=["x0"])
plt.show()
df.plot(x="t", y=["u_0", "u_1"])
plt.show()
plt.plot(df.x0, df.x1, color="Black", label="Drone Position")
plt.plot(
    df.x0, df.x1 + 0.5, color="green", linestyle="dashed", label="0.5m Safety Margin"
)
plt.plot(df.x0, df.x1 - 0.5, color="green", linestyle="dashed")
x = np.arange(-np.pi, 3 * np.pi, 0.01)
cbf1 = np.sin(x) + 1
cbf2 = np.sin(x) - 1
plt.plot(x, cbf1, color="red", linewidth=2, label="Obstacle")
plt.plot(x, cbf2, color="red", linewidth=2)
plt.title("Drone Position")
plt.legend()
# plt.savefig('Drone Figures/DronePD.png')
plt.show()
plt.plot(df.t, ((np.sin(df.x0) + 0.5) - df.x1) * (df.x1 - (np.sin(df.x0) - 0.5)))
plt.title("Drone CBF Value")
plt.savefig("Drone Figures/DroneCBFValue.png")
plt.show()
