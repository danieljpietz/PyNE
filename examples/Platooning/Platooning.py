import pycbf2
from pycbf2 import PyCBF, force, cbf, Link, LinkType
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


class Platoon(pycbf2.NESystem):
    def __init__(self, length, t_safety, v_goal):
        super(Platoon, self).__init__()

        for i in range(length):
            vehicle = Link(
                parent=self,
                mass=1,
                center_of_mass=[0, 0, 0],
                inertia_tensor=np.eye(3),
                index=i,
                axis=[1, 0, 0],
                link_type=LinkType.prismatic,
                rotation_local=np.eye(3),
                position=[0, 0, 0],
            )

        t, x, xdot = self.cbf_vars()

        class Controller(cbf.ControlFunc):
            def __init__(self):
                self.cbf = 1
                for i in range(length - 1):
                    self.cbf *= (x[i + 1] - x[i]) - t_safety * xdot[i]
                #self.cbf = 1
                self.t = 0

                self.clf = 0

            def input_matrix(self, x, xdot):
                return np.eye(length)

            def uref(self, x, xdot):
                return v_goal - xdot

        self.controller = Controller()


import time as tm

length = 4
t_safe = 1
v_goal = 10

s = Platoon(length, t_safe, v_goal).compile()
t1 = tm.time()
df = PyCBF.simulate(s, list(range(length)), np.zeros(length), (1/60), (0, 20))
t2 = tm.time()
print(f"First sim took {t2 - t1} seconds")
df.plot(x="t", y=[f"x{i}" for i in range(length)])
plt.title("Platoon Positions")
plt.savefig("Platoon Figures/NoPositions")
plt.show()
df.plot(x="t", y=[f"xdot{i}" for i in range(length)])
plt.title("Platoon Velocities")
plt.savefig("Platoon Figures/NoVelocities")
plt.show()
df.plot(x="t", y=["cbf"])
plt.title("Platoon CBF")
plt.savefig("Platoon Figures/NoCBF")
plt.show()
