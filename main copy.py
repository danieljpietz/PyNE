import numpy as np

from NE import *
from matplotlib import pyplot as plt

class System(NESystem):
    def __init__(self, x, xdot):
        super(System, self).__init__()

        self.link1 = Link(
            parent=self,
            mass=1,
            COM=np.array([[0], [1], [0]]),
            inertia=np.eye(3),
            rotation=rotx(x[0]),
            position=np.array([[0], [1], [0]]),
            RMap=sparse((3, len(x)), (0, 0)),
            PMap=np.zeros((3, len(x))),
        )

        self.link2 = Link(
            parent=self.link1,
            mass=1,
            COM=np.array([[0], [1], [0]]),
            inertia=np.eye(3),
            rotation=rotx(x[1]),
            position=np.array([[0], [1], [0]]),
            RMap=sparse((3, len(x)), (0, 1)),
            PMap=np.zeros((3, len(x))),
        )

        self.link1.add_force(Gravity(np.array([[0], [0], [-9.81]])))
        self.link2.add_force(Gravity(np.array([[0], [0], [-9.81]])))

        self.link1.add_force(ViscousFriction(1))
        self.link2.add_force(ViscousFriction(1))


x = RungeKutta(System()).dsolve([0, 0], [0, 0], (0, 10)).df()

x.plot(x='time', y=['x0', 'x1'])
plt.show()

