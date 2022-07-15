from NE import *


class System(Root):
    def __init__(self, x, xdot):
        super(System, self).__init__(
            mass=1, COM=np.array([[0], [0], [0]]), inertia=np.eye(3)
        )

        self.position = np.array([[0], [x[0]], [0]])
        self.rotation_global = np.eye(3)
        self.Jacobian = sparse((6, len(x)), (4, 0))
        self.JacobianDot = np.zeros((6, len(x)))

        self.link1 = Link(
            parent=self,
            mass=1,
            COM=np.array([[0], [1], [0]]),
            inertia=np.eye(3),
            rotation=rotx(x[1]),
            position=np.array([[0], [1], [0]]),
            IHat=sparse((3, len(x)), (0, 1)),
            ITilde=np.zeros((3, len(x))),
        )

        self.link2 = Link(
            parent=self.link1,
            mass=1,
            COM=np.array([[0], [1], [0]]),
            inertia=np.eye(3),
            rotation=rotx(x[2]),
            position=np.array([[0], [1], [0]]),
            IHat=sparse((3, len(x)), (0, 2)),
            ITilde=np.zeros((3, len(x))),
        )

        self.link1.add_force(Gravity(np.array([[0], [0], [-1]])))
        self.link2.add_force(Gravity(np.array([[0], [0], [-1]])))

        self.link1.add_force(ViscousFriction(1))
        self.link2.add_force(ViscousFriction(1))


x = RungeKutta(System()).solve([0, 0, 0], [0, 0, 0], (0, 10), 0.01).df()

