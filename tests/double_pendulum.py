import pytest


def test_double_pendulum():
    from pycbf2 import NESystem, Link, LinkType
    from pycbf2.force import JointForce, Friction, Gravity, jointforce
    import numpy as np
    from pycbf2.PyCBF import simulate
    from pycbf2 import cbf
    import sympy as sym

    GOAL = [np.pi / 2, 0]
    SUCCESS_THRESHOLD = 0.01

    class MySystem(NESystem):
        def __init__(self):
            super(MySystem, self).__init__()

            self.link1 = Link(
                parent=self,
                mass=1,
                center_of_mass=[0, 1, 0],
                inertia_tensor=np.eye(3),
                index=0,
                axis=[1, 0, 0],
                link_type=LinkType.rotational,
                rotation_local=np.eye(3),
                position=[0, 0, 0],
            )

            self.link2 = Link(
                parent=self.link1,
                mass=1,
                center_of_mass=[0, 1, 0],
                inertia_tensor=np.eye(3),
                index=1,
                axis=[1, 0, 0],
                link_type=LinkType.rotational,
                rotation_local=np.eye(3),
                position=[0, 1, 0],
            )

            for link in self.links():
                link.add_force(Gravity([0, 0, -9.81]), Friction(0.5))

            t, x, xdot = cbf.cbf_vars(2)
            self.cbf = cbf.compile_symbolic(sym.sin(x[0]), t, x, xdot)

    s = MySystem().compile()

    df = simulate(s, [1, 1], [1, 1], 0.1, (0, 30))

    error = df.iloc[-1].drop("t") - [GOAL[0], GOAL[1], 0, 0]

    assert np.linalg.norm(error) < SUCCESS_THRESHOLD
