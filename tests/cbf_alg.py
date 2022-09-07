import pytest


def test_cbf():
    import pycbf2.func
    import numpy as np
    from pycbf2.type import NESystem, Link, LinkType
    from pycbf2.force import Gravity, Friction

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

    s = MySystem().compile()

    from pycbf2.CBFAlgorithm import differential_kinematics, differential_dynamics
    from pycbf2.NEAlgorithm import recursive_kinematics

    links = s[1]
    l2 = links[1].properties

    differential_dynamics(l2, np.zeros(2))
