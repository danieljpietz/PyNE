from __future__ import annotations
import numba as nb
import numpy as np
import enum
from typing import Union
from .func import skew, skew3

_nbLinkSpec = (
    ("link_type", nb.int64),
    ("dof", nb.int64),
    ("index", nb.int64),
    ("axis", nb.float64[:]),
    ("x", nb.float64),
    ("xdot", nb.float64),
    ("mass", nb.float64),
    ("COM", nb.float64[:]),
    ("GAMMA", nb.float64[:]),
    ("inertia_tensor", nb.float64[:, :]),
    ("IHat", nb.float64[:, :]),
    ("ITilde", nb.float64[:, :]),
    ("rotation_local", nb.float64[:, :]),
    ("rotation_offset", nb.float64[:, :]),
    ("rotation_global", nb.float64[:, :]),
    ("angular_velocity_local", nb.float64[:]),
    ("angular_velocity_global", nb.float64[:]),
    ("position", nb.float64[:]),
    ("position_local_skewed", nb.float64[:, :]),
    ("linear_velocity", nb.float64[:]),
    ("jacobian", nb.float64[:, :]),
    ("JNPrime", nb.float64[:, :]),
    ("dotJNPrime", nb.float64[:, :]),
    ("dotJacobian", nb.float64[:, :]),
    ("H", nb.float64[:, :]),
    ("M", nb.float64[:, :]),
    ("d", nb.float64[:]),
    ("F", nb.float64[:]),
    ("d_rotation_local", nb.float64[:, :, :]),
    ("d_skew_position_local", nb.float64[:, :, :]),
    ("d_rotation_global", nb.float64[:, :, :]),
    ("d_angular_velocity_global", nb.float64[:, :]),
    ("d_angular_velocity_global_skewed", nb.float64[:, :, :]),
    ("d_jacobian", nb.float64[:, :, :]),
    ("d_JNPrime", nb.float64[:, :, :]),
    ("d_dotJNPrime", nb.float64[:, :, :]),
    ("d_dotJacobian", nb.float64[:, :, :]),
    ("d_H", nb.float64[:, :, :]),
    ("d_M", nb.float64[:, :, :]),
    ("d_d", nb.float64[:, :]),
)


@nb.experimental.jitclass(_nbLinkSpec)
class _nbLink:
    def __init__(
        self,
        dof,
        link_type,
        index,
        axis,
        mass,
        GAMMA,
        inertia_tensor,
        position,
        rotation_local,
        IHat,
        ITilde,
    ):
        self.dof = dof
        self.link_type = link_type
        self.index = index
        self.axis = axis
        self.x = 0
        self.xdot = 0
        self.mass = mass
        self.GAMMA = GAMMA
        self.COM = mass * GAMMA
        self.inertia_tensor = inertia_tensor
        self.IHat = IHat
        self.ITilde = ITilde
        self.rotation_local = rotation_local
        self.rotation_offset = rotation_local
        self.rotation_global = np.eye(3)
        self.angular_velocity_local = np.zeros(3)
        self.angular_velocity_global = np.zeros(3)
        self.position = position
        self.position_local_skewed = skew(position)
        self.linear_velocity = np.zeros(3)
        self.jacobian = np.zeros((6, self.dof))
        self.JNPrime = np.zeros((6, 6))
        self.dotJNPrime = np.zeros((6, 6))
        self.dotJacobian = np.zeros((6, self.dof))

        self.H = np.zeros((self.dof, self.dof))
        self.M = np.zeros((6, 6))
        self.d = np.zeros(self.dof)
        self.F = np.zeros(self.dof)

        """
            DIFFERENTIAL PROPERITES 
        """

        self.d_skew_position_local = skew3(self.ITilde)

        self.d_rotation_local = np.zeros((self.dof, 3, 3))
        self.d_rotation_global = np.zeros((self.dof, 3, 3))

        self.d_angular_velocity_global = np.zeros((3, 2 * self.dof))
        self.d_angular_velocity_global_skewed = np.zeros((2 * self.dof, 3, 3))

        self.d_jacobian = np.zeros((self.dof, 6, self.dof))
        self.d_JNPrime = np.zeros((self.dof, 6, self.dof))
        self.d_dotJNPrime = np.zeros((2 * self.dof, 6, self.dof))
        self.d_dotJacobian = np.zeros((2 * self.dof, 6, self.dof))

        self.d_H = np.zeros((self.dof, self.dof, self.dof))
        self.d_M = np.zeros((self.dof, 6, 6))
        self.d_d = np.zeros((self.dof, 2 * self.dof))


link_deferred = nb.deferred_type()


@nb.experimental.jitclass(
    (
        ("parent", nb.optional(link_deferred)),
        ("properties", _nbLink.class_type.instance_type),
    ),
)
class nbLink:
    def __init__(
        self,
        parent,
        dof,
        link_type,
        index,
        axis,
        mass,
        GAMMA,
        inertia_tensor,
        position,
        rotation_local,
        IHat,
        ITilde,
    ):
        self.parent = parent
        self.properties = _nbLink(
            dof,
            link_type,
            index,
            axis,
            mass,
            GAMMA,
            inertia_tensor,
            position,
            rotation_local,
            IHat,
            ITilde,
        )


link_deferred.define(nbLink.class_type.instance_type)


class NESystem(object):
    def __init__(self):
        self.__root = Link(
            parent=None,
            mass=0,
            center_of_mass=[0, 0, 0],
            inertia_tensor=np.zeros((3, 3)),
            link_type=LinkType._root,
            axis=[0, 0, 0],
            index=0,
        )

        self.dof = None
        self.cbf = None

    def __init_subclass__(cls, **kwargs):
        __init__old__ = cls.__init__

        def __init_wrapper__(instance):
            cls.__init__ = __init__old__
            instance.__init__()
            instance._post_init()

        cls.__init__ = __init_wrapper__

    def _post_init(self):
        self.__root.dof = self.get_dof()
        self.__root._assign_dof_branch()
        self.dof = self.__root.dof

    def _add_child(self, child):
        return self.__root._add_child(child)

    def get_dof(self):
        return self.__root._get_local_dof() - 1

    def get_links_flattened(self):
        return self.__root.get_links_flattened()

    def links(self):
        return tuple(self.get_links_flattened()[1:])

    def compile(self):
        links = self.get_links_flattened()
        nbLinks = [link._jit() for link in links]
        for nblink, link in zip(nbLinks[1:], links[1:]):
            nblink.parent = nbLinks[links.index(link.parent)]

        forces = []
        d_forces = []
        forces_link = []

        for i, link in enumerate(links[1:]):
            for force in link.forces:
                _force = force._jit()
                forces.append(_force[0])
                d_forces.append(_force[1])
                forces_link.append(nbLinks[i + 1])

        return (
            int(self.dof),
            tuple(nbLinks),
            len(forces),
            tuple(forces),
            tuple(d_forces),
            tuple(forces_link),
            self.cbf,
        )


class LinkType(enum.IntEnum):
    _root = -1
    prismatic = 0
    rotational = 1


class Link:
    def __init__(
        self,
        parent: Union[None, Link, NESystem],
        mass,
        center_of_mass,
        inertia_tensor,
        index,
        axis,
        link_type,
        rotation_local=np.eye(3),
        position=np.zeros(3),
    ):
        if parent is not None:
            self.parent = parent._add_child(self)
            self.root = self.parent.root
        else:
            self.root = self

        self.mass = float(mass)
        self.GAMMA = self.mass * np.reshape(center_of_mass, 3).astype(float)
        self.inertia_tensor = np.reshape(inertia_tensor, (3, 3)).astype(float)
        self.rotation_offset = np.reshape(rotation_local, (3, 3)).astype(float)
        self.position = np.reshape(position, 3).astype(float)
        self.index = index
        self.axis = np.reshape(axis, 3).astype(float)
        self.link_type = link_type
        self.children = []
        self.forces = []
        self.IHat = np.zeros(0).astype(float)
        self.ITilde = np.zeros(0).astype(float)
        self.dof = None

    def _assign_dof(self):
        self.dof = self.root.dof
        self.IHat = np.zeros((3, self.dof))
        self.ITilde = np.zeros((3, self.dof))
        if self.link_type == LinkType.rotational:
            self.IHat[:, self.index] = self.axis
        elif self.link_type == LinkType.prismatic:
            self.IHat[:, self.index] = self.axis

    def _assign_dof_branch(self):
        self._assign_dof()
        for child in self.children:
            child._assign_dof_branch()

    def _add_child(self, child):
        self.children.append(child)
        return self

    def add_force(self, *forces):
        from .force import Force

        for force in forces:
            if not isinstance(force, Force):
                raise TypeError(f"Unexpected type: {type(force)}")
            self.forces.append(force)

    def _get_local_dof(self):
        return 1 + sum([child._get_local_dof() for child in self.children])

    def get_links_flattened(self):
        return [self] + [
            item
            for sublist in [child.get_links_flattened() for child in self.children]
            for item in sublist
        ]

    def _jit(self) -> nbLink:
        return nbLink(
            parent=None,
            dof=int(self.dof),
            link_type=int(self.link_type),
            mass=self.mass,
            index=self.index,
            axis=self.axis,
            GAMMA=self.GAMMA,
            inertia_tensor=self.inertia_tensor,
            position=self.position,
            rotation_local=self.rotation_offset,
            IHat=self.IHat,
            ITilde=self.ITilde,
        )
