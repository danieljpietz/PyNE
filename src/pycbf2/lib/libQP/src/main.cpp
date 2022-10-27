#include <stdio.h>
#include <math.h>
#include <type.h>

void recursive_kinematics(link_props* parent, link_props* child, Eigen::VectorXd gamma, Eigen::VectorXd dotgamma);

int main() {

    link_props* link;
    link_props* parent;
    size_t dof = 4;
    linktype_t type = rotational;
    int index = 0;
    double _axis[] = {0, 1, 1};
    double mass = 0;
    double _COM[] = {0, 0, 0};
    double _inertia_tensor[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double _rotation_offset[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double _position[] = {0, 0, 0};

    init_link(&parent, NULL, dof, type, index, _axis, mass, _COM, _inertia_tensor, _rotation_offset, _position);
    init_link(&link, parent, dof, type, index + 1, _axis, mass, _COM, _inertia_tensor, _rotation_offset, _position);

    Eigen::VectorXd dotgamma = Eigen::VectorXd::Zero(dof);

    recursive_kinematics(parent, link, dotgamma, dotgamma);

}
