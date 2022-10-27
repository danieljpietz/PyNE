#include "type.h"
#include "func.h"

void init_link(link_props** _link,
               link_props* parent,
               uint32_t dof,
               linktype_t type,
               uint32_t index,
               const double* _axis,
               double mass,
               double* COM,
               const double* _inertia_tensor,
               const double* _rotation_offset,
               const double* _position) {

    link_props* link = (link_props*)malloc(sizeof(link_props));

    link->dof = dof;
    link->type = type;
    link->index = index;
    link->parent = parent;
    link->mass = mass;

    for (size_t i = 0; i < 3; ++i) {
        link->axis[i] = _axis[i];
        link->GAMMA[i] = COM[i];
        link->position[i] = _position[i];
    }

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            link->inertia_tensor(i, j) = _inertia_tensor[3*i + j];
            link->rotation_offset(i, j) = _rotation_offset[3*i + j];
        }
    }

    link->IHat = Eigen::SparseMatrix<double>(3, dof);
    link->ITilde = Eigen::SparseMatrix<double>(3, dof);

    switch (type) {
        case rotational:
            for (size_t i = 0; i < 3; ++i)
                link->IHat.insert(i, index) = link->axis[i];
            break;
        case prismatic:
            for (size_t i = 0; i < 3; ++i)
                link->ITilde.insert(i, index) = link->axis[i];
            break;
        default:
            break;
    }

    link->rotation_global = link->rotation_offset;
    link->angular_velocity_global = Eigen::Vector3d::Zero();
    link->linear_velocity = Eigen::Vector3d::Zero();

    link->jacobian = Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero(6, dof);
    link->dot_jacobian = Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero(6, dof);

    link->jn_star = Eigen::MatrixXd(6, dof);
    link->d_jn_star = Eigen::MatrixXd(6, dof);

    link->jn_star.block(0, 0, 3, dof) = link->IHat;
    link->d_jn_star = Eigen::MatrixXd::Zero(6, dof);

    link->M.block(0, 0, 3, 3) = link->inertia_tensor;
    link->M.block(3, 3, 3, 3) = mass * Eigen::Matrix3d::Identity();

    *_link = link;

}
