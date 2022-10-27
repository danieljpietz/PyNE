#include "func.h"
#include "type.h"


void recursive_kinematics(link_props* parent, link_props* child, Eigen::VectorXd gamma, Eigen::VectorXd dotgamma) {

    child->x = gamma[child->index];
    child->xdot = dotgamma[child->index];

    //
    // Rotation computation
    //

    if (child->type == rotational) {
        child->rotation_local = child->rotation_offset * Eigen::AngleAxisd(child->x, child->axis);
    } else {
        child->rotation_local = child->rotation_offset;
    }

    child->rotation_global = parent->rotation_global * child->rotation_local;

    //
    // Velocities
    //

    child->angular_velocity_local = child->IHat * dotgamma;
    child->linear_velocity = child->ITilde * dotgamma;

    child->jn_prime << child->rotation_local.transpose(), Eigen::Matrix3d::Zero(),
                        -parent->rotation_global * skew(child->position), Eigen::Matrix3d::Identity();


    child->jn_star.block(3, 0, 3, child->dof) = parent->rotation_global * child->ITilde;

    child->jacobian = child->jn_prime * parent->jacobian + child->jn_star;

    child->angular_velocity_global = child->jacobian.block(0, 0, 3, child->dof) * dotgamma;



    child->dot_jn_prime << (child->rotation_local * skew(child->angular_velocity_local)).transpose(),
                            Eigen::Matrix3d::Zero(),
                            (parent->rotation_global * (skew(parent->angular_velocity_global) * skew(child->position)
                                                     + skew(child->linear_velocity))),
                            Eigen::Matrix3d::Zero();


    child->d_jn_star.block(3, 0, 3, child->dof) = parent->rotation_global * (skew(parent->angular_velocity_global) * child->ITilde);

    child->dot_jacobian = child->dot_jn_prime * parent->jacobian + child->jn_prime * parent->dot_jacobian + child->d_jn_star;

}

void link_dynamics(link_props* link, Eigen::VectorXd dotgamma) {

    link->M.block(3, 0, 3, 3) = skew(link->GAMMA) * link->rotation_global.transpose();
    link->M.block(3, 0, 3, 3) = link->M.block(0, 3, 3, 3).transpose();

    Eigen::MatrixXd jtm = link->jacobian.transpose() * link->M;

    link->H = jtm * link->jacobian;

    Eigen::VectorXd d_star;

    d_star << link->angular_velocity_global.cross(link->inertia_tensor * link->angular_velocity_global),
              link->rotation_global * link->angular_velocity_global.cross(link->angular_velocity_global.cross(link->GAMMA));

    link->d = jtm * link->dot_jacobian * dotgamma + link->jacobian.transpose() * d_star;

}