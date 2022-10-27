//
// Created by Daniel Pietz on 9/13/22.
//

#ifndef QP_TYPE_H
#define QP_TYPE_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

typedef enum linktype_t {root, rotational, prismatic} linktype_t;

template <long i, long j, long k>
using TensorXd = Eigen::Matrix<Eigen::Matrix<double, i, j>, 1, k>;


typedef struct link_props {

    linktype_t type;

    link_props* parent;

    uint32_t dof;
    uint32_t index;
    Eigen::Vector3d axis;

    double mass;
    Eigen::Vector3d GAMMA;
    Eigen::Matrix3d inertia_tensor;

    Eigen::Matrix3d rotation_offset;
    Eigen::Vector3d position;

    Eigen::SparseMatrix<double> IHat;
    Eigen::SparseMatrix<double> ITilde;

    double x;
    double xdot;

    Eigen::Matrix3d rotation_local;
    Eigen::Matrix3d rotation_global;

    Eigen::Vector3d angular_velocity_local;
    Eigen::Vector3d angular_velocity_global;

    Eigen::Vector3d linear_velocity;

    Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian;
    Eigen::Matrix<double, 6, 6> jn_prime;
    Eigen::Matrix<double, 6, Eigen::Dynamic> jn_star;


    Eigen::Matrix<double, 6, Eigen::Dynamic> dot_jacobian;
    Eigen::Matrix<double, 6, 6> dot_jn_prime;
    Eigen::Matrix<double, 6, Eigen::Dynamic> d_jn_star;

    Eigen::MatrixXd H;
    Eigen::Matrix<double, 6, 6> M;
    Eigen::VectorXd d;

    TensorXd<Eigen::Dynamic, 3, 3> d_rotation_local;
    TensorXd<Eigen::Dynamic, 3, 3> d_rotation_global;

    Eigen::Matrix3Xd d_angular_velocity_local;
    Eigen::Matrix3Xd d_angular_velocity_global;

    TensorXd<Eigen::Dynamic, 6, Eigen::Dynamic> d_jacobian;
    TensorXd<Eigen::Dynamic, 6, 6> d_jn_prime;

    TensorXd<Eigen::Dynamic, 6, Eigen::Dynamic> d_dot_jacobian;
    TensorXd<Eigen::Dynamic, 6, 6> d_dot_jn_prime;

    TensorXd<Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> d_H;
    TensorXd<Eigen::Dynamic, 6, 6> d_M;
    TensorXd<Eigen::Dynamic, Eigen::Dynamic, 1> d_d;

} link_props;

void init_link(link_props** link,
               link_props* parent,
               uint32_t dof,
               linktype_t type,
               uint32_t index,
               const double* _axis,
               double mass,
               double* _COM,
               const double* _inertia_tensor,
               const double* _rotation_offset,
               const double* _position);


#endif //QP_TYPE_H
