#include "func.h"

Eigen::Matrix3d skew(Eigen::Vector3d v) {
    Eigen::Matrix3d ret;
    ret << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
    return ret;

}
