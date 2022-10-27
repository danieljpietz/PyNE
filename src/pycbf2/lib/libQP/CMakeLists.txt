cmake_minimum_required(VERSION 3.23)
project(QP)

set(CMAKE_CXX_STANDARD 14)

include(ExternalProject)

set(OSQP_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/osqp)
include_directories(${OSQP_INSTALL_LOCATION}/include include)
link_directories(${OSQP_INSTALL_LOCATION}/lib)

ExternalProject_Add(OSQPDownloader
        GIT_REPOSITORY https://github.com/osqp/osqp
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${OSQP_INSTALL_LOCATION}
        )

find_package (Eigen3 3.3 REQUIRED)

add_library(QP SHARED src/qpsolve.c src/newton_euler.cpp src/func.cpp include/func.h include/type.h src/type.cpp)
target_link_libraries(QP osqp Eigen3::Eigen)

add_executable(QPTests src/main.cpp src/qpsolve.c src/newton_euler.cpp src/func.cpp include/func.h include/type.h src/type.cpp)
target_link_libraries(QPTests osqp Eigen3::Eigen)

add_dependencies(${PROJECT_NAME} OSQPDownloader)

