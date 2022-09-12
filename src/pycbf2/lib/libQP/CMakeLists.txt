cmake_minimum_required(VERSION 3.23)
project(QP)

set(CMAKE_C_STANDARD 11)

include(ExternalProject)

set(OSQP_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/osqp)
include_directories(${OSQP_INSTALL_LOCATION}/include)
link_directories(${OSQP_INSTALL_LOCATION}/lib)

ExternalProject_Add(OSQPDownloader
        GIT_REPOSITORY https://github.com/osqp/osqp
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${OSQP_INSTALL_LOCATION}
        )


add_library(QP SHARED main.c qpsolve.c)
target_link_libraries(QP osqp)

add_executable(QPTests main.c)
add_dependencies(${PROJECT_NAME} OSQPDownloader)

