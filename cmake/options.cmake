# ---- set cxx standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- set default build type ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- build tests ----
option(BUILD_TESTS "Build tests." FALSE)

# ---- install options ----
include(GNUInstallDirs)
set(LIBRARY_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(RUNTIME_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(CONFIG_INSTALL_DIR  ${LIBRARY_INSTALL_DIR}/cmake/${PROJECT_NAME})

# ---- 配置proto文件的路径 ----
set(PROTO_INTERFACE_DIR ${PROJECT_SOURCE_DIR}/../message)
set(INNER_PROTO_INTERFACE_DIR ${PROJECT_SOURCE_DIR}/proto)
set(PROTO_TARGET_NAME  ${PROJECT_NAME}_proto_target)

# ---- 添加依赖的库 ----
# find_package(Boost REQUIRED COMPONENTS filesystem system)

# find_package(absl REQUIRED PATHS ${THIRDPARTY_DIR}/lib/cmake/absl)
# find_package(osqp REQUIRED PATHS ${THIRDPARTY_DIR}/lib/cmake/osqp)
# find_package(qpOASES REQUIRED PATHS ${THIRDPARTY_DIR}/lib/cmake/qpOASES)
# find_package(cyber REQUIRED PATHS ${THIRDPARTY_DIR}/lib/cmake/cyber)
# find_package(gflags REQUIRED PATHS ${THIRDPARTY_DIR}/lib/cmake/gflags)
# find_package(glog REQUIRED PATHS ${THIRDPARTY_DIR}/lib/cmake/glog)
find_package(protobuf REQUIRED PATHS /opt/apollo/neo/lib/3rd-protobuf/)
