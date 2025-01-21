# cmake_minimum_required(VERSION 3.14...3.19 FATAL_ERROR)
cmake_minimum_required(VERSION 3.14...3.19 FATAL_ERROR)

# ---- prediction project ----
project(inference_tester
    LANGUAGES CXX C
    VERSION 0.2.1
)

#包含各种配置选项
include(cmake/options.cmake)
#包含各种函数
# include(cmake/macros.cmake)
#生成项目依赖的proto
include(cmake/process_proto.cmake)

# set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA REQUIRED)
find_package(Thrust REQUIRED PATHS /usr/local/cuda-11.8/targets/x86_64-linux/lib/cmake/thrust)

# /usr/local/cuda-11.8/targets/x86_64-linux/lib/cmake/thrust

add_definitions(-DNV_TENSORRT_MAJOR=8)
add_definitions(-DNV_TENSORRT_MINOR=6)

set(CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86)


# ======BEGIN for target adm ======
set(TARGET_NAME "inference_demo")

FILE( GLOB CPPS common/inference/tensorrt/plugins/*.cpp common/inference/tensorrt/plugins/*.cc)
FILE( GLOB HPPS common/inference/tensorrt/plugins/*.hpp common/inference/tensorrt/plugins/*.h )
FILE( GLOB CUDA_SRCS common/inference/tensorrt/plugins/*.cu)
cuda_add_library(cuda_core  ${CUDA_SRCS} ${HPPS})

target_include_directories(cuda_core
PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}
    /usr/local/cuda/include/
    # /usr/local/cuda/include/thrust/
    # /usr/local/cuda/include/thrust/detail
    # ${THIRDPARTY_DIR}/include/
    ${THIRDPARTY_DIR}/include/absl/
    ${THIRDPARTY_DIR}/include/c10/
    ${THIRDPARTY_DIR}/include/cyber/
    ${THIRDPARTY_DIR}/include/glog/
    ${THIRDPARTY_DIR}/include/ymal/
    ${THIRDPARTY_DIR}/include/gflags/
    ${THIRDPARTY_DIR}/include/google/
    # ${THIRDPARTY_DIR}/include/torch/
    ${THIRDPARTY_DIR}/include/fastdds/
    ${THIRDPARTY_DIR}/include/opencv2/
    /usr/include/gtsam/3rdparty/Eigen/
)

FILE( GLOB HPPS common/inference/utils/*.hpp common/inference/utils/*.h )
FILE( GLOB CUDA_SRCS common/inference/utils/*.cu)
cuda_add_library(cuda_utils  ${CUDA_SRCS} ${HPPS})

target_include_directories(cuda_utils
PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}
    /usr/local/cuda/include/
    # /usr/local/cuda/include/thrust/
    # /usr/local/cuda/include/thrust/detail
    # ${THIRDPARTY_DIR}/include/
    ${THIRDPARTY_DIR}/include/absl/
    ${THIRDPARTY_DIR}/include/c10/
    ${THIRDPARTY_DIR}/include/cyber/
    ${THIRDPARTY_DIR}/include/glog/
    ${THIRDPARTY_DIR}/include/ymal/
    ${THIRDPARTY_DIR}/include/gflags/
    ${THIRDPARTY_DIR}/include/google/
    # ${THIRDPARTY_DIR}/include/torch/
    ${THIRDPARTY_DIR}/include/fastdds/
    ${THIRDPARTY_DIR}/include/opencv2/
    /usr/include/gtsam/3rdparty/Eigen/
)


cuda_add_library(cuda_yolo_postprocess  
    camera_detection_multi_stage/detector/yolo/postprocess.cu 
    camera_detection_multi_stage/detector/yolo/postprocess.h
)

target_include_directories(cuda_yolo_postprocess
PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}
    /usr/local/cuda/include/
    # /usr/local/cuda/include/thrust/
    # /usr/local/cuda/include/thrust/detail
    # ${THIRDPARTY_DIR}/include/
    ${THIRDPARTY_DIR}/include/absl/
    ${THIRDPARTY_DIR}/include/c10/
    ${THIRDPARTY_DIR}/include/cyber/
    ${THIRDPARTY_DIR}/include/glog/
    ${THIRDPARTY_DIR}/include/ymal/
    ${THIRDPARTY_DIR}/include/gflags/
    ${THIRDPARTY_DIR}/include/google/
    # ${THIRDPARTY_DIR}/include/torch/
    ${THIRDPARTY_DIR}/include/fastdds/
    ${THIRDPARTY_DIR}/include/opencv2/
    /usr/include/gtsam/3rdparty/Eigen/
)
# add_library(${TARGET_NAME} SHARED)
add_executable(${TARGET_NAME} inference_test.cc)

add_definitions(-DUSE_GPU=1)

add_definitions(-DINT_MAX=0x7fffffff)
add_definitions(-DINT_MIN=-2147483648)
add_definitions(-DUINT_MAX=0xffffffff)
add_definitions(-DCHAR_BIT=8)
# add_definitions(-DCHAR_MIN=0)
# add_definitions(-DSCHAR_MIN=-128)
# add_definitions(-DSCHAR_MIN=-128)
# add_definitions(-DCHAR_MAX=255)
# add_definitions(-DSCHAR_MAX=127)


# include_directories(/usr/include/)


message("PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}")
message("PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")
message("THIRDPARTY_DIR = ${THIRDPARTY_DIR}")
message("GPU_PLATFORM = ${GPU_PLATFORM}")
message("USE_GPU = ${USE_GPU}")

target_include_directories(${TARGET_NAME}
PUBLIC
${PROJECT_SOURCE_DIR}
${PROJECT_BINARY_DIR}
/usr/local/cuda/include/
# /usr/local/cuda/include/thrust/
# /usr/local/cuda/include/thrust/detail
# ${THIRDPARTY_DIR}/include/
${THIRDPARTY_DIR}/include/absl/
${THIRDPARTY_DIR}/include/c10/
${THIRDPARTY_DIR}/include/cyber/
${THIRDPARTY_DIR}/include/glog/
${THIRDPARTY_DIR}/include/ymal/
${THIRDPARTY_DIR}/include/gflags/
${THIRDPARTY_DIR}/include/google/
# ${THIRDPARTY_DIR}/include/torch/
${THIRDPARTY_DIR}/include/fastdds/
${THIRDPARTY_DIR}/include/opencv2/
/usr/include/gtsam/3rdparty/Eigen/
# /opt/apollo/neo/packages/3rd-libtorch-gpu/latest/include/torch/csrc/api/include/

# /usr/include/c++/11/
    # /usr/include/
PRIVATE
    # /usr/lib/gcc/x86_64-linux-gnu/11/include/

    
    # ${PUBLIC_MODULE_DIR}/include
    # ${PUBLIC_MODULE_DIR}/include/common
    # ${THIRDPARTY_DIR}/include/cyber
)

target_link_directories(${TARGET_NAME}
PUBLIC
    # ${PUBLIC_MODULE_DIR}/lib
    # ${THIRDPARTY_DIR}/lib
    /opt/apollo/neo/lib/cyber/
    /opt/apollo/neo/lib/cyber/node/
    /opt/apollo/neo/lib/cyber/blocker/
    /opt/apollo/neo/lib/cyber/plugin_manager/
    /opt/apollo/neo/lib/cyber/class_loader/
    /opt/apollo/neo/lib/cyber/class_loader/utility/
    /opt/apollo/neo/lib/cyber/service_discovery/
    /opt/apollo/neo/lib/cyber/transport/
    /opt/apollo/neo/lib/cyber/message/
    /opt/apollo/neo/lib/cyber/statistics/
    /opt/apollo/neo/lib/cyber/proto/
    /opt/apollo/neo/lib/cyber/task/
    /opt/apollo/neo/lib/cyber/scheduler/
    /opt/apollo/neo/lib/cyber/croutine/
    /opt/apollo/neo/lib/cyber/event/
    /opt/apollo/neo/lib/cyber/time/
    /opt/apollo/neo/lib/cyber/common/

    /opt/apollo/neo/lib/3rd-fastdds-wrap/

    /opt/apollo/neo/lib/3rd-glog/
    /opt/apollo/neo/lib/3rd-gflags

    /opt/apollo/neo/lib/3rd-opencv/
    
    /opt/apollo/neo/lib/3rd-ymal/

    /opt/apollo/neo/lib/3rd-boost/
    /opt/apollo/neo/lib/3rd-absl/
    /usr/local/cuda/targets/x86_64-linux/lib/
    /opt/apollo/neo/packages/3rd-libtorch-gpu/latest/lib/
    /usr/lib/x86_64-linux-gnu/
    # /opt/apollo/neo/lib/3rd-libtorch-cpu/
    # /opt/apollo/neo/lib/3rd-libtorch-gpu/
)

add_subdirectory(common)
add_subdirectory(camera_detection_multi_stage)

target_link_libraries(${TARGET_NAME}
# PRIVATE
    ${PROTO_TARGET_NAME}
    cyber
    cyber_node
    cyber_blocker
    cyber_plugin_manager
    cyber_class_loader
    class_loader_utility
    cyber_service_discovery
    cyber_transport
    cyber_message
    lib_role_attributes_proto_cp_bin.so
    lib_transport_conf_proto_cp_bin.so
    lib_qos_profile_proto_cp_bin.so
    apollo_statistics
    cyber_task
    cyber_scheduler
    cyber_croutine
    cyber_event
    cyber_state
    cyber_time
    cyber_common
    cyber_binary

    fastrtps

    bvar
    /usr/local/cuda/lib64/libcublasLt.so
    /usr/local/cuda/lib64/libcudart.so
    /usr/local/cuda/lib64/libcublas.so
    /usr/lib/x86_64-linux-gnu/libcudnn.so
    glog
    gflags

    nppc
    nppial
    nppicc
    nppidei
    nppif
    nppig
    nppim
    nppist
    nppisu
    nppitc

    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_videoio
    
    boost_filesystem
    absl_strings

    nvinfer
    nvinfer_plugin
    nvonnxparser
    nvparsers
    yaml-cpp
    c10
    
    cuda_core
    cuda_utils
    cuda_yolo_postprocess
# PUBLIC
    # ${}
    # proto_msg
    # common
)

# ======BEGIN for install target ======
install(TARGETS ${TARGET_NAME}
    ARCHIVE DESTINATION ${LIBRARY_INSTALL_DIR}
    RUNTIME DESTINATION ${RUNTIME_INSTALL_DIR}
    LIBRARY DESTINATION ${LIBRARY_INSTALL_DIR}
)


#add_subdirectory(tools)
