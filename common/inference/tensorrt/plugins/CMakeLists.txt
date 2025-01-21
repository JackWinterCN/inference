FILE( GLOB CPPS [^.]*.cpp [^.]*.cc)
FILE( GLOB HPPS [^.]*.hpp [^.]*.h [^.]*.inl)
FILE( GLOB CUDA_SRCS [^.]*.cu)
FILE( GLOB EXCEPT_FILES *test.cc *test.cpp)


LIST(LENGTH EXCEPT_FILES NOTEMPTY)
if (NOTEMPTY)
    list(REMOVE_ITEM CPPS ${EXCEPT_FILES})
endif()

target_sources(${TARGET_NAME}
PRIVATE
    ${HPPS}
    ${CPPS}
    # ${CUDA_SRCS}
)

# cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
 
# project(minist_test LANGUAGES CXX CUDA)

# set(CMAKE_CUDA_STANDARD 14)

# find_package(CUDA REQUIRED)
# find_package(cuDNN REQUIRED)


# cuda_add_library(cuda_core  ${CUDA_SRCS} ${HPPS})