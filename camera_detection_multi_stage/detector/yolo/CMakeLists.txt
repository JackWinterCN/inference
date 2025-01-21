FILE( GLOB CPPS [^.]*.cpp [^.]*.cc)
FILE( GLOB HPPS [^.]*.hpp [^.]*.h [^.]*.inl)
FILE( GLOB CUDA_SRCS [^.]*.cu)
FILE( GLOB EXCEPT_FILES *test.cc *test.cpp)


LIST(LENGTH EXCEPT_FILES NOTEMPTY)
if (NOTEMPTY)
    list(REMOVE_ITEM CPPS ${EXCEPT_FILES})
endif()

message("====> CUDA_SRCS: ${CUDA_SRCS}")
target_sources(${TARGET_NAME}
PRIVATE
    ${HPPS}
    ${CPPS}
    # ${CUDA_SRCS}
)
