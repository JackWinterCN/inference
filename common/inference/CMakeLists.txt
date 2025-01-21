FILE( GLOB CPPS [^.]*.cpp [^.]*.cc)
FILE( GLOB HPPS [^.]*.hpp [^.]*.h [^.]*.inl)
FILE( GLOB EXCEPT_FILES *test.cc *test.cpp)


LIST(LENGTH EXCEPT_FILES NOTEMPTY)
if (NOTEMPTY)
    list(REMOVE_ITEM CPPS ${EXCEPT_FILES})
endif()


target_sources(${TARGET_NAME}
PRIVATE
    ${HPPS}
    ${CPPS}
)

# add_subdirectory(libtorch)
add_subdirectory(onnx)
add_subdirectory(tensorrt)
add_subdirectory(utils)
