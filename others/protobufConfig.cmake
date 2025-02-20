# 定义 Protobuf 的版本
set(PROTOBUF_VERSION 3.14.0)

# 设置 Protobuf 的根目录，根据实际安装路径修改
set(PROTOBUF_ROOT "/opt/apollo/neo/packages/3rd-protobuf/latest/" CACHE PATH "Path to Protobuf installation")

# 查找 Protobuf 的头文件目录
find_path(PROTOBUF_INCLUDE_DIR google/protobuf/message.h
    HINTS /opt/apollo/neo/include
    NO_DEFAULT_PATH
)

# 查找 Protobuf 库文件
find_library(PROTOBUF_LIBRARY NAMES protobuf
    HINTS ${PROTOBUF_ROOT}/lib
    NO_DEFAULT_PATH
)

# 查找 Protobuf 编译器
find_program(PROTOBUF_PROTOC_EXECUTABLE protoc
    HINTS /opt/apollo/neo/lib/3rd-protobuf-new/bin
    NO_DEFAULT_PATH
)

find_program(PROTOC_EXECUTABLE protoc
    HINTS /opt/apollo/neo/lib/3rd-protobuf-new/bin
    NO_DEFAULT_PATH
)

# 查找 libprotoc 库文件
find_library(PROTOBUF_PROTOC_LIBRARY NAMES protoc
    HINTS ${PROTOBUF_ROOT}/lib
    NO_DEFAULT_PATH
)

# 检查是否找到必要的组件
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Protobuf
    REQUIRED_VARS PROTOBUF_INCLUDE_DIR PROTOBUF_LIBRARY PROTOBUF_PROTOC_EXECUTABLE PROTOBUF_PROTOC_LIBRARY
    VERSION_VAR PROTOBUF_VERSION
)

# 如果找到，设置相关变量
if(true)
    set(PROTOBUF_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIR})
    set(PROTOBUF_LIBRARIES ${PROTOBUF_LIBRARY})
    if(NOT TARGET Protobuf::libprotobuf)
        add_library(Protobuf::libprotobuf UNKNOWN IMPORTED)
        set_target_properties(Protobuf::libprotobuf PROPERTIES
            IMPORTED_LOCATION "${PROTOBUF_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PROTOBUF_INCLUDE_DIR}"
        )
    endif()
    if(NOT TARGET Protobuf::protoc)
        add_executable(Protobuf::protoc IMPORTED)
        set_target_properties(Protobuf::protoc PROPERTIES
            IMPORTED_LOCATION "${PROTOBUF_PROTOC_EXECUTABLE}"
        )
    endif()
    if(NOT TARGET Protobuf::libprotoc)
        add_library(Protobuf::libprotoc UNKNOWN IMPORTED)
        set_target_properties(Protobuf::libprotoc PROPERTIES
            IMPORTED_LOCATION "${PROTOBUF_PROTOC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PROTOBUF_INCLUDE_DIR}"
        )
    endif()
endif()

# 标记这些变量为高级变量，在 CMake GUI 中默认隐藏
mark_as_advanced(PROTOBUF_INCLUDE_DIR PROTOBUF_LIBRARY PROTOBUF_PROTOC_EXECUTABLE PROTOBUF_PROTOC_LIBRARY)