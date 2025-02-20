include(cmake/function_proto.cmake)

# ---- 自定义proto文件的生成 ----
file(GLOB_RECURSE inner_proto_files "${INNER_PROTO_INTERFACE_DIR}/*.proto")
foreach(inner_proto_abs ${inner_proto_files})
    message("====> find proto file: ${inner_proto_abs}")
    get_filename_component(inner_proto_rel ${inner_proto_abs} NAME)
    list(APPEND inner_proto_rel_files proto/${inner_proto_rel})
endforeach()
message("====> all proto file: ${inner_proto_rel_files}")

generate_proto_files(
    custom_proto_file_list
PROTO_OUTPUT_DIR
    ${PROJECT_BINARY_DIR}
PROTO_INPUT_DIR
    ${PROJECT_SOURCE_DIR}
PROTO_PATHS
    ${PROJECT_SOURCE_DIR}
    # ${THIRDPARTY_DIR}/include
PROTO_FILES
    ${inner_proto_rel_files}
)
message("====> all proto source file: ${custom_proto_file_list}")

# ---- 生成 object目标 ----
add_library(${PROTO_TARGET_NAME} OBJECT)

target_sources(${PROTO_TARGET_NAME}
PRIVATE
    ${custom_proto_file_list}
)

target_include_directories(${PROTO_TARGET_NAME}
PRIVATE
    ${PROJECT_BINARY_DIR}
    ${THIRDPARTY_DIR}/include
)

target_link_libraries(${PROTO_TARGET_NAME}
PUBLIC
    ${PROTOBUF_PROTOC_LIBRARY}
    ${PROTOBUF_LIBRARY}
    # protobuf::libprotoc
)

target_compile_options(${PROTO_TARGET_NAME}
PRIVATE
    "-fPIC"
)
