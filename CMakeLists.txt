cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Set extension name here
set(TARGET_NAME onnx)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

# Include directories
include_directories(src/include)
include_directories(src/onnx/codegen)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
find_package(OpenSSL CONFIG REQUIRED)
find_package(protobuf CONFIG REQUIRED)
message(STATUS "Protobuf_FOUND: ${Protobuf_FOUND}")
message(STATUS "Protobuf_VERSION: ${Protobuf_VERSION}")
message(STATUS "Protobuf_INCLUDE_DIRS: ${Protobuf_INCLUDE_DIRS}")
message(STATUS "Protobuf_LIBRARIES: ${Protobuf_LIBRARIES}")
message(STATUS "Protobuf_PROTOC_EXECUTABLE: ${Protobuf_PROTOC_EXECUTABLE}")

# 确保生成的目录路径是绝对路径
set(PROTO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx/protos/onnx.proto3)
set(GENERATED_PROTO_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${GENERATED_PROTO_DIR})
message(STATUS "Generated directory: ${GENERATED_PROTO_DIR}")
add_custom_command(
        OUTPUT
        "${GENERATED_PROTO_DIR}/onnx.proto3.pb.cc"
        "${GENERATED_PROTO_DIR}/onnx.proto3.pb.h"
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src/onnx/protos
        --cpp_out=${GENERATED_PROTO_DIR}
        ${PROTO_FILES}
        DEPENDS ${PROTO_FILES}
        COMMENT "Generate onnx.proto3.pb.cc 和 onnx.proto3.pb.h"
        VERBATIM
)

set(PROTO_SRCS "${GENERATED_PROTO_DIR}/onnx.proto3.pb.cc")
set(PROTO_HDRS "${GENERATED_PROTO_DIR}/onnx.proto3.pb.h")

message(STATUS "Generated proto sources: ${PROTO_SRCS}")
message(STATUS "Generated proto headers: ${PROTO_HDRS}")

add_custom_target(generate_proto DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})

add_library(onnx_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
add_dependencies(onnx_proto generate_proto)  # 确保文件先被生成
target_link_libraries(onnx_proto PRIVATE protobuf::libprotobuf)
target_include_directories(onnx_proto PRIVATE ${GENERATED_PROTO_DIR})

include_directories(${GENERATED_PROTO_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)

list(APPEND EXTENSION_SOURCES ${PROTO_SRCS})

add_subdirectory(unit_test)
add_subdirectory(src)

# Build the DuckDB static and loadable extensions
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

add_dependencies(${EXTENSION_NAME} generate_proto)
add_dependencies(${LOADABLE_EXTENSION_NAME} generate_proto)


if (APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)

    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)

    target_link_libraries(${EXTENSION_NAME}
            OpenSSL::SSL
            OpenSSL::Crypto
            protobuf::libprotobuf
            ${FOUNDATION_FRAMEWORK}
            ${COREFOUNDATION_FRAMEWORK}
            "-ObjC"
    )

    target_link_libraries(${LOADABLE_EXTENSION_NAME}
            OpenSSL::SSL
            OpenSSL::Crypto
            protobuf::libprotobuf
            ${FOUNDATION_FRAMEWORK}
            ${COREFOUNDATION_FRAMEWORK}
            "-ObjC"
    )

    foreach (TARGET_LIB ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
        target_compile_options(${TARGET_LIB} PRIVATE -fobjc-arc)
    endforeach ()
endif ()

if (UNIX AND NOT APPLE)
    foreach (TARGET_LIB ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
        target_compile_options(${TARGET_LIB} PRIVATE -fPIC)
    endforeach ()
endif ()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
