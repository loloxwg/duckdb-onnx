cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME onnx)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
find_package(OpenSSL REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)
set(EXTENSION_SOURCES src/onnx_extension.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})


if (APPLE)
    ### 1.download and unzip
    ### https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.3/onnxruntime-osx-arm64-static_lib-1.17.3.zip
    set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.3/onnxruntime-osx-arm64-static_lib-1.17.3.zip")
    set(ONNXRUNTIME_ZIP "onnxruntime-osx-arm64-static_lib-1.17.3.zip")
    set(ONNXRUNTIME_DIR "${CMAKE_BINARY_DIR}/onnxruntime-osx-arm64-static_lib-1.17.3")

    if (NOT EXISTS ${CMAKE_BINARY_DIR}/${ONNXRUNTIME_ZIP})
        message(STATUS "Downloading ONNX Runtime from ${ONNXRUNTIME_URL}...")
        execute_process(
                COMMAND wget ${ONNXRUNTIME_URL} -O ${CMAKE_BINARY_DIR}/${ONNXRUNTIME_ZIP}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                RESULT_VARIABLE WGET_RESULT
        )
        if (WGET_RESULT)
            message(FATAL_ERROR "Failed to download ONNX Runtime!")
        endif ()
    endif ()

    if (NOT EXISTS ${ONNXRUNTIME_DIR})
        message(STATUS "Extracting ONNX Runtime...")
        execute_process(
                COMMAND unzip ${CMAKE_BINARY_DIR}/${ONNXRUNTIME_ZIP} -d ${CMAKE_BINARY_DIR}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                RESULT_VARIABLE UNZIP_RESULT
        )
        if (UNZIP_RESULT)
            message(FATAL_ERROR "Failed to extract ONNX Runtime!")
        endif ()
    endif ()


    include_directories(${ONNXRUNTIME_DIR}/include)
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.a)

    enable_language(OBJC)
    enable_language(OBJCXX)

    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)

    target_link_libraries(${EXTENSION_NAME}
            OpenSSL::SSL
            OpenSSL::Crypto
            ${ONNXRUNTIME_LIB}
            ${FOUNDATION_FRAMEWORK}
            ${COREFOUNDATION_FRAMEWORK}
            "-ObjC"
    )

    target_link_libraries(${LOADABLE_EXTENSION_NAME}
            OpenSSL::SSL
            OpenSSL::Crypto
            ${ONNXRUNTIME_LIB}
            ${FOUNDATION_FRAMEWORK}
            ${COREFOUNDATION_FRAMEWORK}
            "-ObjC"
    )

    foreach (TARGET_LIB ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
        target_compile_options(${TARGET_LIB} PRIVATE -fobjc-arc)
    endforeach ()
endif ()

if (UNIX AND NOT APPLE)
    # onnxruntime-linux-x64-glibc2_17-Debug-1.17.3.zip
    # onnxruntime-linux-x64-glibc2_17-Release-1.17.3.zip
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.3/onnxruntime-linux-x64-glibc2_17-Debug-1.17.3.zip")
        set(ONNXRUNTIME_ZIP "onnxruntime-linux-x64-glibc2_17-Debug-1.17.3.zip")
        set(ONNXRUNTIME_DIR "${CMAKE_BINARY_DIR}/onnxruntime-linux-x64-glibc2_17-Debug-1.17.3")
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        set(ONNXRUNTIME_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.3/onnxruntime-linux-x64-glibc2_17-Release-1.17.3.zip")
        set(ONNXRUNTIME_ZIP "onnxruntime-linux-x64-glibc2_17-Release-1.17.3.zip")
        set(ONNXRUNTIME_DIR "${CMAKE_BINARY_DIR}/onnxruntime-linux-x64-glibc2_17-Release-1.17.3")
    else ()
        message(FATAL_ERROR "Unsupported or unspecified CMAKE_BUILD_TYPE. Please specify Debug or Release.")
    endif ()


    if (NOT EXISTS ${CMAKE_BINARY_DIR}/${ONNXRUNTIME_ZIP})
        message(STATUS "Downloading ONNX Runtime from ${ONNXRUNTIME_URL}...")
        execute_process(
                COMMAND wget ${ONNXRUNTIME_URL} -O ${CMAKE_BINARY_DIR}/${ONNXRUNTIME_ZIP}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                RESULT_VARIABLE WGET_RESULT
        )
        if (WGET_RESULT)
            message(FATAL_ERROR "Failed to download ONNX Runtime!")
        endif ()
    endif ()

    if (NOT EXISTS ${ONNXRUNTIME_DIR})
        message(STATUS "Extracting ONNX Runtime...")
        execute_process(
                COMMAND unzip -o ${CMAKE_BINARY_DIR}/${ONNXRUNTIME_ZIP} -d ${CMAKE_BINARY_DIR}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                RESULT_VARIABLE UNZIP_RESULT
        )
        if (UNZIP_RESULT)
            message(FATAL_ERROR "Failed to extract ONNX Runtime!")
        endif ()
    endif ()

    include_directories(${ONNXRUNTIME_DIR}/include)
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.a)

    target_link_libraries(${EXTENSION_NAME}
            ${ONNXRUNTIME_LIB}
    )

    target_link_libraries(${LOADABLE_EXTENSION_NAME}
            ${ONNXRUNTIME_LIB}
    )

    foreach (TARGET_LIB ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
        target_compile_options(${TARGET_LIB} PRIVATE -fPIC)
    endforeach ()
endif ()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
