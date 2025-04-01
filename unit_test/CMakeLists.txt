cmake_minimum_required(VERSION 3.10)

project(MNIST_Inference)


set(INPUT_DIR "${CMAKE_SOURCE_DIR}/../unit_test/mnist")
set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/extension/onnx/unit_test/mnist")
message(STATUS "Input Directory: ${INPUT_DIR}")
message(STATUS "Output Directory: ${OUTPUT_DIR}")
file(MAKE_DIRECTORY ${OUTPUT_DIR})

set(CMAKE_CXX_STANDARD 11)

include_directories(${OpenCV_INCLUDE_DIRS})
find_package(OpenCV CONFIG REQUIRED)

add_executable(mnist_inference mnist_inference.cpp)
add_custom_command(TARGET mnist_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${INPUT_DIR} ${OUTPUT_DIR}
        COMMENT "Copying files from ${INPUT_DIR} to ${OUTPUT_DIR}"
)

set_target_properties(mnist_inference PROPERTIES
        BUILD_RPATH ${DUCKDB_LIB_DIR}
        INSTALL_RPATH ${DUCKDB_LIB_DIR}
)


target_link_libraries(mnist_inference PRIVATE duckdb ${OpenCV_LIBS})


