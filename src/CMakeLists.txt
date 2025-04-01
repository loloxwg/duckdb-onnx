add_subdirectory(onnx)
add_subdirectory(core)
set(EXTENSION_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/error.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/onnx_extension.cpp
        ${EXTENSION_SOURCES}
        PARENT_SCOPE)