add_subdirectory(model)
add_subdirectory(ops)
set(EXTENSION_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/model/graph.cpp ${EXTENSION_SOURCES}
        PARENT_SCOPE)