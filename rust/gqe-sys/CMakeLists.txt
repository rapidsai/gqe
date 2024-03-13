set(CXX_GQE_BINDINGS_DIR "" CACHE PATH "Path to the bindings directory")
message(STATUS "Using bindings at ${CXX_GQE_BINDINGS_DIR}")

add_library(gqe-cxx STATIC EXCLUDE_FROM_ALL)

target_include_directories(gqe-cxx
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CXX_GQE_BINDINGS_DIR}/crate
        ${CXX_GQE_BINDINGS_DIR}/include
        ${CXX_GQE_BINDINGS_DIR}/sources
)

target_sources(gqe-cxx
    PRIVATE
        cpp/api.cpp
        cpp/cxx_bindings.cpp
        cpp/executor.cpp
        cpp/logical.cpp
        cpp/physical.cpp
        cpp/storage.cpp
)

target_link_libraries(gqe-cxx PUBLIC gqe)
