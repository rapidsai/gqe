add_library(libperfect STATIC masked_join.cu unique_indices.cu scatter_aggregate.cu memory_pool.cpp)
target_include_directories(libperfect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(libperfect PUBLIC cudf::cudf)
