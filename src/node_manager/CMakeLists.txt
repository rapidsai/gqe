# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

include(${PROJECT_SOURCE_DIR}/cmake/arrow_flight_sql.cmake)

add_executable(gqe_node_manager main.cpp node_manager.cpp service.cpp spawn.cpp
                                nvshmem_bootstrap.cpp context.cpp)

target_include_directories(gqe_node_manager BEFORE
                           PRIVATE ${ARROW_FLIGHT_SQL_INCLUDE_DIRS})

target_link_libraries(gqe_node_manager PRIVATE ${ARROW_FLIGHT_SQL_LIBS})

# gqe PUBLIC includes propagate transitively, but Arrow includes must come first
# (BEFORE) so Arrow headers win over any system-installed copies.

# Sibling-lib-dir INSTALL_RPATH. $ORIGIN/../${CMAKE_INSTALL_LIBDIR} resolves
# arrow/nvcomp/etc. installed to the same LIBDIR.
set_target_properties(
  gqe_node_manager PROPERTIES INSTALL_RPATH
                              "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS gqe_node_manager RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
