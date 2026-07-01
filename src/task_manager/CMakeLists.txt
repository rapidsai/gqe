# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

include(${PROJECT_SOURCE_DIR}/cmake/arrow_flight_sql.cmake)

add_executable(gqe_task_manager main.cpp task_manager.cpp service.cpp
                                plan_executor.cpp remote_table_cache.cpp)

target_include_directories(gqe_task_manager BEFORE
                           PUBLIC ${ARROW_FLIGHT_SQL_INCLUDE_DIRS})

target_link_libraries(gqe_task_manager PRIVATE ${ARROW_FLIGHT_SQL_LIBS})

# Sibling-lib-dir INSTALL_RPATH. $ORIGIN/../${CMAKE_INSTALL_LIBDIR} resolves
# arrow/nvcomp/etc. installed to the same LIBDIR.
set_target_properties(
  gqe_task_manager PROPERTIES INSTALL_RPATH
                              "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS gqe_task_manager RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
