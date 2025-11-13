/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <gqe/types.hpp>

#include <shared_mutex>

namespace gqe {

/**
 * @brief Manages and updates table statistics in a thread-safe manner.
 */
class table_statistics_manager {
 public:
  table_statistics_manager(table_statistics statistics) : _statistics(statistics) {}

  table_statistics_manager(const table_statistics_manager&)            = delete;
  table_statistics_manager& operator=(const table_statistics_manager&) = delete;

  /**
   * @brief Increments row count in the table statistics in a thread-safe manner.
   *
   * @param[in] rows Number of rows to increment the row count by.
   */
  void add_rows(int rows);

  /**
   * @brief Retrieves table statistics in a thread-safe manner.
   *
   * @return The current table statistics.
   */
  table_statistics statistics();

 private:
  table_statistics _statistics;
  std::shared_mutex _statistics_latch;  //> The latch always needs to be acquired in order to
                                        // guarantee thread-safe access to `_statistics`
};

}  // namespace gqe
