
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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

  table_statistics_manager(const table_statistics_manager&) = delete;
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
