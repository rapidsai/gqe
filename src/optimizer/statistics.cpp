/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/optimizer/statistics.hpp>

#include <mutex>
#include <type_traits>
#include <utility>

namespace gqe {

void table_statistics_manager::add_rows(int rows)
{
  std::scoped_lock latch_guard(_statistics_latch);
  _statistics.num_rows += rows;
}

void table_statistics_manager::append_table_statistics(const table_statistics& table_stats)
{
  static_assert(std::is_nothrow_move_assignable_v<table_statistics>,
                "Publishing the prepared statistics must not throw");

  std::scoped_lock latch_guard(_statistics_latch);
  // Merge into a copy, so that a throwing merge leaves the published statistics unchanged.
  auto updated = _statistics;
  updated.append_table_statistics(table_stats);
  _statistics = std::move(updated);
}

table_statistics table_statistics_manager::statistics()
{
  std::shared_lock latch_guard(_statistics_latch);
  return _statistics;
}

}  // namespace gqe
