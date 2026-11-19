/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/storage/descriptor.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/storage/table_provider.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gqe {
class task_manager_context;
}

namespace gqe::task_manager {

/**
 * @brief Cache for remote shared tables.
 *
 * The cache persists across queries to avoid repeatedly mapping the same tables into the local
 * process. The task manager refreshes a cached table when the table descriptor changes.
 *
 * # Design Limitations
 *
 * To avoid the need for a cache coherence protocol, the design makes the following simplifying
 * assumptions:
 *
 * - Table mappings are explicitly allowed to become stale. The node manager is assumed to manage
 * table lifetimes and never execute a query plan that references non-existing table.
 * - The cache can grow to an unbounded size, because descriptors are never deleted. The table
 * mappings are assumed to be rather small. The lifetime of the task manager process ultimately
 * bounds the cache size.
 */
class remote_table_cache : public storage::table_provider {
 public:
  /**
   * @brief Construct a new remote table cache.
   *
   * @param[in] ctx Task manager context used to construct in-memory tables.
   */
  explicit remote_table_cache(task_manager_context* ctx);

  /**
   * @brief Update the cache with a new set of descriptors.
   *
   * New descriptors are added to the cache. If an existing descriptor for a table has changed, the
   * cached table is evicted and eagerly recreated from the new descriptor.
   *
   * @param[in] descriptors The descriptors received from the node manager.
   */
  void update(std::vector<storage::descriptor> descriptors);

  /** @copydoc storage::table_provider::get_table */
  [[nodiscard]] std::shared_ptr<storage::table> get_table(
    std::string_view table_name) const override;

 private:
  task_manager_context* _ctx;
  std::unordered_set<storage::descriptor> _descriptors;
  std::unordered_map<std::string, std::shared_ptr<storage::table>> _tables;
};

}  // namespace gqe::task_manager
