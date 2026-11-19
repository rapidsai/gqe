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

#pragma once

#include <gqe/executor/read.hpp>
#include <gqe/storage/readable_view.hpp>

namespace gqe {
namespace storage {
class in_memory_table;
class in_memory_table_interface;
class shared_in_memory_table;
}  // namespace storage
}  // namespace gqe

namespace gqe {

namespace storage {

/**
 * @brief Data access method to read an in-memory table.
 *
 * `get_read_tasks` acquires a shared read-only latch on the row groups
 * collection in the table for the duration of creating new read tasks.
 */
class in_memory_readable_view : public readable_view {
 public:
  friend class in_memory_table;
  friend class shared_in_memory_table;

  /**
   * @copydoc gqe::storage::readable_view::get_read_tasks()
   */
  std::vector<std::unique_ptr<read_task_base>> get_read_tasks(
    std::vector<readable_view::task_parameters>&& task_parameters,
    context_reference ctx_ref,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types) override;

  /**
   * @copydoc gqe::storage::readable_view::get_read_pipeline()
   */
  [[nodiscard]] qep_read_pipeline get_read_pipeline(
    source_reader_type reader,
    std::string table_name,
    table_provider const* provider,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types,
    std::optional<cudf::size_type> partition_column_idx = std::nullopt,
    std::unique_ptr<gqe::expression> partial_filter     = nullptr) override;

 private:
  in_memory_readable_view(in_memory_table_interface* non_owning_table);

  in_memory_table_interface* _non_owning_table;
};

}  // namespace storage

}  // namespace gqe
