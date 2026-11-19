/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/qep/task.hpp>

#include <memory>
#include <optional>

namespace gqe {
namespace qep {

/**
 * @brief Apply an optional valid mask and materialize indirect columns.
 *
 * Accepts every column-bearing `logical_table_view` relation and count-only state. It applies and
 * removes an optional relation-level valid mask, compacts distinct indirect position lists before
 * gathering, and materializes every logical column as a regular cuDF state. Indirect columns that
 * share a gather source are materialized by one gather operation. Regular-table and count-only
 * inputs pass through by shared ownership. An empty input container is an executor-contract
 * violation.
 */
class gather_task : public stateless_transform_task {
 public:
  /** @brief Construct a task that preserves relative row order. */
  gather_task() : stateless_transform_task(task_ordering::preserve) {}

  /** @copydoc stateless_transform_task::next */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;
};

}  // namespace qep
}  // namespace gqe
