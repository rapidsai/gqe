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

#include <cudf/types.hpp>

#include <cstddef>

namespace gqe {
namespace qep {

/**
 * @brief Portable description of one independently fetchable unit of source work.
 *
 * A source reference is the addressing half of a source read. It names a memory-bounded unit of
 * a data source (for example one Parquet row-group span or one in-memory row group) without
 * holding any of its data. A reference-generating iterate task mints references, and a resolve task
 * resolves them into `deferred_column` state.
 *
 * # Design
 *
 * References are the work-distribution currency of source pipelines. They are deliberately
 * host-only and free of rank-local pointers such as readers, datasources, device buffers, or
 * storage addresses, so a reference can be claimed, queued, and eventually serialized to an
 * equivalent task on another rank. Concrete subclasses per source kind (in-memory, Parquet, and
 * future sources such as network shuffles or spilled partitions) carry the source-specific
 * addressing fields.
 *
 * # Scheduling Metadata
 *
 * The base interface exposes host-side metadata a scheduler can read without touching the source:
 * the exact retained row count.
 *
 * # Lifetime
 *
 * References are immutable after construction and shared via `shared_ptr<const>`. They own all of
 * their fields.
 */
class source_reference {
 public:
  virtual ~source_reference() = default;

  /**
   * @brief Return the exact number of rows the referenced chunk yields after pruning.
   *
   * Never triggers device work.
   */
  [[nodiscard]] virtual cudf::size_type num_rows() const noexcept = 0;
};

}  // namespace qep
}  // namespace gqe
