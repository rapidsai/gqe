/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/query_context.hpp>

#include <gqe/utility/error.hpp>

#include <cstdlib>
#include <iomanip>

namespace gqe {

query_context::query_context(optimization_parameters parameters,
                             optimizer::optimization_configuration logical_rule_config)
  : parameters(parameters), logical_rule_config(std::move(logical_rule_config))
{
  if (parameters.use_customized_io) {
    // Allocate page-locked CPU bounce buffers
    io_bounce_buffer_mr = std::make_unique<
      rmm::mr::fixed_size_memory_resource<gqe::memory_resource::pinned_memory_resource>>(
      &_pinned_mr,
      static_cast<int64_t>(parameters.io_bounce_buffer_size) * 1024 * 1024 * 1024,
      parameters.max_num_workers);
  }
}

}  // namespace gqe
