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

#include <gqe/query_context.hpp>

#include "query_stream_manager.hpp"

#include <gqe/utility/error.hpp>

#include <cuda/memory_resource>

#include <cstdlib>
#include <iomanip>

namespace gqe {

query_context::query_context(optimization_parameters parameters,
                             optimizer::optimization_configuration logical_rule_config)
  : parameters(parameters),
    logical_rule_config(std::move(logical_rule_config)),
    _stream_manager(std::make_unique<detail::query_stream_manager>())
{
  if (parameters.use_customized_io) {
    io_bounce_buffer_mr = std::make_unique<rmm::mr::fixed_size_memory_resource>(
      cuda::mr::any_resource<cuda::mr::device_accessible>{_pinned_mr},
      static_cast<int64_t>(parameters.io_bounce_buffer_size) * 1024 * 1024 * 1024,
      parameters.num_workers);
  }
}

query_context::query_context(utility::uuid query_id,
                             optimization_parameters parameters,
                             optimizer::optimization_configuration logical_rule_config)
  : query_context(parameters, std::move(logical_rule_config))
{
  this->query_id = query_id;
}

query_context::query_context(query_context&& other) noexcept = default;

query_context::~query_context() = default;

query_stream query_context::create_stream(std::string_view semantic_name)
{
  auto const stream_query_id = query_id == utility::uuid{} ? std::optional<utility::uuid>{}
                                                           : std::optional<utility::uuid>{query_id};
  return _stream_manager->create_stream(stream_query_id, semantic_name);
}

void query_context::close_and_synchronize_streams() { _stream_manager->close_and_synchronize(); }

}  // namespace gqe
