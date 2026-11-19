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

#include "query_stream_manager.hpp"

#include <gqe/utility/error.hpp>

#include <nvtx3/nvToolsExtCudaRt.h>

#include <fmt/format.h>

#include <stdexcept>
#include <string>

namespace gqe::detail {

query_stream query_stream_manager::create_stream(std::optional<utility::uuid> query_id,
                                                 std::string_view semantic_name)
{
  std::scoped_lock guard{_mutex};

  GQE_EXPECTS(!_is_stream_creation_closed,
              "query_stream_manager::create_stream: stream creation is closed",
              std::logic_error);
  GQE_EXPECTS(!semantic_name.empty(),
              "query_stream_manager::create_stream: semantic name must not be empty",
              std::invalid_argument);

  auto const stream_id = _next_stream_id;
  auto const nvtx_name =
    query_id.has_value() ? fmt::format("query={}/stream={}/{}", *query_id, stream_id, semantic_name)
                         : fmt::format("stream={}/{}", stream_id, semantic_name);
  auto stream                = std::make_shared<rmm::cuda_stream>();
  cuda::stream_ref const ref = *stream;

  nvtxNameCudaStreamA(ref.get(), nvtx_name.c_str());

  _streams.push_back(stream);
  ++_next_stream_id;
  return query_stream{std::move(stream), stream_id};
}

void query_stream_manager::close_and_synchronize()
{
  {
    std::scoped_lock guard{_mutex};
    _is_stream_creation_closed = true;
  }

  // The stream collection is immutable once creation is closed, so the potentially long-running
  // synchronization does not need to hold the manager mutex.
  for (auto const& stream : _streams) {
    static_cast<cuda::stream_ref>(*stream).sync();
  }
}

}  // namespace gqe::detail
