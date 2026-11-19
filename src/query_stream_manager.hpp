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

#include <gqe/query_context.hpp>

#include <rmm/cuda_stream.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

namespace gqe::detail {

/**
 * @brief Owns dynamically created CUDA streams for one query context.
 *
 * Streams are created on demand and retained until the manager is destroyed. Returned
 * `query_stream` capabilities share ownership so stream-dependent state may safely outlive the
 * manager. During normal query execution, the manager's own references also cover QEP-owned opaque
 * state that is destroyed before its `query_context` and therefore does not retain a capability.
 * The manager does not currently reuse streams.
 *
 * # Thread Safety
 *
 * `create_stream()` may be called concurrently until `close_and_synchronize()` begins. Closing
 * permanently disables stream creation. Destruction requires that no calls remain in progress.
 * Returned capabilities keep their streams alive independently of the manager.
 */
class query_stream_manager {
 public:
  query_stream_manager()                                       = default;
  ~query_stream_manager()                                      = default;
  query_stream_manager(query_stream_manager const&)            = delete;
  query_stream_manager& operator=(query_stream_manager const&) = delete;
  query_stream_manager(query_stream_manager&&)                 = delete;
  query_stream_manager& operator=(query_stream_manager&&)      = delete;

  /**
   * @brief Create and retain one named CUDA stream.
   *
   * @param[in] query_id Optional identifier included in the NVTX stream name.
   * @param[in] semantic_name Description of the stream's role in query execution.
   *
   * @return A capability that shares ownership of the newly created stream.
   *
   * @throws std::logic_error If stream creation has been closed by `close_and_synchronize()`.
   */
  [[nodiscard]] query_stream create_stream(std::optional<utility::uuid> query_id,
                                           std::string_view semantic_name);

  /**
   * @brief Block until previously submitted work on every owned stream completes.
   *
   * The caller must prevent concurrent submissions through previously returned stream
   * capabilities until this function returns. This operation permanently closes stream creation.
   * A concurrent `create_stream()` call fails once synchronization begins. Repeated calls are
   * permitted.
   */
  void close_and_synchronize();

 private:
  std::mutex _mutex;
  bool _is_stream_creation_closed{false};
  std::uint64_t _next_stream_id{0};
  std::vector<std::shared_ptr<rmm::cuda_stream>> _streams;
};

}  // namespace gqe::detail
