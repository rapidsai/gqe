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

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/optimizer/optimization_configuration.hpp>
#include <gqe/utility/timer.hpp>
#include <gqe/utility/uuid.hpp>

#include <cuda/stream>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/fixed_size_memory_resource.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

namespace gqe {

namespace detail {
class query_stream_manager;
}

/**
 * @brief Copyable capability that keeps an explicit CUDA stream alive for QEP state.
 *
 * An explicit `query_stream` can only be created by the stream manager owned by a `query_context`.
 * Copies retain shared ownership of that stream, so state that records the capability cannot
 * outlive its deallocation stream. `default_stream()` is the runtime-owned exception and must be
 * acquired on the worker thread where it will be used so cuDF's per-thread default-stream behavior
 * is preserved.
 *
 * This type does not synchronize the stream.
 */
class query_stream {
 public:
  query_stream() = delete;

  /**
   * @brief Return a capability for cuDF's default CUDA stream on the calling thread.
   *
   * The CUDA runtime owns the stream. The CUDA context must outlive every use of the returned
   * capability and any stream-dependent state created from it.
   */
  [[nodiscard]] static query_stream default_stream()
  {
    return query_stream{cudf::get_default_stream()};
  }

  /** @brief Return the non-owning CUDA stream reference. */
  [[nodiscard]] cuda::stream_ref ref() const noexcept { return _ref; }

  /** @brief Return the identifier of a query-owned stream, or no value for the default stream. */
  [[nodiscard]] std::optional<std::uint64_t> id() const noexcept { return _id; }

 private:
  friend class detail::query_stream_manager;

  explicit query_stream(cuda::stream_ref ref) : _ref{ref} {}

  query_stream(std::shared_ptr<rmm::cuda_stream> stream, std::uint64_t id)
    : _stream{std::move(stream)}, _ref{*_stream}, _id{id}
  {
  }

  std::shared_ptr<rmm::cuda_stream> _stream;
  cuda::stream_ref _ref;
  std::optional<std::uint64_t> _id;
};

/**
 * @brief Query context of the current query.
 *
 * The query context centralizes all important resources and parameters that
 * are relevant for query execution. This enables optimizations that rely on
 * context, as well as logging and debugging.
 *
 * The context retains every CUDA stream returned by `create_stream()` for the query lifetime. The
 * executor destroys all QEP-owned stream-dependent state before destroying the context. State that
 * is intentionally allowed to outlive the context must instead retain copies of every corresponding
 * `query_stream`; column-bearing `shared_state` does so through its deallocation-stream metadata.
 *
 * Implementation note: Add a comment for each struct member documenting its purpose.
 * Each member must have a default setting.
 *
 * FIXME: Single source of truth for optimization parameters. There is another instance in
 * `task_manager_context`.
 */
struct query_context {
  query_context(query_context&& other) noexcept;
  explicit query_context(optimization_parameters parameters,
                         optimizer::optimization_configuration logical_rule_config =
                           optimizer::optimization_configuration());
  explicit query_context(utility::uuid query_id,
                         optimization_parameters parameters,
                         optimizer::optimization_configuration logical_rule_config);
  ~query_context();

  query_context()                                = delete;
  query_context(query_context const&)            = delete;
  query_context& operator=(query_context const&) = delete;

  /**
   * @brief Create a query-owned CUDA stream.
   *
   * The caller supplies the semantic portion of the stream's NVTX name. A non-nil query identifier
   * and a unique query-local stream identifier are added automatically. Streams are not reused.
   * The context and every returned capability share ownership of the stream.
   *
   * This method is thread-safe. Each stream is created on the calling thread's current CUDA
   * device. Once `close_and_synchronize_streams()` begins, stream creation is permanently closed
   * for this context.
   *
   * @param[in] semantic_name Description of the stream's role in query execution.
   *
   * @return A capability that shares ownership of the newly created stream.
   *
   * @throws std::invalid_argument If `semantic_name` is empty.
   * @throws std::logic_error If stream creation has been closed by
   *                          `close_and_synchronize_streams()`.
   */
  [[nodiscard]] query_stream create_stream(std::string_view semantic_name);

  /**
   * @brief Block until previously submitted work on every query-owned stream completes.
   *
   * This operation establishes a host-side boundary before query results are consumed outside
   * this context, such as conversion to Arrow on a caller-owned stream. It does not synchronize
   * runtime-owned default streams. The executor must finish task submission before calling this
   * method. Result consumption may then use a previously acquired stream and must complete before
   * context destruction. Calling this method permanently closes stream creation for this context.
   * Repeated calls are permitted.
   */
  void close_and_synchronize_streams();

  utility::uuid query_id;

  gqe::optimization_parameters parameters;

  gqe::optimizer::optimization_configuration logical_rule_config;

  // Memory resource used to allocate host-accessible bounce buffers for the customized Parquet
  // reader.
  std::unique_ptr<rmm::mr::fixed_size_memory_resource> io_bounce_buffer_mr = nullptr;

  utility::bandwidth_timer disk_timer{"disk_read"}, h2d_timer{"h2d"}, decomp_timer{"decompress"},
    decode_timer{"decode"};

 private:
  gqe::memory_resource::pinned_memory_resource _pinned_mr;

  // Retains all dynamically created query streams for the query lifetime.
  std::unique_ptr<detail::query_stream_manager> _stream_manager;
};

}  // namespace gqe
