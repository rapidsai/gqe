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

#include <gqe/storage/chunked_parquet_writer.hpp>

#include <gqe/context_reference.hpp>
#include <gqe/qep/shapes/write_statistics.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe {
namespace storage {

namespace {

struct chunked_parquet_writer_private : public qep::task_private_state {
  explicit chunked_parquet_writer_private(query_stream stream) : writer_stream{std::move(stream)} {}

  std::mutex writer_mutex;     ///< Serializes `writer.write()`. cuDF's `chunked_parquet_writer`
                               ///< is not thread-safe and mutates persistent internal state
                               ///< (`_table_meta`, `_agg_meta`, sink position) on every call.
  std::mutex events_mutex;     ///< Protects insertions into `thread_events`.
  query_stream writer_stream;  ///< Dedicated stream used by `writer` for all device work.
  std::unordered_map<std::thread::id, cudaEvent_t>
    thread_events;  ///< One event per host thread, reused across that thread's `next` calls.
  std::unique_ptr<cudf::io::chunked_parquet_writer> writer;  ///< Open chunked writer.
  std::atomic<bool> written_chunk{false};  ///< True after at least one chunk, including a zero-row
                                           ///< chunk, is written successfully.
  std::atomic<int64_t> total_rows_written{
    0};  ///< Cumulative rows written; read single-threaded in `finalize`.

  ~chunked_parquet_writer_private() override
  {
    // Best-effort destruction — destructors must not throw. No locking: by the QEP fold
    // contract `finalize` runs single-threaded after all `next` calls have completed.
    for (auto& [tid, event] : thread_events) {
      if (event != nullptr) { std::ignore = cudaEventDestroy(event); }
    }
  }
};

/**
 * @brief Return this thread's `cudaEvent_t`, lazily creating it on first use.
 *
 * Briefly locks `events_mutex` for the map lookup/insert. The returned handle is stable
 * for the lifetime of the private state.
 */
[[nodiscard]] cudaEvent_t get_or_create_thread_event(chunked_parquet_writer_private& state)
{
  auto const tid = std::this_thread::get_id();
  std::lock_guard<std::mutex> const guard{state.events_mutex};
  auto it = state.thread_events.find(tid);
  if (it == state.thread_events.end()) {
    cudaEvent_t event{};
    GQE_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    it = state.thread_events.emplace(tid, event).first;
  }
  return it->second;
}

}  // namespace

chunked_parquet_write_task::chunked_parquet_write_task(std::string file_path,
                                                       std::vector<std::string> column_names,
                                                       std::vector<cudf::data_type> data_types)
  : _file_path(std::move(file_path)),
    _column_names(std::move(column_names)),
    _data_types(std::move(data_types))
{
  GQE_EXPECTS(_column_names.size() == _data_types.size(),
              "chunked_parquet_write_task: column_names and data_types must have the same length",
              std::invalid_argument);
  GQE_EXPECTS(!_column_names.empty(),
              "chunked_parquet_write_task: column_names must be non-empty",
              std::invalid_argument);
  GQE_EXPECTS(
    std::ranges::none_of(_data_types,
                         [](cudf::data_type type) {
                           return type.id() != cudf::type_id::EMPTY && cudf::is_nested(type);
                         }),
    "chunked_parquet_write_task: nested LIST and STRUCT columns are not supported because "
    "data_types does not describe their child schemas",
    std::invalid_argument);
}

std::unique_ptr<qep::task> chunked_parquet_write_task::clone() const
{
  return std::make_unique<chunked_parquet_write_task>(_file_path, _column_names, _data_types);
}

qep::state_container chunked_parquet_write_task::initialize(
  context_reference ctx_ref,
  gqe::query_stream const& /* stream_handle */,
  rmm::device_async_resource_ref /* mr */) const
{
  gqe::utility::nvtx_scoped_range range{"chunked_parquet_write_task::initialize"};

  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "chunked_parquet_write_task::initialize: query context must not be null");
  auto state = std::make_unique<chunked_parquet_writer_private>(
    ctx_ref._query_context->create_stream("parquet-writer"));

  // Column names are baked into the writer's table_input_metadata. Per-chunk schema
  // (column count and types) is validated against `_data_types` in `next`.
  cudf::io::table_input_metadata metadata;
  metadata.column_metadata.resize(_column_names.size());
  for (std::size_t i = 0; i < _column_names.size(); ++i) {
    metadata.column_metadata[i].set_name(_column_names[i]);
  }

  auto sink    = cudf::io::sink_info(_file_path);
  auto options = cudf::io::chunked_parquet_writer_options::builder(sink).metadata(metadata).build();

  state->writer =
    std::make_unique<cudf::io::chunked_parquet_writer>(options, state->writer_stream.ref());

  return qep::state_container_builder()
    .add_state(qep::state_kind::task_private{std::move(state)})
    .build();
}

void chunked_parquet_write_task::next(qep::state_container_view inputs,
                                      qep::state_container_view accumulator,
                                      context_reference /* ctx_ref */,
                                      gqe::query_stream const& stream_handle,
                                      rmm::device_async_resource_ref /* mr */) const
{
  auto const stream = stream_handle.ref();
  gqe::utility::nvtx_scoped_range range{"chunked_parquet_write_task::next"};

  auto* private_slot = std::get_if<qep::state_kind::task_private>(accumulator[0].get());
  assert(private_slot != nullptr);

  auto const table = qep::to_table_view(inputs);
  GQE_EXPECTS(table.has_value(),
              "chunked_parquet_write_task::next: input is not a plain table",
              std::invalid_argument);

  // Per-chunk schema validation: column count and per-column type against the
  // constructor-supplied output schema.
  GQE_EXPECTS(static_cast<std::size_t>(table->num_columns()) == _data_types.size(),
              "chunked_parquet_write_task::next: input chunk has wrong number of columns",
              std::invalid_argument);
  for (std::size_t i = 0; i < _data_types.size(); ++i) {
    if (table->column(i).type() != _data_types[i]) {
      throw std::invalid_argument(std::format(
        "chunked_parquet_write_task::next: input chunk column type does not match the "
        "expected output schema at column {}: expected ({}, scale={}), got ({}, scale={})",
        i,
        cudf::type_to_name(_data_types[i]),
        _data_types[i].scale(),
        cudf::type_to_name(table->column(i).type()),
        table->column(i).type().scale()));
    }
  }

  // The `task_private` slot was populated in `initialize` with this concrete type, so a
  // `static_cast` is sufficient.
  auto& state = static_cast<chunked_parquet_writer_private&>(*private_slot->data);

  // Capture the input stream's current point on this thread's reusable event. No lock is
  // needed — the event is thread-owned.
  cudaEvent_t const event = get_or_create_thread_event(state);
  GQE_CUDA_TRY(cudaEventRecord(event, stream.get()));

  {
    std::lock_guard<std::mutex> const guard{state.writer_mutex};
    GQE_CUDA_TRY(cudaStreamWaitEvent(state.writer_stream.ref().get(), event, 0));
    state.writer->write(*table);

    // As of cuDF 26.06, the current cuDF implementation synchronizes internally in write(), but its
    // public contract does not promise that. Publish completion back to the caller stream so any
    // subsequent work on that stream waits until the writer has finished consuming this chunk.
    GQE_CUDA_TRY(cudaEventRecord(event, state.writer_stream.ref().get()));
    GQE_CUDA_TRY(cudaStreamWaitEvent(stream.get(), event, 0));
  }

  auto const chunk_rows = static_cast<int64_t>(table->num_rows());
  state.total_rows_written.fetch_add(chunk_rows, std::memory_order_relaxed);
  state.written_chunk.store(true, std::memory_order_relaxed);
}

qep::state_container chunked_parquet_write_task::finalize(
  qep::state_container&& accumulator,
  context_reference /* ctx_ref */,
  gqe::query_stream const& /* stream_handle */,
  rmm::device_async_resource_ref /* mr */) const
{
  gqe::utility::nvtx_scoped_range range{"chunked_parquet_write_task::finalize"};

  auto* private_slot = std::get_if<qep::state_kind::task_private>(accumulator[0].get());
  assert(private_slot != nullptr);
  auto& state = static_cast<chunked_parquet_writer_private&>(*private_slot->data);

  // cuDF needs one successful write to initialize the Parquet schema and emit a valid footer. A QEP
  // fold may be finalized without receiving any chunks, so synthesize a typed zero-row chunk for
  // that lifecycle. Keep its owning container alive until close() completes.
  qep::state_container empty_input;
  if (!state.written_chunk.load(std::memory_order_relaxed)) {
    empty_input            = qep::make_empty_state_container(_data_types);
    auto const empty_table = qep::to_table_view(empty_input);
    assert(empty_table.has_value());
    state.writer->write(*empty_table);
  }

  state.writer->close();

  auto const total_rows       = state.total_rows_written.load(std::memory_order_relaxed);
  auto const parquet_metadata = cudf::io::read_parquet_metadata(cudf::io::source_info(_file_path));
  GQE_EXPECTS(parquet_metadata.num_rows() == total_rows,
              "chunked_parquet_write_task::finalize: Parquet metadata row count does not match "
              "the rows submitted to the writer",
              std::logic_error);

  auto statistics           = table_statistics{total_rows, _data_types};
  statistics.num_row_groups = static_cast<std::size_t>(parquet_metadata.num_rowgroups());
  return qep::make_write_statistics_container(std::move(statistics));
}

}  // namespace storage
}  // namespace gqe
