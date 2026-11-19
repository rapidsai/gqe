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

#include <gqe/executor/qep_adapter.hpp>

#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <atomic>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {

namespace {

/**
 * @brief Concatenate the chunks a streaming iterate task produced into one result container.
 *
 * The chunks share a schema; the first chunk's shape selects the path:
 *
 *  - A single chunk is returned verbatim.
 *  - Count-only chunks are summed (concatenating column-less tables would report zero rows).
 *  - Column-bearing chunks are concatenated.
 *  - Masked-table chunks concatenate the mask column and the data columns in lockstep.
 *
 * Add an `else if` arm as new container shapes gain support.
 *
 * @param[in] chunks Non-empty sequence of chunks, in order. Consumed (moved from).
 * @param[in] stream_handle CUDA stream lifetime capability.
 * @param[in] mr Memory resource.
 *
 * @pre `chunks` is not empty.
 *
 * @throws std::logic_error If a chunk has an unsupported shape.
 *
 * @return A single result container. A single input chunk is returned verbatim; concatenated
 *         column-bearing chunks produce owned columns.
 */
qep::state_container concatenate_chunks(std::vector<qep::state_container> chunks,
                                        query_stream const& stream_handle,
                                        rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  // A keyed accumulator is pipeline-breaker state between the task that builds it and its
  // successors. It must flow through `initialize`/`finalize`, never as a pipelined chunk.
  // Checked before the single-chunk fast path so the invariant holds for every chunk count.
  if (!chunks.front().empty() && chunks.front().front() != nullptr &&
      std::holds_alternative<qep::state_kind::keyed_accumulator>(*chunks.front().front())) {
    throw std::logic_error(
      "iterate_adapter_task: keyed_accumulator state must not travel as a pipelined chunk");
  }

  // A single chunk needs no concatenation: return it verbatim (zero copy), preserving its shape.
  if (chunks.size() == 1) { return std::move(chunks.front()); }

  // The chunks share a schema, so the first chunk's shape selects the path.
  if (qep::try_row_count(qep::state_container_view(chunks.front())).has_value()) {
    // Count-only: `cudf::concatenate` would collapse the column-less tables to zero rows, so sum
    // the per-chunk counts instead. Accumulate in a 64-bit total -- each per-chunk count fits in
    // `cudf::size_type`, but their sum may not -- and reject an overflowing result rather than
    // narrowing a too-large value back into `cudf::size_type`.
    std::int64_t total = 0;
    for (auto const& chunk : chunks) {
      auto const count = qep::try_row_count(qep::state_container_view(chunk));
      GQE_EXPECTS(count.has_value(),
                  "iterate_adapter_task: iterate task emitted chunks with inconsistent shapes");
      total += *count;
    }
    GQE_EXPECTS(total <= std::numeric_limits<cudf::size_type>::max(),
                "iterate_adapter_task: summed row count exceeds cudf::size_type");
    return qep::make_row_count_container(static_cast<cudf::size_type>(total));
  } else if (qep::to_table_view(qep::state_container_view(chunks.front())).has_value()) {
    // Column-bearing: concatenate the column data across chunks. Lifetimes:
    //  - `views` borrow the column data referenced by `chunks`.
    //  - `chunks` retains any source shared_state slots for this `cudf::concatenate` call.
    //  - `cudf::concatenate` returns an independent owned table.
    std::vector<cudf::table_view> views;
    views.reserve(chunks.size());
    for (auto const& chunk : chunks) {
      auto const view = qep::to_table_view(qep::state_container_view(chunk));
      GQE_EXPECTS(view.has_value(),
                  "iterate_adapter_task: iterate task emitted chunks with inconsistent shapes");
      views.push_back(*view);
    }
    auto concatenated = cudf::concatenate(views, stream, mr);
    return qep::state_container_builder()
      .add_state(std::move(*concatenated), stream_handle)
      .build();
  } else if (qep::masked_table_view::try_from(qep::state_container_view(chunks.front()))) {
    // Masked table (`[valid_mask, columns...]`): concatenate the mask column and the data columns
    // in lockstep (same chunk order) so the merged mask stays aligned with the merged rows.
    std::vector<cudf::column_view> mask_views;
    std::vector<cudf::table_view> data_views;
    mask_views.reserve(chunks.size());
    data_views.reserve(chunks.size());
    for (auto const& chunk : chunks) {
      auto const masked = qep::masked_table_view::try_from(qep::state_container_view(chunk));
      GQE_EXPECTS(masked.has_value(),
                  "iterate_adapter_task: iterate task emitted chunks with inconsistent shapes");
      mask_views.push_back(masked->mask);
      data_views.push_back(masked->columns);
    }
    auto mask = cudf::concatenate(mask_views, stream, mr);
    auto data = cudf::concatenate(data_views, stream, mr);
    return qep::state_container_builder()
      .add_state(qep::state_kind::valid_mask{std::move(mask)}, stream_handle)
      .add_state(std::move(*data), stream_handle)
      .build();
  } else {
    throw std::logic_error(
      "iterate_adapter_task: wrapped iterate task emitted a chunk of unsupported shape");
  }
}

/**
 * @brief Concatenate pipelined predecessors' results into one `next` input, watching for an
 *        upstream early exit.
 *
 * Every executed task emits a result, so a missing result is an executor bug. A predecessor that
 * early-exited emitted an empty container, because a predecessor with real output emits at least
 * the schema. When every predecessor exited there is no input left, and the caller must skip the
 * wrapped `next`. A partial exit among several predecessors is rejected. The zipped input is only
 * meaningful when all of them produced output, so proceeding would bind shifted columns.
 *
 * @param[in] predecessors The pipelined predecessors, in dependency order.
 * @param[in] context Adapter name used in error messages.
 *
 * @throws std::logic_error If a predecessor emitted no result, or only a subset of several
 *         predecessors early-exited.
 *
 * @return The concatenated input, or `std::nullopt` when every predecessor early-exited.
 */
std::optional<qep::state_container> concatenate_pipelined_results(
  std::span<gqe::task* const> predecessors, std::string_view context)
{
  qep::state_container combined;
  std::size_t exited = 0;
  for (auto* const predecessor : predecessors) {
    auto result = predecessor->qep_state_result();
    GQE_EXPECTS(result.has_value(), std::format("{}: predecessor did not emit a result", context));
    if (result->empty()) { ++exited; }
    combined.insert(combined.end(),
                    std::make_move_iterator(result->begin()),
                    std::make_move_iterator(result->end()));
  }
  if (exited == predecessors.size()) { return std::nullopt; }
  GQE_EXPECTS(exited == 0,
              std::format("{}: only a subset of the pipelined predecessors early-exited", context));
  return combined;
}

/**
 * @brief Release every unprepared same-stage task in `tasks` on behalf of one successor.
 *
 * Tasks that already carry a result are left untouched. Cross-stage tasks are stage roots that
 * the scheduler executes independently, so they are never released. Requires only the public
 * task surface, so the adapters' `cancel` implementations share it from the anonymous namespace.
 *
 * @param[in] tasks The candidate tasks, typically an adapter's dependencies.
 * @param[in] stage_id The releasing adapter's stage.
 *
 * @throws std::logic_error If an unprepared same-stage task does not implement
 *         `cancellable_task_trait`.
 */
void release_same_stage_tasks(std::span<task* const> tasks, int32_t stage_id)
{
  for (auto* const candidate : tasks) {
    if (candidate->stage_id() != stage_id) { continue; }
    if (candidate->qep_state_result().has_value()) { continue; }
    auto* const cancellable = dynamic_cast<cancellable_task_trait*>(candidate);
    GQE_EXPECTS(cancellable != nullptr,
                "release_same_stage_tasks: unprepared same-stage task is not cancellable");
    cancellable->release();
  }
}

/**
 * @brief Register `successor` with every same-stage cancellable task in `tasks`.
 *
 * Called from the adapter constructors, one edge at a time, so a task's release count accounts
 * for every same-stage successor regardless of whether that successor can itself be cancelled.
 */
void register_successor_with_same_stage_tasks(std::span<task* const> tasks, int32_t stage_id)
{
  for (auto* const candidate : tasks) {
    if (candidate->stage_id() != stage_id) { continue; }
    if (auto* const cancellable = dynamic_cast<cancellable_task_trait*>(candidate)) {
      cancellable->register_successor();
    }
  }
}

}  // namespace

namespace detail {

/**
 * @brief The accumulator of one fold, shared across its N accumulate adapters.
 *
 * Lets the accumulate adapters of a single fold cooperatively build one fold result, which the
 * paired finalize adapter then consumes.
 */
struct shared_fold_accumulator : public qep::task_private_state {
  std::once_flag init_flag;          ///< Ensures exactly one adapter seeds the accumulator.
  qep::state_container accumulator;  ///< The running fold result every accumulate adapter updates.
  cudaEvent_t init_done{};           ///< Recorded on the seeding adapter's stream after
                                     ///< `initialize`; every adapter's stream waits on it before
                                     ///< `next` so the (possibly async) device init is ordered
                                     ///< before all folds. Created once under `init_flag`.

  ~shared_fold_accumulator() override
  {
    // Best-effort: never throw from a destructor. A null event (never seeded) is a valid no-op.
    if (init_done != nullptr) { GQE_CUDA_TRY_NO_THROW(cudaEventDestroy(init_done)); }
  }
};

/**
 * @brief The accumulator of one stateful transform, shared across its N streaming-side adapters.
 *
 * Built once from the build-side predecessor; every streaming-side adapter then reads it in its
 * `next` call.
 */
struct shared_stateful_transform_accumulator : public qep::task_private_state {
  std::once_flag init_flag;          ///< Ensures exactly one adapter builds the accumulator.
  qep::state_container accumulator;  ///< The accumulator every streaming-side adapter reads.
  cudaEvent_t init_done{};           ///< Recorded on the building adapter's stream after
                                     ///< `initialize`; every adapter's stream waits on it before
                                     ///< `next` so the (possibly async) device init is ordered
                                     ///< before all reads. Created once under `init_flag`.
  std::atomic<int32_t> pending_adapters{0};  ///< Adapters not yet past `next`; each registers at
                                             ///< construction and decrements after its
                                             ///< stream-synced `remove_dependencies`. The one
                                             ///< reaching zero finalizes (all `next` device work
                                             ///< drained).
  std::atomic<bool> exhausted{false};  ///< Raised when a `next` call returns `std::nullopt` (the
                                       ///< QEP early-exit signal). Siblings that execute afterwards
                                       ///< skip their wrapped `next` call. The flag is monotonic
                                       ///< and is never cleared once raised.
  std::atomic<bool> initialized{false};  ///< Set after the `call_once` initialization. A cancelled
                                         ///< last arrival consults it: only an initialized
                                         ///< accumulator has state to finalize.

  ~shared_stateful_transform_accumulator() override
  {
    // Best-effort: never throw from a destructor. A null event (never built) is a valid no-op.
    if (init_done != nullptr) { GQE_CUDA_TRY_NO_THROW(cudaEventDestroy(init_done)); }
  }
};

}  // namespace detail

stateless_transform_adapter_task::stateless_transform_adapter_task(
  context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::vector<std::shared_ptr<task>> dependencies,
  std::unique_ptr<qep::stateless_transform_task> qep_task)
  : task(ctx_ref, task_id, stage_id, std::move(dependencies), {}), _qep_task(std::move(qep_task))
{
  GQE_EXPECTS(_qep_task != nullptr, "stateless_transform_adapter_task: qep_task must not be null");

  // Account for this adapter as one same-stage successor of each cancellable dependency, so a
  // dependency shared by several successors is only cancelled by the last release.
  register_successor_with_same_stage_tasks(this->dependencies(), this->stage_id());
  register_successor_with_same_stage_tasks(this->subqueries(), this->stage_id());
}

void stateless_transform_adapter_task::execute()
{
  prepare_dependencies();
  utility::nvtx_scoped_range range{"stateless_transform_adapter_task"};

  // Use this worker's cuDF default stream for every operation performed by this adapter.
  auto const stream_handle                = gqe::query_stream::default_stream();
  rmm::device_async_resource_ref const mr = cudf::get_current_device_resource_ref();

  auto const deps = dependencies();
  GQE_EXPECTS(!deps.empty(), "stateless_transform_adapter_task: expected at least one predecessor");

  std::optional<qep::state_container> result;
  {
    // `stateless_transform_task::next` is N-ary over its pipelined inputs: horizontally concatenate
    // every predecessor's qep_state container into one `next` input, preserving dependency order.
    // A disengaged input means every predecessor early-exited, so the wrapped `next` is skipped
    // and the exit passes on.
    //
    // Scoped so the input's shared_state refs to upstream columns drop as soon as `next` returns.
    // A column the result keeps alive holds its own ref via the result container; one the operator
    // did not carry forward loses its ref here, then its last ref when `remove_dependencies()`
    // runs below.
    auto next_input = concatenate_pipelined_results(deps, "stateless_transform_adapter_task");
    if (next_input.has_value()) {
      result = _qep_task->next(
        qep::state_container_view(*next_input), get_context_reference(), stream_handle, mr);
    }
  }

  // Drop upstream refs before publishing our result so any input column not carried forward in
  // `result` is released before downstream consumers start.
  _qep_task.reset();
  remove_dependencies(stream_handle.ref());

  // `next` returns `nullopt` to signal an early exit (see the class contract). Every task-graph
  // task must emit exactly one result, so represent the exit as an empty container.
  emit_result(result.has_value() ? std::move(*result) : qep::state_container{},
              stream_handle.ref());
}

void stateless_transform_adapter_task::cancel()
{
  if (qep_state_result().has_value()) { return; }
  auto const stream_handle = gqe::query_stream::default_stream();
  release_same_stage_tasks(dependencies(), stage_id());
  release_same_stage_tasks(subqueries(), stage_id());
  remove_dependencies(stream_handle.ref());
  emit_result(qep::state_container{}, stream_handle.ref());
}

qep::shared_state fold_accumulate_adapter_task::make_shared_accumulator()
{
  return qep::make_shared_state(
    qep::state_kind::task_private{std::make_unique<detail::shared_fold_accumulator>()});
}

fold_accumulate_adapter_task::fold_accumulate_adapter_task(
  context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::vector<std::shared_ptr<task>> dependencies,
  qep::shared_state shared_accumulator,
  std::unique_ptr<qep::fold_task> qep_task)
  : task(ctx_ref, task_id, stage_id, std::move(dependencies), {}),
    _shared_accumulator(std::move(shared_accumulator)),
    _qep_task(std::move(qep_task))
{
  GQE_EXPECTS(_shared_accumulator != nullptr,
              "fold_accumulate_adapter_task: shared_accumulator must not be null");
  GQE_EXPECTS(_qep_task != nullptr, "fold_accumulate_adapter_task: qep_task must not be null");

  auto* private_state = std::get_if<qep::state_kind::task_private>(_shared_accumulator.get());
  GQE_EXPECTS(private_state != nullptr,
              "fold_accumulate_adapter_task: shared_accumulator does not hold task_private");
  _shared_fold = static_cast<detail::shared_fold_accumulator*>(private_state->data.get());
  GQE_EXPECTS(_shared_fold != nullptr,
              "fold_accumulate_adapter_task: shared_accumulator task_private data is null");

  // Account for this adapter as one same-stage successor of each cancellable dependency, so a
  // dependency shared by several successors is only cancelled by the last release.
  register_successor_with_same_stage_tasks(this->dependencies(), this->stage_id());
  register_successor_with_same_stage_tasks(this->subqueries(), this->stage_id());
}

void fold_accumulate_adapter_task::execute()
{
  prepare_dependencies();
  utility::nvtx_scoped_range range{"fold_accumulate_adapter_task"};

  // Defensive: pin the stream / MR once and pass them everywhere so the QEP task lifecycle can't
  // drift onto different defaults.
  auto const stream_handle                = gqe::query_stream::default_stream();
  cuda::stream_ref const stream           = stream_handle.ref();
  rmm::device_async_resource_ref const mr = cudf::get_current_device_resource_ref();

  // Lazy, thread-safe one-time initialization. Sibling clones produce interchangeable accumulators
  // (the QEP task is stateless), so whichever adapter wins seeds the accumulator and the others
  // reuse it.
  //
  // `call_once` orders only the host side: it runs the seed once and publishes its host writes. The
  // seed's device work may still be running on the seeding adapter's stream, so also record an
  // event for the cross-stream wait below.
  std::call_once(_shared_fold->init_flag, [&]() {
    _shared_fold->accumulator = _qep_task->initialize(get_context_reference(), stream_handle, mr);
    GQE_CUDA_TRY(cudaEventCreateWithFlags(&_shared_fold->init_done, cudaEventDisableTiming));
    GQE_CUDA_TRY(cudaEventRecord(_shared_fold->init_done, stream.get()));
  });

  // Sibling adapters may run on different workers' per-thread default streams, so the seed is not
  // implicitly ordered before a sibling's `next`. Wait on the event to enforce that ordering, as
  // the `fold_task` contract requires. The seeding adapter waits on its own stream: a near no-op.
  GQE_CUDA_TRY(cudaStreamWaitEvent(stream.get(), _shared_fold->init_done));

  // Fold this adapter's chunk into the shared accumulator. `next` is N-ary over its pipelined
  // inputs, so horizontally concatenate every predecessor's chunk into one `next` input, preserving
  // dependency order. Concurrent calls from sibling accumulate adapters are sound per the
  // `qep::fold_task::next` thread-safety contract.
  auto const deps = dependencies();
  GQE_EXPECTS(!deps.empty(), "fold_accumulate_adapter_task: expected at least one predecessor");
  {
    // Scope `input` so its shared_state refs to upstream's columns drop as soon as `next`
    // returns. Combined with the `remove_dependencies()` below, the upstream tasks' columns
    // can be reclaimed before this adapter emits its result. A disengaged input means every
    // predecessor early-exited, so there is nothing to fold and the wrapped `next` is skipped.
    auto input = concatenate_pipelined_results(deps, "fold_accumulate_adapter_task");
    if (input.has_value()) {
      _qep_task->next(qep::state_container_view(*input),
                      qep::state_container_view(_shared_fold->accumulator),
                      get_context_reference(),
                      stream_handle,
                      mr);
    }
  }

  // Snapshot the shared accumulator (a shallow copy aliasing its inner state) before the reset
  // below: that reset may drop the last ref to the outer `shared_fold_accumulator` and dangle
  // `_shared_fold`. The finalize adapter reads the fully-folded accumulator through any accumulate
  // adapter's emitted result.
  auto result = qep::make_mutable_state_copy(qep::state_container_view(_shared_fold->accumulator));

  _shared_accumulator.reset();
  _qep_task.reset();

  // Drop upstream refs before publishing our result so any input column not carried into the
  // accumulator is released before downstream consumers start.
  remove_dependencies(stream_handle.ref());

  emit_result(std::move(result), stream_handle.ref());
}

fold_finalize_adapter_task::fold_finalize_adapter_task(
  context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::vector<std::shared_ptr<task>> dependencies,
  std::unique_ptr<qep::fold_task> qep_task)
  : task(ctx_ref, task_id, stage_id, std::move(dependencies), {}), _qep_task(std::move(qep_task))
{
  GQE_EXPECTS(_qep_task != nullptr, "fold_finalize_adapter_task: qep_task must not be null");

  // Account for this adapter as one same-stage successor of each cancellable dependency, so a
  // dependency shared by several successors is only cancelled by the last release.
  register_successor_with_same_stage_tasks(this->dependencies(), this->stage_id());
  register_successor_with_same_stage_tasks(this->subqueries(), this->stage_id());
}

void fold_finalize_adapter_task::execute()
{
  prepare_dependencies();
  utility::nvtx_scoped_range range{"fold_finalize_adapter_task"};

  auto const stream_handle                = gqe::query_stream::default_stream();
  rmm::device_async_resource_ref const mr = cudf::get_current_device_resource_ref();

  // Task graph dependency edges guarantee every accumulate adapter ran before this task. Their
  // emitted containers all alias the same accumulator (shared inner state), so any predecessor's
  // result is the fully-folded accumulator.
  auto const deps = dependencies();
  GQE_EXPECTS(!deps.empty(), "fold_finalize_adapter_task: expected at least one predecessor");
  auto accumulator = deps.front()->qep_state_result();
  GQE_EXPECTS(accumulator.has_value(),
              "fold_finalize_adapter_task: predecessor did not emit a result");

  // Drop our refs to the accumulate adapters before finalize so their emitted containers release
  // their share of the accumulator's inner state. The local `accumulator` keeps that state alive
  // until `finalize` consumes it.
  remove_dependencies(stream_handle.ref());

  // No cross-stream wait before finalize: `prepare_dependencies()` above blocks until every
  // accumulate adapter is `finished`, and a task reaches `finished` only after its `emit_result`
  // host-syncs its stream (see `task::emit_result`). So every `next`'s device work has drained,
  // even though the adapters may have run concurrently on different streams.
  auto result =
    _qep_task->finalize(std::move(*accumulator), get_context_reference(), stream_handle, mr);
  _qep_task.reset();

  // Keep the moved accumulator alive through `emit_result`'s stream synchronization. A finalize
  // implementation may retain asynchronous input operands there until its result is ready.
  emit_result(std::move(result), stream_handle.ref());
}

qep::shared_state stateful_transform_adapter_task::make_shared_accumulator()
{
  return qep::make_shared_state(qep::state_kind::task_private{
    std::make_unique<detail::shared_stateful_transform_accumulator>()});
}

stateful_transform_adapter_task::stateful_transform_adapter_task(
  context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::vector<std::shared_ptr<task>> dependencies,
  std::size_t num_materialized_inputs,
  qep::shared_state shared_accumulator,
  std::unique_ptr<qep::stateful_transform_task> qep_task)
  : task(ctx_ref, task_id, stage_id, std::move(dependencies), {}),
    _num_materialized_inputs(num_materialized_inputs),
    _shared_accumulator(std::move(shared_accumulator)),
    _qep_task(std::move(qep_task))
{
  GQE_EXPECTS(_shared_accumulator != nullptr,
              "stateful_transform_adapter_task: shared_accumulator must not be null");
  GQE_EXPECTS(_qep_task != nullptr, "stateful_transform_adapter_task: qep_task must not be null");

  auto* private_state = std::get_if<qep::state_kind::task_private>(_shared_accumulator.get());
  GQE_EXPECTS(private_state != nullptr,
              "stateful_transform_adapter_task: shared_accumulator does not hold task_private");
  _shared_transform =
    static_cast<detail::shared_stateful_transform_accumulator*>(private_state->data.get());
  GQE_EXPECTS(_shared_transform != nullptr,
              "stateful_transform_adapter_task: shared_accumulator task_private data is null");

  // Register with the shared accumulator so the last adapter to finish `next` can finalize it. All
  // adapters are constructed before the graph executes, so the count reaches N before any
  // decrement.
  _shared_transform->pending_adapters.fetch_add(1, std::memory_order_relaxed);

  // Account for this adapter as one same-stage successor of each cancellable dependency, so a
  // dependency shared by several successors is only cancelled by the last release.
  register_successor_with_same_stage_tasks(this->dependencies(), this->stage_id());
  register_successor_with_same_stage_tasks(this->subqueries(), this->stage_id());
}

void stateful_transform_adapter_task::execute()
{
  // NVTX range name shared by the early-exit and regular paths, as the C string nvtx3 takes.
  constexpr char const* range_name = "stateful_transform_adapter_task";

  // Defensive: pin the stream / MR once and pass them everywhere so the QEP task lifecycle can't
  // drift onto different defaults.
  auto const stream_handle                = gqe::query_stream::default_stream();
  cuda::stream_ref const stream           = stream_handle.ref();
  rmm::device_async_resource_ref const mr = cudf::get_current_device_resource_ref();

  // Cancel this adapter once a sibling observed the early-exit signal:
  //
  //  - Checking the flag before `prepare_dependencies` matters. Execution is pull-driven, so
  //    cancellation elides the whole unprepared upstream subtree, such as decoding rows a full
  //    fetch discards.
  //  - The flag is raised only after a `next` call, so a cancelling adapter is never the first
  //    sibling and the shared accumulator is already initialized.
  //  - `finalize` still runs exactly once through the shared `shutdown` tail, for this transform
  //    and for every cancelled upstream transform (see `cancellable_task_trait`).
  if (_shared_transform->exhausted.load(std::memory_order_acquire)) {
    utility::nvtx_scoped_range range{range_name};
    cancel();
    return;
  }

  prepare_dependencies();
  // The range opens after `prepare_dependencies`, so the inline upstream execution is attributed
  // to the upstream tasks' own ranges instead of this adapter's.
  utility::nvtx_scoped_range range{range_name};

  // Per the QEP contract, the dependencies are grouped `[materialized..., pipelined...]`: the
  // leading `_num_materialized_inputs` feed `initialize`, the rest feed `next`.
  auto const deps = dependencies();
  GQE_EXPECTS(_num_materialized_inputs < deps.size(),
              "stateful_transform_adapter_task: expected at least one pipelined predecessor");

  // Concatenate the materialized predecessors' results into the `initialize` input, preserving
  // dependency order. Materialized inputs come from pipeline breakers and are never empty, so no
  // early-exit handling applies. The columns stay alive through this task because their adapters
  // remain dependencies until `remove_dependencies()` below.
  auto const concat_materialized_results = [&]() {
    qep::state_container combined;
    for (std::size_t i = 0; i < _num_materialized_inputs; ++i) {
      auto input = deps[i]->qep_state_result();
      GQE_EXPECTS(input.has_value(),
                  "stateful_transform_adapter_task: predecessor did not emit a result");
      combined.insert(combined.end(),
                      std::make_move_iterator(input->begin()),
                      std::make_move_iterator(input->end()));
    }
    return combined;
  };

  // Lazy, thread-safe one-time initialisation from the materialized inputs (an empty container when
  // there are none). Any sibling clone produces an interchangeable accumulator (the QEP task is
  // stateless), so whichever adapter wins the race builds it; the others reuse it.
  //
  // `call_once` orders only the host side: it runs the build once and publishes its host writes.
  // The build's device work may still be running on the building adapter's stream, so also record
  // an event for the cross-stream wait below.
  std::call_once(_shared_transform->init_flag, [&]() {
    auto materialized              = concat_materialized_results();
    _shared_transform->accumulator = _qep_task->initialize(
      qep::state_container_view(materialized), get_context_reference(), stream_handle, mr);
    GQE_CUDA_TRY(cudaEventCreateWithFlags(&_shared_transform->init_done, cudaEventDisableTiming));
    GQE_CUDA_TRY(cudaEventRecord(_shared_transform->init_done, stream.get()));
    _shared_transform->initialized.store(true, std::memory_order_release);
  });

  // Sibling adapters may run on different workers' per-thread default streams, so the build is not
  // implicitly ordered before a sibling's `next`. Wait on the event to enforce that ordering.
  // The building adapter waits on its own stream: a near no-op.
  GQE_CUDA_TRY(cudaStreamWaitEvent(stream.get(), _shared_transform->init_done));

  // Drive `next` on the pipelined inputs. Concurrent calls from sibling adapters on the shared
  // accumulator are sound per the `qep::stateful_transform_task::next` thread-safety contract.
  //
  // Skip the call entirely when the stream is already over (see the class contract):
  //
  //  - A sibling observed the early-exit signal and raised `exhausted` while this adapter was in
  //    flight. A sibling already past this check races the flag instead, which the QEP contract
  //    permits.
  //  - The pipelined input is disengaged, because every pipelined predecessor early-exited.
  auto pipelined = concatenate_pipelined_results(std::span(deps).subspan(_num_materialized_inputs),
                                                 "stateful_transform_adapter_task");
  std::optional<qep::state_container> result;
  if (pipelined.has_value() && !_shared_transform->exhausted.load(std::memory_order_acquire)) {
    result = _qep_task->next(qep::state_container_view(*pipelined),
                             qep::state_container_view(_shared_transform->accumulator),
                             get_context_reference(),
                             stream_handle,
                             mr);
    if (!result.has_value()) {
      _shared_transform->exhausted.store(true, std::memory_order_release);
    }
  }

  shutdown(std::move(result), stream_handle, mr);
}

iterate_adapter_task::iterate_adapter_task(context_reference ctx_ref,
                                           int32_t task_id,
                                           int32_t stage_id,
                                           std::vector<std::shared_ptr<task>> dependencies,
                                           std::unique_ptr<qep::iterate_task> qep_task)
  : task(ctx_ref, task_id, stage_id, std::move(dependencies), {}), _qep_task(std::move(qep_task))
{
  GQE_EXPECTS(_qep_task != nullptr, "iterate_adapter_task: qep_task must not be null");

  // Account for this adapter as one same-stage successor of each cancellable dependency, so a
  // dependency shared by several successors is only cancelled by the last release.
  register_successor_with_same_stage_tasks(this->dependencies(), this->stage_id());
  register_successor_with_same_stage_tasks(this->subqueries(), this->stage_id());
}

void stateful_transform_adapter_task::cancel()
{
  if (qep_state_result().has_value()) { return; }

  // Cancel the unprepared upstream subtree first, mirroring the dependency-before-self order of
  // the execute path, then publish the empty early-exit marker through the shared tail.
  release_same_stage_tasks(dependencies(), stage_id());
  release_same_stage_tasks(subqueries(), stage_id());
  shutdown(
    std::nullopt, gqe::query_stream::default_stream(), cudf::get_current_device_resource_ref());
}

void stateful_transform_adapter_task::shutdown(std::optional<qep::state_container>&& result,
                                               gqe::query_stream const& stream_handle,
                                               rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  // Drop upstream refs before publishing our result so upstream column state is released before
  // downstream consumers start.
  remove_dependencies(stream_handle.ref());

  emit_result(result.has_value() ? std::move(*result) : qep::state_container{},
              stream_handle.ref());

  // The last adapter to arrive finalizes the shared accumulator exactly once, whether it
  // executed or was cancelled, because cancelled adapters participate in the count:
  //  - `remove_dependencies` synced this adapter's stream, so its `next` device work is drained
  //    before the decrement; the last arrival (all siblings likewise synced) thus orders
  //    `finalize` after every `next` -- no pipeline breaker, and the build state is freed at
  //    once.
  //  - On failure an adapter throws before decrementing, so the count never reaches zero,
  //    `finalize` is skipped, and the destructor reclaims the build state.
  //  - The wrapped `finalize` runs only when a sibling initialized the accumulator; one that no
  //    adapter initialized holds no state to tear down. The event wait orders the (possibly
  //    asynchronous) initialization before `finalize`; on the execute path it is a near no-op
  //    repeat of the wait before `next`.
  if (_shared_transform->pending_adapters.fetch_sub(1, std::memory_order_acq_rel) == 1 &&
      _shared_transform->initialized.load(std::memory_order_acquire)) {
    GQE_CUDA_TRY(cudaStreamWaitEvent(stream.get(), _shared_transform->init_done));
    _qep_task->finalize(
      std::move(_shared_transform->accumulator), get_context_reference(), stream_handle, mr);
    stream.sync();
  }

  // Release this adapter's hold on the shared accumulator and the wrapped task. The accumulator's
  // device resources are freed once the last adapter drops its reference (the finalizing adapter
  // above).
  _shared_accumulator.reset();
  _qep_task.reset();
}

void iterate_adapter_task::execute()
{
  prepare_dependencies();
  utility::nvtx_scoped_range range{"iterate_adapter_task"};

  // Defensive: pin the stream / MR once and pass them everywhere so the QEP task lifecycle
  // and `cudf::concatenate` can't drift onto different defaults.
  auto const stream_handle                = gqe::query_stream::default_stream();
  rmm::device_async_resource_ref const mr = cudf::get_current_device_resource_ref();

  qep::state_container iter_state;
  {
    // Concatenate every predecessor's qep_state container into a single `initialize` input,
    // preserving dependency order. Scoped so temporary shared_state refs drop as soon as
    // `initialize` returns. The iterator may retain source shared_state slots in its private state
    // when emitted borrowed views require them.
    qep::state_container init_input;
    for (auto const& dep : dependencies()) {
      auto pred_result = dep->qep_state_result();
      GQE_EXPECTS(pred_result.has_value(),
                  "iterate_adapter_task: predecessor did not emit a result");
      init_input.insert(init_input.end(),
                        std::make_move_iterator(pred_result->begin()),
                        std::make_move_iterator(pred_result->end()));
    }
    iter_state = _qep_task->initialize(
      qep::state_container_view(init_input), get_context_reference(), stream_handle, mr);
  }

  // Drain the iterator into chunks; stop on the first `nullopt`.
  std::vector<qep::state_container> chunks;
  while (true) {
    auto chunk = _qep_task->next(
      qep::state_container_view(iter_state), get_context_reference(), stream_handle, mr);
    if (!chunk) break;
    chunks.emplace_back(std::move(*chunk));
  }
  _qep_task->finalize(std::move(iter_state), get_context_reference(), stream_handle, mr);

  // An empty stream loses the schema; the wrapped iterate task must emit at least one
  // schema-correct chunk (empty rows OK).
  GQE_EXPECTS(!chunks.empty(), "iterate_adapter_task: wrapped iterate task emitted no chunks");

  // Drop the predecessor tasks before publishing our result. Source state slots retained by
  // borrowed chunks remain alive through their shared_state refs; input states that were already
  // borrowed retain their external-owner requirement.
  _qep_task.reset();
  remove_dependencies(stream_handle.ref());

  emit_result(concatenate_chunks(std::move(chunks), stream_handle, mr), stream_handle.ref());
}

void iterate_adapter_task::cancel()
{
  if (qep_state_result().has_value()) { return; }
  auto const stream_handle = gqe::query_stream::default_stream();
  release_same_stage_tasks(dependencies(), stage_id());
  release_same_stage_tasks(subqueries(), stage_id());
  remove_dependencies(stream_handle.ref());
  emit_result(qep::state_container{}, stream_handle.ref());
}

}  // namespace gqe
