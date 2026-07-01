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

#include "utilities.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/utility/error.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_scalar.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;

/**
 * @brief Iterate task that emits the supplied chunks one by one, then `nullopt`.
 *
 * A second `clone()` yields a fresh copy with the same chunks (suitable for re-running the
 * test through the adapter's clone path).
 */
class chunked_iterate_task : public gqe::qep::iterate_task {
 public:
  explicit chunked_iterate_task(std::vector<std::vector<int64_t>> chunks)
    : _chunks(std::move(chunks))
  {
  }

  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    if (_cursor >= _chunks.size()) { return std::nullopt; }
    auto const& values = _chunks[_cursor++];
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper(values.begin(), values.end()).release());
    return gqe::qep::state_container_builder().add_state(cudf::table{std::move(columns)}).build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<chunked_iterate_task>(_chunks);
  }

 private:
  std::vector<std::vector<int64_t>> _chunks;
  mutable std::size_t _cursor = 0;
};

/**
 * @brief Iterate task that throws if its `initialize` input does not contain exactly the
 *        expected number of slots. Used to verify the adapter's N-ary input concatenation.
 *
 * The body emits a single one-row chunk so `execute()` can complete cleanly when the input
 * count check passes; otherwise the `initialize` throw surfaces through `execute()`.
 */
class expects_input_slots_iterate_task : public gqe::qep::iterate_task {
 public:
  explicit expects_input_slots_iterate_task(std::size_t expected_slots)
    : _expected_slots(expected_slots)
  {
  }

  gqe::qep::state_container initialize(gqe::qep::state_container_view inputs,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    GQE_EXPECTS(inputs.size() == _expected_slots,
                "expects_input_slots_iterate_task: unexpected number of input slots");
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    if (_emitted) { return std::nullopt; }
    _emitted = true;
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper{1}.release());
    return gqe::qep::state_container_builder().add_state(cudf::table{std::move(columns)}).build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<expects_input_slots_iterate_task>(_expected_slots);
  }

 private:
  std::size_t _expected_slots;
  mutable bool _emitted = false;
};

/**
 * @brief Iterate task that emits no chunks (immediate `nullopt`). Loses schema; the adapter
 *        must reject this.
 */
class empty_iterate_task : public gqe::qep::iterate_task {
 public:
  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    return std::nullopt;
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<empty_iterate_task>();
  }
};

/**
 * @brief Iterate task that emits each supplied row count as a count-only chunk
 *        (`[row_count{N}]`, no columns), then `nullopt`.
 */
class count_only_iterate_task : public gqe::qep::iterate_task {
 public:
  explicit count_only_iterate_task(std::vector<cudf::size_type> counts) : _counts(std::move(counts))
  {
  }

  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    if (_cursor >= _counts.size()) { return std::nullopt; }
    return gqe::qep::make_row_count_container(_counts[_cursor++]);
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<count_only_iterate_task>(_counts);
  }

 private:
  std::vector<cudf::size_type> _counts;
  mutable std::size_t _cursor = 0;
};

/**
 * @brief Iterate task that emits each `(data, mask)` pair as a masked-table chunk
 *        (`[valid_mask, column]`), then `nullopt`.
 */
class masked_iterate_task : public gqe::qep::iterate_task {
 public:
  using chunk = std::pair<std::vector<int64_t>, std::vector<bool>>;

  explicit masked_iterate_task(std::vector<chunk> chunks) : _chunks(std::move(chunks)) {}

  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    if (_cursor >= _chunks.size()) { return std::nullopt; }
    auto const& [values, mask_bits] = _chunks[_cursor++];
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper(values.begin(), values.end()).release());
    auto mask = bool_column_wrapper(mask_bits.begin(), mask_bits.end()).release();
    return gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{std::move(mask)})
      .add_state(cudf::table{std::move(columns)})
      .build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<masked_iterate_task>(_chunks);
  }

 private:
  std::vector<chunk> _chunks;
  mutable std::size_t _cursor = 0;
};

/**
 * @brief Iterate task that emits each `(values, validity)` pair as a plain single-column table
 *        whose column carries a null bitmap, then `nullopt`.
 */
class nullable_iterate_task : public gqe::qep::iterate_task {
 public:
  using chunk = std::pair<std::vector<int64_t>, std::vector<bool>>;

  explicit nullable_iterate_task(std::vector<chunk> chunks) : _chunks(std::move(chunks)) {}

  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    if (_cursor >= _chunks.size()) { return std::nullopt; }
    auto const& [values, validity] = _chunks[_cursor++];
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(
      int64_column_wrapper(values.begin(), values.end(), validity.begin()).release());
    return gqe::qep::state_container_builder().add_state(cudf::table{std::move(columns)}).build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<nullable_iterate_task>(_chunks);
  }

 private:
  std::vector<chunk> _chunks;
  mutable std::size_t _cursor = 0;
};

class IterateAdapterTest : public gqe::test::BaseFixture {
 protected:
  IterateAdapterTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  /**
   * @brief Build an upstream `executed_task` that emits a single-column `int64` table.
   *
   * `executed_task` stores the result as `owned`; `task::qep_state_result()` wraps it as a
   * one-`cudf_column_view`-slot container.
   */
  std::shared_ptr<gqe::test::executed_task> make_upstream(std::vector<int64_t> values,
                                                          int32_t task_id)
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper(values.begin(), values.end()).release());
    return std::make_shared<gqe::test::executed_task>(
      ctx_ref, task_id, /*stage_id=*/0, std::make_unique<cudf::table>(std::move(columns)));
  }

  gqe::context_reference ctx_ref;
};

/**
 * @brief A single chunk passes through the adapter and is emitted as the result.
 */
TEST_F(IterateAdapterTest, SingleChunkEmitsResult)
{
  auto adapter = std::make_unique<gqe::iterate_adapter_task>(
    ctx_ref,
    /*task_id=*/0,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{},
    std::make_unique<chunked_iterate_task>(std::vector<std::vector<int64_t>>{{10, 20, 30}}));

  adapter->execute();

  auto state = adapter->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto tv = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(tv.has_value());
  ASSERT_EQ(tv->num_columns(), 1);
  EXPECT_EQ(tv->num_rows(), 3);

  int64_column_wrapper const expected{10, 20, 30};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(tv->column(0), expected);
}

/**
 * @brief Multiple chunks are concatenated into a single output table.
 */
TEST_F(IterateAdapterTest, MultipleChunksAreConcatenated)
{
  auto adapter = std::make_unique<gqe::iterate_adapter_task>(
    ctx_ref,
    /*task_id=*/0,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{},
    std::make_unique<chunked_iterate_task>(
      std::vector<std::vector<int64_t>>{{10, 20}, {30}, {40, 50, 60}}));

  adapter->execute();

  auto state = adapter->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto tv = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(tv.has_value());
  ASSERT_EQ(tv->num_columns(), 1);
  EXPECT_EQ(tv->num_rows(), 6);

  int64_column_wrapper const expected{10, 20, 30, 40, 50, 60};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(tv->column(0), expected);
}

/**
 * @brief Multiple count-only chunks (column-less `[row_count{N}]`) are summed into a single
 *        count-only result. A plain `cudf::concatenate` would collapse the zero-column tables to
 *        zero rows, so the adapter must sum the per-chunk counts instead.
 */
TEST_F(IterateAdapterTest, CountOnlyChunksAreSummed)
{
  auto adapter = std::make_unique<gqe::iterate_adapter_task>(
    ctx_ref,
    /*task_id=*/0,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{},
    std::make_unique<count_only_iterate_task>(std::vector<cudf::size_type>{2, 3, 5}));

  adapter->execute();

  auto state = adapter->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto const count = gqe::qep::try_row_count(*state);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 10);
}

/**
 * @brief Count-only chunks whose summed row count exceeds `cudf::size_type` are rejected rather
 *        than silently narrowed/overflowed.
 */
TEST_F(IterateAdapterTest, CountOnlyOverflowThrows)
{
  auto const max_count = std::numeric_limits<cudf::size_type>::max();
  auto adapter         = std::make_unique<gqe::iterate_adapter_task>(
    ctx_ref,
    /*task_id=*/0,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{},
    std::make_unique<count_only_iterate_task>(std::vector<cudf::size_type>{max_count, 1}));

  EXPECT_THROW(adapter->execute(), std::logic_error);
}

/**
 * @brief Masked-table chunks are concatenated mask-and-data in lockstep: the result is a masked
 *        table whose mask and columns are the per-chunk masks and columns in order.
 */
TEST_F(IterateAdapterTest, MaskedChunksAreConcatenated)
{
  auto adapter = std::make_unique<gqe::iterate_adapter_task>(
    ctx_ref,
    /*task_id=*/0,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{},
    // Row 1 is masked out (mask=false); its data value is a don't-care, so use a sentinel.
    std::make_unique<masked_iterate_task>(
      std::vector<masked_iterate_task::chunk>{{{10, -999}, {true, false}}, {{30}, {true}}}));

  adapter->execute();

  auto state = adapter->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto const masked = gqe::qep::masked_table_view::try_from(*state);
  ASSERT_TRUE(masked.has_value());
  EXPECT_EQ(masked->row_count, 3);
  ASSERT_EQ(masked->columns.num_columns(), 1);

  int64_column_wrapper const expected_data{10, -999, 30};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(masked->columns.column(0), expected_data);
  bool_column_wrapper const expected_mask{true, false, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(masked->mask, expected_mask);
}

/**
 * @brief Concatenating column-bearing chunks preserves each column's null bitmap: a null in an
 *        input chunk stays null in the merged result.
 */
TEST_F(IterateAdapterTest, ChunksWithNullsPreserveNullBitmap)
{
  auto adapter = std::make_unique<gqe::iterate_adapter_task>(
    ctx_ref,
    /*task_id=*/0,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{},
    // Row 1 is null; its value is a don't-care, so use a sentinel.
    std::make_unique<nullable_iterate_task>(
      std::vector<nullable_iterate_task::chunk>{{{10, -999}, {true, false}}, {{30}, {true}}}));

  adapter->execute();

  auto state = adapter->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto tv = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(tv.has_value());
  ASSERT_EQ(tv->num_columns(), 1);
  EXPECT_EQ(tv->num_rows(), 3);

  int64_column_wrapper const expected({10, -999, 30}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(tv->column(0), expected);
}

/**
 * @brief An iterator that emits no chunks loses its output schema; the adapter must refuse.
 */
TEST_F(IterateAdapterTest, EmptyStreamThrows)
{
  auto adapter =
    std::make_unique<gqe::iterate_adapter_task>(ctx_ref,
                                                /*task_id=*/0,
                                                /*stage_id=*/0,
                                                std::vector<std::shared_ptr<gqe::task>>{},
                                                std::make_unique<empty_iterate_task>());

  EXPECT_THROW(adapter->execute(), std::logic_error);
}

/**
 * @brief Multiple predecessors' state containers are concatenated (slot-wise, in dependency
 *        order) into a single `initialize` input. The wrapped iterator's input-slot
 *        precondition would throw on a mismatch.
 */
TEST_F(IterateAdapterTest, NAryPredecessorsConcatenated)
{
  // Two upstream tasks, each emitting a one-column table → one slot per upstream.
  std::vector<std::shared_ptr<gqe::task>> deps;
  deps.push_back(make_upstream({1, 2, 3}, /*task_id=*/0));
  deps.push_back(make_upstream({4, 5}, /*task_id=*/1));

  auto adapter = std::make_unique<gqe::iterate_adapter_task>(
    ctx_ref,
    /*task_id=*/2,
    /*stage_id=*/0,
    std::move(deps),
    std::make_unique<expects_input_slots_iterate_task>(/*expected_slots=*/2));

  EXPECT_NO_THROW(adapter->execute());
}

/**
 * @brief Minimal fold task that counts the total number of input rows.
 *
 * `initialize` seeds a `[row_count{0}]` accumulator, `next` adds each input's row count to it, and
 * `finalize` (the base default) returns the accumulator unchanged.
 */
class row_count_fold_task : public gqe::qep::fold_task {
 public:
  gqe::qep::state_container initialize(gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return gqe::qep::make_row_count_container(0);
  }

  void next(gqe::qep::state_container_view inputs,
            gqe::qep::state_container_view accumulator,
            gqe::context_reference,
            rmm::cuda_stream_view,
            rmm::device_async_resource_ref) const override
  {
    // The accumulator is a single `row_count` slot; bump it by this input's row count. The inner
    // shared_ptr is const but its pointee is mutable, per the `state_container_view` contract.
    auto& count = std::get<gqe::qep::state_kind::row_count>(*accumulator.front());
    count.value += gqe::qep::get_row_count(inputs);
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<row_count_fold_task>();
  }
};

/**
 * @brief Fold task whose accumulator counts the number of input columns its `next` has seen.
 *
 * Used to check that multiple pipelined inputs are horizontally concatenated into a single `next`
 * call: with two single-column inputs, `next` sees a two-column container.
 */
class input_column_count_fold_task : public gqe::qep::fold_task {
 public:
  gqe::qep::state_container initialize(gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return gqe::qep::make_row_count_container(0);
  }

  void next(gqe::qep::state_container_view inputs,
            gqe::qep::state_container_view accumulator,
            gqe::context_reference,
            rmm::cuda_stream_view,
            rmm::device_async_resource_ref) const override
  {
    auto const table = gqe::qep::to_table_view(inputs);
    auto& count      = std::get<gqe::qep::state_kind::row_count>(*accumulator.front());
    count.value += table.has_value() ? table->num_columns() : 0;
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<input_column_count_fold_task>();
  }
};

/**
 * @brief Accumulator state for `delayed_seed_fold_task`, carried in the fold's `task_private` slot.
 *
 * Holds the running count in device memory plus a mutex that serializes the cross-thread
 * read-modify-write in `next` (the `fold_task` external thread-safety contract). One instance is
 * shared by every accumulate adapter of the fold via the shared accumulator container, so the fold
 * task itself stays stateless.
 */
struct device_count_accumulator : public gqe::qep::task_private_state {
  explicit device_count_accumulator(rmm::cuda_stream_view stream) : running_count{stream} {}

  rmm::device_scalar<std::int64_t> running_count;
  std::mutex mutex;
};

/**
 * @brief Fold task that exposes a missing `initialize`-before-`next` stream ordering.
 *
 * Counts rows; `initialize` poisons the accumulator and defers the real seed behind a delay, so a
 * `next` not stream-ordered after `initialize` folds the poison. See @ref
 * FoldAdapterTest_ConcurrentAccumulatesAreOrderedAfterInitializeDeviceWork_Test for the full
 * scenario, diagram, and limitations.
 */
class delayed_seed_fold_task : public gqe::qep::fold_task {
 public:
  gqe::qep::state_container initialize(gqe::context_reference,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref) const override
  {
    auto accumulator = std::make_unique<device_count_accumulator>(stream);

    // Poison synchronously, so an unordered `next` reads a detectably wrong value rather than
    // uninitialized memory that might happen to equal the seed.
    accumulator->running_count.set_value_async(poison_value, stream);
    stream.synchronize();

    // Delay, then write the seed: it stays pending behind the delay, so an unordered sibling `next`
    // observes the poison instead.
    GQE_CUDA_TRY(cudaLaunchHostFunc(stream.value(), &sleep_for_race_window, nullptr));
    accumulator->running_count.set_value_async(seed_value, stream);

    return gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::task_private{std::move(accumulator)})
      .build();
  }

  void next(gqe::qep::state_container_view inputs,
            gqe::qep::state_container_view accumulator,
            gqe::context_reference,
            rmm::cuda_stream_view stream,
            rmm::device_async_resource_ref) const override
  {
    auto const count = static_cast<std::int64_t>(gqe::qep::get_row_count(inputs));
    auto& acc        = as_device_count_accumulator(accumulator);

    // Serialize the read-modify-write across sibling adapters. `value()` copies device->host and
    // syncs the stream; the trailing sync commits the write before the lock is released.
    std::lock_guard<std::mutex> const lock{acc.mutex};
    std::int64_t const total = acc.running_count.value(stream) + count;
    acc.running_count.set_value_async(total, stream);
    stream.synchronize();
  }

  gqe::qep::state_container finalize(gqe::qep::state_container&& accumulator,
                                     gqe::context_reference,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref) const override
  {
    auto& acc = as_device_count_accumulator(accumulator);
    return gqe::qep::make_row_count_container(
      static_cast<cudf::size_type>(acc.running_count.value(stream)));
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<delayed_seed_fold_task>();
  }

 private:
  static constexpr std::int64_t poison_value = -1;  ///< Wrong value an unordered `next` would fold.
  static constexpr std::int64_t seed_value   = 0;   ///< Correct initial accumulator value.
  ///< Margin that keeps the seed pending past when a racing `next` issues its read.
  static constexpr std::chrono::milliseconds race_window{25};

  static void CUDART_CB sleep_for_race_window(void*) { std::this_thread::sleep_for(race_window); }

  template <typename Container>
  static device_count_accumulator& as_device_count_accumulator(Container&& accumulator)
  {
    auto& state = std::get<gqe::qep::state_kind::task_private>(*accumulator.front());
    return *static_cast<device_count_accumulator*>(state.data.get());
  }
};

class FoldAdapterTest : public gqe::test::BaseFixture {
 protected:
  FoldAdapterTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  /**
   * @brief Build an upstream `executed_task` emitting a single-column int64 table with `num_rows`
   *        rows. The fold only reads the row count, so the column values are arbitrary.
   */
  std::shared_ptr<gqe::test::executed_task> make_upstream(cudf::size_type num_rows, int32_t task_id)
  {
    std::vector<int64_t> const values(static_cast<std::size_t>(num_rows), 0);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper(values.begin(), values.end()).release());
    return std::make_shared<gqe::test::executed_task>(
      ctx_ref, task_id, /*stage_id=*/0, std::make_unique<cudf::table>(std::move(columns)));
  }

  gqe::context_reference ctx_ref;
};

/**
 * @brief A single accumulate adapter folds its one upstream partition; the finalize adapter then
 *        produces the fold result over that partition.
 */
TEST_F(FoldAdapterTest, SinglePartitionFoldCountsRows)
{
  auto upstream    = make_upstream(/*num_rows=*/3, /*task_id=*/0);
  auto accumulator = gqe::fold_accumulate_adapter_task::make_shared_accumulator();

  auto accumulate = std::make_shared<gqe::fold_accumulate_adapter_task>(
    ctx_ref,
    /*task_id=*/1,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{upstream},
    accumulator,
    std::make_unique<row_count_fold_task>());
  accumulate->execute();

  auto finalize = std::make_unique<gqe::fold_finalize_adapter_task>(
    ctx_ref,
    /*task_id=*/2,
    /*stage_id=*/1,
    std::vector<std::shared_ptr<gqe::task>>{accumulate},
    std::make_unique<row_count_fold_task>());
  finalize->execute();

  auto state = finalize->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(3));
}

/**
 * @brief Two accumulate adapters share one accumulator; the finalize adapter returns the combined
 *        fold over both partitions. The accumulates are executed sequentially here (as a single
 *        stage worker would), so no concurrent `next` is exercised.
 */
TEST_F(FoldAdapterTest, FinalizeCombinesAllAccumulatePartitions)
{
  auto accumulator = gqe::fold_accumulate_adapter_task::make_shared_accumulator();

  auto upstream_a = make_upstream(/*num_rows=*/3, /*task_id=*/0);
  auto upstream_b = make_upstream(/*num_rows=*/2, /*task_id=*/1);

  auto accumulate_a = std::make_shared<gqe::fold_accumulate_adapter_task>(
    ctx_ref,
    /*task_id=*/2,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{upstream_a},
    accumulator,
    std::make_unique<row_count_fold_task>());
  auto accumulate_b = std::make_shared<gqe::fold_accumulate_adapter_task>(
    ctx_ref,
    /*task_id=*/3,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{upstream_b},
    accumulator,
    std::make_unique<row_count_fold_task>());

  accumulate_a->execute();
  accumulate_b->execute();

  auto finalize = std::make_unique<gqe::fold_finalize_adapter_task>(
    ctx_ref,
    /*task_id=*/4,
    /*stage_id=*/1,
    std::vector<std::shared_ptr<gqe::task>>{accumulate_a, accumulate_b},
    std::make_unique<row_count_fold_task>());

  finalize->execute();

  auto state = finalize->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(5));
}

/**
 * @brief Multiple pipelined inputs are horizontally concatenated into a single `next` call: two
 *        single-column partitions are seen by the fold's `next` as one two-column input.
 */
TEST_F(FoldAdapterTest, MultiplePipelinedInputsAreConcatenated)
{
  auto upstream_a = make_upstream(/*num_rows=*/3, /*task_id=*/0);
  auto upstream_b = make_upstream(/*num_rows=*/3, /*task_id=*/1);

  auto accumulator = gqe::fold_accumulate_adapter_task::make_shared_accumulator();
  auto accumulate  = std::make_unique<gqe::fold_accumulate_adapter_task>(
    ctx_ref,
    /*task_id=*/2,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{upstream_a, upstream_b},
    accumulator,
    std::make_unique<input_column_count_fold_task>());

  accumulate->execute();

  auto state = accumulate->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(2));
}

/**
 * @brief A fold's `initialize` device work is ordered before every concurrent accumulate `next`.
 *
 * Two accumulate adapters share one accumulator and run on separate threads, so each is on its own
 * per-thread default stream -- putting the fold's `initialize` device work and a sibling's `next`
 * on different streams. `delayed_seed_fold_task` poisons the accumulator, then defers the real seed
 * behind a delay; a `next` not stream-ordered after `initialize` folds the poison and the final
 * count is wrong. With the adapter's cross-stream ordering, every `next` observes the seed and the
 * count is the exact row total. Without the ordering, the invalid interleaving is:
 *
 *     seeder stream:   poison ──▶│■■■ delay ■■■│──▶ seed
 *     sibling stream:       next: read ──▶ fold (-1)
 *                                 ▲
 *                                 └─ reads the poison: nothing orders this read after the
 *                                    seed write, and the seed is still stuck behind the delay
 *
 * # Limitation
 *
 * Detection is probabilistic: a failing run proves the ordering is missing, but a passing run does
 * not prove it present (hence the repeated trials). The delay only widens the race window; it
 * cannot force the order in which the two streams' device work completes, and neither host
 * scheduling nor stream priority can.
 *
 * The strawman that would be deterministic -- record an event after the sibling's racing read and
 * have the seed `cudaStreamWaitEvent` on it, so the seed provably runs after the read -- deadlocks
 * the correct adapter: its fix already makes the read wait on the seed (via the init-done event),
 * so "seed waits on read" plus "read waits on seed" is a cycle. That opposition is fundamental, so
 * the timing margin is the best trigger available.
 */
TEST_F(FoldAdapterTest, ConcurrentAccumulatesAreOrderedAfterInitializeDeviceWork)
{
  constexpr cudf::size_type rows_a = 3;
  constexpr cudf::size_type rows_b = 2;
  constexpr int num_trials         = 4;

  for (int trial = 0; trial < num_trials; ++trial) {
    auto accumulator = gqe::fold_accumulate_adapter_task::make_shared_accumulator();
    auto upstream_a  = make_upstream(rows_a, /*task_id=*/0);
    auto upstream_b  = make_upstream(rows_b, /*task_id=*/1);

    auto accumulate_a = std::make_shared<gqe::fold_accumulate_adapter_task>(
      ctx_ref,
      /*task_id=*/2,
      /*stage_id=*/0,
      std::vector<std::shared_ptr<gqe::task>>{upstream_a},
      accumulator,
      std::make_unique<delayed_seed_fold_task>());
    auto accumulate_b = std::make_shared<gqe::fold_accumulate_adapter_task>(
      ctx_ref,
      /*task_id=*/3,
      /*stage_id=*/0,
      std::vector<std::shared_ptr<gqe::task>>{upstream_b},
      accumulator,
      std::make_unique<delayed_seed_fold_task>());

    // Distinct threads give the two adapters distinct per-thread default streams.
    std::thread thread_a{[&] { accumulate_a->execute(); }};
    std::thread thread_b{[&] { accumulate_b->execute(); }};
    thread_a.join();
    thread_b.join();

    auto finalize = std::make_unique<gqe::fold_finalize_adapter_task>(
      ctx_ref,
      /*task_id=*/4,
      /*stage_id=*/1,
      std::vector<std::shared_ptr<gqe::task>>{accumulate_a, accumulate_b},
      std::make_unique<delayed_seed_fold_task>());
    finalize->execute();

    auto state = finalize->qep_state_result();
    ASSERT_TRUE(state.has_value());
    EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(rows_a + rows_b));
  }
}

/**
 * @brief optional_transform task that keeps only the input rows whose value is `> threshold`.
 *
 * When no row passes, it emits an empty (zero-row) table that still carries the schema, rather
 * than `nullopt`.
 */
class greater_than_filter_transform_task : public gqe::qep::optional_transform_task {
 public:
  explicit greater_than_filter_transform_task(int64_t threshold) : _threshold(threshold) {}

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view inputs,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr) const override
  {
    auto const tv = gqe::qep::to_table_view(inputs);
    GQE_EXPECTS(tv.has_value(), "greater_than_filter_transform_task: input is not a table");

    cudf::numeric_scalar<int64_t> const threshold{_threshold, /*is_valid=*/true, stream, mr};
    auto const mask = cudf::binary_operation(tv->column(0),
                                             threshold,
                                             cudf::binary_operator::GREATER,
                                             cudf::data_type{cudf::type_id::BOOL8},
                                             stream,
                                             mr);
    auto filtered   = cudf::apply_boolean_mask(*tv, mask->view(), stream, mr);
    return gqe::qep::state_container_builder().add_state(std::move(*filtered)).build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<greater_than_filter_transform_task>(_threshold);
  }

 private:
  int64_t _threshold;
};

/**
 * @brief optional_transform task that produces no output at all.
 */
class nullopt_transform_task : public gqe::qep::optional_transform_task {
 public:
  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    return std::nullopt;
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<nullopt_transform_task>();
  }
};

/**
 * @brief optional_transform task that horizontally concatenates its inputs — its output places the
 *        input columns side by side, in dependency order.
 *
 * With N pipelined predecessors contributing one column each, the output carries N columns, one per
 * predecessor. Used to verify the adapter's N-ary input concatenation.
 */
class horizontal_concat_transform_task : public gqe::qep::optional_transform_task {
 public:
  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view inputs,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr) const override
  {
    auto const tv = gqe::qep::to_table_view(inputs);
    GQE_EXPECTS(tv.has_value(), "horizontal_concat_transform_task: inputs are not columns");
    return gqe::qep::state_container_builder().add_state(cudf::table{*tv, stream, mr}).build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<horizontal_concat_transform_task>();
  }
};

class OptionalTransformAdapterTest : public gqe::test::BaseFixture {
 protected:
  OptionalTransformAdapterTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  /**
   * @brief Build an upstream `executed_task` that emits a single-column `int64` table.
   */
  std::shared_ptr<gqe::test::executed_task> make_upstream(std::vector<int64_t> values,
                                                          int32_t task_id)
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper(values.begin(), values.end()).release());
    return std::make_shared<gqe::test::executed_task>(
      ctx_ref, task_id, /*stage_id=*/0, std::make_unique<cudf::table>(std::move(columns)));
  }

  gqe::context_reference ctx_ref;
};

/**
 * @brief The adapter runs the wrapped transform on its predecessor's result and emits the output.
 */
TEST_F(OptionalTransformAdapterTest, RunsTransformOnPredecessorResult)
{
  auto upstream = make_upstream({10, 20, 30}, /*task_id=*/0);

  auto adapter = std::make_unique<gqe::optional_transform_adapter_task>(
    ctx_ref,
    /*task_id=*/1,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{upstream},
    std::make_unique<greater_than_filter_transform_task>(/*threshold=*/15));

  adapter->execute();

  auto state = adapter->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto tv = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(tv.has_value());
  ASSERT_EQ(tv->num_columns(), 1);
  EXPECT_EQ(tv->num_rows(), 2);

  int64_column_wrapper const expected{20, 30};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(tv->column(0), expected);
}

/**
 * @brief A filter that drops every row still emits a schema-correct empty (zero-row) table.
 */
TEST_F(OptionalTransformAdapterTest, FilterDroppingAllRowsEmitsEmptyTable)
{
  auto upstream = make_upstream({1, 2, 3}, /*task_id=*/0);

  auto adapter = std::make_unique<gqe::optional_transform_adapter_task>(
    ctx_ref,
    /*task_id=*/1,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{upstream},
    std::make_unique<greater_than_filter_transform_task>(/*threshold=*/100));

  adapter->execute();

  auto state = adapter->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto tv = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(tv.has_value());
  EXPECT_EQ(tv->num_columns(), 1);
  EXPECT_EQ(tv->num_rows(), 0);
}

/**
 * @brief A `nullopt` result (no output at all) is forwarded as an empty container, not an error.
 */
TEST_F(OptionalTransformAdapterTest, NulloptResultEmitsEmptyContainer)
{
  auto upstream = make_upstream({1, 2, 3}, /*task_id=*/0);

  auto adapter = std::make_unique<gqe::optional_transform_adapter_task>(
    ctx_ref,
    /*task_id=*/1,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{upstream},
    std::make_unique<nullopt_transform_task>());

  adapter->execute();

  auto state = adapter->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_TRUE(state->empty());
}

/**
 * @brief The adapter concatenates multiple pipelined predecessors into a single `next` input,
 *        preserving dependency order.
 */
TEST_F(OptionalTransformAdapterTest, ConcatenatesMultiplePipelinedInputs)
{
  auto upstream_a = make_upstream({10, 20, 30}, /*task_id=*/0);
  auto upstream_b = make_upstream({40, 50, 60}, /*task_id=*/1);

  auto adapter = std::make_unique<gqe::optional_transform_adapter_task>(
    ctx_ref,
    /*task_id=*/2,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{upstream_a, upstream_b},
    std::make_unique<horizontal_concat_transform_task>());

  adapter->execute();

  auto state = adapter->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto tv = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(tv.has_value());
  ASSERT_EQ(tv->num_columns(), 2);
  EXPECT_EQ(tv->num_rows(), 3);

  // Column order follows dependency order: predecessor A then predecessor B.
  int64_column_wrapper const expected_a{10, 20, 30};
  int64_column_wrapper const expected_b{40, 50, 60};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(tv->column(0), expected_a);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(tv->column(1), expected_b);
}

/**
 * @brief Minimal stateful transform that, per probe partition, emits the build-side row count plus
 *        that partition's row count.
 *
 * `initialize` records the build input's row count in a `[row_count]` accumulator; `next` reads
 * that count back and emits it added to the probe partition's row count.
 */
class build_plus_probe_count_stateful_transform_task : public gqe::qep::stateful_transform_task {
 public:
  gqe::qep::state_container initialize(gqe::qep::state_container_view inputs,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return gqe::qep::make_row_count_container(gqe::qep::get_row_count(inputs));
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view inputs,
                                                gqe::qep::state_container_view accumulator,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    auto const build_count = gqe::qep::get_row_count(accumulator);
    auto const probe_count = gqe::qep::get_row_count(inputs);
    return gqe::qep::make_row_count_container(build_count + probe_count);
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<build_plus_probe_count_stateful_transform_task>();
  }
};

/**
 * @brief Stateful transform whose `next` emits the number of columns in its pipelined input.
 *
 * Used to check that multiple pipelined inputs are horizontally concatenated into a single `next`
 * call: with two single-column inputs, `next` sees a two-column container.
 */
class input_column_count_stateful_transform_task : public gqe::qep::stateful_transform_task {
 public:
  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return gqe::qep::make_row_count_container(0);
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view inputs,
                                                gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    auto const table = gqe::qep::to_table_view(inputs);
    return gqe::qep::make_row_count_container(table.has_value() ? table->num_columns() : 0);
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<input_column_count_stateful_transform_task>();
  }
};

/**
 * @brief Stateful transform that counts `finalize` invocations through a shared counter.
 *
 * Every adapter holds a clone sharing the same counter, so the test can assert the executor calls
 * `finalize` exactly once for the shared accumulator (after the last `next`).
 */
class finalize_counting_stateful_transform_task : public gqe::qep::stateful_transform_task {
 public:
  explicit finalize_counting_stateful_transform_task(
    std::shared_ptr<std::atomic<int>> finalize_calls)
    : _finalize_calls{std::move(finalize_calls)}
  {
  }

  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return gqe::qep::make_row_count_container(0);
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view inputs,
                                                gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    return gqe::qep::make_row_count_container(gqe::qep::get_row_count(inputs));
  }

  void finalize(gqe::qep::state_container&&,
                gqe::context_reference,
                rmm::cuda_stream_view,
                rmm::device_async_resource_ref) const override
  {
    _finalize_calls->fetch_add(1, std::memory_order_relaxed);
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<finalize_counting_stateful_transform_task>(_finalize_calls);
  }

 private:
  std::shared_ptr<std::atomic<int>> _finalize_calls;
};

/**
 * @brief Stateful transform that captures the accumulator `finalize` receives.
 *
 * `initialize` seeds the accumulator with the materialized build input's row count; `next` passes
 * that count through unchanged; `finalize` records the accumulator's row count into a shared slot.
 * Lets the test assert `finalize` is handed the initialized accumulator, not an empty or default
 * one.
 */
class finalize_capturing_stateful_transform_task : public gqe::qep::stateful_transform_task {
 public:
  explicit finalize_capturing_stateful_transform_task(
    std::shared_ptr<std::optional<cudf::size_type>> captured)
    : _captured{std::move(captured)}
  {
  }

  gqe::qep::state_container initialize(gqe::qep::state_container_view inputs,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return gqe::qep::make_row_count_container(gqe::qep::get_row_count(inputs));
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::qep::state_container_view accumulator,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    return gqe::qep::make_row_count_container(gqe::qep::get_row_count(accumulator));
  }

  void finalize(gqe::qep::state_container&& accumulator,
                gqe::context_reference,
                rmm::cuda_stream_view,
                rmm::device_async_resource_ref) const override
  {
    *_captured = gqe::qep::get_row_count(gqe::qep::state_container_view(accumulator));
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<finalize_capturing_stateful_transform_task>(_captured);
  }

 private:
  std::shared_ptr<std::optional<cudf::size_type>> _captured;
};

/**
 * @brief Build accumulator for `delayed_seed_stateful_transform_task`, carried in the transform's
 *        `task_private` slot.
 *
 * Holds the build value in device memory. `next` only reads it, so -- unlike a fold accumulator --
 * no mutex is needed; one instance is shared by every streaming-side adapter.
 */
struct device_build_accumulator : public gqe::qep::task_private_state {
  explicit device_build_accumulator(rmm::cuda_stream_view stream) : value{stream} {}

  rmm::device_scalar<std::int64_t> value;
};

/**
 * @brief Reach the `device_build_accumulator` carried in a transform accumulator's `task_private`
 *        slot.
 */
inline device_build_accumulator& as_device_build_accumulator(
  gqe::qep::state_container_view accumulator)
{
  auto& state = std::get<gqe::qep::state_kind::task_private>(*accumulator.front());
  return *static_cast<device_build_accumulator*>(state.data.get());
}

/**
 * @brief Stateful transform that exposes a missing `initialize`-before-`next` stream ordering.
 *
 * Emits the build value plus the probe row count; `initialize` poisons the build value and defers
 * the real value behind a delay, so a `next` not stream-ordered after `initialize` reads the
 * poison. See @ref
 * StatefulTransformAdapterTest_ConcurrentNextIsOrderedAfterInitializeDeviceWork_Test for the full
 * scenario, diagram, and limitations.
 */
class delayed_seed_stateful_transform_task : public gqe::qep::stateful_transform_task {
 public:
  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref) const override
  {
    auto accumulator = std::make_unique<device_build_accumulator>(stream);

    // Poison synchronously, so an unordered `next` reads a detectably wrong value rather than
    // uninitialized memory that might happen to equal the seed.
    accumulator->value.set_value_async(poison_value, stream);
    stream.synchronize();

    // Delay, then write the seed: it stays pending behind the delay, so an unordered sibling `next`
    // observes the poison instead.
    GQE_CUDA_TRY(cudaLaunchHostFunc(stream.value(), &sleep_for_race_window, nullptr));
    accumulator->value.set_value_async(seed_value, stream);

    return gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::task_private{std::move(accumulator)})
      .build();
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view inputs,
                                                gqe::qep::state_container_view accumulator,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref) const override
  {
    auto const build = as_device_build_accumulator(accumulator).value.value(stream);  // reads+syncs
    auto const probe = static_cast<std::int64_t>(gqe::qep::get_row_count(inputs));
    return gqe::qep::make_row_count_container(static_cast<cudf::size_type>(build + probe));
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<delayed_seed_stateful_transform_task>();
  }

 private:
  static constexpr std::int64_t poison_value = -1;  ///< Wrong value an unordered `next` would read.
  static constexpr std::int64_t seed_value   = 0;   ///< Correct build value.
  ///< Margin that keeps the seed pending past when a racing `next` issues its read.
  static constexpr std::chrono::milliseconds race_window{25};

  static void CUDART_CB sleep_for_race_window(void*) { std::this_thread::sleep_for(race_window); }
};

/**
 * @brief Per-adapter output slot for `delayed_read_stateful_transform_task`, carried in the
 * result's `task_private` slot.
 *
 * Holds the build value `next` copies out of the shared accumulator. The copy is issued behind a
 * delay and left pending on `next`'s stream, so it settles only when the adapter synchronizes that
 * stream after `next`.
 */
struct device_output_value : public gqe::qep::task_private_state {
  explicit device_output_value(rmm::cuda_stream_view stream) : value{stream} {}

  rmm::device_scalar<std::int64_t> value;
};

/**
 * @brief Stateful transform that exposes a missing `next`-before-`finalize` stream ordering.
 *
 *  - `initialize` seeds the build value.
 *  - `next` copies the build value into a per-adapter output device scalar behind a delay, leaving
 *    the copy pending on `next`'s stream until the adapter synchronizes after `next`.
 *  - `finalize` poisons the build value (standing in for freeing it).
 *
 * A `finalize` not stream-ordered after every `next` poisons the value while a sibling's copy is
 * still pending, and that copy captures the poison. See @ref
 * StatefulTransformAdapterTest_ConcurrentNextIsOrderedBeforeFinalizeDeviceWork_Test for the full
 * scenario, diagram, and limitations.
 */
class delayed_read_stateful_transform_task : public gqe::qep::stateful_transform_task {
 public:
  static constexpr std::int64_t seed_value = 7;  ///< Correct build value every `next` must copy.

  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref) const override
  {
    auto accumulator = std::make_unique<device_build_accumulator>(stream);
    accumulator->value.set_value_async(seed_value, stream);
    stream.synchronize();
    return gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::task_private{std::move(accumulator)})
      .build();
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::qep::state_container_view accumulator,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref) const override
  {
    auto output = std::make_unique<device_output_value>(stream);

    // Delay, then copy the build value: the copy stays pending behind the delay, so only the
    // adapter's post-`next` stream sync drains it. A `finalize` not ordered after this `next` would
    // poison the build value while this copy is pending, and the copy would capture the poison.
    GQE_CUDA_TRY(cudaLaunchHostFunc(stream.value(), &sleep_for_race_window, nullptr));
    GQE_CUDA_TRY(cudaMemcpyAsync(output->value.data(),
                                 as_device_build_accumulator(accumulator).value.data(),
                                 sizeof(std::int64_t),
                                 cudaMemcpyDeviceToDevice,
                                 stream.value()));

    return gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::task_private{std::move(output)})
      .build();
  }

  void finalize(gqe::qep::state_container&& accumulator,
                gqe::context_reference,
                rmm::cuda_stream_view,
                rmm::device_async_resource_ref) const override
  {
    // Poison the build value (standing in for freeing it) on a fresh stream, so the write lands
    // immediately rather than behind this adapter's own pending `next` copy -- mirroring a real
    // free. A `finalize` not ordered after every `next` then catches a sibling's pending copy.
    rmm::cuda_stream poison_stream;
    as_device_build_accumulator(gqe::qep::state_container_view(accumulator))
      .value.set_value_async(poison_value, poison_stream.view());
    poison_stream.synchronize();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<delayed_read_stateful_transform_task>();
  }

 private:
  static constexpr std::int64_t poison_value = -1;  ///< Value a copy racing `finalize` would read.
  ///< Margin that keeps a sibling's `next` copy pending past when the finalizer issues its poison.
  static constexpr std::chrono::milliseconds race_window{25};

  static void CUDART_CB sleep_for_race_window(void*) { std::this_thread::sleep_for(race_window); }
};

/**
 * @brief Read the build value a `delayed_read_stateful_transform_task` adapter copied into its
 *        result. The adapter has finished, so its `next` stream is already synced.
 */
inline std::int64_t output_build_value(gqe::task& adapter)
{
  auto state = adapter.qep_state_result();
  auto& slot =
    std::get<gqe::qep::state_kind::task_private>(*gqe::qep::state_container_view(*state).front());
  return static_cast<device_output_value*>(slot.data.get())
    ->value.value(cudf::get_default_stream());
}

class StatefulTransformAdapterTest : public gqe::test::BaseFixture {
 protected:
  StatefulTransformAdapterTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  /**
   * @brief Build an upstream `executed_task` emitting a single-column int64 table with `num_rows`
   *        rows. The stub only reads the row count, so the column values are arbitrary.
   */
  std::shared_ptr<gqe::test::executed_task> make_upstream(cudf::size_type num_rows, int32_t task_id)
  {
    std::vector<int64_t> const values(static_cast<std::size_t>(num_rows), 0);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper(values.begin(), values.end()).release());
    return std::make_shared<gqe::test::executed_task>(
      ctx_ref, task_id, /*stage_id=*/0, std::make_unique<cudf::table>(std::move(columns)));
  }

  gqe::context_reference ctx_ref;
};

/**
 * @brief A single streaming-side adapter initializes the shared accumulator from its build
 *        predecessor (deps[0]) and runs `next` on its probe partition (deps[1]).
 */
TEST_F(StatefulTransformAdapterTest, SingleProbePartitionAppliesBuildState)
{
  auto build = make_upstream(/*num_rows=*/4, /*task_id=*/0);
  auto probe = make_upstream(/*num_rows=*/3, /*task_id=*/1);

  auto accumulator = gqe::stateful_transform_adapter_task::make_shared_accumulator();
  auto adapter     = std::make_unique<gqe::stateful_transform_adapter_task>(
    ctx_ref,
    /*task_id=*/2,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{build, probe},
    /*num_materialized_inputs=*/1,
    accumulator,
    std::make_unique<build_plus_probe_count_stateful_transform_task>());

  adapter->execute();

  auto state = adapter->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(7));
}

/**
 * @brief A transform's streaming-side adapters share one accumulator, built once, so every probe
 *        partition sees the same build state.
 *
 * Each of the two adapters emits the build count plus its own probe partition's count. They run
 * sequentially here (as a single stage worker would), so concurrent `next` is not exercised.
 */
TEST_F(StatefulTransformAdapterTest, SharedAccumulatorAcrossProbePartitions)
{
  auto build   = make_upstream(/*num_rows=*/4, /*task_id=*/0);
  auto probe_a = make_upstream(/*num_rows=*/3, /*task_id=*/1);
  auto probe_b = make_upstream(/*num_rows=*/2, /*task_id=*/2);

  auto accumulator = gqe::stateful_transform_adapter_task::make_shared_accumulator();

  auto adapter_a = std::make_unique<gqe::stateful_transform_adapter_task>(
    ctx_ref,
    /*task_id=*/3,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{build, probe_a},
    /*num_materialized_inputs=*/1,
    accumulator,
    std::make_unique<build_plus_probe_count_stateful_transform_task>());
  auto adapter_b = std::make_unique<gqe::stateful_transform_adapter_task>(
    ctx_ref,
    /*task_id=*/4,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{build, probe_b},
    /*num_materialized_inputs=*/1,
    accumulator,
    std::make_unique<build_plus_probe_count_stateful_transform_task>());

  adapter_a->execute();
  adapter_b->execute();

  auto state_a = adapter_a->qep_state_result();
  ASSERT_TRUE(state_a.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state_a), testing::Optional(7));

  auto state_b = adapter_b->qep_state_result();
  ASSERT_TRUE(state_b.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state_b), testing::Optional(6));
}

/**
 * @brief `finalize` runs exactly once for a shared accumulator, regardless of the adapter count.
 *
 * Two adapters share one accumulator; whichever finishes `next` last finalizes it. The shared
 * counter must read exactly 1 -- never zero (finalize skipped) nor per-adapter.
 */
TEST_F(StatefulTransformAdapterTest, FinalizeRunsExactlyOncePerAccumulator)
{
  auto finalize_calls = std::make_shared<std::atomic<int>>(0);

  auto accumulator = gqe::stateful_transform_adapter_task::make_shared_accumulator();
  auto probe_a     = make_upstream(/*num_rows=*/3, /*task_id=*/0);
  auto probe_b     = make_upstream(/*num_rows=*/2, /*task_id=*/1);

  auto adapter_a = std::make_unique<gqe::stateful_transform_adapter_task>(
    ctx_ref,
    /*task_id=*/2,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{probe_a},
    /*num_materialized_inputs=*/0,
    accumulator,
    std::make_unique<finalize_counting_stateful_transform_task>(finalize_calls));
  auto adapter_b = std::make_unique<gqe::stateful_transform_adapter_task>(
    ctx_ref,
    /*task_id=*/3,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{probe_b},
    /*num_materialized_inputs=*/0,
    accumulator,
    std::make_unique<finalize_counting_stateful_transform_task>(finalize_calls));

  adapter_a->execute();
  adapter_b->execute();

  EXPECT_EQ(finalize_calls->load(), 1);
}

/**
 * @brief Multiple pipelined inputs (no materialized input) are horizontally concatenated into a
 *        single `next` call: two single-column partitions are seen by `next` as one two-column
 *        input.
 */
TEST_F(StatefulTransformAdapterTest, MultiplePipelinedInputsAreConcatenated)
{
  auto pipelined_a = make_upstream(/*num_rows=*/3, /*task_id=*/0);
  auto pipelined_b = make_upstream(/*num_rows=*/3, /*task_id=*/1);

  auto accumulator = gqe::stateful_transform_adapter_task::make_shared_accumulator();
  auto adapter     = std::make_unique<gqe::stateful_transform_adapter_task>(
    ctx_ref,
    /*task_id=*/2,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{pipelined_a, pipelined_b},
    /*num_materialized_inputs=*/0,
    accumulator,
    std::make_unique<input_column_count_stateful_transform_task>());

  adapter->execute();

  auto state = adapter->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(2));
}

/**
 * @brief A stateful transform's `initialize` device work is ordered before every concurrent `next`.
 *
 * The scenario, using `delayed_seed_stateful_transform_task` on two adapters that share one build
 * accumulator:
 *
 *   1. Each adapter runs on its own thread, hence its own per-thread default stream, so the
 *      transform's `initialize` device work and a sibling's `next` land on different streams.
 *   2. `initialize` poisons the build value, then defers the real seed behind a delay, leaving the
 *      seed write pending.
 *   3. Each `next` reads the build value; ordered after `initialize`, it reads the seed and emits
 *      its probe row count.
 *
 * The adapter makes every `next` stream wait on `initialize`'s completion event, so the step-3 read
 * sees the seed. Without that ordering, a sibling's `next` reads the still-pending poison:
 *
 *     builder stream:   poison ──▶│■■■ delay ■■■│──▶ seed
 *     sibling stream:        next: read ──▶ output (probe - 1)
 *                                  ▲
 *                                  └─ reads the poison: nothing orders this read after the
 *                                     seed write, and the seed is still stuck behind the delay
 *
 * # Limitation
 *
 * Detection is probabilistic: a failing run proves the ordering is missing, but a passing run does
 * not (hence the repeated trials). The delay only widens the race window; it cannot force the order
 * in which the two streams' work completes. The one mechanism that could -- making the seed wait on
 * the racing read -- would deadlock the correct adapter, whose ordering already makes that read
 * wait on the seed.
 */
TEST_F(StatefulTransformAdapterTest, ConcurrentNextIsOrderedAfterInitializeDeviceWork)
{
  constexpr cudf::size_type rows_a = 3;
  constexpr cudf::size_type rows_b = 2;
  constexpr int num_trials         = 4;

  for (int trial = 0; trial < num_trials; ++trial) {
    auto accumulator = gqe::stateful_transform_adapter_task::make_shared_accumulator();
    auto probe_a     = make_upstream(rows_a, /*task_id=*/0);
    auto probe_b     = make_upstream(rows_b, /*task_id=*/1);

    auto adapter_a = std::make_shared<gqe::stateful_transform_adapter_task>(
      ctx_ref,
      /*task_id=*/2,
      /*stage_id=*/0,
      std::vector<std::shared_ptr<gqe::task>>{probe_a},
      /*num_materialized_inputs=*/0,
      accumulator,
      std::make_unique<delayed_seed_stateful_transform_task>());
    auto adapter_b = std::make_shared<gqe::stateful_transform_adapter_task>(
      ctx_ref,
      /*task_id=*/3,
      /*stage_id=*/0,
      std::vector<std::shared_ptr<gqe::task>>{probe_b},
      /*num_materialized_inputs=*/0,
      accumulator,
      std::make_unique<delayed_seed_stateful_transform_task>());

    // Distinct threads give the two adapters distinct per-thread default streams.
    std::thread thread_a{[&] { adapter_a->execute(); }};
    std::thread thread_b{[&] { adapter_b->execute(); }};
    thread_a.join();
    thread_b.join();

    auto state_a = adapter_a->qep_state_result();
    ASSERT_TRUE(state_a.has_value());
    EXPECT_THAT(gqe::qep::try_row_count(*state_a), testing::Optional(rows_a));

    auto state_b = adapter_b->qep_state_result();
    ASSERT_TRUE(state_b.has_value());
    EXPECT_THAT(gqe::qep::try_row_count(*state_b), testing::Optional(rows_b));
  }
}

/**
 * @brief `finalize` is handed the accumulator `initialize` built, not an empty or default one.
 */
TEST_F(StatefulTransformAdapterTest, FinalizeReceivesInitializedAccumulator)
{
  auto captured = std::make_shared<std::optional<cudf::size_type>>();

  auto build = make_upstream(/*num_rows=*/5, /*task_id=*/0);
  auto probe = make_upstream(/*num_rows=*/3, /*task_id=*/1);

  auto accumulator = gqe::stateful_transform_adapter_task::make_shared_accumulator();
  auto adapter     = std::make_unique<gqe::stateful_transform_adapter_task>(
    ctx_ref,
    /*task_id=*/2,
    /*stage_id=*/0,
    std::vector<std::shared_ptr<gqe::task>>{build, probe},
    /*num_materialized_inputs=*/1,
    accumulator,
    std::make_unique<finalize_capturing_stateful_transform_task>(captured));

  adapter->execute();

  // `finalize` saw the accumulator seeded from the build input (5 rows).
  ASSERT_TRUE(captured->has_value());
  EXPECT_EQ(*captured, 5);
}

/**
 * @brief A stateful transform's `finalize` is ordered after every concurrent `next`'s device work.
 *
 * The scenario, using `delayed_read_stateful_transform_task` on two adapters that share one build
 * accumulator:
 *
 *   1. Each adapter runs on its own thread, hence its own per-thread default stream, so a sibling's
 *      `next` and the finalizing adapter's `finalize` land on different streams.
 *   2. Each adapter's `next` leaves its build-value copy pending behind a delay; only the adapter's
 *      post-`next` `remove_dependencies` stream sync drains it.
 *   3. The last adapter to finish `next` runs `finalize`, which poisons the build value (standing
 * in for freeing it).
 *
 * The adapter syncs each stream in `remove_dependencies` before its `pending_adapters` decrement,
 * so the `finalize` in step 3 runs only after every sibling's `next` has drained, and every copy
 * reads the seed. Without that ordering, a sibling's still-pending copy captures the poison:
 *
 *     finalizer stream:  finalize: poison      (immediate, on a fresh stream)
 *     sibling stream:    next: │■■ delay ■■│──▶ copy
 *                                               ▲
 *                                               └─ copy reads the poison: nothing orders it
 *                                                  ahead of the finalizer's poison write
 *
 * # Limitation
 *
 * Detection is probabilistic, as for the initialize-ordering test: a failing run proves the
 * ordering is missing, but a passing run does not prove it present (hence the repeated trials). The
 * delay only widens the race window; it cannot force the order in which the two streams' work
 * completes.
 */
TEST_F(StatefulTransformAdapterTest, ConcurrentNextIsOrderedBeforeFinalizeDeviceWork)
{
  constexpr int num_trials = 4;

  for (int trial = 0; trial < num_trials; ++trial) {
    auto accumulator = gqe::stateful_transform_adapter_task::make_shared_accumulator();
    auto probe_a     = make_upstream(/*num_rows=*/3, /*task_id=*/0);
    auto probe_b     = make_upstream(/*num_rows=*/2, /*task_id=*/1);

    auto adapter_a = std::make_shared<gqe::stateful_transform_adapter_task>(
      ctx_ref,
      /*task_id=*/2,
      /*stage_id=*/0,
      std::vector<std::shared_ptr<gqe::task>>{probe_a},
      /*num_materialized_inputs=*/0,
      accumulator,
      std::make_unique<delayed_read_stateful_transform_task>());
    auto adapter_b = std::make_shared<gqe::stateful_transform_adapter_task>(
      ctx_ref,
      /*task_id=*/3,
      /*stage_id=*/0,
      std::vector<std::shared_ptr<gqe::task>>{probe_b},
      /*num_materialized_inputs=*/0,
      accumulator,
      std::make_unique<delayed_read_stateful_transform_task>());

    // Distinct threads give the two adapters distinct per-thread default streams.
    std::thread thread_a{[&] { adapter_a->execute(); }};
    std::thread thread_b{[&] { adapter_b->execute(); }};
    thread_a.join();
    thread_b.join();

    // Every adapter's `next` copied the seed, never the poison `finalize` writes after all `next`.
    EXPECT_EQ(output_build_value(*adapter_a), delayed_read_stateful_transform_task::seed_value);
    EXPECT_EQ(output_build_value(*adapter_b), delayed_read_stateful_transform_task::seed_value);
  }
}
