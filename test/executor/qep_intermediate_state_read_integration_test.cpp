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

#include "../utility.hpp"

#include <gqe/executor/qep_task_graph_transformation.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/qep/operator/intermediate_state_read.hpp>
#include <gqe/qep/pipeline_normalization.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/error.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/table/table.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

/** @brief State-container shape emitted by `synthetic_source_task`. */
enum class source_shape { plain, masked, count_only };

/**
 * @brief Construct a single-column table that owns `column`.
 *
 * @param[in] column Column whose ownership transfers to the returned table.
 *
 * @return A table containing `column`.
 */
cudf::table one_column_source_table(std::unique_ptr<cudf::column> column)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return cudf::table{std::move(columns)};
}

/** @brief Construct one materialized state container of the selected shape. */
gqe::qep::state_container make_source_state(source_shape shape,
                                            gqe::query_stream const& stream_handle)
{
  switch (shape) {
    case source_shape::plain:
      return gqe::qep::state_container_builder()
        .add_state(
          one_column_source_table(int64_column_wrapper{10, 20, 30, 40, 50, 60, 70}.release()),
          stream_handle)
        .build();
    case source_shape::masked:
      return gqe::qep::state_container_builder()
        .add_state(
          gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true, true}.release()},
          stream_handle)
        .add_state(one_column_source_table(int64_column_wrapper{10, 20, 30, 40}.release()),
                   stream_handle)
        .build();
    case source_shape::count_only: return gqe::qep::make_row_count_container(7);
  }
  throw std::logic_error("make_source_state: unsupported source shape");
}

/** @brief Test source that emits one materialized state container of a selected shape. */
class synthetic_source_task final : public gqe::qep::iterate_task {
 public:
  explicit synthetic_source_task(source_shape shape) : _shape(shape) {}

  /** @copydoc gqe::qep::iterate_task::initialize */
  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       gqe::query_stream const&,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  /** @copydoc gqe::qep::iterate_task::next */
  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                gqe::query_stream const& stream_handle,
                                                rmm::device_async_resource_ref) const override
  {
    if (_emitted) { return std::nullopt; }
    _emitted = true;
    return make_source_state(_shape, stream_handle);
  }

  /** @copydoc gqe::qep::task::clone */
  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<synthetic_source_task>(_shape);
  }

 private:
  source_shape _shape;
  mutable bool _emitted = false;
};

/**
 * @brief Accumulator used by `materializing_fold_task` to retain its single input chunk.
 *
 * The mutex protects the check and write performed by concurrent `next` calls. The task lifecycle
 * ensures `finalize` reads `value` only after those calls complete.
 */
class materialized_state final : public gqe::qep::task_private_state {
 public:
  std::mutex mutex;                 ///< Protects the one-time write to `value` in `next`.
  gqe::qep::state_container value;  ///< Input chunk returned after accumulation completes.
};

/** @brief Test fold that retains one input chunk as materialized state. */
class materializing_fold_task final : public gqe::qep::fold_task {
 public:
  /** @copydoc gqe::qep::fold_task::initialize */
  gqe::qep::state_container initialize(gqe::context_reference,
                                       gqe::query_stream const&,
                                       rmm::device_async_resource_ref) const override
  {
    return gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::task_private{std::make_unique<materialized_state>()})
      .build();
  }

  /** @copydoc gqe::qep::fold_task::next */
  void next(gqe::qep::state_container_view inputs,
            gqe::qep::state_container_view accumulator,
            gqe::context_reference,
            gqe::query_stream const&,
            rmm::device_async_resource_ref) const override
  {
    auto* private_slot = std::get_if<gqe::qep::state_kind::task_private>(accumulator.front().get());
    GQE_EXPECTS(private_slot != nullptr && private_slot->data != nullptr,
                "materializing_fold_task: accumulator is not task-private state");
    auto* state = dynamic_cast<materialized_state*>(private_slot->data.get());
    GQE_EXPECTS(state != nullptr, "materializing_fold_task: accumulator belongs to another task");
    std::lock_guard lock{state->mutex};
    GQE_EXPECTS(state->value.empty(),
                "materializing_fold_task: synthetic source emitted multiple chunks");
    state->value = gqe::qep::make_mutable_state_copy(inputs);
  }

  /** @copydoc gqe::qep::fold_task::finalize */
  gqe::qep::state_container finalize(gqe::qep::state_container&& accumulator,
                                     gqe::context_reference,
                                     gqe::query_stream const&,
                                     rmm::device_async_resource_ref) const override
  {
    auto* private_slot = std::get_if<gqe::qep::state_kind::task_private>(accumulator.front().get());
    GQE_EXPECTS(private_slot != nullptr && private_slot->data != nullptr,
                "materializing_fold_task: accumulator is not task-private state");
    auto* state = dynamic_cast<materialized_state*>(private_slot->data.get());
    GQE_EXPECTS(state != nullptr, "materializing_fold_task: accumulator belongs to another task");
    return std::move(state->value);
  }

  /** @copydoc gqe::qep::task::clone */
  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<materializing_fold_task>();
  }
};

/** @brief Fixture that runs a source, materializing fold, and reader through QEP lowering. */
class QepIntermediateStateReadIntegrationTest : public gqe::test::BaseFixture {
 protected:
  QepIntermediateStateReadIntegrationTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  /**
   * @brief Execute synthetic input through materialization, reader, and sink tasks.
   *
   * @param[in] shape State-container shape emitted by the synthetic source.
   * @param[in] reader Configured reader whose ownership transfers to the QEP.
   *
   * @return The state container materialized downstream of the reader.
   */
  gqe::qep::state_container execute(source_shape shape,
                                    std::unique_ptr<gqe::qep::intermediate_state_read_task> reader)
  {
    auto source            = std::make_unique<synthetic_source_task>(shape);
    auto fold              = std::make_unique<materializing_fold_task>();
    auto sink              = std::make_unique<materializing_fold_task>();
    auto const* source_ptr = source.get();
    auto const* fold_ptr   = fold.get();
    auto const* reader_ptr = reader.get();
    auto const* sink_ptr   = sink.get();
    auto qep               = gqe::qep::query_execution_plan_builder()
                 .add_task(std::move(source))
                 .add_task(std::move(fold))
                 .add_task(std::move(reader))
                 .add_task(std::move(sink))
                 .add_successor(source_ptr, fold_ptr)
                 .add_successor(fold_ptr, reader_ptr)
                 .add_successor(reader_ptr, sink_ptr)
                 .build();
    auto normalized =
      gqe::qep::pipeline_normalization(qep, {}, gqe_test::make_test_task_stable_keys(qep));
    auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);
    GQE_EXPECTS(graph != nullptr,
                "QepIntermediateStateReadIntegrationTest: lowering returned a null graph");

    for (auto const& stage : graph->stage_root_tasks) {
      for (auto const& weak_root : stage) {
        auto root = weak_root.lock();
        GQE_EXPECTS(root != nullptr, "QepIntermediateStateReadIntegrationTest: stage root expired");
        root->execute();
      }
    }

    auto result = graph->root_tasks.front()->qep_state_result();
    GQE_EXPECTS(result.has_value(),
                "QepIntermediateStateReadIntegrationTest: reader emitted no result");
    return std::move(*result);
  }

  /** @brief Return row counts emitted directly by a reader inserted during normalization. */
  std::vector<cudf::size_type> read_inserted_reader_chunks(
    source_shape shape, std::optional<cudf::size_type> morsel_max_rows = std::nullopt)
  {
    auto source            = std::make_unique<synthetic_source_task>(shape);
    auto fold              = std::make_unique<materializing_fold_task>();
    auto sink              = std::make_unique<materializing_fold_task>();
    auto const* source_ptr = source.get();
    auto const* fold_ptr   = fold.get();
    auto const* sink_ptr   = sink.get();
    auto qep               = gqe::qep::query_execution_plan_builder()
                 .add_task(std::move(source))
                 .add_task(std::move(fold))
                 .add_task(std::move(sink))
                 .add_successor(source_ptr, fold_ptr)
                 .add_successor(fold_ptr, sink_ptr)
                 .build();
    auto normalized = gqe::qep::pipeline_normalization(
      qep, {}, gqe_test::make_test_task_stable_keys(qep), {}, morsel_max_rows);

    gqe::qep::intermediate_state_read_task const* inserted_reader = nullptr;
    for (auto const* task : normalized.plan().tasks()) {
      auto const* reader = dynamic_cast<gqe::qep::intermediate_state_read_task const*>(task);
      if (reader == nullptr) { continue; }
      GQE_EXPECTS(inserted_reader == nullptr,
                  "QepIntermediateStateReadIntegrationTest: normalization inserted multiple "
                  "readers");
      inserted_reader = reader;
    }
    GQE_EXPECTS(inserted_reader != nullptr,
                "QepIntermediateStateReadIntegrationTest: normalization inserted no reader");

    auto const stream_handle = get_query_ctx()->create_stream("inserted-reader-direct");
    auto input               = make_source_state(shape, stream_handle);
    auto iterator            = inserted_reader->initialize(input, ctx_ref, stream_handle);
    std::vector<cudf::size_type> rows;
    while (auto chunk = inserted_reader->next(iterator, ctx_ref, stream_handle)) {
      rows.push_back(gqe::qep::get_row_count(*chunk));
    }
    inserted_reader->finalize(std::move(iterator), ctx_ref, stream_handle);
    return rows;
  }

  gqe::context_reference ctx_ref;
};

TEST_F(QepIntermediateStateReadIntegrationTest, SinglePlainOutputSurvivesLoweringAndRelease)
{
  auto result = execute(source_shape::plain,
                        std::make_unique<gqe::qep::intermediate_state_read_task>(
                          gqe::qep::intermediate_state_read_task::single_output));

  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column_view>(*result.front()));
  int64_column_wrapper expected{10, 20, 30, 40, 50, 60, 70};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(result)->column(0), expected);
}

TEST_F(QepIntermediateStateReadIntegrationTest, BoundedPlainOutputIsRecombined)
{
  auto result =
    execute(source_shape::plain, std::make_unique<gqe::qep::intermediate_state_read_task>(3));

  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*result.front()));
  int64_column_wrapper expected{10, 20, 30, 40, 50, 60, 70};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(result)->column(0), expected);
}

TEST_F(QepIntermediateStateReadIntegrationTest, NormalizationDefaultsInsertedReaderToSingleOutput)
{
  EXPECT_EQ(read_inserted_reader_chunks(source_shape::plain), (std::vector<cudf::size_type>{7}));
}

TEST_F(QepIntermediateStateReadIntegrationTest, NormalizationBoundsInsertedReaderOutputRows)
{
  EXPECT_EQ(read_inserted_reader_chunks(source_shape::plain, 3),
            (std::vector<cudf::size_type>{3, 3, 1}));
}

TEST_F(QepIntermediateStateReadIntegrationTest, SingleMaskedOutputPreservesBorrowedShape)
{
  auto result = execute(source_shape::masked,
                        std::make_unique<gqe::qep::intermediate_state_read_task>(
                          gqe::qep::intermediate_state_read_task::single_output));

  ASSERT_TRUE(std::holds_alternative<gqe::qep::state_kind::valid_mask_view>(*result.front()));
  auto masked = gqe::qep::masked_table_view::try_from(result);
  ASSERT_TRUE(masked.has_value());
  bool_column_wrapper expected_mask{true, false, true, true};
  int64_column_wrapper expected_data{10, 20, 30, 40};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked->mask, expected_mask);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked->columns.column(0), expected_data);
}

TEST_F(QepIntermediateStateReadIntegrationTest, CountOnlyOutputIgnoresRowBound)
{
  auto result =
    execute(source_shape::count_only, std::make_unique<gqe::qep::intermediate_state_read_task>(1));
  auto const row_count = gqe::qep::try_row_count(result);
  ASSERT_TRUE(row_count.has_value());
  EXPECT_EQ(*row_count, 7);
}

TEST_F(QepIntermediateStateReadIntegrationTest, NormalizedCountOnlyOutputIgnoresRowBound)
{
  EXPECT_EQ(read_inserted_reader_chunks(source_shape::count_only, 1),
            (std::vector<cudf::size_type>{7}));
}
