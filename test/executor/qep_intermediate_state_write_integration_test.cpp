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
#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/operator/intermediate_state_write.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/qep/traits/static_iterate_split_trait.hpp>
#include <gqe/utility/error.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace qep = gqe::qep;

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;

/** @brief State-container shapes emitted by the test iterate tasks. */
enum class output_shape { count_only, indirect, masked, regular };

/**
 * @brief Emit one chunk using the selected QEP state-container shape.
 *
 * The task owns the input values and emits them exactly once from `next` as a regular, masked,
 * indirect, or count-only state container.
 */
class one_chunk_iterate_task final : public qep::iterate_task {
 public:
  one_chunk_iterate_task(std::vector<int64_t> values, output_shape shape)
    : _values(std::move(values)), _shape(shape)
  {
  }

  /** @copydoc gqe::qep::iterate_task::initialize */
  qep::state_container initialize(qep::state_container_view,
                                  gqe::context_reference,
                                  gqe::query_stream const&,
                                  rmm::device_async_resource_ref) const override
  {
    return {};
  }

  /** @copydoc gqe::qep::iterate_task::next */
  std::optional<qep::state_container> next(qep::state_container_view,
                                           gqe::context_reference,
                                           gqe::query_stream const&,
                                           rmm::device_async_resource_ref) const override
  {
    if (_emitted) { return std::nullopt; }
    _emitted = true;
    if (_shape == output_shape::count_only) {
      return qep::make_row_count_container(static_cast<cudf::size_type>(_values.size()));
    }

    if (_shape == output_shape::masked) {
      GQE_EXPECTS(_values.size() == 3,
                  "one_chunk_iterate_task: masked test input must contain three rows");
      return qep::state_container_builder()
        .add_state(qep::state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
                   gqe::query_stream::default_stream())
        .add_state(
          qep::state_kind::cudf_column{
            int64_column_wrapper(_values.begin(), _values.end()).release()},
          gqe::query_stream::default_stream())
        .build();
    }

    if (_shape == output_shape::indirect) {
      GQE_EXPECTS(_values.size() == 3,
                  "one_chunk_iterate_task: indirect test input must contain three rows");
      auto source = qep::make_shared_state(
        qep::state_kind::cudf_column{
          int64_column_wrapper(_values.begin(), _values.end()).release()},
        gqe::query_stream::default_stream());
      auto relation      = qep::indirect_table_source_state::make({source});
      auto gather_source = qep::indirect_gather_source_state::make(
        relation,
        qep::make_position_list(int32_column_wrapper{2, 0}.release(),
                                gqe::query_stream::default_stream()));
      return qep::state_container{qep::indirect_column_state::make(gather_source, 0)};
    }

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper(_values.begin(), _values.end()).release());
    return qep::state_container_builder()
      .add_state(cudf::table{std::move(columns)}, gqe::query_stream::default_stream())
      .build();
  }

  /** @copydoc gqe::qep::task::clone */
  std::unique_ptr<qep::task> clone() const override
  {
    return std::make_unique<one_chunk_iterate_task>(_values, _shape);
  }

 private:
  std::vector<int64_t> _values;
  output_shape _shape;
  mutable bool _emitted = false;
};

/**
 * @brief Split an owned sequence of partitions into one-chunk iterate tasks.
 *
 * Each split emits exactly one configured partition using the selected state-container shape.
 */
class partitioned_iterate_task final : public qep::iterate_task,
                                       public qep::static_iterate_split_trait {
 public:
  partitioned_iterate_task(std::vector<std::vector<int64_t>> partitions, output_shape shape)
    : _partitions(std::move(partitions)), _shape(shape)
  {
  }

  /** @copydoc gqe::qep::iterate_task::initialize */
  qep::state_container initialize(qep::state_container_view,
                                  gqe::context_reference,
                                  gqe::query_stream const&,
                                  rmm::device_async_resource_ref) const override
  {
    return {};
  }

  /** @copydoc gqe::qep::iterate_task::next */
  std::optional<qep::state_container> next(qep::state_container_view,
                                           gqe::context_reference,
                                           gqe::query_stream const&,
                                           rmm::device_async_resource_ref) const override
  {
    return std::nullopt;
  }

  /** @copydoc gqe::qep::task::clone */
  std::unique_ptr<qep::task> clone() const override
  {
    return std::make_unique<partitioned_iterate_task>(_partitions, _shape);
  }

  /** @copydoc gqe::qep::static_iterate_split_trait::max_splits */
  cudf::size_type max_splits() const noexcept override
  {
    return static_cast<cudf::size_type>(_partitions.size());
  }

  /** @copydoc gqe::qep::static_iterate_split_trait::split */
  std::unique_ptr<qep::iterate_task> split(cudf::size_type split_index,
                                           cudf::size_type num_splits) const override
  {
    GQE_EXPECTS(num_splits == static_cast<cudf::size_type>(_partitions.size()),
                "partitioned_iterate_task: unexpected split count");
    return std::make_unique<one_chunk_iterate_task>(
      _partitions[static_cast<std::size_t>(split_index)], _shape);
  }

 private:
  std::vector<std::vector<int64_t>> _partitions;
  output_shape _shape;
};

/** @brief Integration fixture for executing intermediate-state writer QEPs. */
class QepIntermediateStateWriteIntegrationTest : public gqe::test::BaseFixture {
 protected:
  QepIntermediateStateWriteIntegrationTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  /**
   * @brief Execute partitioned input through gather and intermediate-state-write tasks.
   *
   * @param[in] partitions Values emitted independently by the source splits.
   * @param[in] shape State-container shape emitted by each split.
   *
   * @return The state container materialized by the writer.
   */
  qep::state_container execute(std::vector<std::vector<int64_t>> partitions, output_shape shape)
  {
    auto source = std::make_unique<partitioned_iterate_task>(std::move(partitions), shape);
    auto gather = std::make_unique<qep::gather_task>();
    auto output_data_types = shape == output_shape::count_only
                               ? std::vector<cudf::data_type>{}
                               : std::vector{cudf::data_type{cudf::type_id::INT64}};
    auto writer =
      std::make_unique<qep::intermediate_state_write_task>(std::move(output_data_types));
    auto* source_ptr = source.get();
    auto* gather_ptr = gather.get();
    auto* writer_ptr = writer.get();
    auto plan        = qep::query_execution_plan_builder()
                  .add_task(std::move(source))
                  .add_task(std::move(gather))
                  .add_task(std::move(writer))
                  .add_successor(source_ptr, gather_ptr)
                  .add_successor(gather_ptr, writer_ptr)
                  .build();
    auto normalized =
      qep::pipeline_normalization(plan, {}, gqe_test::make_test_task_stable_keys(plan));
    auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);
    GQE_EXPECTS(graph != nullptr,
                "QepIntermediateStateWriteIntegrationTest: lowering returned a null graph");

    for (auto const& stage : graph->stage_root_tasks) {
      for (auto const& weak_root : stage) {
        auto root = weak_root.lock();
        GQE_EXPECTS(root != nullptr,
                    "QepIntermediateStateWriteIntegrationTest: stage root expired");
        root->execute();
      }
    }

    auto result = graph->root_tasks.front()->qep_state_result();
    GQE_EXPECTS(result.has_value(),
                "QepIntermediateStateWriteIntegrationTest: writer emitted no result");
    return std::move(*result);
  }

  gqe::context_reference ctx_ref;
};

TEST_F(QepIntermediateStateWriteIntegrationTest, RecombinesMultipleInputPartitions)
{
  auto result = execute({{1, 2}, {3}, {4, 5}}, output_shape::regular);
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto sorted = cudf::sort(*table);
  int64_column_wrapper expected{1, 2, 3, 4, 5};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected);
}

TEST_F(QepIntermediateStateWriteIntegrationTest, PreservesZeroRowSchema)
{
  auto result = execute({{}}, output_shape::regular);
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  EXPECT_EQ(table->num_columns(), 1);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->column(0).type().id(), cudf::type_id::INT64);
}

TEST_F(QepIntermediateStateWriteIntegrationTest, RecombinesCountOnlyPartitions)
{
  auto result = execute({{1, 2}, {3}, {4, 5}}, output_shape::count_only);
  ASSERT_TRUE(qep::try_row_count(result).has_value());
  EXPECT_EQ(*qep::try_row_count(result), 5);
}

TEST_F(QepIntermediateStateWriteIntegrationTest, GathersMaskedInputBeforeWriting)
{
  auto result = execute({{10, 20, 30}}, output_shape::masked);
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected{10, 30};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected);
}

TEST_F(QepIntermediateStateWriteIntegrationTest, GathersIndirectInputBeforeWriting)
{
  auto result = execute({{10, 20, 30}}, output_shape::indirect);
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected{30, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected);
}
