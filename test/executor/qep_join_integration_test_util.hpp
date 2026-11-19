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

#include "../utility.hpp"

#include <gqe/executor/qep_task_graph_transformation.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/qep/physical_qep_transformation.hpp>
#include <gqe/qep/pipeline_normalization.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/storage/table_provider.hpp>
#include <gqe/utility/error.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gqe_test {

namespace physical = gqe::physical;
namespace qep      = gqe::qep;
namespace storage  = gqe::storage;

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

inline auto const qep_join_temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

/** @brief Resolves fixed Parquet inputs and the optional writer output for one integration plan. */
class join_table_provider final : public storage::table_provider {
 public:
  join_table_provider(std::shared_ptr<storage::table> left,
                      std::shared_ptr<storage::table> right,
                      std::shared_ptr<storage::table> output)
    : _left{std::move(left)}, _right{std::move(right)}, _output{std::move(output)}
  {
  }

  [[nodiscard]] std::shared_ptr<storage::table> get_table(
    std::string_view table_name) const override
  {
    if (table_name == "left") { return _left; }
    if (table_name == "right") { return _right; }
    if (table_name == "output") { return _output; }
    throw std::logic_error("join_table_provider: unknown table " + std::string{table_name});
  }

 private:
  std::shared_ptr<storage::table> _left;
  std::shared_ptr<storage::table> _right;
  std::shared_ptr<storage::table> _output;
};

inline std::shared_ptr<physical::read_relation> make_join_read(
  std::string table_name, std::vector<std::string> column_names)
{
  auto const int64_type = cudf::data_type{cudf::type_id::INT64};
  auto data_types       = std::vector<cudf::data_type>(column_names.size(), int64_type);
  return std::make_shared<physical::read_relation>(
    std::vector<std::shared_ptr<physical::relation>>{},
    std::move(column_names),
    std::move(table_name),
    nullptr,
    std::move(data_types));
}

inline std::vector<std::unique_ptr<cudf::table>> make_left_partitions()
{
  std::vector<std::unique_ptr<cudf::table>> partitions;
  partitions.push_back(make_table_from_columns(int64_column_wrapper{100, 300}.release(),
                                               int64_column_wrapper{1, 3}.release(),
                                               int64_column_wrapper{1000, 3000}.release()));
  partitions.push_back(make_table_from_columns(int64_column_wrapper{200, 400}.release(),
                                               int64_column_wrapper{2, 4}.release(),
                                               int64_column_wrapper{2000, 4000}.release()));
  return partitions;
}

inline std::vector<std::unique_ptr<cudf::table>> make_right_partitions()
{
  std::vector<std::unique_ptr<cudf::table>> partitions;
  partitions.push_back(make_table_from_columns(int64_column_wrapper{10, 30}.release(),
                                               int64_column_wrapper{11, 33}.release(),
                                               int64_column_wrapper{2, 4}.release()));
  partitions.push_back(make_table_from_columns(int64_column_wrapper{20, 40}.release(),
                                               int64_column_wrapper{22, 44}.release(),
                                               int64_column_wrapper{1, 5}.release()));
  return partitions;
}

/** @brief Selects the physical join implementation and supported runtime options for one test. */
struct qep_join_test_configuration {
  physical::broadcast_policy policy;

  gqe::join_type_type join_type;

  bool is_perfect_hashing;

  bool is_hash_map_cache_enabled;

  std::optional<gqe::unique_keys_policy> unique_keys = std::nullopt;

  bool is_mark_join{false};
};

/**
 * @brief Builds an optional successor topology above the tested join.
 *
 * The callback receives the owned join root followed by its logical left and right inputs. It
 * returns the owned complete physical root, which retains the entire relation graph through
 * lowering and pipeline normalization.
 */
using qep_join_root_transform =
  std::function<std::shared_ptr<physical::relation>(std::shared_ptr<physical::relation>,
                                                    std::shared_ptr<physical::relation>,
                                                    std::shared_ptr<physical::relation>)>;

/** @brief Optional inputs and successor topology for one join integration execution. */
struct qep_join_execution_options {
  /** Left input partitions. Defaults to `make_left_partitions()`. */
  std::vector<std::unique_ptr<cudf::table>> left_partitions = make_left_partitions();

  /** Right input partitions. Defaults to `make_right_partitions()`. */
  std::vector<std::unique_ptr<cudf::table>> right_partitions = make_right_partitions();

  /**
   * Append `column[0] + column[2]` as column 3 of each input after its optional filter.
   * The combined schema becomes `[left0, left1, left2, left_sum, right0, right1, right2,
   * right_sum]`. Predicate and output indices must address this extended schema.
   */
  bool has_input_sum_projection{false};

  /**
   * Output column indices in the combined `[left columns..., right columns...]` schema. The default
   * selects `{right_payload, left_value, right_payload, left_payload}` to exercise reordered and
   * repeated projections for the default three-column inputs.
   */
  std::vector<cudf::size_type> projection_indices{4, 0, 4, 2};

  /** Optional predicate using left-input column indices. Null leaves the input unfiltered. */
  std::unique_ptr<gqe::expression> left_filter;

  /** Optional predicate using right-input column indices. Null leaves the input unfiltered. */
  std::unique_ptr<gqe::expression> right_filter;

  /**
   * Optional predicate using column indices from the combined `[left columns..., right columns...]`
   * schema. Null compares the `join_key` columns at combined indices 1 and 5, or 1 and 6
   * when `has_input_sum_projection` is enabled.
   */
  std::unique_ptr<gqe::expression> join_condition;

  /** Optional successor builder. An empty callback leaves the tested join as the execution root. */
  qep_join_root_transform transform_root;
};

/** @brief Executes normalized join-rooted QEPs against partitioned Parquet inputs. */
class QepJoinIntegrationTestBase : public gqe::test::BaseFixture {
 protected:
  static constexpr cudf::size_type max_num_partitions = 2;

  QepJoinIntegrationTestBase() : ctx_ref{get_task_manager_ctx(), get_query_ctx()}
  {
    get_query_ctx()->parameters.num_partitions = max_num_partitions;
  }

  std::shared_ptr<storage::parquet_table> write_partitions(
    std::string_view name,
    std::vector<std::string> const& column_names,
    std::vector<std::unique_ptr<cudf::table>> partitions)
  {
    std::vector<std::string> paths;
    paths.reserve(partitions.size());
    for (std::size_t i = 0; i < partitions.size(); ++i) {
      auto path = qep_join_temp_env->get_temp_filepath(std::string{name} + "_" + std::to_string(i) +
                                                       ".parquet");
      write_table_to_file(partitions[i]->view(), column_names, path);
      paths.push_back(std::move(path));
    }
    return std::make_shared<storage::parquet_table>(std::move(paths));
  }

  qep::state_container execute_join(qep_join_test_configuration configuration,
                                    std::string_view name,
                                    qep_join_execution_options options = {})
  {
    get_query_ctx()->parameters.join_use_perfect_hash = configuration.is_perfect_hashing;

    std::vector<std::string> const left_columns{"left_value", "join_key", "left_payload"};
    std::vector<std::string> const right_columns{"right_value", "right_payload", "join_key"};
    auto const prefix      = std::string{name};
    auto const output_path = qep_join_temp_env->get_temp_filepath(prefix + "_output.parquet");
    join_table_provider provider{
      write_partitions(prefix + "_left", left_columns, std::move(options.left_partitions)),
      write_partitions(prefix + "_right", right_columns, std::move(options.right_partitions)),
      std::make_shared<storage::parquet_table>(std::vector<std::string>{output_path})};

    std::shared_ptr<physical::relation> left_input = make_join_read("left", left_columns);
    if (options.left_filter != nullptr) {
      left_input = std::make_shared<physical::filter_relation>(
        left_input,
        std::vector<std::shared_ptr<physical::relation>>{},
        std::move(options.left_filter),
        std::vector<cudf::size_type>{0, 1, 2});
    }
    std::shared_ptr<physical::relation> right_input = make_join_read("right", right_columns);
    if (options.right_filter != nullptr) {
      right_input = std::make_shared<physical::filter_relation>(
        right_input,
        std::vector<std::shared_ptr<physical::relation>>{},
        std::move(options.right_filter),
        std::vector<cudf::size_type>{0, 1, 2});
    }
    if (options.has_input_sum_projection) {
      auto make_project = [](std::shared_ptr<physical::relation> input) {
        std::vector<std::unique_ptr<gqe::expression>> expressions;
        for (cudf::size_type index = 0; index < 3; ++index) {
          expressions.push_back(std::make_unique<gqe::column_reference_expression>(index));
        }
        expressions.push_back(std::make_unique<gqe::add_expression>(
          std::make_shared<gqe::column_reference_expression>(0),
          std::make_shared<gqe::column_reference_expression>(2)));
        return std::make_shared<physical::project_relation>(
          std::move(input),
          std::vector<std::shared_ptr<physical::relation>>{},
          std::move(expressions));
      };
      left_input  = make_project(std::move(left_input));
      right_input = make_project(std::move(right_input));
    }

    auto const unique_keys_policy = configuration.unique_keys.value_or(
      configuration.is_perfect_hashing ? (configuration.policy == physical::broadcast_policy::left
                                            ? gqe::unique_keys_policy::left
                                            : gqe::unique_keys_policy::right)
                                       : gqe::unique_keys_policy::none);
    if (options.join_condition == nullptr) {
      auto const right_start = options.has_input_sum_projection ? 4 : 3;
      options.join_condition = std::make_unique<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::column_reference_expression>(right_start + 2));
    }
    auto join = std::make_shared<physical::broadcast_join_relation>(
      left_input,
      right_input,
      std::vector<std::shared_ptr<physical::relation>>{},
      configuration.join_type,
      std::move(options.join_condition),
      std::move(options.projection_indices),
      configuration.policy,
      unique_keys_policy,
      configuration.is_perfect_hashing,
      nullptr,
      nullptr,
      configuration.is_hash_map_cache_enabled,
      configuration.is_mark_join);
    std::shared_ptr<physical::relation> root = join;
    if (options.transform_root) { root = options.transform_root(join, left_input, right_input); }
    GQE_EXPECTS(root != nullptr, "QepJoinIntegrationTestBase: root transform returned null");

    auto identified_plan = gqe_test::make_borrowed_identified_plan(root.get());
    auto lowered =
      qep::physical_qep_transformation(identified_plan, &provider, get_query_ctx()->parameters);
    auto normalized = qep::pipeline_normalization(
      lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations);
    auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);
    GQE_EXPECTS(graph != nullptr,
                "QepJoinIntegrationTestBase: task-graph transformation returned null");
    gqe::execute_task_graph_single_gpu(ctx_ref, graph.get());
    GQE_EXPECTS(graph->root_tasks.size() == 1,
                "QepJoinIntegrationTestBase: expected one root task");
    auto result = graph->root_tasks.front()->qep_state_result();
    GQE_EXPECTS(result.has_value(), "QepJoinIntegrationTestBase: root emitted no result");
    return std::move(*result);
  }

  gqe::context_reference ctx_ref;
};

}  // namespace gqe_test
