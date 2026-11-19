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

#include <gqe/qep/physical_qep_transformation.hpp>

#include "../utility.hpp"

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/bounded_hash_aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/set.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/top_k.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/qep/operator/bounded_hash_aggregate.hpp>
#include <gqe/qep/operator/conditional_inner_join.hpp>
#include <gqe/qep/operator/conditional_semi_anti_join.hpp>
#include <gqe/qep/operator/equality_semi_anti_join.hpp>
#include <gqe/qep/operator/fetch.hpp>
#include <gqe/qep/operator/filter.hpp>
#include <gqe/qep/operator/filter_join_indices.hpp>
#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/operator/hash_aggregate.hpp>
#include <gqe/qep/operator/hash_join.hpp>
#include <gqe/qep/operator/intermediate_state_read.hpp>
#include <gqe/qep/operator/intermediate_state_write.hpp>
#include <gqe/qep/operator/join_materialization.hpp>
#include <gqe/qep/operator/mark_join.hpp>
#include <gqe/qep/operator/merge_sort.hpp>
#include <gqe/qep/operator/mixed_semi_anti_join.hpp>
#include <gqe/qep/operator/perfect_hash_aggregate.hpp>
#include <gqe/qep/operator/perfect_hash_join.hpp>
#include <gqe/qep/operator/project.hpp>
#include <gqe/qep/operator/reduce_aggregate.hpp>
#include <gqe/qep/operator/top_k.hpp>
#include <gqe/qep/operator/unique_key_inner_join.hpp>
#include <gqe/qep/pipeline_normalization.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/storage/chunked_in_memory_reader.hpp>
#include <gqe/storage/chunked_parquet_reader.hpp>
#include <gqe/storage/chunked_parquet_writer.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/in_memory_reference_iterate.hpp>
#include <gqe/storage/in_memory_resolve.hpp>
#include <gqe/storage/oneshot_in_memory_reader.hpp>
#include <gqe/storage/oneshot_parquet_reader.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/storage/parquet_reference_iterate.hpp>
#include <gqe/storage/parquet_resolve.hpp>
#include <gqe/storage/table_provider.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

namespace {

namespace physical = gqe::physical;
namespace qep      = gqe::qep;
namespace storage  = gqe::storage;

using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

cudf::data_type int32_type() { return cudf::data_type{cudf::type_id::INT32}; }

cudf::data_type int64_type() { return cudf::data_type{cudf::type_id::INT64}; }

qep::physical_qep_transformation_result make_physical_qep_transformation(
  physical::relation* plan,
  storage::table_provider const* table_provider,
  gqe::optimization_parameters const& parameters =
    gqe::make_optimization_parameters(/*only_defaults=*/true))
{
  auto identified_plan = gqe_test::make_borrowed_identified_plan(plan);
  return qep::physical_qep_transformation(identified_plan, table_provider, parameters);
}

/**
 * @brief Minimal table provider backed by a name-to-table map.
 */
class map_table_provider : public storage::table_provider {
 public:
  void add_table(std::string name, std::shared_ptr<storage::table> table)
  {
    _tables.emplace(std::move(name), std::move(table));
  }

  [[nodiscard]] std::shared_ptr<storage::table> get_table(
    std::string_view table_name) const override
  {
    auto const it = _tables.find(std::string{table_name});
    if (it == _tables.end()) {
      throw std::logic_error("map_table_provider: unknown table " + std::string{table_name});
    }
    return it->second;
  }

 private:
  std::map<std::string, std::shared_ptr<storage::table>> _tables;
};

map_table_provider make_provider_with_parquet_table(std::string const& name)
{
  map_table_provider provider;
  provider.add_table(
    name, std::make_shared<storage::parquet_table>(std::vector<std::string>{"unused.parquet"}));
  return provider;
}

std::shared_ptr<physical::read_relation> make_read(
  std::string table_name,
  std::size_t num_columns,
  std::unique_ptr<gqe::expression> partial_filter,
  std::vector<std::shared_ptr<physical::relation>> subqueries = {},
  cudf::data_type data_type                                   = int64_type())
{
  std::vector<std::string> column_names;
  std::vector<cudf::data_type> data_types;
  for (std::size_t i = 0; i < num_columns; ++i) {
    column_names.push_back("col_" + std::to_string(i));
    data_types.push_back(data_type);
  }
  return std::make_shared<physical::read_relation>(std::move(subqueries),
                                                   std::move(column_names),
                                                   std::move(table_name),
                                                   std::move(partial_filter),
                                                   std::move(data_types));
}

std::unique_ptr<gqe::expression> col_ref(cudf::size_type idx)
{
  return std::make_unique<gqe::column_reference_expression>(idx);
}

/**
 * @brief A simple `col(lhs_idx) < col(rhs_idx)` filter condition.
 */
std::unique_ptr<gqe::expression> less_condition(cudf::size_type lhs_idx, cudf::size_type rhs_idx)
{
  return std::make_unique<gqe::less_expression>(
    std::make_shared<gqe::column_reference_expression>(lhs_idx),
    std::make_shared<gqe::column_reference_expression>(rhs_idx));
}

std::unique_ptr<gqe::expression> equal_condition(cudf::size_type lhs_idx, cudf::size_type rhs_idx)
{
  return std::make_unique<gqe::equal_expression>(
    std::make_shared<gqe::column_reference_expression>(lhs_idx),
    std::make_shared<gqe::column_reference_expression>(rhs_idx));
}

std::unique_ptr<gqe::expression> nulls_equal_condition(cudf::size_type lhs_idx,
                                                       cudf::size_type rhs_idx)
{
  return std::make_unique<gqe::nulls_equal_expression>(
    std::make_shared<gqe::column_reference_expression>(lhs_idx),
    std::make_shared<gqe::column_reference_expression>(rhs_idx));
}

std::vector<cudf::size_type> identity_indices(std::size_t count)
{
  std::vector<cudf::size_type> indices(count);
  for (std::size_t i = 0; i < count; ++i) {
    indices[i] = static_cast<cudf::size_type>(i);
  }
  return indices;
}

template <typename... Columns>
qep::state_container make_table(Columns&&... input_columns)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(sizeof...(input_columns));
  (columns.push_back(std::forward<Columns>(input_columns)), ...);
  return qep::state_container_builder()
    .add_state(cudf::table{std::move(columns)}, gqe::query_stream::default_stream())
    .build();
}

template <typename TaskType>
std::vector<qep::task const*> tasks_of_type(qep::query_execution_plan const& plan)
{
  std::vector<qep::task const*> matches;
  for (auto const* task : plan.tasks()) {
    if (dynamic_cast<TaskType const*>(task) != nullptr) { matches.push_back(task); }
  }
  return matches;
}

template <typename TaskType>
qep::task const* single_task_of_type(qep::query_execution_plan const& plan)
{
  auto const matches = tasks_of_type<TaskType>(plan);
  EXPECT_EQ(matches.size(), 1);
  return matches.empty() ? nullptr : matches.front();
}

/** @brief Check that a completed probe feeds the join materialization task. */
qep::task const* join_materialization_after(qep::query_execution_plan const& plan,
                                            qep::task const* probe)
{
  auto const* materialize = single_task_of_type<qep::join_materialization_task>(plan);
  EXPECT_NE(materialize, nullptr);
  if (materialize != nullptr) {
    EXPECT_EQ(plan.predecessors(materialize), std::vector<qep::task const*>{probe});
    EXPECT_EQ(plan.successors(probe), std::vector<qep::task const*>{materialize});
  }
  return materialize;
}

qep::pipeline_normalization_result lower_and_normalize(
  physical::relation* root, storage::table_provider const* table_provider)
{
  auto lowered = make_physical_qep_transformation(root, table_provider);
  return qep::pipeline_normalization(
    lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations);
}

void verify_perfect_hash_join_lowering(physical::broadcast_policy policy,
                                       gqe::unique_keys_policy unique_keys_policy)
{
  auto const provider  = make_provider_with_parquet_table("t");
  auto const left_read = make_read("t", 3, nullptr);
  auto const left_input =
    std::make_shared<physical::filter_relation>(left_read,
                                                std::vector<std::shared_ptr<physical::relation>>{},
                                                less_condition(0, 2),
                                                identity_indices(3));
  auto const right_input = make_read("t", 3, nullptr);
  physical::broadcast_join_relation join{left_input,
                                         right_input,
                                         {},
                                         gqe::join_type_type::inner,
                                         equal_condition(1, 5),
                                         {4, 0, 4, 2},
                                         policy,
                                         unique_keys_policy,
                                         /*perfect_hashing=*/true};

  auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_perfect_hash = true;
  auto lowered                     = make_physical_qep_transformation(&join, &provider, parameters);
  auto const& plan                 = lowered.plan;

  auto const* left_filter = single_task_of_type<qep::filter_task>(plan);
  auto const* build_task  = dynamic_cast<qep::perfect_hash_join_build_task const*>(
    single_task_of_type<qep::perfect_hash_join_build_task>(plan));
  auto const* probe_task = dynamic_cast<qep::perfect_hash_join_probe_task const*>(
    single_task_of_type<qep::perfect_hash_join_probe_task>(plan));
  auto const reads = tasks_of_type<storage::oneshot_parquet_read_task>(plan);
  ASSERT_NE(left_filter, nullptr);
  ASSERT_NE(build_task, nullptr);
  ASSERT_NE(probe_task, nullptr);
  ASSERT_EQ(reads.size(), 2);
  ASSERT_EQ(plan.predecessors(left_filter).size(), 1);
  auto const* left_read_task      = plan.predecessors(left_filter).front();
  auto const* right_read_task     = reads.front() == left_read_task ? reads.back() : reads.front();
  bool const materialized_is_left = policy == physical::broadcast_policy::left;
  auto const* materialized_input_task = materialized_is_left ? left_filter : right_read_task;
  auto const* streaming_input_task    = materialized_is_left ? right_read_task : left_filter;
  EXPECT_EQ(plan.predecessors(build_task), std::vector<qep::task const*>{materialized_input_task});
  EXPECT_EQ(plan.predecessors(probe_task),
            (std::vector<qep::task const*>{build_task, streaming_input_task}));
  EXPECT_TRUE(plan.successors(join_materialization_after(plan, probe_task)).empty());
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
  EXPECT_TRUE(lowered.output_data_types.at(build_task).empty());
  EXPECT_EQ(lowered.output_data_types.at(probe_task), join.output_data_types());
  EXPECT_EQ(lowered.physical_relations.at(build_task), &join);
  EXPECT_EQ(lowered.physical_relations.at(probe_task), &join);
}

void verify_hash_join_lowering(
  physical::broadcast_policy policy,
  gqe::join_type_type join_type,
  bool use_hash_map_cache,
  bool nulls_equal,
  bool relation_perfect_hashing              = false,
  gqe::unique_keys_policy unique_keys_policy = gqe::unique_keys_policy::none,
  bool join_use_perfect_hash                 = false)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 3, nullptr);
  auto const right    = make_read("t", 3, nullptr);
  auto condition      = nulls_equal ? nulls_equal_condition(1, 5) : equal_condition(1, 5);
  physical::broadcast_join_relation join{left,
                                         right,
                                         {},
                                         join_type,
                                         std::move(condition),
                                         {4, 0, 4, 2},
                                         policy,
                                         unique_keys_policy,
                                         relation_perfect_hashing,
                                         /*left_filter_condition=*/nullptr,
                                         /*right_filter_condition=*/nullptr,
                                         use_hash_map_cache};

  auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_perfect_hash = join_use_perfect_hash;
  parameters.join_use_unique_keys  = false;
  auto lowered                     = make_physical_qep_transformation(&join, &provider, parameters);
  auto const& plan                 = lowered.plan;
  auto const* build                = single_task_of_type<qep::hash_join_build_task>(plan);
  auto const* probe                = single_task_of_type<qep::hash_join_probe_task>(plan);
  auto const reads                 = tasks_of_type<storage::oneshot_parquet_read_task>(plan);
  ASSERT_NE(build, nullptr);
  ASSERT_NE(probe, nullptr);
  ASSERT_EQ(reads.size(), 2);

  auto const* build_relation = policy == physical::broadcast_policy::right
                                 ? static_cast<physical::relation const*>(right.get())
                                 : static_cast<physical::relation const*>(left.get());
  auto const* probe_relation = policy == physical::broadcast_policy::right
                                 ? static_cast<physical::relation const*>(left.get())
                                 : static_cast<physical::relation const*>(right.get());
  auto const build_read_it   = std::ranges::find_if(reads, [&](qep::task const* task) {
    return lowered.physical_relations.at(task) == build_relation;
  });
  ASSERT_NE(build_read_it, reads.end()) << "Missing read task for the hash-join build relation";
  auto const* build_read   = *build_read_it;
  auto const probe_read_it = std::ranges::find_if(reads, [&](qep::task const* task) {
    return lowered.physical_relations.at(task) == probe_relation;
  });
  ASSERT_NE(probe_read_it, reads.end()) << "Missing read task for the hash-join probe relation";
  auto const* probe_read = *probe_read_it;
  EXPECT_EQ(plan.predecessors(build), std::vector<qep::task const*>{build_read});
  EXPECT_EQ(plan.predecessors(probe), (std::vector<qep::task const*>{build, probe_read}));
  EXPECT_TRUE(plan.successors(join_materialization_after(plan, probe)).empty());
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
  EXPECT_TRUE(tasks_of_type<qep::perfect_hash_join_build_task>(plan).empty());
  EXPECT_TRUE(tasks_of_type<qep::perfect_hash_join_probe_task>(plan).empty());
  EXPECT_TRUE(lowered.output_data_types.at(build).empty());
  EXPECT_EQ(lowered.output_data_types.at(probe), join.output_data_types());
  EXPECT_EQ(lowered.physical_relations.at(build), &join);
  EXPECT_EQ(lowered.physical_relations.at(probe), &join);
}

void verify_unique_key_join_lowering(physical::broadcast_join_relation& join,
                                     storage::table_provider const& provider,
                                     bool enabled,
                                     bool is_unique)
{
  auto parameters                 = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_unique_keys = enabled;
  auto lowered                    = make_physical_qep_transformation(&join, &provider, parameters);
  EXPECT_EQ(tasks_of_type<qep::unique_key_inner_join_build_task>(lowered.plan).size(),
            is_unique ? 1 : 0);
  EXPECT_EQ(tasks_of_type<qep::unique_key_inner_join_probe_task>(lowered.plan).size(),
            is_unique ? 1 : 0);
  EXPECT_EQ(tasks_of_type<qep::hash_join_build_task>(lowered.plan).size(), is_unique ? 0 : 1);
  EXPECT_EQ(tasks_of_type<qep::hash_join_probe_task>(lowered.plan).size(), is_unique ? 0 : 1);
}

std::shared_ptr<physical::broadcast_join_relation> make_inner_join(
  std::shared_ptr<physical::relation> left,
  std::shared_ptr<physical::relation> right,
  cudf::size_type right_key,
  std::vector<cudf::size_type> projection,
  bool is_perfect)
{
  auto const policy = physical::broadcast_policy::right;
  return std::make_shared<physical::broadcast_join_relation>(
    std::move(left),
    std::move(right),
    std::vector<std::shared_ptr<physical::relation>>{},
    gqe::join_type_type::inner,
    equal_condition(1, right_key),
    std::move(projection),
    policy,
    is_perfect ? gqe::unique_keys_policy::right : gqe::unique_keys_policy::none,
    is_perfect);
}

void verify_equality_semi_anti_join_lowering(gqe::join_type_type join_type, bool nulls_equal)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 3, nullptr);
  auto const right    = make_read("t", 3, nullptr);
  auto condition      = nulls_equal ? nulls_equal_condition(1, 5) : equal_condition(1, 5);
  physical::broadcast_join_relation join{left,
                                         right,
                                         {},
                                         join_type,
                                         std::move(condition),
                                         {2, 0, 2},
                                         physical::broadcast_policy::right,
                                         gqe::unique_keys_policy::none,
                                         /*perfect_hashing=*/false,
                                         /*left_filter_condition=*/nullptr,
                                         /*right_filter_condition=*/nullptr,
                                         /*use_hash_map_cache=*/true};

  auto lowered = make_physical_qep_transformation(
    &join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
  auto const& plan  = lowered.plan;
  auto const* build = single_task_of_type<qep::equality_semi_anti_join_build_task>(plan);
  auto const* probe = single_task_of_type<qep::equality_semi_anti_join_probe_task>(plan);
  auto const reads  = tasks_of_type<storage::oneshot_parquet_read_task>(plan);
  ASSERT_NE(build, nullptr);
  ASSERT_NE(probe, nullptr);
  ASSERT_EQ(reads.size(), 2);
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
  EXPECT_TRUE(plan.successors(join_materialization_after(plan, probe)).empty());

  auto const right_read_it = std::ranges::find_if(reads, [&](qep::task const* task) {
    return lowered.physical_relations.at(task) == right.get();
  });
  ASSERT_NE(right_read_it, reads.end());
  auto const left_read_it = std::ranges::find_if(reads, [&](qep::task const* task) {
    return lowered.physical_relations.at(task) == left.get();
  });
  ASSERT_NE(left_read_it, reads.end());
  EXPECT_EQ(plan.predecessors(build), std::vector<qep::task const*>{*right_read_it});
  EXPECT_EQ(plan.predecessors(probe), (std::vector<qep::task const*>{build, *left_read_it}));
  EXPECT_TRUE(tasks_of_type<qep::hash_join_build_task>(plan).empty());
  EXPECT_TRUE(tasks_of_type<qep::hash_join_probe_task>(plan).empty());
  EXPECT_TRUE(tasks_of_type<qep::perfect_hash_join_build_task>(plan).empty());
  EXPECT_TRUE(tasks_of_type<qep::perfect_hash_join_probe_task>(plan).empty());
  EXPECT_TRUE(lowered.output_data_types.at(build).empty());
  EXPECT_EQ(lowered.output_data_types.at(probe), join.output_data_types());
  EXPECT_EQ(lowered.physical_relations.at(build), &join);
  EXPECT_EQ(lowered.physical_relations.at(probe), &join);
}

}  // namespace

TEST(PhysicalQepTransformationTest, KeepsTaskMetadataForOwnedTasksAfterMove)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);

  auto initial     = make_physical_qep_transformation(read.get(), &provider);
  auto lowered     = std::move(initial);
  auto const tasks = lowered.plan.tasks();

  ASSERT_EQ(tasks.size(), 1);
  auto const found = lowered.output_data_types.find(tasks.front());
  ASSERT_NE(found, lowered.output_data_types.end());
  EXPECT_EQ(found->second, read->output_data_types());
  EXPECT_EQ(lowered.physical_relations.at(tasks.front()), read.get());
}

TEST(PhysicalQepTransformationTest, TracksThePhysicalRelationThatProducedEachTask)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::filter_relation filter{read, {}, less_condition(0, 1), identity_indices(2)};

  auto lowered     = make_physical_qep_transformation(&filter, &provider);
  auto const tasks = lowered.plan.tasks();
  ASSERT_EQ(tasks.size(), 2);
  ASSERT_EQ(lowered.physical_relations.size(), tasks.size());
  for (auto const* task : tasks) {
    auto const* expected_relation =
      dynamic_cast<storage::oneshot_parquet_read_task const*>(task) != nullptr
        ? static_cast<physical::relation const*>(read.get())
        : static_cast<physical::relation const*>(&filter);
    EXPECT_EQ(lowered.physical_relations.at(task), expected_relation);
  }
}

class QepSourceReaderTest : public gqe::test::BaseFixture {};

// Lowers a read under a filter for each reader and storage kind. A deferred read adds a resolver
// after the source, and the filter consumes the resolver. Parquet reads are pruned, so a haystack
// feeds the source; in-memory QEP reads don't support partition-column pruning.
TEST_F(QepSourceReaderTest, LowersTheSelectedReader)
{
  map_table_provider in_memory;
  in_memory.add_table("t",
                      std::make_shared<storage::in_memory_table>(
                        gqe::memory_kind::device{rmm::get_current_cuda_device()},
                        std::vector<std::string>{"col_0", "col_1"},
                        std::vector<cudf::data_type>{int64_type(), int64_type()},
                        get_task_manager_ctx()));
  auto const parquet = make_provider_with_parquet_table("t");
  struct expected_read {
    gqe::source_reader_type reader;
    map_table_provider const* provider;
    std::type_info const* source;
    std::type_info const* resolve;
  };
  using reader = gqe::source_reader_type;
  for (auto const& expected : std::vector<expected_read>{
         {reader::oneshot, &in_memory, &typeid(storage::oneshot_in_memory_read_task), nullptr},
         {reader::chunked, &in_memory, &typeid(storage::chunked_in_memory_read_task), nullptr},
         {reader::deferred,
          &in_memory,
          &typeid(storage::in_memory_reference_iterate_task),
          &typeid(storage::in_memory_resolve_task)},
         {reader::oneshot, &parquet, &typeid(storage::oneshot_parquet_read_task), nullptr},
         {reader::chunked, &parquet, &typeid(storage::chunked_parquet_read_task), nullptr},
         {reader::deferred,
          &parquet,
          &typeid(storage::parquet_reference_iterate_task),
          &typeid(storage::parquet_resolve_task)}}) {
    SCOPED_TRACE(expected.source->name());
    bool const pruned = expected.provider == &parquet;
    std::vector<std::shared_ptr<gqe::expression>> needles{col_ref(0)};
    auto read = pruned ? make_read("t",
                                   2,
                                   std::make_unique<gqe::in_predicate_expression>(needles, 0),
                                   {make_read("t", 1, nullptr)})
                       : make_read("t", 2, nullptr);
    physical::filter_relation filter{read, {}, less_condition(0, 1), identity_indices(2)};
    auto parameters          = gqe::make_optimization_parameters(true);
    parameters.source_reader = expected.reader;
    auto const lowered = make_physical_qep_transformation(&filter, expected.provider, parameters);
    auto const& plan   = lowered.plan;

    qep::task const* source  = nullptr;
    qep::task const* resolve = nullptr;
    for (auto const* task : plan.tasks()) {
      if (lowered.physical_relations.at(task) != read.get()) continue;
      auto const role = lowered.stable_keys.at(task).role;
      if (role == qep::lowering_role::read) source = task;
      if (role == qep::lowering_role::read_resolve) resolve = task;
    }
    ASSERT_NE(source, nullptr);
    auto const& source_task = *source;
    EXPECT_EQ(typeid(source_task), *expected.source);
    auto const haystack_edge = plan.predecessors(source);
    ASSERT_EQ(haystack_edge.size(), pruned ? 1 : 0);
    if (pruned) {
      EXPECT_NE(dynamic_cast<qep::gather_task const*>(haystack_edge.front()), nullptr);
    }
    if (expected.resolve != nullptr) {
      ASSERT_NE(resolve, nullptr);
      auto const& resolve_task = *resolve;
      EXPECT_EQ(typeid(resolve_task), *expected.resolve);
      EXPECT_EQ(plan.predecessors(resolve), std::vector<qep::task const*>{source});
    } else {
      EXPECT_EQ(resolve, nullptr);
    }
    auto const* filter_task = single_task_of_type<qep::filter_task>(plan);
    ASSERT_NE(filter_task, nullptr);
    EXPECT_EQ(plan.predecessors(filter_task),
              std::vector<qep::task const*>{expected.resolve != nullptr ? resolve : source});
  }
}

TEST(PhysicalQepTransformationTest, ReadLowersToCompletePipeline)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);

  auto normalized  = lower_and_normalize(read.get(), &provider);
  auto const& plan = normalized.plan();

  ASSERT_EQ(plan.tasks().size(), 3);
  auto const* read_task   = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const* gather_task = single_task_of_type<qep::gather_task>(plan);
  auto const* writer_task = single_task_of_type<qep::intermediate_state_write_task>(plan);
  ASSERT_NE(read_task, nullptr);
  EXPECT_TRUE(plan.predecessors(read_task).empty());
  EXPECT_EQ(plan.predecessors(gather_task), std::vector<qep::task const*>{read_task});
  EXPECT_EQ(plan.predecessors(writer_task), std::vector<qep::task const*>{gather_task});
  EXPECT_TRUE(plan.successors(writer_task).empty());
}

TEST(PhysicalQepTransformationTest, FilterAppendsFinalWriterBoundary)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::filter_relation filter{read, {}, less_condition(0, 1), identity_indices(2)};

  auto normalized  = lower_and_normalize(&filter, &provider);
  auto const& plan = normalized.plan();

  ASSERT_EQ(plan.tasks().size(), 4);
  auto const* read_task   = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const* filter_task = single_task_of_type<qep::filter_task>(plan);
  auto const* gather_task = single_task_of_type<qep::gather_task>(plan);
  auto const* writer_task = single_task_of_type<qep::intermediate_state_write_task>(plan);
  ASSERT_NE(filter_task, nullptr);
  ASSERT_NE(gather_task, nullptr);
  ASSERT_NE(writer_task, nullptr);
  EXPECT_EQ(plan.predecessors(filter_task), std::vector<qep::task const*>{read_task});
  EXPECT_EQ(plan.predecessors(gather_task), std::vector<qep::task const*>{filter_task});
  EXPECT_EQ(plan.predecessors(writer_task), std::vector<qep::task const*>{gather_task});
  EXPECT_TRUE(plan.successors(writer_task).empty());
}

/**
 * @brief Test the terminal-only gather emitted while lowering Top-K relations.
 *
 * These tests inspect pre-normalization lowering to verify gather placement before normalization
 * inserts intermediate-state readers and writers. The normalized execution path is covered by the
 * QEP Top-K executor integration test.
 */
TEST(PhysicalQepTransformationTest, TopKLowersToFoldAndTerminalGather)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::top_k_relation top_k{read, 0, cudf::order::DESCENDING, cudf::null_order::AFTER, 1, 2};

  auto lowered     = make_physical_qep_transformation(&top_k, &provider);
  auto const& plan = lowered.plan;

  ASSERT_EQ(plan.tasks().size(), 3);
  auto const* read_task   = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const* top_k_task  = single_task_of_type<qep::top_k_task>(plan);
  auto const* gather_task = single_task_of_type<qep::gather_task>(plan);
  ASSERT_NE(top_k_task, nullptr);
  ASSERT_NE(gather_task, nullptr);
  EXPECT_EQ(plan.predecessors(top_k_task), std::vector<qep::task const*>{read_task});
  EXPECT_EQ(plan.predecessors(gather_task), std::vector<qep::task const*>{top_k_task});
  EXPECT_TRUE(plan.successors(gather_task).empty());
  EXPECT_EQ(lowered.output_data_types.at(gather_task), top_k.output_data_types());
  EXPECT_EQ(lowered.physical_relations.at(gather_task), &top_k);
}

TEST(PhysicalQepTransformationTest, TopKConsumesMaskedInputWithoutInteriorGather)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  auto filter =
    std::make_shared<physical::filter_relation>(read,
                                                std::vector<std::shared_ptr<physical::relation>>{},
                                                less_condition(0, 1),
                                                identity_indices(2));
  physical::top_k_relation top_k{filter, 0, cudf::order::ASCENDING, cudf::null_order::AFTER, 0, 2};

  auto lowered     = make_physical_qep_transformation(&top_k, &provider);
  auto const& plan = lowered.plan;

  ASSERT_EQ(plan.tasks().size(), 4);
  auto const gathers = tasks_of_type<qep::gather_task>(plan);
  ASSERT_EQ(gathers.size(), 1);
  auto const* top_k_task = single_task_of_type<qep::top_k_task>(plan);
  ASSERT_NE(top_k_task, nullptr);
  ASSERT_EQ(plan.predecessors(top_k_task).size(), 1);
  EXPECT_NE(dynamic_cast<qep::filter_task const*>(plan.predecessors(top_k_task).front()), nullptr);
  ASSERT_EQ(plan.successors(top_k_task).size(), 1);
  auto const* output_gather = plan.successors(top_k_task).front();
  EXPECT_NE(dynamic_cast<qep::gather_task const*>(output_gather), nullptr);
  EXPECT_EQ(lowered.stable_keys.at(output_gather).role, qep::lowering_role::materialize_output);
}

TEST(PhysicalQepTransformationTest, StableTaskKeysPreserveIdentifiedPlanNodeIds)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  auto filter =
    std::make_shared<physical::filter_relation>(read,
                                                std::vector<std::shared_ptr<physical::relation>>{},
                                                less_condition(0, 1),
                                                identity_indices(2));
  physical::identified_plan identified_plan{filter, {{filter.get(), 17}, {read.get(), 29}}};

  auto lowered = qep::physical_qep_transformation(
    identified_plan, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
  auto const* read_task   = single_task_of_type<storage::oneshot_parquet_read_task>(lowered.plan);
  auto const* filter_task = single_task_of_type<qep::filter_task>(lowered.plan);
  ASSERT_NE(read_task, nullptr);
  ASSERT_NE(filter_task, nullptr);

  EXPECT_EQ(lowered.stable_keys.at(read_task).relation_id, 29);
  EXPECT_EQ(lowered.stable_keys.at(filter_task).relation_id, 17);
}

TEST(PhysicalQepTransformationTest, NonTerminalTopKRemainsIndirectForFetch)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  auto top_k          = std::make_shared<physical::top_k_relation>(
    read, 0, cudf::order::ASCENDING, cudf::null_order::AFTER, 0, 2);
  physical::fetch_relation fetch{top_k, /*offset=*/0, /*count=*/1};

  auto lowered = make_physical_qep_transformation(
    &fetch, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
  auto const& plan = lowered.plan;

  auto const* top_k_task = single_task_of_type<qep::top_k_task>(plan);
  auto const* fetch_task = single_task_of_type<qep::fetch_task>(plan);
  ASSERT_NE(top_k_task, nullptr);
  ASSERT_NE(fetch_task, nullptr);
  EXPECT_EQ(plan.predecessors(fetch_task), std::vector<qep::task const*>{top_k_task});
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
}

TEST(PhysicalQepTransformationTest, NonIdentityFilterProjectionStaysMaskedUntilTerminal)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::filter_relation filter{read, {}, less_condition(0, 1), {1}};

  auto normalized  = lower_and_normalize(&filter, &provider);
  auto const& plan = normalized.plan();

  ASSERT_EQ(plan.tasks().size(), 5);
  auto const* filter_task  = single_task_of_type<qep::filter_task>(plan);
  auto const* project_task = single_task_of_type<qep::project_task>(plan);
  auto const* gather_task  = single_task_of_type<qep::gather_task>(plan);
  auto const* writer_task  = single_task_of_type<qep::intermediate_state_write_task>(plan);
  ASSERT_NE(project_task, nullptr);
  ASSERT_NE(gather_task, nullptr);
  EXPECT_EQ(plan.predecessors(project_task), std::vector<qep::task const*>{filter_task});
  EXPECT_EQ(plan.predecessors(gather_task), std::vector<qep::task const*>{project_task});
  EXPECT_EQ(plan.predecessors(writer_task), std::vector<qep::task const*>{gather_task});
}

TEST(PhysicalQepTransformationTest, ProjectOnPlainInputGetsFinalWriterBoundary)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> exprs;
  exprs.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::column_reference_expression>(1)));
  physical::project_relation project{read, {}, std::move(exprs)};

  auto normalized  = lower_and_normalize(&project, &provider);
  auto const& plan = normalized.plan();

  ASSERT_EQ(plan.tasks().size(), 4);
  auto const* project_task = single_task_of_type<qep::project_task>(plan);
  auto const* gather_task  = single_task_of_type<qep::gather_task>(plan);
  auto const* writer_task  = single_task_of_type<qep::intermediate_state_write_task>(plan);
  ASSERT_NE(project_task, nullptr);
  EXPECT_EQ(plan.predecessors(gather_task), std::vector<qep::task const*>{project_task});
  EXPECT_EQ(plan.predecessors(writer_task), std::vector<qep::task const*>{gather_task});
  EXPECT_TRUE(plan.successors(writer_task).empty());
}

TEST(PhysicalQepTransformationTest, IdentityProjectIsElided)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> exprs;
  exprs.push_back(col_ref(0));
  exprs.push_back(col_ref(1));
  physical::project_relation project{read, {}, std::move(exprs)};

  auto normalized  = lower_and_normalize(&project, &provider);
  auto const& plan = normalized.plan();

  // The identity project is elided. Pipeline normalization completes the read pipeline.
  ASSERT_EQ(plan.tasks().size(), 3);
  EXPECT_NE(single_task_of_type<storage::oneshot_parquet_read_task>(plan), nullptr);
  EXPECT_NE(single_task_of_type<qep::gather_task>(plan), nullptr);
  EXPECT_NE(single_task_of_type<qep::intermediate_state_write_task>(plan), nullptr);
}

TEST(PhysicalQepTransformationTest, ReorderingProjectIsNotElided)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> exprs;
  exprs.push_back(col_ref(1));
  exprs.push_back(col_ref(0));
  physical::project_relation project{read, {}, std::move(exprs)};

  auto normalized  = lower_and_normalize(&project, &provider);
  auto const& plan = normalized.plan();

  ASSERT_EQ(plan.tasks().size(), 4);
  EXPECT_NE(single_task_of_type<qep::project_task>(plan), nullptr);
  EXPECT_NE(single_task_of_type<qep::gather_task>(plan), nullptr);
  EXPECT_NE(single_task_of_type<qep::intermediate_state_write_task>(plan), nullptr);
}

TEST(PhysicalQepTransformationTest, PerfectHashJoinMaterializesUniqueLeft)
{
  verify_perfect_hash_join_lowering(physical::broadcast_policy::left,
                                    gqe::unique_keys_policy::left);
}

TEST(PhysicalQepTransformationTest, PerfectHashJoinMaterializesUniqueRight)
{
  verify_perfect_hash_join_lowering(physical::broadcast_policy::right,
                                    gqe::unique_keys_policy::right);
}

TEST(PhysicalQepTransformationTest, PerfectHashJoinMaterializesRightWhileBuildingUniqueLeft)
{
  verify_perfect_hash_join_lowering(physical::broadcast_policy::right,
                                    gqe::unique_keys_policy::left);
}

TEST(PhysicalQepTransformationTest, PerfectHashJoinMaterializesLeftWhileBuildingUniqueRight)
{
  verify_perfect_hash_join_lowering(physical::broadcast_policy::left,
                                    gqe::unique_keys_policy::right);
}

TEST(PhysicalQepTransformationTest, PerfectHashJoinEitherMaterializesLeft)
{
  verify_perfect_hash_join_lowering(physical::broadcast_policy::left,
                                    gqe::unique_keys_policy::either);
}

TEST(PhysicalQepTransformationTest, PerfectHashJoinEitherMaterializesRight)
{
  verify_perfect_hash_join_lowering(physical::broadcast_policy::right,
                                    gqe::unique_keys_policy::either);
}

TEST(PhysicalQepTransformationTest, PerfectHashJoinDisabledGateFallsBackToHashJoin)
{
  verify_hash_join_lowering(physical::broadcast_policy::right,
                            gqe::join_type_type::inner,
                            /*use_hash_map_cache=*/false,
                            /*nulls_equal=*/false,
                            /*relation_perfect_hashing=*/true,
                            gqe::unique_keys_policy::right,
                            /*join_use_perfect_hash=*/false);
}

TEST(PhysicalQepTransformationTest, PerfectHashJoinWithoutUniqueSideFallsBackToHashJoin)
{
  verify_hash_join_lowering(physical::broadcast_policy::right,
                            gqe::join_type_type::inner,
                            /*use_hash_map_cache=*/false,
                            /*nulls_equal=*/false,
                            /*relation_perfect_hashing=*/true,
                            gqe::unique_keys_policy::none,
                            /*join_use_perfect_hash=*/true);
}

TEST(PhysicalQepTransformationTest, HashJoinBuildLeftLowersExpectedTopology)
{
  verify_hash_join_lowering(physical::broadcast_policy::left,
                            gqe::join_type_type::inner,
                            /*use_hash_map_cache=*/false,
                            /*nulls_equal=*/false);
}

TEST(PhysicalQepTransformationTest, CachedHashJoinBuildRightAcceptsNullEquals)
{
  verify_hash_join_lowering(physical::broadcast_policy::right,
                            gqe::join_type_type::inner,
                            /*use_hash_map_cache=*/true,
                            /*nulls_equal=*/true);
}

TEST(PhysicalQepTransformationTest, LeftHashJoinBuildsLogicalRight)
{
  verify_hash_join_lowering(physical::broadcast_policy::right,
                            gqe::join_type_type::left,
                            /*use_hash_map_cache=*/true,
                            /*nulls_equal=*/false);
}

TEST(PhysicalQepTransformationTest, PerfectJoinFilterProjectReduceNeedsNoGather)
{
  auto provider = make_provider_with_parquet_table("t");

  auto join =
    make_inner_join(make_read("t", 3, nullptr), make_read("t", 3, nullptr), 5, {4, 0, 4, 2}, true);
  auto filter =
    std::make_shared<physical::filter_relation>(join,
                                                std::vector<std::shared_ptr<physical::relation>>{},
                                                less_condition(0, 1),
                                                identity_indices(4));
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(col_ref(0));
  expressions.push_back(col_ref(1));
  auto project = std::make_shared<physical::project_relation>(
    filter, std::vector<std::shared_ptr<physical::relation>>{}, std::move(expressions));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    project, {}, std::vector<std::unique_ptr<gqe::expression>>{}, std::move(values)};

  auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_perfect_hash = true;
  auto lowered     = make_physical_qep_transformation(&aggregate, &provider, parameters);
  auto const& plan = lowered.plan;
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
  auto const* probe        = single_task_of_type<qep::perfect_hash_join_probe_task>(plan);
  auto const* filter_task  = single_task_of_type<qep::filter_task>(plan);
  auto const* project_task = single_task_of_type<qep::project_task>(plan);
  auto const* reduce       = single_task_of_type<qep::reduce_aggregate_task>(plan);
  EXPECT_EQ(plan.predecessors(filter_task),
            std::vector<qep::task const*>{join_materialization_after(plan, probe)});
  EXPECT_EQ(plan.predecessors(project_task), std::vector<qep::task const*>{filter_task});
  EXPECT_EQ(plan.predecessors(reduce), std::vector<qep::task const*>{project_task});
}

TEST(PhysicalQepTransformationTest, PerfectJoinProjectGroupedAggregateNeedsNoGather)
{
  auto provider = make_provider_with_parquet_table("t");

  auto join =
    make_inner_join(make_read("t", 3, nullptr), make_read("t", 3, nullptr), 5, {4, 0, 4, 2}, true);
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(col_ref(0));
  expressions.push_back(col_ref(1));
  auto project = std::make_shared<physical::project_relation>(
    join, std::vector<std::shared_ptr<physical::relation>>{}, std::move(expressions));
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    project, {}, std::move(keys), std::move(values), nullptr, /*perfect_hashing=*/true};

  auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_perfect_hash = true;
  parameters.aggregation_use_perfect_hash = true;
  auto lowered     = make_physical_qep_transformation(&aggregate, &provider, parameters);
  auto const& plan = lowered.plan;
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
  auto const* probe        = single_task_of_type<qep::perfect_hash_join_probe_task>(plan);
  auto const* project_task = single_task_of_type<qep::project_task>(plan);
  auto const* partial      = single_task_of_type<qep::perfect_hash_partial_aggregate_task>(plan);
  EXPECT_EQ(plan.predecessors(project_task),
            std::vector<qep::task const*>{join_materialization_after(plan, probe)});
  EXPECT_EQ(plan.predecessors(partial), std::vector<qep::task const*>{project_task});
}

TEST(PhysicalQepTransformationTest, HashJoinProjectReduceNeedsNoGather)
{
  auto provider = make_provider_with_parquet_table("t");

  auto join =
    make_inner_join(make_read("t", 3, nullptr), make_read("t", 3, nullptr), 5, {4, 0, 4, 2}, false);
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(col_ref(0));
  expressions.push_back(col_ref(1));
  auto project = std::make_shared<physical::project_relation>(
    join, std::vector<std::shared_ptr<physical::relation>>{}, std::move(expressions));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    project, {}, std::vector<std::unique_ptr<gqe::expression>>{}, std::move(values)};

  auto lowered = make_physical_qep_transformation(
    &aggregate, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(lowered.plan).empty());
}

TEST(PhysicalQepTransformationTest, JoinFetchTopKGathersAtTerminalBoundary)
{
  auto provider = make_provider_with_parquet_table("t");

  auto join =
    make_inner_join(make_read("t", 3, nullptr), make_read("t", 3, nullptr), 5, {4, 0, 4, 2}, true);
  auto fetch = std::make_shared<physical::fetch_relation>(join, /*offset=*/0, /*count=*/3);
  physical::top_k_relation top_k{fetch, 0, cudf::order::DESCENDING, cudf::null_order::AFTER, 0, 1};

  auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_perfect_hash = true;
  auto lowered       = make_physical_qep_transformation(&top_k, &provider, parameters);
  auto const& plan   = lowered.plan;
  auto const gathers = tasks_of_type<qep::gather_task>(plan);
  ASSERT_EQ(gathers.size(), 1);
  auto const* probe      = single_task_of_type<qep::perfect_hash_join_probe_task>(plan);
  auto const* fetch_task = single_task_of_type<qep::fetch_task>(plan);
  auto const* top_k_task = single_task_of_type<qep::top_k_task>(plan);
  EXPECT_EQ(plan.predecessors(fetch_task),
            std::vector<qep::task const*>{join_materialization_after(plan, probe)});
  EXPECT_EQ(plan.predecessors(top_k_task), std::vector<qep::task const*>{fetch_task});
  EXPECT_EQ(plan.predecessors(gathers.front()), std::vector<qep::task const*>{top_k_task});
}

TEST(PhysicalQepTransformationTest, JoinWriterGathersBeforeStorageBoundary)
{
  auto provider = make_provider_with_parquet_table("t");
  provider.add_table(
    "out", std::make_shared<storage::parquet_table>(std::vector<std::string>{"out.parquet"}));

  auto join =
    make_inner_join(make_read("t", 3, nullptr), make_read("t", 3, nullptr), 5, {4, 0, 4, 2}, false);
  physical::write_relation write{join, {"a", "b", "c", "d"}, "out"};

  auto lowered = make_physical_qep_transformation(
    &write, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
  auto const& plan   = lowered.plan;
  auto const* probe  = single_task_of_type<qep::hash_join_probe_task>(plan);
  auto const* gather = single_task_of_type<qep::gather_task>(plan);
  auto const* writer = single_task_of_type<storage::chunked_parquet_write_task>(plan);
  EXPECT_EQ(plan.predecessors(gather),
            std::vector<qep::task const*>{join_materialization_after(plan, probe)});
  EXPECT_EQ(plan.predecessors(writer), std::vector<qep::task const*>{gather});
}

class PhysicalQepChainedJoinTest : public ::testing::TestWithParam<bool> {};

TEST_P(PhysicalQepChainedJoinTest, NeedsNoGather)
{
  auto provider                = make_provider_with_parquet_table("t");
  bool const second_is_perfect = GetParam();
  auto left                    = make_read("t", 3, nullptr);
  auto first =
    make_inner_join(left, make_read("t", 3, nullptr), 5, {4, 0, 4, 2}, /*is_perfect=*/true);
  auto second = make_inner_join(first, left, 4, {0, 1, 4}, /*is_perfect=*/second_is_perfect);

  auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_perfect_hash = true;
  auto lowered = make_physical_qep_transformation(second.get(), &provider, parameters);
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(lowered.plan).empty());
  EXPECT_EQ(tasks_of_type<qep::perfect_hash_join_probe_task>(lowered.plan).size(),
            second_is_perfect ? 2 : 1);
  EXPECT_EQ(tasks_of_type<qep::hash_join_probe_task>(lowered.plan).size(),
            second_is_perfect ? 0 : 1);
}

INSTANTIATE_TEST_SUITE_P(SuccessorTopologies,
                         PhysicalQepChainedJoinTest,
                         ::testing::Values(true, false),
                         [](auto const& info) { return info.param ? "PerfectJoin" : "HashJoin"; });

namespace {

/** @brief Default optimization parameters with late materialization disabled. */
gqe::optimization_parameters early_materialization_parameters()
{
  auto parameters                     = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.use_late_materialization = false;
  return parameters;
}

/**
 * @brief Expect `producer` to have exactly one successor, a gather of the same relation.
 *
 * @param role Lowering role the gather must carry.
 *
 * @return The gather, or null when the expectation fails.
 */
qep::task const* expect_output_gather(
  qep::physical_qep_transformation_result const& lowered,
  qep::task const* producer,
  qep::lowering_role role = qep::lowering_role::materialize_output)
{
  EXPECT_NE(producer, nullptr);
  if (producer == nullptr) { return nullptr; }
  auto const successors = lowered.plan.successors(producer);
  EXPECT_EQ(successors.size(), 1);
  if (successors.size() != 1) { return nullptr; }
  auto const* gather = successors.front();
  EXPECT_NE(dynamic_cast<qep::gather_task const*>(gather), nullptr);
  EXPECT_EQ(lowered.stable_keys.at(gather).role, role);
  EXPECT_EQ(lowered.physical_relations.at(gather), lowered.physical_relations.at(producer));
  return gather;
}

}  // namespace

TEST(PhysicalQepTransformationTest, LateMaterializationIsEnabledByDefault)
{
  EXPECT_TRUE(gqe::make_optimization_parameters(/*only_defaults=*/true).use_late_materialization);
}

TEST(PhysicalQepTransformationTest, EarlyMaterializationGathersFilterAfterItsProjection)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::filter_relation filter{read, {}, less_condition(0, 1), {1}};

  auto lowered =
    make_physical_qep_transformation(&filter, &provider, early_materialization_parameters());
  auto const& plan = lowered.plan;

  ASSERT_EQ(tasks_of_type<qep::gather_task>(plan).size(), 1);
  auto const* filter_task  = single_task_of_type<qep::filter_task>(plan);
  auto const* project_task = single_task_of_type<qep::project_task>(plan);
  EXPECT_EQ(plan.predecessors(project_task), std::vector<qep::task const*>{filter_task});
  auto const* gather = expect_output_gather(lowered, project_task);
  ASSERT_NE(gather, nullptr);
  EXPECT_TRUE(plan.successors(gather).empty());
  EXPECT_EQ(lowered.output_data_types.at(gather), filter.output_data_types());

  // Normalization appends its own terminal gather, which is zero-copy for the early boundary's
  // ordinary output.
  auto normalized = qep::pipeline_normalization(
    lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations);
  auto const* writer = single_task_of_type<qep::intermediate_state_write_task>(normalized.plan());
  ASSERT_NE(writer, nullptr);
  auto const writer_input = normalized.plan().predecessors(writer);
  ASSERT_EQ(writer_input.size(), 1);
  auto const terminal_input = normalized.plan().predecessors(writer_input.front());
  ASSERT_EQ(terminal_input.size(), 1);
  EXPECT_NE(dynamic_cast<qep::gather_task const*>(terminal_input.front()), nullptr);
}

TEST(PhysicalQepTransformationTest, EarlyMaterializationGathersIdentityProjectionFilter)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::filter_relation filter{read, {}, less_condition(0, 1), {0, 1}};

  auto lowered =
    make_physical_qep_transformation(&filter, &provider, early_materialization_parameters());
  auto const& plan = lowered.plan;

  EXPECT_TRUE(tasks_of_type<qep::project_task>(plan).empty());
  ASSERT_EQ(tasks_of_type<qep::gather_task>(plan).size(), 1);
  auto const* gather = expect_output_gather(lowered, single_task_of_type<qep::filter_task>(plan));
  ASSERT_NE(gather, nullptr);
  EXPECT_EQ(lowered.output_data_types.at(gather), filter.output_data_types());
}

TEST(PhysicalQepTransformationTest, EarlyMaterializationGathersAggregateInputCondition)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    read,
    {},
    std::vector<std::unique_ptr<gqe::expression>>{},
    std::move(values),
    less_condition(0, 1)};

  for (bool const is_late : {true, false}) {
    auto parameters                     = early_materialization_parameters();
    parameters.use_late_materialization = is_late;
    auto lowered            = make_physical_qep_transformation(&aggregate, &provider, parameters);
    auto const& plan        = lowered.plan;
    auto const* filter_task = single_task_of_type<qep::filter_task>(plan);
    auto const* reduce      = single_task_of_type<qep::reduce_aggregate_task>(plan);
    if (is_late) {
      EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
      EXPECT_EQ(plan.predecessors(reduce), std::vector<qep::task const*>{filter_task});
      continue;
    }
    // The gather materializes the aggregate's input, not its result.
    auto const* gather =
      expect_output_gather(lowered, filter_task, qep::lowering_role::materialize_input);
    ASSERT_NE(gather, nullptr);
    EXPECT_EQ(plan.predecessors(reduce), std::vector<qep::task const*>{gather});
  }
}

TEST(PhysicalQepTransformationTest, EarlyMaterializationGathersNonTerminalTopK)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  auto top_k          = std::make_shared<physical::top_k_relation>(
    read, 0, cudf::order::ASCENDING, cudf::null_order::AFTER, 0, 2);
  physical::fetch_relation fetch{top_k, /*offset=*/0, /*count=*/1};

  auto lowered =
    make_physical_qep_transformation(&fetch, &provider, early_materialization_parameters());
  auto const& plan = lowered.plan;

  ASSERT_EQ(tasks_of_type<qep::gather_task>(plan).size(), 1);
  auto const* gather = expect_output_gather(lowered, single_task_of_type<qep::top_k_task>(plan));
  ASSERT_NE(gather, nullptr);
  EXPECT_EQ(plan.predecessors(single_task_of_type<qep::fetch_task>(plan)),
            std::vector<qep::task const*>{gather});
}

TEST(PhysicalQepTransformationTest, EarlyMaterializationKeepsOneTerminalTopKGather)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::top_k_relation top_k{read, 0, cudf::order::DESCENDING, cudf::null_order::AFTER, 1, 2};

  auto lowered =
    make_physical_qep_transformation(&top_k, &provider, early_materialization_parameters());
  auto const& plan = lowered.plan;

  ASSERT_EQ(tasks_of_type<qep::gather_task>(plan).size(), 1);
  auto const* gather = expect_output_gather(lowered, single_task_of_type<qep::top_k_task>(plan));
  ASSERT_NE(gather, nullptr);
  EXPECT_TRUE(plan.successors(gather).empty());
}

namespace {

/**
 * @brief Expect early materialization to gather each join family's output exactly once.
 *
 * @tparam ExitTask Task type that terminates the join's lowering: the probe, or the residual
 *   filter of a mixed join.
 */
template <typename ExitTask>
void verify_early_join_gather(physical::broadcast_join_relation& join,
                              storage::table_provider const* provider,
                              gqe::optimization_parameters parameters)
{
  parameters.use_late_materialization = true;
  auto const late = make_physical_qep_transformation(&join, provider, parameters);
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(late.plan).empty());

  parameters.use_late_materialization = false;
  auto const early = make_physical_qep_transformation(&join, provider, parameters);
  ASSERT_EQ(tasks_of_type<qep::gather_task>(early.plan).size(), 1);
  auto const* gather = expect_output_gather(early, single_task_of_type<ExitTask>(early.plan));
  ASSERT_NE(gather, nullptr);
  EXPECT_TRUE(early.plan.successors(gather).empty());
  EXPECT_EQ(early.output_data_types.at(gather), join.output_data_types());
}

}  // namespace

TEST(PhysicalQepTransformationTest, EarlyMaterializationGathersHashJoinOutput)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto join =
    make_inner_join(make_read("t", 3, nullptr), make_read("t", 3, nullptr), 5, {4, 0, 4, 2}, false);
  verify_early_join_gather<qep::hash_join_probe_task>(
    *join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
}

TEST(PhysicalQepTransformationTest, EarlyMaterializationGathersPerfectHashJoinOutput)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto join =
    make_inner_join(make_read("t", 3, nullptr), make_read("t", 3, nullptr), 5, {4, 0, 4, 2}, true);
  auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_perfect_hash = true;
  verify_early_join_gather<qep::perfect_hash_join_probe_task>(*join, &provider, parameters);
}

TEST(PhysicalQepTransformationTest, EarlyMaterializationGathersConditionalInnerJoinOutput)
{
  auto const provider = make_provider_with_parquet_table("t");
  physical::broadcast_join_relation join{make_read("t", 3, nullptr),
                                         make_read("t", 3, nullptr),
                                         {},
                                         gqe::join_type_type::inner,
                                         less_condition(2, 4),
                                         {4, 0, 4, 2},
                                         physical::broadcast_policy::right,
                                         gqe::unique_keys_policy::none,
                                         false};
  verify_early_join_gather<qep::conditional_inner_join_probe_task>(
    join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
}

TEST(PhysicalQepTransformationTest, EarlyMaterializationGathersConditionalSemiJoinOutput)
{
  auto const provider = make_provider_with_parquet_table("t");
  physical::broadcast_join_relation join{make_read("t", 3, nullptr),
                                         make_read("t", 3, nullptr),
                                         {},
                                         gqe::join_type_type::left_semi,
                                         less_condition(2, 4),
                                         {2, 0, 2},
                                         physical::broadcast_policy::right,
                                         gqe::unique_keys_policy::right,
                                         true};
  auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_perfect_hash = true;
  verify_early_join_gather<qep::conditional_semi_anti_join_probe_task>(join, &provider, parameters);
}

TEST(PhysicalQepTransformationTest, EarlyMaterializationGathersMixedJoinAfterResidualFilter)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto condition      = std::make_unique<gqe::logical_and_expression>(
    std::shared_ptr<gqe::expression>{nulls_equal_condition(0, 2)},
    std::shared_ptr<gqe::expression>{less_condition(1, 3)});
  physical::broadcast_join_relation join{make_read("t", 2, nullptr),
                                         make_read("t", 2, nullptr),
                                         {},
                                         gqe::join_type_type::inner,
                                         std::move(condition),
                                         {0, 3},
                                         physical::broadcast_policy::right,
                                         gqe::unique_keys_policy::none,
                                         /*perfect_hashing=*/false};
  verify_early_join_gather<qep::filter_join_indices_task>(
    join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
}

TEST(PhysicalQepTransformationTest, EqualityLeftSemiJoinBuildsLogicalRight)
{
  verify_equality_semi_anti_join_lowering(gqe::join_type_type::left_semi,
                                          /*nulls_equal=*/false);
}

TEST(PhysicalQepTransformationTest, EqualityLeftAntiJoinBuildsLogicalRightAndAcceptsNullEquals)
{
  verify_equality_semi_anti_join_lowering(gqe::join_type_type::left_anti,
                                          /*nulls_equal=*/true);
}

TEST(PhysicalQepTransformationTest, HashJoinRejectsUnsupportedJoinTypesAndLeftBuild)
{
  auto const provider           = make_provider_with_parquet_table("t");
  auto const left               = make_read("t", 2, nullptr);
  auto const right              = make_read("t", 2, nullptr);
  auto parameters               = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_mark_join = false;
  for (auto const join_type : {gqe::join_type_type::full, gqe::join_type_type::single}) {
    physical::broadcast_join_relation join{
      left, right, {}, join_type, equal_condition(0, 2), {0}, physical::broadcast_policy::right};
    EXPECT_THROW(std::ignore = make_physical_qep_transformation(&join, &provider),
                 std::logic_error);
  }
  for (auto const join_type : {gqe::join_type_type::left,
                               gqe::join_type_type::left_semi,
                               gqe::join_type_type::left_anti}) {
    SCOPED_TRACE(static_cast<int>(join_type));
    physical::broadcast_join_relation invalid_left{
      left, right, {}, join_type, equal_condition(0, 2), {0}, physical::broadcast_policy::left};
    EXPECT_THROW(
      std::ignore = make_physical_qep_transformation(&invalid_left, &provider, parameters),
      std::logic_error);
  }
}

TEST(PhysicalQepTransformationTest, EqualitySemiAntiJoinsRejectNestedKeysOnEitherSide)
{
  auto const provider = make_provider_with_parquet_table("t");
  for (auto const type : {cudf::type_id::LIST, cudf::type_id::STRUCT}) {
    for (auto const mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
      auto const nested_type = cudf::data_type(type);
      for (auto const& [left_type, right_type] : {std::pair{nested_type, int64_type()},
                                                  std::pair{int64_type(), nested_type},
                                                  std::pair{nested_type, nested_type}}) {
        auto left  = make_read("t", 1, nullptr, {}, left_type);
        auto right = make_read("t", 1, nullptr, {}, right_type);
        physical::broadcast_join_relation join(
          left, right, {}, mode, equal_condition(0, 1), {}, physical::broadcast_policy::right);
        EXPECT_THROW(std::ignore = make_physical_qep_transformation(
                       &join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true)),
                     std::logic_error);
      }
    }
  }
}

TEST(PhysicalQepTransformationTest, HashJoinRejectsMixedNullPolicies)
{
  auto const provider  = make_provider_with_parquet_table("t");
  auto const left      = make_read("t", 2, nullptr);
  auto const right     = make_read("t", 2, nullptr);
  auto mixed_condition = std::make_unique<gqe::logical_and_expression>(
    std::shared_ptr<gqe::expression>{equal_condition(0, 2)},
    std::shared_ptr<gqe::expression>{nulls_equal_condition(1, 3)});
  physical::broadcast_join_relation mixed{left,
                                          right,
                                          {},
                                          gqe::join_type_type::inner,
                                          std::move(mixed_condition),
                                          {0},
                                          physical::broadcast_policy::right};
  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&mixed, &provider), std::logic_error);
}

TEST(PhysicalQepTransformationTest, MixedJoinUsesSelectedProbeFilterAndDefersPayloads)
{
  for (bool const is_perfect_hashing : {false, true}) {
    SCOPED_TRACE(is_perfect_hashing);
    auto const provider = make_provider_with_parquet_table("t");
    auto const left     = make_read("t", 3, nullptr);
    auto const right    = make_read("t", 3, nullptr);
    auto condition      = std::make_unique<gqe::logical_and_expression>(
      std::shared_ptr<gqe::expression>{equal_condition(1, 5)},
      std::shared_ptr<gqe::expression>{less_condition(2, 4)});
    physical::broadcast_join_relation join{left,
                                           right,
                                           {},
                                           gqe::join_type_type::inner,
                                           std::move(condition),
                                           {4, 0, 4, 2},
                                           physical::broadcast_policy::right,
                                           gqe::unique_keys_policy::right,
                                           /*perfect_hashing=*/is_perfect_hashing,
                                           /*left_filter_condition=*/nullptr,
                                           /*right_filter_condition=*/nullptr,
                                           /*use_hash_map_cache=*/false};

    auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
    parameters.join_use_perfect_hash = is_perfect_hashing;
    parameters.join_use_unique_keys  = true;
    auto lowered           = make_physical_qep_transformation(&join, &provider, parameters);
    auto const& plan       = lowered.plan;
    qep::task const* build = is_perfect_hashing
                               ? static_cast<qep::task const*>(
                                   single_task_of_type<qep::perfect_hash_join_build_task>(plan))
                               : single_task_of_type<qep::unique_key_inner_join_build_task>(plan);
    qep::task const* probe = is_perfect_hashing
                               ? static_cast<qep::task const*>(
                                   single_task_of_type<qep::perfect_hash_join_probe_task>(plan))
                               : single_task_of_type<qep::unique_key_inner_join_probe_task>(plan);
    auto const* filter     = single_task_of_type<qep::filter_join_indices_task>(plan);
    EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
    ASSERT_NE(build, nullptr);
    ASSERT_NE(probe, nullptr);
    ASSERT_NE(filter, nullptr);
    EXPECT_EQ(plan.predecessors(filter), std::vector<qep::task const*>{probe});
    EXPECT_EQ(lowered.output_data_types.at(probe).size(), 3);
    EXPECT_EQ(lowered.output_data_types.at(filter), join.output_data_types());
  }
}

TEST(PhysicalQepTransformationTest, MixedNullEqualJoinRetainsRegularHashProbe)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 2, nullptr);
  auto const right    = make_read("t", 2, nullptr);
  auto condition      = std::make_unique<gqe::logical_and_expression>(
    std::shared_ptr<gqe::expression>{nulls_equal_condition(0, 2)},
    std::shared_ptr<gqe::expression>{less_condition(1, 3)});
  physical::broadcast_join_relation join{left,
                                         right,
                                         {},
                                         gqe::join_type_type::inner,
                                         std::move(condition),
                                         {0, 3},
                                         physical::broadcast_policy::right,
                                         gqe::unique_keys_policy::none,
                                         /*perfect_hashing=*/true};
  auto const lowered = make_physical_qep_transformation(
    &join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
  auto const* probe  = single_task_of_type<qep::hash_join_probe_task>(lowered.plan);
  auto const* filter = single_task_of_type<qep::filter_join_indices_task>(lowered.plan);
  ASSERT_NE(probe, nullptr);
  ASSERT_NE(filter, nullptr);
  EXPECT_EQ(lowered.plan.predecessors(filter), std::vector<qep::task const*>{probe});
  EXPECT_TRUE(tasks_of_type<qep::perfect_hash_join_probe_task>(lowered.plan).empty());
}

TEST(PhysicalQepTransformationTest, MarkJoinSelectionUsesRuntimeFlagAndBroadcastSide)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 2, nullptr);
  auto const right    = make_read("t", 2, nullptr);
  for (auto const mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    for (auto const policy :
         {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
      for (bool const runtime_mark : {false, true}) {
        for (bool const plan_mark : {false, true}) {
          SCOPED_TRACE(static_cast<int>(mode));
          SCOPED_TRACE(static_cast<int>(policy));
          SCOPED_TRACE(runtime_mark);
          SCOPED_TRACE(plan_mark);
          auto parameters               = gqe::make_optimization_parameters(/*only_defaults=*/true);
          parameters.join_use_mark_join = runtime_mark;
          physical::broadcast_join_relation join{left,
                                                 right,
                                                 {},
                                                 mode,
                                                 equal_condition(0, 2),
                                                 {1},
                                                 policy,
                                                 gqe::unique_keys_policy::none,
                                                 /*perfect_hashing=*/false,
                                                 nullptr,
                                                 nullptr,
                                                 /*use_hash_map_cache=*/false,
                                                 plan_mark};
          if (policy == physical::broadcast_policy::left && !runtime_mark) {
            try {
              std::ignore = make_physical_qep_transformation(&join, &provider, parameters);
              FAIL() << "Expected non-mark broadcast-left semi/anti to be rejected";
            } catch (std::logic_error const& error) {
              EXPECT_NE(std::string{error.what()}.find("must build their logical right input"),
                        std::string::npos);
            }
            continue;
          }
          auto lowered = make_physical_qep_transformation(&join, &provider, parameters);
          qep::task const* build;
          if (policy == physical::broadcast_policy::left) {
            build = single_task_of_type<qep::mark_join_build_task>(lowered.plan);
            EXPECT_NE(single_task_of_type<qep::mark_join_probe_task>(lowered.plan), nullptr);
            EXPECT_NE(single_task_of_type<qep::mark_join_scan_task>(lowered.plan), nullptr);
            EXPECT_TRUE(
              tasks_of_type<qep::equality_semi_anti_join_build_task>(lowered.plan).empty());
          } else {
            build = single_task_of_type<qep::equality_semi_anti_join_build_task>(lowered.plan);
            EXPECT_NE(single_task_of_type<qep::equality_semi_anti_join_probe_task>(lowered.plan),
                      nullptr);
            EXPECT_TRUE(tasks_of_type<qep::mark_join_build_task>(lowered.plan).empty());
          }
          ASSERT_NE(build, nullptr);
          auto const predecessors = lowered.plan.predecessors(build);
          ASSERT_EQ(predecessors.size(), 1);
          EXPECT_EQ(lowered.physical_relations.at(predecessors.front()),
                    policy == physical::broadcast_policy::left ? left.get() : right.get());
          EXPECT_NO_THROW(std::ignore = qep::pipeline_normalization(lowered.plan,
                                                                    lowered.output_data_types,
                                                                    lowered.stable_keys,
                                                                    lowered.physical_relations));
        }
      }
    }
  }
}

TEST(PhysicalQepTransformationTest, MarkJoinUsesEffectivePerfectHashSetting)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 2, nullptr);
  auto const right    = make_read("t", 2, nullptr);
  for (auto const mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    for (bool const runtime_perfect : {false, true}) {
      for (bool const plan_perfect : {false, true}) {
        SCOPED_TRACE(static_cast<int>(mode));
        SCOPED_TRACE(runtime_perfect);
        SCOPED_TRACE(plan_perfect);
        auto parameters               = gqe::make_optimization_parameters(/*only_defaults=*/true);
        parameters.join_use_mark_join = true;
        parameters.join_use_perfect_hash = runtime_perfect;
        physical::broadcast_join_relation join{left,
                                               right,
                                               {},
                                               mode,
                                               equal_condition(0, 2),
                                               {1},
                                               physical::broadcast_policy::left,
                                               gqe::unique_keys_policy::left,
                                               plan_perfect,
                                               nullptr,
                                               nullptr,
                                               /*use_hash_map_cache=*/false,
                                               /*use_mark_join=*/true};
        if (runtime_perfect && plan_perfect) {
          EXPECT_THROW(std::ignore = make_physical_qep_transformation(&join, &provider, parameters),
                       std::logic_error);
        } else {
          auto lowered = make_physical_qep_transformation(&join, &provider, parameters);
          EXPECT_NE(single_task_of_type<qep::mark_join_build_task>(lowered.plan), nullptr);
          EXPECT_TRUE(tasks_of_type<qep::perfect_hash_join_build_task>(lowered.plan).empty());
        }
      }
    }
  }
}

TEST(PhysicalQepTransformationTest, MarkJoinHasDistinctStableKeysForInputsAndPhases)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 2, nullptr);
  auto const right    = make_read("t", 2, nullptr);
  for (auto const join_type : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    SCOPED_TRACE(static_cast<int>(join_type));
    auto join = std::make_shared<physical::broadcast_join_relation>(
      left,
      right,
      std::vector<std::shared_ptr<physical::relation>>{},
      join_type,
      equal_condition(0, 2),
      std::vector<cudf::size_type>{1},
      physical::broadcast_policy::left,
      gqe::unique_keys_policy::none,
      /*perfect_hashing=*/false,
      /*left_filter_condition=*/nullptr,
      /*right_filter_condition=*/nullptr,
      /*use_hash_map_cache=*/true,
      /*use_mark_join=*/true);
    physical::identified_plan identified_plan{
      join, {{join.get(), 17}, {left.get(), 29}, {right.get(), 31}}};
    auto lowered = qep::physical_qep_transformation(
      identified_plan, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
    ASSERT_EQ(lowered.stable_keys.size(), lowered.plan.tasks().size());
    std::map<qep::lowering_role, qep::task const*> join_tasks;
    for (auto const* task : lowered.plan.tasks()) {
      if (lowered.physical_relations.at(task) != join.get()) { continue; }
      auto const key = lowered.stable_keys.at(task);
      EXPECT_EQ(key.relation_id, 17);
      EXPECT_TRUE(join_tasks.emplace(key.role, task).second);
    }
    // Late materialization is on by default, so the scan feeds a join materialization step.
    ASSERT_EQ(join_tasks.size(), 5);
    auto const* build_task = join_tasks.at(qep::lowering_role::join_build);
    auto const* probe_task = join_tasks.at(qep::lowering_role::join_probe);
    EXPECT_EQ(lowered.stable_keys.at(lowered.plan.predecessors(build_task).front()).relation_id,
              29);
    auto const probe_inputs = lowered.plan.predecessors(probe_task);
    EXPECT_TRUE(std::ranges::any_of(probe_inputs, [&](auto const* task) {
      return lowered.stable_keys.at(task).relation_id == 31;
    }));
    EXPECT_TRUE(tasks_of_type<qep::gather_task>(lowered.plan).empty());
    EXPECT_NE(
      dynamic_cast<qep::mark_join_build_task const*>(join_tasks.at(qep::lowering_role::join_build)),
      nullptr);
    EXPECT_NE(
      dynamic_cast<qep::mark_join_probe_task const*>(join_tasks.at(qep::lowering_role::join_probe)),
      nullptr);
    EXPECT_NE(dynamic_cast<qep::mark_join_complete_task const*>(
                join_tasks.at(qep::lowering_role::join_complete)),
              nullptr);
    EXPECT_NE(
      dynamic_cast<qep::mark_join_scan_task const*>(join_tasks.at(qep::lowering_role::join_scan)),
      nullptr);
    auto const* materialize_task = join_tasks.at(qep::lowering_role::join_materialize);
    EXPECT_NE(dynamic_cast<qep::join_materialization_task const*>(materialize_task), nullptr);
    EXPECT_EQ(lowered.plan.predecessors(materialize_task),
              std::vector<qep::task const*>{join_tasks.at(qep::lowering_role::join_scan)});
    EXPECT_NO_THROW(
      std::ignore = qep::pipeline_normalization(
        lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations));
  }
}

TEST(PhysicalQepTransformationTest, MarkJoinRejectsNonNumericKeys)
{
  auto const provider           = make_provider_with_parquet_table("t");
  auto parameters               = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_mark_join = true;
  for (auto const type :
       {cudf::type_id::STRING, cudf::type_id::TIMESTAMP_DAYS, cudf::type_id::DECIMAL32}) {
    SCOPED_TRACE(static_cast<int>(type));
    auto const left  = make_read("t", 1, nullptr, {}, cudf::data_type{type});
    auto const right = make_read("t", 1, nullptr, {}, cudf::data_type{type});
    for (auto const mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
      SCOPED_TRACE(static_cast<int>(mode));
      physical::broadcast_join_relation join{
        left, right, {}, mode, equal_condition(0, 1), {0}, physical::broadcast_policy::left};
      EXPECT_THROW(std::ignore = make_physical_qep_transformation(&join, &provider, parameters),
                   std::logic_error);
    }
  }
}

TEST(PhysicalQepTransformationTest, MarkJoinRejectsUnsupportedResidual)
{
  auto const provider           = make_provider_with_parquet_table("t");
  auto const left               = make_read("t", 2, nullptr);
  auto const right              = make_read("t", 2, nullptr);
  auto parameters               = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_mark_join = true;
  for (auto const mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    SCOPED_TRACE(static_cast<int>(mode));
    auto residual = std::make_shared<gqe::less_expression>(
      std::make_shared<gqe::cast_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                             cudf::data_type{cudf::type_id::STRING}),
      std::make_shared<gqe::cast_expression>(std::make_shared<gqe::column_reference_expression>(3),
                                             cudf::data_type{cudf::type_id::STRING}));
    auto condition = std::make_unique<gqe::logical_and_expression>(
      std::shared_ptr<gqe::expression>{equal_condition(0, 2)}, std::move(residual));
    physical::broadcast_join_relation join{
      left, right, {}, mode, std::move(condition), {0}, physical::broadcast_policy::left};
    EXPECT_THROW(std::ignore = make_physical_qep_transformation(&join, &provider, parameters),
                 std::logic_error);
  }
}

TEST(PhysicalQepTransformationTest, MarkJoinRejectsMixedNullPolicies)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 2, nullptr);
  auto const right    = make_read("t", 2, nullptr);
  for (auto const join_type : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    SCOPED_TRACE(static_cast<int>(join_type));
    auto condition = std::make_unique<gqe::logical_and_expression>(
      std::shared_ptr<gqe::expression>{equal_condition(0, 2)},
      std::shared_ptr<gqe::expression>{nulls_equal_condition(1, 3)});
    physical::broadcast_join_relation join{left,
                                           right,
                                           {},
                                           join_type,
                                           std::move(condition),
                                           {0},
                                           physical::broadcast_policy::left,
                                           gqe::unique_keys_policy::none,
                                           false,
                                           nullptr,
                                           nullptr,
                                           true,
                                           true};
    try {
      std::ignore = make_physical_qep_transformation(&join, &provider);
      FAIL() << "Expected mixed null policies to be rejected";
    } catch (std::logic_error const& error) {
      EXPECT_NE(std::string{error.what()}.find("join equality keys use mixed null policies"),
                std::string::npos);
    }
  }
}

TEST(PhysicalQepTransformationTest, MarkJoinRejectsFusedInputFilters)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 2, nullptr);
  auto const right    = make_read("t", 2, nullptr);
  for (auto const join_type : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    for (auto const [has_left_filter, has_right_filter] :
         {std::pair{true, false}, std::pair{false, true}, std::pair{true, true}}) {
      SCOPED_TRACE(static_cast<int>(join_type));
      SCOPED_TRACE(has_left_filter);
      SCOPED_TRACE(has_right_filter);
      physical::broadcast_join_relation join{left,
                                             right,
                                             {},
                                             join_type,
                                             equal_condition(0, 2),
                                             {0},
                                             physical::broadcast_policy::left,
                                             gqe::unique_keys_policy::none,
                                             /*perfect_hashing=*/false,
                                             has_left_filter ? less_condition(0, 1) : nullptr,
                                             has_right_filter ? less_condition(2, 3) : nullptr,
                                             /*use_hash_map_cache=*/true,
                                             /*use_mark_join=*/true};
      try {
        std::ignore = make_physical_qep_transformation(&join, &provider);
        FAIL() << "Expected embedded input filters to be rejected";
      } catch (std::logic_error const& error) {
        EXPECT_NE(std::string{error.what()}.find("broadcast join input filters are not supported"),
                  std::string::npos);
      }
    }
  }
}

TEST(PhysicalQepTransformationTest, HashJoinRejectsFusedInputFilters)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 2, nullptr);
  auto const right    = make_read("t", 2, nullptr);
  physical::broadcast_join_relation join{left,
                                         right,
                                         {},
                                         gqe::join_type_type::inner,
                                         equal_condition(0, 2),
                                         {0},
                                         physical::broadcast_policy::right,
                                         gqe::unique_keys_policy::none,
                                         /*perfect_hashing=*/false,
                                         less_condition(0, 1)};
  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&join, &provider), std::logic_error);
}

TEST(PhysicalQepTransformationTest, PerfectHashJoinInputFilterThrowsUnsupported)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 2, nullptr);
  auto const right    = make_read("t", 2, nullptr);
  physical::broadcast_join_relation join{left,
                                         right,
                                         {},
                                         gqe::join_type_type::inner,
                                         equal_condition(0, 2),
                                         {0},
                                         physical::broadcast_policy::right,
                                         gqe::unique_keys_policy::right,
                                         /*perfect_hashing=*/true,
                                         /*left_filter_condition=*/nullptr,
                                         /*right_filter_condition=*/less_condition(2, 2)};

  auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_perfect_hash = true;
  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&join, &provider, parameters),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, PerfectHashJoinUnsupportedKeyTypeThrows)
{
  auto const provider      = make_provider_with_parquet_table("t");
  auto const duration_type = cudf::data_type{cudf::type_id::DURATION_SECONDS};
  auto const left          = make_read("t", 1, nullptr, {}, duration_type);
  auto const right         = make_read("t", 1, nullptr, {}, duration_type);
  physical::broadcast_join_relation join{left,
                                         right,
                                         {},
                                         gqe::join_type_type::inner,
                                         equal_condition(0, 1),
                                         {0},
                                         physical::broadcast_policy::right,
                                         gqe::unique_keys_policy::right,
                                         /*perfect_hashing=*/true};

  auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_perfect_hash = true;
  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&join, &provider, parameters),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, KeylessAggregateLowersToReduceAggregate)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  values.emplace_back(cudf::aggregation::COUNT_ALL, col_ref(0));
  physical::concatenate_aggregate_relation aggregate{read, {}, /*keys=*/{}, std::move(values)};

  auto normalized  = lower_and_normalize(&aggregate, &provider);
  auto const& plan = normalized.plan();

  ASSERT_EQ(plan.tasks().size(), 2);
  auto const* read_task   = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const* reduce_task = single_task_of_type<qep::reduce_aggregate_task>(plan);
  ASSERT_NE(reduce_task, nullptr);
  EXPECT_EQ(plan.predecessors(reduce_task), std::vector<qep::task const*>{read_task});
  EXPECT_TRUE(plan.successors(reduce_task).empty());
}

TEST(PhysicalQepTransformationTest, ProjectAfterAggregateGetsIntermediateReader)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  auto aggregate = std::make_shared<physical::concatenate_aggregate_relation>(
    read,
    std::vector<std::shared_ptr<physical::relation>>{},
    std::vector<std::unique_ptr<gqe::expression>>{},
    std::move(values));
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::column_reference_expression>(0)));
  physical::project_relation project{aggregate, {}, std::move(expressions)};

  auto normalized          = lower_and_normalize(&project, &provider);
  auto const& plan         = normalized.plan();
  auto const* reduce_task  = single_task_of_type<qep::reduce_aggregate_task>(plan);
  auto const* reader_task  = single_task_of_type<qep::intermediate_state_read_task>(plan);
  auto const* project_task = single_task_of_type<qep::project_task>(plan);
  ASSERT_NE(reduce_task, nullptr);
  ASSERT_NE(reader_task, nullptr);
  ASSERT_NE(project_task, nullptr);
  EXPECT_EQ(plan.predecessors(reader_task), std::vector<qep::task const*>{reduce_task});
  EXPECT_EQ(plan.predecessors(project_task), std::vector<qep::task const*>{reader_task});
}

TEST(PhysicalQepTransformationTest, AggregateConditionLowersToUpstreamFilter)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    read, {}, /*keys=*/{}, std::move(values), less_condition(0, 1)};

  auto normalized  = lower_and_normalize(&aggregate, &provider);
  auto const& plan = normalized.plan();

  // The reduce aggregate consumes the filter's masked output natively — no gather in between,
  // and the aggregate's plain output needs no terminal gather either.
  ASSERT_EQ(plan.tasks().size(), 3);
  auto const* filter_task = single_task_of_type<qep::filter_task>(plan);
  auto const* reduce_task = single_task_of_type<qep::reduce_aggregate_task>(plan);
  ASSERT_NE(filter_task, nullptr);
  ASSERT_NE(reduce_task, nullptr);
  EXPECT_EQ(plan.predecessors(reduce_task), std::vector<qep::task const*>{filter_task});
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
}

TEST(PhysicalQepTransformationTest, ComputedAggregateValueThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM,
                      std::make_unique<gqe::multiply_expression>(
                        std::make_shared<gqe::column_reference_expression>(0),
                        std::make_shared<gqe::column_reference_expression>(1)));
  physical::concatenate_aggregate_relation aggregate{read, {}, /*keys=*/{}, std::move(values)};

  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&aggregate, &provider),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, PerfectHashGroupByLowersAndCombinesCountPartialsWithSum)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::COUNT_VALID, col_ref(1));
  values.emplace_back(cudf::aggregation::COUNT_ALL, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    read, {}, std::move(keys), std::move(values), nullptr, /*perfect_hashing=*/true};

  auto parameters = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.aggregation_use_perfect_hash = true;
  auto lowered    = make_physical_qep_transformation(&aggregate, &provider, parameters);
  auto normalized = qep::pipeline_normalization(
    lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations);
  auto const& plan = normalized.plan();

  ASSERT_EQ(plan.tasks().size(), 3);
  auto const* read_task    = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const* partial_base = single_task_of_type<qep::perfect_hash_partial_aggregate_task>(plan);
  auto const* combine_base = single_task_of_type<qep::perfect_hash_aggregate_task>(plan);
  ASSERT_NE(read_task, nullptr);
  ASSERT_NE(partial_base, nullptr);
  ASSERT_NE(combine_base, nullptr);
  EXPECT_EQ(plan.predecessors(partial_base), std::vector<qep::task const*>{read_task});
  EXPECT_EQ(plan.predecessors(combine_base), std::vector<qep::task const*>{partial_base});
  EXPECT_TRUE(plan.successors(combine_base).empty());

  auto const* partial = dynamic_cast<qep::perfect_hash_partial_aggregate_task const*>(partial_base);
  auto const* combine = dynamic_cast<qep::perfect_hash_aggregate_task const*>(combine_base);
  ASSERT_NE(partial, nullptr);
  ASSERT_NE(combine, nullptr);

  auto chunk1 =
    make_table(int64_column_wrapper{1, 1}.release(), int64_column_wrapper{10, 11}.release());
  auto chunk2 =
    make_table(int64_column_wrapper{1, 1, 1}.release(), int64_column_wrapper{12, 13, 14}.release());
  gqe::context_reference ctx_ref{};
  auto partial1 = partial->next(chunk1, ctx_ref);
  auto partial2 = partial->next(chunk2, ctx_ref);
  ASSERT_TRUE(partial1.has_value());
  ASSERT_TRUE(partial2.has_value());

  auto accumulator = combine->initialize(ctx_ref);
  combine->next(*partial1, accumulator, ctx_ref);
  combine->next(*partial2, accumulator, ctx_ref);
  auto result      = combine->finalize(std::move(accumulator), ctx_ref);
  auto result_view = qep::to_table_view(result);
  ASSERT_TRUE(result_view.has_value());

  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(int64_column_wrapper{1}.release());
  expected_columns.push_back(int64_column_wrapper{5}.release());
  expected_columns.push_back(int64_column_wrapper{5}.release());
  auto expected = cudf::table{std::move(expected_columns)};
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result_view, expected.view());
}

TEST(PhysicalQepTransformationTest, PerfectHashGroupByDisabledGateFallsBackToHashAggregate)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    read, {}, std::move(keys), std::move(values), nullptr, /*perfect_hashing=*/true};

  auto parameters = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.aggregation_use_perfect_hash = false;
  auto lowered = make_physical_qep_transformation(&aggregate, &provider, parameters);

  EXPECT_NE(single_task_of_type<qep::hash_partial_aggregate_task>(lowered.plan), nullptr);
  EXPECT_NE(single_task_of_type<qep::hash_aggregate_task>(lowered.plan), nullptr);
  EXPECT_TRUE(tasks_of_type<qep::perfect_hash_partial_aggregate_task>(lowered.plan).empty());
  EXPECT_TRUE(tasks_of_type<qep::perfect_hash_aggregate_task>(lowered.plan).empty());
}

TEST(PhysicalQepTransformationTest, PerfectHashAggregateConditionDoesNotInsertGather)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{read,
                                                     {},
                                                     std::move(keys),
                                                     std::move(values),
                                                     less_condition(0, 1),
                                                     /*perfect_hashing=*/true};

  auto parameters = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.aggregation_use_perfect_hash = true;
  auto lowered    = make_physical_qep_transformation(&aggregate, &provider, parameters);
  auto normalized = qep::pipeline_normalization(
    lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations);
  auto const& plan = normalized.plan();

  ASSERT_EQ(plan.tasks().size(), 4);
  auto const* filter_task  = single_task_of_type<qep::filter_task>(plan);
  auto const* partial_task = single_task_of_type<qep::perfect_hash_partial_aggregate_task>(plan);
  auto const* combine_task = single_task_of_type<qep::perfect_hash_aggregate_task>(plan);
  ASSERT_NE(filter_task, nullptr);
  ASSERT_NE(partial_task, nullptr);
  ASSERT_NE(combine_task, nullptr);
  EXPECT_EQ(plan.predecessors(partial_task), std::vector<qep::task const*>{filter_task});
  EXPECT_EQ(plan.predecessors(combine_task), std::vector<qep::task const*>{partial_task});
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
}

TEST(PhysicalQepTransformationTest, HashGroupByLowersWithPartialAndFinalSchemas)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr, {}, int32_type());
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::COUNT_VALID, col_ref(1));
  values.emplace_back(cudf::aggregation::MAX, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{read, {}, std::move(keys), std::move(values)};

  auto lowered             = make_physical_qep_transformation(&aggregate, &provider);
  auto const* read_task    = single_task_of_type<storage::oneshot_parquet_read_task>(lowered.plan);
  auto const* partial_base = single_task_of_type<qep::hash_partial_aggregate_task>(lowered.plan);
  auto const* combine_base = single_task_of_type<qep::hash_aggregate_task>(lowered.plan);
  ASSERT_NE(read_task, nullptr);
  ASSERT_NE(partial_base, nullptr);
  ASSERT_NE(combine_base, nullptr);
  EXPECT_EQ(lowered.plan.predecessors(partial_base), std::vector<qep::task const*>{read_task});
  EXPECT_EQ(lowered.plan.predecessors(combine_base), std::vector<qep::task const*>{partial_base});
  EXPECT_EQ(lowered.output_data_types.at(partial_base),
            (std::vector<cudf::data_type>{int32_type(), int32_type(), int32_type()}));
  EXPECT_EQ(lowered.output_data_types.at(combine_base),
            (std::vector<cudf::data_type>{int32_type(), int64_type(), int32_type()}));

  auto const* partial = dynamic_cast<qep::hash_partial_aggregate_task const*>(partial_base);
  auto const* combine = dynamic_cast<qep::hash_aggregate_task const*>(combine_base);
  ASSERT_NE(partial, nullptr);
  ASSERT_NE(combine, nullptr);
  auto chunk1 =
    make_table(int32_column_wrapper{1, 1}.release(), int32_column_wrapper{10, 11}.release());
  auto chunk2 =
    make_table(int32_column_wrapper{1, 1, 2}.release(), int32_column_wrapper{12, 13, 20}.release());
  auto partial1 = partial->next(chunk1, gqe::context_reference{});
  auto partial2 = partial->next(chunk2, gqe::context_reference{});
  ASSERT_TRUE(partial1.has_value());
  ASSERT_TRUE(partial2.has_value());
  auto accumulator = combine->initialize(gqe::context_reference{});
  combine->next(*partial1, accumulator, gqe::context_reference{});
  combine->next(*partial2, accumulator, gqe::context_reference{});
  auto result      = combine->finalize(std::move(accumulator), gqe::context_reference{});
  auto result_view = qep::to_table_view(result);
  ASSERT_TRUE(result_view.has_value());
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(int32_column_wrapper{1, 2}.release());
  expected_columns.push_back(int64_column_wrapper{4, 1}.release());
  expected_columns.push_back(int32_column_wrapper{13, 20}.release());
  auto expected        = cudf::table{std::move(expected_columns)};
  auto sorted_result   = cudf::sort(*result_view);
  auto sorted_expected = cudf::sort(expected.view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_result->view(), sorted_expected->view());
}

TEST(PhysicalQepTransformationTest, HashAggregateConditionFeedsPartialWithoutGather)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    read, {}, std::move(keys), std::move(values), less_condition(0, 1)};

  auto normalized  = lower_and_normalize(&aggregate, &provider);
  auto const& plan = normalized.plan();
  ASSERT_EQ(plan.tasks().size(), 4);
  auto const* filter  = single_task_of_type<qep::filter_task>(plan);
  auto const* partial = single_task_of_type<qep::hash_partial_aggregate_task>(plan);
  auto const* combine = single_task_of_type<qep::hash_aggregate_task>(plan);
  ASSERT_NE(filter, nullptr);
  ASSERT_NE(partial, nullptr);
  ASSERT_NE(combine, nullptr);
  EXPECT_EQ(plan.predecessors(partial), std::vector<qep::task const*>{filter});
  EXPECT_EQ(plan.predecessors(combine), std::vector<qep::task const*>{partial});
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
}

TEST(PhysicalQepTransformationTest, HashAggregateRejectsUnsupportedKinds)
{
  auto const provider = make_provider_with_parquet_table("t");
  for (auto const kind : {cudf::aggregation::PRODUCT, cudf::aggregation::MEAN}) {
    auto read = make_read("t", 2, nullptr);
    std::vector<std::unique_ptr<gqe::expression>> keys;
    keys.push_back(col_ref(0));
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
    values.emplace_back(kind, col_ref(1));
    physical::concatenate_aggregate_relation aggregate{
      read, {}, std::move(keys), std::move(values)};
    EXPECT_THROW(std::ignore = make_physical_qep_transformation(&aggregate, &provider),
                 std::logic_error);
  }
}

TEST(PhysicalQepTransformationTest, HashAggregateRejectsNestedDomains)
{
  auto const provider = make_provider_with_parquet_table("t");

  {
    auto read = make_read("t", 1, nullptr, {}, cudf::data_type{cudf::type_id::LIST});
    std::vector<std::unique_ptr<gqe::expression>> keys;
    keys.push_back(col_ref(0));
    physical::concatenate_aggregate_relation aggregate{read, {}, std::move(keys), /*values=*/{}};
    EXPECT_THROW(std::ignore = make_physical_qep_transformation(&aggregate, &provider),
                 std::logic_error);
  }

  {
    auto read = std::make_shared<physical::read_relation>(
      std::vector<std::shared_ptr<physical::relation>>{},
      std::vector<std::string>{"key", "value"},
      "t",
      nullptr,
      std::vector<cudf::data_type>{int64_type(), cudf::data_type{cudf::type_id::STRUCT}});
    std::vector<std::unique_ptr<gqe::expression>> keys;
    keys.push_back(col_ref(0));
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
    values.emplace_back(cudf::aggregation::MIN, col_ref(1));
    physical::concatenate_aggregate_relation aggregate{
      read, {}, std::move(keys), std::move(values)};
    EXPECT_THROW(std::ignore = make_physical_qep_transformation(&aggregate, &provider),
                 std::logic_error);
  }
}

TEST(PhysicalQepTransformationTest, ProjectAfterHashAggregateGetsIntermediateReader)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  auto aggregate = std::make_shared<physical::concatenate_aggregate_relation>(
    read, std::vector<std::shared_ptr<physical::relation>>{}, std::move(keys), std::move(values));
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::column_reference_expression>(1)));
  physical::project_relation project{aggregate, {}, std::move(expressions)};

  auto normalized          = lower_and_normalize(&project, &provider);
  auto const& plan         = normalized.plan();
  auto const* combine_task = single_task_of_type<qep::hash_aggregate_task>(plan);
  auto const* reader_task  = single_task_of_type<qep::intermediate_state_read_task>(plan);
  auto const* project_task = single_task_of_type<qep::project_task>(plan);
  ASSERT_NE(combine_task, nullptr);
  ASSERT_NE(reader_task, nullptr);
  ASSERT_NE(project_task, nullptr);
  EXPECT_EQ(plan.predecessors(reader_task), std::vector<qep::task const*>{combine_task});
  EXPECT_EQ(plan.predecessors(project_task), std::vector<qep::task const*>{reader_task});
}

TEST(PhysicalQepTransformationTest, PerfectHashComputedKeyThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::column_reference_expression>(1)));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  physical::concatenate_aggregate_relation aggregate{
    read, {}, std::move(keys), std::move(values), nullptr, /*perfect_hashing=*/true};

  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&aggregate, &provider),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, PerfectHashComputedValueThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM,
                      std::make_unique<gqe::multiply_expression>(
                        std::make_shared<gqe::column_reference_expression>(0),
                        std::make_shared<gqe::column_reference_expression>(1)));
  physical::concatenate_aggregate_relation aggregate{
    read, {}, std::move(keys), std::move(values), nullptr, /*perfect_hashing=*/true};

  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&aggregate, &provider),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, BoundedHashAggregateRelationLowersToBuildAndScanTasks)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::bounded_hash_aggregate_relation aggregate(
    read,
    /*key_indices=*/{0},
    /*values=*/{{cudf::aggregation::SUM, 1, gqe::nullability::nullable}},
    /*group_count_bound=*/1000,
    /*desired_load_factor=*/0.7);

  auto lowered     = make_physical_qep_transformation(&aggregate, &provider);
  auto const& plan = lowered.plan;

  ASSERT_EQ(plan.tasks().size(), 3);
  auto const* read_task  = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const* build_task = single_task_of_type<qep::bounded_hash_aggregate_build_task>(plan);
  auto const* scan_task  = single_task_of_type<qep::bounded_hash_aggregate_scan_task>(plan);
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
  ASSERT_NE(build_task, nullptr);
  ASSERT_NE(scan_task, nullptr);
  EXPECT_EQ(plan.predecessors(build_task), std::vector<qep::task const*>{read_task});
  EXPECT_EQ(plan.predecessors(scan_task), std::vector<qep::task const*>{build_task});
  EXPECT_TRUE(plan.successors(scan_task).empty());
  EXPECT_TRUE(lowered.output_data_types.at(build_task).empty());
  EXPECT_EQ(lowered.output_data_types.at(scan_task), aggregate.output_data_types());
  EXPECT_EQ(lowered.physical_relations.at(build_task), &aggregate);
  EXPECT_EQ(lowered.physical_relations.at(scan_task), &aggregate);
  EXPECT_EQ(lowered.stable_keys.at(build_task).role, qep::lowering_role::aggregate_build);
  EXPECT_EQ(lowered.stable_keys.at(scan_task).role, qep::lowering_role::aggregate_scan);
}

/**
 * A pre-aggregation filter can preserve indirect payloads. The bounded build task consumes
 * its logical output directly and materializes only its required columns.
 */
TEST(PhysicalQepTransformationTest, BoundedHashAggregateRelationConsumesFilteredInput)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  auto filter =
    std::make_shared<physical::filter_relation>(read,
                                                std::vector<std::shared_ptr<physical::relation>>{},
                                                less_condition(0, 1),
                                                identity_indices(2));
  physical::bounded_hash_aggregate_relation aggregate(
    filter,
    /*key_indices=*/{0},
    /*values=*/{{cudf::aggregation::SUM, 1, gqe::nullability::nullable}},
    /*group_count_bound=*/1000);

  auto lowered     = make_physical_qep_transformation(&aggregate, &provider);
  auto const& plan = lowered.plan;

  ASSERT_EQ(plan.tasks().size(), 4);
  auto const* filter_task = single_task_of_type<qep::filter_task>(plan);
  auto const* build_task  = single_task_of_type<qep::bounded_hash_aggregate_build_task>(plan);
  ASSERT_NE(filter_task, nullptr);
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
  ASSERT_NE(build_task, nullptr);
  EXPECT_EQ(plan.predecessors(build_task), std::vector<qep::task const*>{filter_task});
}

/**
 * The dedicated relation asserts the bounded lowering: an unsupported type combination (here a
 * FLOAT64 key, which the operator rejects) propagates the operator's construction error
 * instead of falling back.
 */
TEST(PhysicalQepTransformationTest, BoundedHashAggregateRelationUnsupportedTypeThrows)
{
  auto const provider   = make_provider_with_parquet_table("t");
  auto const float_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto read             = make_read("t", 2, nullptr, {}, float_type);
  physical::bounded_hash_aggregate_relation aggregate(
    read,
    /*key_indices=*/{0},
    /*values=*/{{cudf::aggregation::SUM, 1, gqe::nullability::nullable}},
    /*group_count_bound=*/1000);

  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&aggregate, &provider),
               std::logic_error);
}

/** A DECIMAL64 SUM lowers via the DECIMAL128-widened output schema the operator requires. */
TEST(PhysicalQepTransformationTest, BoundedHashAggregateRelationLowersDecimalSum)
{
  auto const provider     = make_provider_with_parquet_table("t");
  auto const decimal_type = cudf::data_type{cudf::type_id::DECIMAL64, -2};
  auto read               = make_read("t", 2, nullptr, {}, decimal_type);
  physical::bounded_hash_aggregate_relation aggregate(
    read,
    /*key_indices=*/{0},
    /*values=*/{{cudf::aggregation::SUM, 1, gqe::nullability::nullable}},
    /*group_count_bound=*/1000);

  auto lowered     = make_physical_qep_transformation(&aggregate, &provider);
  auto const& plan = lowered.plan;

  auto const* scan_task = single_task_of_type<qep::bounded_hash_aggregate_scan_task>(plan);
  ASSERT_NE(scan_task, nullptr);
  auto const expected =
    std::vector<cudf::data_type>{decimal_type, cudf::data_type{cudf::type_id::DECIMAL128, -2}};
  EXPECT_EQ(lowered.output_data_types.at(scan_task), expected);
}

/** COUNT_ALL is the only value shape whose column index is nullopt. */
TEST(PhysicalQepTransformationTest, BoundedHashAggregateRelationLowersCountAll)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::bounded_hash_aggregate_relation aggregate(
    read,
    /*key_indices=*/{0},
    /*values=*/{{cudf::aggregation::COUNT_ALL, std::nullopt, gqe::nullability::nullable}},
    /*group_count_bound=*/1000);

  auto lowered     = make_physical_qep_transformation(&aggregate, &provider);
  auto const& plan = lowered.plan;

  ASSERT_EQ(plan.tasks().size(), 3);
  auto const* build_task = single_task_of_type<qep::bounded_hash_aggregate_build_task>(plan);
  auto const* scan_task  = single_task_of_type<qep::bounded_hash_aggregate_scan_task>(plan);
  ASSERT_NE(build_task, nullptr);
  ASSERT_NE(scan_task, nullptr);
  EXPECT_EQ(lowered.output_data_types.at(scan_task), aggregate.output_data_types());
}

/** The relation constructs, but the lowering supports only a single grouping key. */
TEST(PhysicalQepTransformationTest, BoundedHashAggregateRelationMultiKeyThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 3, nullptr);
  physical::bounded_hash_aggregate_relation aggregate(
    read,
    /*key_indices=*/{0, 1},
    /*values=*/{{cudf::aggregation::SUM, 2, gqe::nullability::nullable}},
    /*group_count_bound=*/1000);

  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&aggregate, &provider),
               std::logic_error);
}

/** The relation constructs, but the lowering supports only a single aggregate value. */
TEST(PhysicalQepTransformationTest, BoundedHashAggregateRelationMultiValueThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 3, nullptr);
  physical::bounded_hash_aggregate_relation aggregate(
    read,
    /*key_indices=*/{0},
    /*values=*/
    {{cudf::aggregation::SUM, 1, gqe::nullability::nullable},
     {cudf::aggregation::MIN, 2, gqe::nullability::nullable}},
    /*group_count_bound=*/1000);

  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&aggregate, &provider),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, SubqueryInFilterConditionThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 1, nullptr);
  auto haystack       = make_read("t", 1, nullptr);

  std::vector<std::shared_ptr<gqe::expression>> needles;
  needles.push_back(std::make_shared<gqe::column_reference_expression>(0));
  auto condition = std::make_unique<gqe::in_predicate_expression>(std::move(needles), 0);
  physical::filter_relation filter{read, {haystack}, std::move(condition), identity_indices(1)};

  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&filter, &provider),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, UnsupportedRelationKindThrows)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 1, nullptr);
  physical::union_all_relation union_all{read, make_read("t", 1, nullptr)};

  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&union_all, &provider),
               std::logic_error);
}

TEST(PhysicalQepTransformationTest, FetchLowersToStatefulTransformOnReadPipeline)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  physical::fetch_relation fetch{read, /*offset=*/1, /*count=*/10};

  auto lowered     = make_physical_qep_transformation(&fetch, &provider);
  auto const& plan = lowered.plan;

  ASSERT_EQ(plan.tasks().size(), 2);
  auto const* fetch_task = single_task_of_type<qep::fetch_task>(plan);
  auto const* read_task  = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  ASSERT_NE(fetch_task, nullptr);
  ASSERT_NE(read_task, nullptr);
  EXPECT_EQ(plan.predecessors(fetch_task), std::vector<qep::task const*>{read_task});
  EXPECT_TRUE(plan.successors(fetch_task).empty());
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
  EXPECT_EQ(lowered.output_data_types.at(fetch_task), fetch.output_data_types());
  EXPECT_EQ(lowered.physical_relations.at(fetch_task), &fetch);
}

TEST(PhysicalQepTransformationTest, FetchValidatesAndClampsWindowBounds)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 1, nullptr);

  // No producer validates the bounds, so the lowering must reject a negative offset before any
  // arithmetic on it.
  physical::fetch_relation negative_offset{read, /*offset=*/-1, /*count=*/10};
  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&negative_offset, &provider),
               std::logic_error);

  // A frontend may emit the largest count. Together with a positive offset it must clamp to the
  // largest expressible window instead of failing the query.
  physical::fetch_relation oversized{read, /*offset=*/1, std::numeric_limits<int64_t>::max()};
  EXPECT_NO_THROW(std::ignore = make_physical_qep_transformation(&oversized, &provider));
}

TEST(PhysicalQepTransformationTest, FetchConsumesMaskedInputWithoutInteriorGather)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  auto filter =
    std::make_shared<physical::filter_relation>(read,
                                                std::vector<std::shared_ptr<physical::relation>>{},
                                                less_condition(0, 1),
                                                identity_indices(2));
  physical::fetch_relation fetch{filter, /*offset=*/0, /*count=*/5};

  auto lowered     = make_physical_qep_transformation(&fetch, &provider);
  auto const& plan = lowered.plan;

  auto const* filter_task = single_task_of_type<qep::filter_task>(plan);
  auto const* fetch_task  = single_task_of_type<qep::fetch_task>(plan);
  ASSERT_NE(filter_task, nullptr);
  ASSERT_NE(fetch_task, nullptr);
  EXPECT_EQ(plan.predecessors(fetch_task), std::vector<qep::task const*>{filter_task});
  EXPECT_TRUE(plan.successors(fetch_task).empty());
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(plan).empty());
}

TEST(PhysicalQepTransformationTest, InPredicatePartialFilterWiresHaystackPredecessor)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto haystack       = make_read("t", 1, nullptr);

  std::vector<std::shared_ptr<gqe::expression>> needles;
  needles.push_back(std::make_shared<gqe::column_reference_expression>(0));
  auto partial_filter = std::make_unique<gqe::in_predicate_expression>(std::move(needles), 0);
  auto read           = make_read("t", 1, std::move(partial_filter), {haystack});

  auto normalized  = lower_and_normalize(read.get(), &provider);
  auto const& plan = normalized.plan();

  // Each read pipeline ends at an intermediate writer. The first writer feeds the pruned read.
  auto const reads   = tasks_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const writers = tasks_of_type<qep::intermediate_state_write_task>(plan);
  ASSERT_EQ(plan.tasks().size(), 7);
  ASSERT_EQ(reads.size(), 2);
  ASSERT_EQ(writers.size(), 2);
  EXPECT_EQ(tasks_of_type<qep::gather_task>(plan).size(), 3);
  auto const* pruned = plan.predecessors(reads[0]).empty() ? reads[1] : reads[0];
  ASSERT_EQ(plan.predecessors(pruned).size(), 1);
  EXPECT_NE(
    dynamic_cast<qep::intermediate_state_write_task const*>(plan.predecessors(pruned).front()),
    nullptr);
  EXPECT_FALSE(plan.successors(pruned).empty());
}

TEST(PhysicalQepTransformationTest, MaskedHaystackIsMaterializedBeforePrunedRead)
{
  auto const provider  = make_provider_with_parquet_table("t");
  auto haystack_source = make_read("t", 1, nullptr);
  auto haystack =
    std::make_shared<physical::filter_relation>(haystack_source,
                                                std::vector<std::shared_ptr<physical::relation>>{},
                                                less_condition(0, 0),
                                                identity_indices(1));

  std::vector<std::shared_ptr<gqe::expression>> needles;
  needles.push_back(std::make_shared<gqe::column_reference_expression>(0));
  auto partial_filter = std::make_unique<gqe::in_predicate_expression>(std::move(needles), 0);
  auto read           = make_read("t", 1, std::move(partial_filter), {haystack});

  auto normalized  = lower_and_normalize(read.get(), &provider);
  auto const& plan = normalized.plan();

  // The unconditional haystack gather remains. Pipeline normalization then inserts the writer
  // boundary required before the pruned read and another writer boundary after it.
  ASSERT_EQ(plan.tasks().size(), 8);
  EXPECT_EQ(tasks_of_type<qep::gather_task>(plan).size(), 3);
  EXPECT_EQ(tasks_of_type<qep::intermediate_state_write_task>(plan).size(), 2);
  auto const reads = tasks_of_type<storage::oneshot_parquet_read_task>(plan);
  ASSERT_EQ(reads.size(), 2);
  auto const* pruned = plan.predecessors(reads[0]).empty() ? reads[1] : reads[0];
  ASSERT_EQ(plan.predecessors(pruned).size(), 1);
  EXPECT_NE(
    dynamic_cast<qep::intermediate_state_write_task const*>(plan.predecessors(pruned).front()),
    nullptr);
}

TEST(PhysicalQepTransformationTest, ConcatenateSortLowersColumnReferenceKeys)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(1));
  keys.push_back(col_ref(0));
  physical::concatenate_sort_relation sort(read,
                                           {},
                                           std::move(keys),
                                           {cudf::order::DESCENDING, cudf::order::ASCENDING},
                                           {cudf::null_order::BEFORE, cudf::null_order::AFTER});

  auto lowered     = make_physical_qep_transformation(&sort, &provider);
  auto const tasks = lowered.plan.tasks();

  ASSERT_EQ(tasks.size(), 3);
  auto const* read_task   = single_task_of_type<storage::oneshot_parquet_read_task>(lowered.plan);
  auto const* gather_task = single_task_of_type<qep::gather_task>(lowered.plan);
  auto const* sort_task   = single_task_of_type<qep::merge_sort_task>(lowered.plan);
  ASSERT_NE(read_task, nullptr);
  ASSERT_NE(gather_task, nullptr);
  ASSERT_NE(sort_task, nullptr);
  EXPECT_EQ(lowered.plan.predecessors(gather_task), std::vector<qep::task const*>{read_task});
  EXPECT_EQ(lowered.plan.predecessors(sort_task), std::vector<qep::task const*>{gather_task});
  EXPECT_TRUE(lowered.plan.successors(sort_task).empty());
  EXPECT_EQ(lowered.output_data_types.at(sort_task), sort.output_data_types());
  EXPECT_EQ(lowered.physical_relations.at(sort_task), &sort);

  auto normalized = qep::pipeline_normalization(
    lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations);
  EXPECT_EQ(normalized.plan().tasks().size(), 3);
}

TEST(PhysicalQepTransformationTest, ConcatenateSortRejectsInvalidConfiguration)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);

  physical::concatenate_sort_relation empty_keys(read, {}, {}, {}, {});
  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&empty_keys, &provider),
               std::invalid_argument);

  std::vector<std::unique_ptr<gqe::expression>> null_keys(1);
  physical::concatenate_sort_relation null_key(
    read, {}, std::move(null_keys), {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&null_key, &provider),
               std::invalid_argument);

  std::vector<std::unique_ptr<gqe::expression>> mismatched_keys;
  mismatched_keys.push_back(col_ref(0));
  physical::concatenate_sort_relation mismatched_orders(
    read, {}, std::move(mismatched_keys), {}, {cudf::null_order::AFTER});
  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&mismatched_orders, &provider),
               std::invalid_argument);

  std::vector<std::unique_ptr<gqe::expression>> mismatched_null_precedence_keys;
  mismatched_null_precedence_keys.push_back(col_ref(0));
  physical::concatenate_sort_relation mismatched_null_precedences(
    read, {}, std::move(mismatched_null_precedence_keys), {cudf::order::ASCENDING}, {});
  EXPECT_THROW(
    std::ignore = make_physical_qep_transformation(&mismatched_null_precedences, &provider),
    std::invalid_argument);

  std::vector<std::unique_ptr<gqe::expression>> out_of_range_keys;
  out_of_range_keys.push_back(col_ref(2));
  physical::concatenate_sort_relation out_of_range_key(
    read, {}, std::move(out_of_range_keys), {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&out_of_range_key, &provider),
               std::invalid_argument);
}

TEST(PhysicalQepTransformationTest, ConcatenateSortRejectsComputedKeys)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::column_reference_expression>(1)));
  physical::concatenate_sort_relation sort(
    read, {}, std::move(keys), {cudf::order::ASCENDING}, {cudf::null_order::AFTER});

  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&sort, &provider), std::logic_error);
}

TEST(PhysicalQepTransformationTest, ConcatenateSortRejectsSubqueryKeys)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);

  std::vector<std::shared_ptr<gqe::expression>> needles;
  needles.push_back(std::make_shared<gqe::column_reference_expression>(0));
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(std::make_unique<gqe::in_predicate_expression>(std::move(needles), 0));
  physical::concatenate_sort_relation sort(read,
                                           std::vector<std::shared_ptr<physical::relation>>{read},
                                           std::move(keys),
                                           {cudf::order::ASCENDING},
                                           {cudf::null_order::AFTER});

  EXPECT_THROW(std::ignore = make_physical_qep_transformation(&sort, &provider), std::logic_error);
}

TEST(PhysicalQepTransformationTest, ConcatenateSortMaterializesMaskedInput)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto read           = make_read("t", 2, nullptr);
  auto filter =
    std::make_shared<physical::filter_relation>(read,
                                                std::vector<std::shared_ptr<physical::relation>>{},
                                                less_condition(0, 1),
                                                identity_indices(2));
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(col_ref(0));
  physical::concatenate_sort_relation sort(
    filter, {}, std::move(keys), {cudf::order::ASCENDING}, {cudf::null_order::AFTER});

  auto normalized  = lower_and_normalize(&sort, &provider);
  auto const& plan = normalized.plan();

  ASSERT_EQ(plan.tasks().size(), 4);
  auto const* filter_task = single_task_of_type<qep::filter_task>(plan);
  auto const* gather_task = single_task_of_type<qep::gather_task>(plan);
  auto const* sort_task   = single_task_of_type<qep::merge_sort_task>(plan);
  ASSERT_NE(filter_task, nullptr);
  ASSERT_NE(gather_task, nullptr);
  ASSERT_NE(sort_task, nullptr);
  EXPECT_EQ(plan.predecessors(gather_task), std::vector<qep::task const*>{filter_task});
  EXPECT_EQ(plan.predecessors(sort_task), std::vector<qep::task const*>{gather_task});
}

TEST(PhysicalQepTransformationTest, PlainInputToWriteGetsGatherBoundary)
{
  auto provider = make_provider_with_parquet_table("t");
  provider.add_table(
    "out", std::make_shared<storage::parquet_table>(std::vector<std::string>{"out.parquet"}));
  auto read = make_read("t", 2, nullptr);
  physical::write_relation write{read, {"col_0", "col_1"}, "out"};

  auto lowered = make_physical_qep_transformation(
    &write, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
  auto const* lowered_gather = single_task_of_type<qep::gather_task>(lowered.plan);
  ASSERT_NE(lowered_gather, nullptr);
  EXPECT_EQ(lowered.output_data_types.at(lowered_gather), read->output_data_types());
  EXPECT_EQ(lowered.physical_relations.at(lowered_gather), &write);
  auto normalized = qep::pipeline_normalization(
    lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations);
  auto const& plan = normalized.plan();

  ASSERT_EQ(plan.tasks().size(), 3);
  auto const* read_task   = single_task_of_type<storage::oneshot_parquet_read_task>(plan);
  auto const* gather_task = single_task_of_type<qep::gather_task>(plan);
  auto const* fold_task   = single_task_of_type<storage::chunked_parquet_write_task>(plan);
  ASSERT_NE(gather_task, nullptr);
  ASSERT_NE(fold_task, nullptr);
  EXPECT_EQ(plan.predecessors(gather_task), std::vector<qep::task const*>{read_task});
  EXPECT_EQ(plan.predecessors(fold_task), std::vector<qep::task const*>{gather_task});
  EXPECT_TRUE(plan.successors(fold_task).empty());
}

TEST(PhysicalQepTransformationTest, AggregateBeforeWriteGetsIntermediateReader)
{
  auto provider = make_provider_with_parquet_table("t");
  provider.add_table(
    "out", std::make_shared<storage::parquet_table>(std::vector<std::string>{"out.parquet"}));
  auto read = make_read("t", 2, nullptr);
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM, col_ref(1));
  auto aggregate = std::make_shared<physical::concatenate_aggregate_relation>(
    read,
    std::vector<std::shared_ptr<physical::relation>>{},
    std::vector<std::unique_ptr<gqe::expression>>{},
    std::move(values));
  physical::write_relation write{aggregate, {"sum"}, "out"};

  auto normalized         = lower_and_normalize(&write, &provider);
  auto const& plan        = normalized.plan();
  auto const* reduce_task = single_task_of_type<qep::reduce_aggregate_task>(plan);
  auto const* reader_task = single_task_of_type<qep::intermediate_state_read_task>(plan);
  auto const* gather_task = single_task_of_type<qep::gather_task>(plan);
  auto const* writer_task = single_task_of_type<storage::chunked_parquet_write_task>(plan);
  ASSERT_NE(reduce_task, nullptr);
  ASSERT_NE(reader_task, nullptr);
  ASSERT_NE(gather_task, nullptr);
  ASSERT_NE(writer_task, nullptr);
  EXPECT_EQ(plan.predecessors(reader_task), std::vector<qep::task const*>{reduce_task});
  EXPECT_EQ(plan.predecessors(gather_task), std::vector<qep::task const*>{reader_task});
  EXPECT_EQ(plan.predecessors(writer_task), std::vector<qep::task const*>{gather_task});
}

TEST(PhysicalQepTransformationTest, MaskedInputToWriteIsMaterialized)
{
  auto provider = make_provider_with_parquet_table("t");
  provider.add_table(
    "out", std::make_shared<storage::parquet_table>(std::vector<std::string>{"out.parquet"}));
  auto read = make_read("t", 2, nullptr);
  auto filter =
    std::make_shared<physical::filter_relation>(read,
                                                std::vector<std::shared_ptr<physical::relation>>{},
                                                less_condition(0, 1),
                                                identity_indices(2));
  physical::write_relation write{filter, {"col_0", "col_1"}, "out"};

  auto normalized  = lower_and_normalize(&write, &provider);
  auto const& plan = normalized.plan();

  // read -> filter -> gather -> write: the writer fold rejects masked input.
  ASSERT_EQ(plan.tasks().size(), 4);
  auto const* gather_task = single_task_of_type<qep::gather_task>(plan);
  auto const* fold_task   = single_task_of_type<storage::chunked_parquet_write_task>(plan);
  ASSERT_NE(gather_task, nullptr);
  ASSERT_NE(fold_task, nullptr);
  EXPECT_EQ(plan.predecessors(fold_task), std::vector<qep::task const*>{gather_task});
}

TEST(PhysicalQepTransformationTest, SelectsUniqueKeyTasksDuringLowering)
{
  auto const provider = make_provider_with_parquet_table("t");
  for (auto policy : {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
    for (auto unique_keys : {gqe::unique_keys_policy::none,
                             gqe::unique_keys_policy::left,
                             gqe::unique_keys_policy::right}) {
      for (bool enabled : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "policy=" << static_cast<int>(policy)
                                          << ", unique_keys=" << static_cast<int>(unique_keys)
                                          << ", enabled=" << enabled);
        auto left  = make_read("t", 3, nullptr);
        auto right = make_read("t", 3, nullptr);
        physical::broadcast_join_relation join(left,
                                               right,
                                               {},
                                               gqe::join_type_type::inner,
                                               equal_condition(1, 5),
                                               {4, 0},
                                               policy,
                                               unique_keys,
                                               false);
        bool const is_unique = enabled && unique_keys == (policy == physical::broadcast_policy::left
                                                            ? gqe::unique_keys_policy::left
                                                            : gqe::unique_keys_policy::right);
        verify_unique_key_join_lowering(join, provider, enabled, is_unique);
      }
    }
  }
}

TEST(PhysicalQepTransformationTest, UniqueKeySelectionChecksEveryKeyType)
{
  auto const provider = make_provider_with_parquet_table("t");
  for (auto type : {cudf::type_id::INT64,
                    cudf::type_id::FLOAT64,
                    cudf::type_id::BOOL8,
                    cudf::type_id::DECIMAL64,
                    cudf::type_id::STRING,
                    cudf::type_id::TIMESTAMP_DAYS,
                    cudf::type_id::DURATION_SECONDS}) {
    for (auto policy : {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
      for (bool enabled : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "type=" << static_cast<int>(type) << ", policy="
                                          << static_cast<int>(policy) << ", enabled=" << enabled);
        auto make_input = [&] {
          return std::make_shared<physical::read_relation>(
            std::vector<std::shared_ptr<physical::relation>>{},
            std::vector<std::string>{"col_0", "col_1"},
            "t",
            nullptr,
            std::vector<cudf::data_type>{int64_type(), cudf::data_type{type}});
        };
        auto left      = make_input();
        auto right     = make_input();
        auto condition = std::make_unique<gqe::logical_and_expression>(equal_condition(0, 2),
                                                                       equal_condition(1, 3));
        physical::broadcast_join_relation join(left,
                                               right,
                                               {},
                                               gqe::join_type_type::inner,
                                               std::move(condition),
                                               {0, 2},
                                               policy,
                                               policy == physical::broadcast_policy::right
                                                 ? gqe::unique_keys_policy::right
                                                 : gqe::unique_keys_policy::left,
                                               false);
        bool const is_unique =
          enabled && (type == cudf::type_id::INT64 || type == cudf::type_id::FLOAT64 ||
                      type == cudf::type_id::BOOL8 || type == cudf::type_id::DECIMAL64);
        verify_unique_key_join_lowering(join, provider, enabled, is_unique);
      }
    }
  }
}

TEST(PhysicalQepTransformationTest, LeftJoinDoesNotSelectUniqueKeyTasks)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto left           = make_read("t", 1, nullptr);
  auto right          = make_read("t", 1, nullptr);
  physical::broadcast_join_relation join(left,
                                         right,
                                         {},
                                         gqe::join_type_type::left,
                                         equal_condition(0, 1),
                                         {0, 1},
                                         physical::broadcast_policy::right,
                                         gqe::unique_keys_policy::right,
                                         false);
  verify_unique_key_join_lowering(join, provider, /*enabled=*/true, /*is_unique=*/false);
}

TEST(PhysicalQepTransformationTest, MixedSemiAntiUsesDedicatedFamilyAndRejectsOtherShapes)
{
  auto const provider              = make_provider_with_parquet_table("t");
  auto const left                  = make_read("t", 3, nullptr);
  auto const right                 = make_read("t", 3, nullptr);
  auto parameters                  = gqe::make_optimization_parameters(/*only_defaults=*/true);
  parameters.join_use_perfect_hash = true;
  parameters.join_use_unique_keys  = true;
  enum class join_shape {
    mixed_broadcast_right,
    equality_only,
    residual_only,
    mixed_broadcast_left
  };
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    for (auto shape : {join_shape::mixed_broadcast_right,
                       join_shape::equality_only,
                       join_shape::residual_only,
                       join_shape::mixed_broadcast_left}) {
      std::unique_ptr<gqe::expression> condition;
      if (shape == join_shape::equality_only)
        condition = equal_condition(1, 5);
      else if (shape == join_shape::residual_only)
        condition = less_condition(2, 4);
      else
        condition = std::make_unique<gqe::logical_and_expression>(
          std::shared_ptr<gqe::expression>{equal_condition(1, 5)},
          std::shared_ptr<gqe::expression>{less_condition(2, 4)});
      physical::broadcast_join_relation join{left,
                                             right,
                                             {},
                                             mode,
                                             std::move(condition),
                                             {2, 0, 2},
                                             shape == join_shape::mixed_broadcast_left
                                               ? physical::broadcast_policy::left
                                               : physical::broadcast_policy::right,
                                             gqe::unique_keys_policy::right,
                                             true};
      // Equality-only and left-build joins are unsupported by these task families.
      if (shape == join_shape::equality_only || shape == join_shape::mixed_broadcast_left) {
        EXPECT_THROW(std::ignore = make_physical_qep_transformation(&join, &provider, parameters),
                     std::logic_error);
        continue;
      }
      auto lowered = make_physical_qep_transformation(&join, &provider, parameters);
      if (shape == join_shape::residual_only) {
        EXPECT_EQ(tasks_of_type<qep::conditional_semi_anti_join_build_task>(lowered.plan).size(),
                  1);
        EXPECT_EQ(tasks_of_type<qep::conditional_semi_anti_join_probe_task>(lowered.plan).size(),
                  1);
        EXPECT_TRUE(tasks_of_type<qep::mixed_semi_anti_join_probe_task>(lowered.plan).empty());
        continue;
      }
      EXPECT_EQ(tasks_of_type<qep::mixed_semi_anti_join_build_task>(lowered.plan).size(), 1);
      EXPECT_EQ(tasks_of_type<qep::mixed_semi_anti_join_probe_task>(lowered.plan).size(), 1);
      EXPECT_TRUE(tasks_of_type<qep::hash_join_probe_task>(lowered.plan).empty());
      EXPECT_TRUE(tasks_of_type<qep::unique_key_inner_join_probe_task>(lowered.plan).empty());
      EXPECT_TRUE(tasks_of_type<qep::perfect_hash_join_probe_task>(lowered.plan).empty());
    }
  }
}

TEST(PhysicalQepTransformationTest, MixedSemiAntiRejectsMixedEqualityNullPolicies)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 3, nullptr);
  auto const right    = make_read("t", 3, nullptr);
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    auto keys = std::make_shared<gqe::logical_and_expression>(
      std::shared_ptr<gqe::expression>{equal_condition(0, 3)},
      std::shared_ptr<gqe::expression>{nulls_equal_condition(1, 4)});
    auto condition = std::make_unique<gqe::logical_and_expression>(
      std::move(keys), std::shared_ptr<gqe::expression>{less_condition(2, 5)});
    physical::broadcast_join_relation join{
      left, right, {}, mode, std::move(condition), {0}, physical::broadcast_policy::right};
    EXPECT_THROW(std::ignore = make_physical_qep_transformation(
                   &join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true)),
                 std::logic_error);
  }
}

TEST(PhysicalQepTransformationTest, ConditionalRejectsUnsupportedModesAndBroadcastLeft)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 3, nullptr);
  auto const right    = make_read("t", 3, nullptr);
  for (auto mode : {gqe::join_type_type::left,
                    gqe::join_type_type::left_semi,
                    gqe::join_type_type::left_anti}) {
    physical::broadcast_join_relation join{left,
                                           right,
                                           {},
                                           mode,
                                           less_condition(0, 3),
                                           {0},
                                           mode == gqe::join_type_type::left
                                             ? physical::broadcast_policy::right
                                             : physical::broadcast_policy::left};
    EXPECT_THROW(std::ignore = make_physical_qep_transformation(
                   &join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true)),
                 std::logic_error);
  }
}

TEST(PhysicalQepTransformationTest, ConditionalSemiAntiPredicatesMustReferenceBothInputs)
{
  auto const provider = make_provider_with_parquet_table("t");
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    for (int side : {-1, 0, 1}) {
      std::shared_ptr<gqe::expression> operand =
        side < 0
          ? std::shared_ptr<gqe::expression>{std::make_shared<gqe::literal_expression<int64_t>>(1)}
          : std::shared_ptr<gqe::expression>{col_ref(side * 2)};
      auto condition = std::make_unique<gqe::less_expression>(
        operand, std::make_shared<gqe::literal_expression<int64_t>>(2));
      physical::broadcast_join_relation join{make_read("t", 2, nullptr),
                                             make_read("t", 2, nullptr),
                                             {},
                                             mode,
                                             std::move(condition),
                                             {0},
                                             physical::broadcast_policy::right};
      EXPECT_THROW(std::ignore = make_physical_qep_transformation(
                     &join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true)),
                   std::logic_error);
    }
  }
}

TEST(PhysicalQepTransformationTest, ConditionalInnerUsesDedicatedTasksForEitherBuildSide)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 3, nullptr);
  auto const right    = make_read("t", 3, nullptr);
  for (auto policy : {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
    physical::broadcast_join_relation join{left,
                                           right,
                                           {},
                                           gqe::join_type_type::inner,
                                           less_condition(2, 4),
                                           {4, 0, 4, 2},
                                           policy,
                                           gqe::unique_keys_policy::none,
                                           false};
    auto lowered = make_physical_qep_transformation(
      &join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
    auto const* build = single_task_of_type<qep::conditional_inner_join_build_task>(lowered.plan);
    auto const* probe = single_task_of_type<qep::conditional_inner_join_probe_task>(lowered.plan);
    ASSERT_NE(build, nullptr);
    ASSERT_NE(probe, nullptr);
    EXPECT_EQ(lowered.plan.predecessors(build).size(), 1);
    EXPECT_EQ(lowered.plan.predecessors(probe).size(), 2);
    EXPECT_EQ(lowered.output_data_types.at(probe), join.output_data_types());
    EXPECT_TRUE(tasks_of_type<qep::gather_task>(lowered.plan).empty());
    EXPECT_TRUE(tasks_of_type<qep::hash_join_probe_task>(lowered.plan).empty());
    EXPECT_TRUE(tasks_of_type<qep::unique_key_inner_join_probe_task>(lowered.plan).empty());
    EXPECT_TRUE(tasks_of_type<qep::conditional_semi_anti_join_probe_task>(lowered.plan).empty());
  }
}

TEST(PhysicalQepTransformationTest, EquiAndMixedInnerJoinsDoNotSelectConditionalTasks)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 3, nullptr);
  auto const right    = make_read("t", 3, nullptr);
  for (bool is_mixed : {false, true}) {
    std::unique_ptr<gqe::expression> condition = equal_condition(1, 5);
    if (is_mixed) {
      condition = std::make_unique<gqe::logical_and_expression>(
        std::shared_ptr<gqe::expression>{std::move(condition)},
        std::shared_ptr<gqe::expression>{less_condition(2, 4)});
    }
    physical::broadcast_join_relation join{left,
                                           right,
                                           {},
                                           gqe::join_type_type::inner,
                                           std::move(condition),
                                           {4, 0, 4},
                                           physical::broadcast_policy::right,
                                           gqe::unique_keys_policy::none,
                                           false};
    auto lowered = make_physical_qep_transformation(
      &join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true));
    EXPECT_TRUE(tasks_of_type<qep::conditional_inner_join_build_task>(lowered.plan).empty());
    EXPECT_TRUE(tasks_of_type<qep::conditional_inner_join_probe_task>(lowered.plan).empty());
    EXPECT_EQ(tasks_of_type<qep::hash_join_probe_task>(lowered.plan).size(), 1);
    EXPECT_EQ(tasks_of_type<qep::filter_join_indices_task>(lowered.plan).size(), is_mixed ? 1 : 0);
  }
}

TEST(PhysicalQepTransformationTest, ConditionalInnerRejectsUnsupportedPredicateOperands)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto const left     = make_read("t", 1, nullptr);
  auto const right    = make_read("t", 1, nullptr);
  for (auto policy : {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
    auto condition = std::make_unique<gqe::less_expression>(
      std::make_shared<gqe::cast_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                             cudf::data_type{cudf::type_id::STRING}),
      std::make_shared<gqe::column_reference_expression>(1));
    physical::broadcast_join_relation join{left,
                                           right,
                                           {},
                                           gqe::join_type_type::inner,
                                           std::move(condition),
                                           {0},
                                           policy,
                                           gqe::unique_keys_policy::none,
                                           false};
    EXPECT_THROW(std::ignore = make_physical_qep_transformation(
                   &join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true)),
                 std::logic_error);
  }
}

TEST(PhysicalQepTransformationTest, ConditionalInnerRejectsConsumedNestedColumnsOnEitherSide)
{
  auto const provider = make_provider_with_parquet_table("t");
  for (auto policy : {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
    for (bool is_nested_left : {false, true}) {
      auto const left = make_read(
        "t", 1, nullptr, {}, is_nested_left ? cudf::data_type{cudf::type_id::LIST} : int64_type());
      auto const right = make_read(
        "t", 1, nullptr, {}, is_nested_left ? int64_type() : cudf::data_type{cudf::type_id::LIST});
      physical::broadcast_join_relation join{left,
                                             right,
                                             {},
                                             gqe::join_type_type::inner,
                                             less_condition(0, 1),
                                             {},
                                             policy,
                                             gqe::unique_keys_policy::none,
                                             false};
      EXPECT_THROW(std::ignore = make_physical_qep_transformation(
                     &join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true)),
                   std::logic_error);
    }
  }
}

TEST(PhysicalQepTransformationTest, ConditionalInnerPredicatesMustReferenceBothInputs)
{
  auto const provider = make_provider_with_parquet_table("t");
  for (auto policy : {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
    for (int side : {-1, 0, 1}) {
      std::shared_ptr<gqe::expression> operand =
        side < 0
          ? std::shared_ptr<gqe::expression>{std::make_shared<gqe::literal_expression<int64_t>>(1)}
          : std::shared_ptr<gqe::expression>{col_ref(side * 2)};
      auto condition = std::make_unique<gqe::less_expression>(
        operand, std::make_shared<gqe::literal_expression<int64_t>>(2));
      physical::broadcast_join_relation join{make_read("t", 2, nullptr),
                                             make_read("t", 2, nullptr),
                                             {},
                                             gqe::join_type_type::inner,
                                             std::move(condition),
                                             {0},
                                             policy};
      EXPECT_THROW(std::ignore = make_physical_qep_transformation(
                     &join, &provider, gqe::make_optimization_parameters(/*only_defaults=*/true)),
                   std::logic_error);
    }
  }
}

TEST(PhysicalQepTransformationTest, JoinMaterializationFollowsResidualFilteringAndCanBeDisabled)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto predicate =
    std::make_unique<gqe::logical_and_expression>(equal_condition(0, 2), less_condition(1, 3));
  physical::broadcast_join_relation join{make_read("t", 2, nullptr),
                                         make_read("t", 2, nullptr),
                                         {},
                                         gqe::join_type_type::inner,
                                         std::move(predicate),
                                         {0, 3},
                                         physical::broadcast_policy::right};
  auto parameters = gqe::make_optimization_parameters(true);
  for (double threshold : {0.0, 0.1}) {
    parameters.join_materialization_threshold = threshold;
    auto lowered          = make_physical_qep_transformation(&join, &provider, parameters);
    auto materializations = tasks_of_type<qep::join_materialization_task>(lowered.plan);
    if (threshold == 0.0) {
      EXPECT_TRUE(materializations.empty());
    } else {
      ASSERT_EQ(materializations.size(), 1);
      auto const* residual = single_task_of_type<qep::filter_join_indices_task>(lowered.plan);
      EXPECT_EQ(lowered.plan.predecessors(materializations.front()),
                std::vector<qep::task const*>{residual});
      EXPECT_TRUE(lowered.plan.successors(materializations.front()).empty());
      EXPECT_EQ(lowered.stable_keys.at(materializations.front()).role,
                qep::lowering_role::join_materialize);
    }
  }
}

TEST(PhysicalQepTransformationTest, SharedJoinConsumersUseOneJoinMaterialization)
{
  auto const provider = make_provider_with_parquet_table("t");
  auto shared         = std::make_shared<physical::broadcast_join_relation>(
    make_read("t", 2, nullptr),
    make_read("t", 2, nullptr),
    std::vector<std::shared_ptr<physical::relation>>{},
    gqe::join_type_type::inner,
    equal_condition(0, 2),
    std::vector<cudf::size_type>{0, 1},
    physical::broadcast_policy::right);
  physical::broadcast_join_relation consumer{shared,
                                             shared,
                                             {},
                                             gqe::join_type_type::inner,
                                             equal_condition(0, 2),
                                             {0, 3},
                                             physical::broadcast_policy::right};
  auto lowered          = make_physical_qep_transformation(&consumer, &provider);
  auto materializations = tasks_of_type<qep::join_materialization_task>(lowered.plan);
  ASSERT_EQ(materializations.size(), 2);
  auto const it = std::ranges::find_if(materializations, [&](auto const* materialize) {
    return lowered.physical_relations.at(materialize) == shared.get();
  });
  ASSERT_NE(it, materializations.end());
  EXPECT_EQ(lowered.plan.successors(*it).size(), 2);
  auto const predecessors = lowered.plan.predecessors(*it);
  ASSERT_EQ(predecessors.size(), 1);
  EXPECT_EQ(lowered.plan.successors(predecessors.front()), std::vector<qep::task const*>{*it});
}

TEST(PhysicalQepTransformationTest, EarlyMaterializationSkipsJoinMaterialization)
{
  auto const provider = make_provider_with_parquet_table("t");
  physical::broadcast_join_relation join{make_read("t", 2, nullptr),
                                         make_read("t", 2, nullptr),
                                         {},
                                         gqe::join_type_type::inner,
                                         equal_condition(0, 2),
                                         {0, 3},
                                         physical::broadcast_policy::right};
  auto const parameters = early_materialization_parameters();
  ASSERT_GT(parameters.join_materialization_threshold, 0.0);
  auto lowered = make_physical_qep_transformation(&join, &provider, parameters);
  EXPECT_TRUE(tasks_of_type<qep::join_materialization_task>(lowered.plan).empty());
}

TEST(PhysicalQepTransformationTest, JoinMaterializationReceivesConfiguredThreshold)
{
  auto const provider = make_provider_with_parquet_table("t");
  physical::broadcast_join_relation join{make_read("t", 2, nullptr),
                                         make_read("t", 2, nullptr),
                                         {},
                                         gqe::join_type_type::inner,
                                         equal_condition(0, 2),
                                         {0, 3},
                                         physical::broadcast_policy::right};
  auto parameters                           = gqe::make_optimization_parameters(true);
  parameters.join_materialization_threshold = 0.25;
  auto lowered            = make_physical_qep_transformation(&join, &provider, parameters);
  auto const* materialize = dynamic_cast<qep::join_materialization_task const*>(
    single_task_of_type<qep::join_materialization_task>(lowered.plan));
  ASSERT_NE(materialize, nullptr);
  EXPECT_EQ(materialize->threshold(), 0.25);
}

TEST(PhysicalQepTransformationTest, CountOnlyJoinSkipsJoinMaterialization)
{
  auto const provider = make_provider_with_parquet_table("t");
  physical::broadcast_join_relation join{make_read("t", 2, nullptr),
                                         make_read("t", 2, nullptr),
                                         {},
                                         gqe::join_type_type::inner,
                                         equal_condition(0, 2),
                                         {},
                                         physical::broadcast_policy::right};
  auto lowered = make_physical_qep_transformation(&join, &provider);
  EXPECT_TRUE(join.output_data_types().empty());
  EXPECT_TRUE(tasks_of_type<qep::join_materialization_task>(lowered.plan).empty());
}
