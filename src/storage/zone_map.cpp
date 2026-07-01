/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/storage/zone_map.hpp>

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/is_null.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/scalar_function.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/utility/boost.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/cudf_to_arrow.hpp>
#include <gqe/utility/multi_process_helpers.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/interop.hpp>
#include <cudf/reduction.hpp>
#include <parquet/arrow/writer.h>

#include <arrow/array.h>
#include <arrow/c/bridge.h>
#include <arrow/compute/api.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/record_batch.h>
#include <arrow/scalar.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <functional>
#include <numeric>
#include <optional>
#include <string>

std::optional<arrow::compute::Expression> gqe::zone_map_expression_transformer::transform(
  const gqe::expression& input_expression)
{
  zone_map_expression_transformer transformer;
  input_expression.accept(transformer);
  return transformer._transformation_map[&input_expression];
}

void gqe::zone_map_expression_transformer::visitEqualExpression(
  const binary_op_expression* expression, const gqe::expression* lhs, const gqe::expression* rhs)
{
  auto less_and_greater_equal_expression = std::make_shared<logical_and_expression>(
    std::make_shared<less_equal_expression>(lhs->clone(), rhs->clone()),
    std::make_shared<greater_equal_expression>(lhs->clone(), rhs->clone()));
  // Store the expanded expression, so it is valid until this transformer is destroyed.
  _expanded_expressions.push_back(less_and_greater_equal_expression);
  visit(less_and_greater_equal_expression.get());
  _transformation_map.emplace(expression,
                              _transformation_map[less_and_greater_equal_expression.get()]);
}

// column <> value could be transformed into MIN <> MAX OR MIN <> value. However, this has two
// problems:
// (1) The <> operations are nested, so we cannot just visit the rewritten expression. This could be
//     fixed by reformulating as: MIN < MAX OR MIN > MAX OR MIN < value OR MIN > value
// (2) We have to explicitly identify which side is the column reference and which is the value. (In
//     contrast, the other code for binary operators is oblivious to this distinction.)
// Since <> would not prune a lot of data anyway, this seems too complicated. Instead, we just
// ignore this operator.
void gqe::zone_map_expression_transformer::visitNotEqualExpression(
  const binary_op_expression* expression)
{
  GQE_LOG_DEBUG("Ignoring not-equal expression during zone map filter transformation");
  _transformation_map.emplace(expression, std::nullopt);
}

void gqe::zone_map_expression_transformer::visitComparisonExpression(
  const binary_op_expression* expression,
  const cudf::binary_operator op,
  const gqe::expression* lhs,
  const gqe::expression* rhs)
{
  transform_comparisons(op, [&] {
    _side = side::lhs;
    lhs->accept(*this);
    _side = side::rhs;
    rhs->accept(*this);
  });
  auto& transformed_lhs = _transformation_map[lhs];
  auto& transformed_rhs = _transformation_map[rhs];
  if (!transformed_lhs || !transformed_rhs) {
    GQE_LOG_DEBUG(
      "Ignoring comparison expression during zone map filter transformation as it has an ignored "
      "child");
    _transformation_map.emplace(expression, std::nullopt);
    return;
  }
  const char* func = nullptr;
  switch (op) {
    case cudf::binary_operator::LESS: func = "less"; break;
    case cudf::binary_operator::LESS_EQUAL: func = "less_equal"; break;
    case cudf::binary_operator::GREATER_EQUAL: func = "greater_equal"; break;
    case cudf::binary_operator::GREATER: func = "greater"; break;
    default: throw std::logic_error("Invalid comparison operator. Should never be here.");
  }
  _transformation_map.emplace(expression,
                              arrow::compute::call(func, {*transformed_lhs, *transformed_rhs}));
}

// If any of the children of an AND expression are not supported, the other child can still be used
// to filter out partitions. The unsupported child would further restrict the qualifying partitions.
// If it is ignored, the transformed filter produces more false negatives (qualifying partitions),
// but no false positives (incorrectly pruned partitions).
void gqe::zone_map_expression_transformer::visitAndExpression(
  const binary_op_expression* expression, const gqe::expression* lhs, const gqe::expression* rhs)
{
  lhs->accept(*this);
  rhs->accept(*this);
  auto& transformed_lhs = _transformation_map[lhs];
  auto& transformed_rhs = _transformation_map[rhs];
  if (transformed_lhs && transformed_rhs) {
    _transformation_map.emplace(expression,
                                arrow::compute::call("and", {*transformed_lhs, *transformed_rhs}));
  } else if (transformed_lhs) {
    _transformation_map.emplace(expression, transformed_lhs);
  } else if (transformed_rhs) {
    _transformation_map.emplace(expression, transformed_rhs);
  } else {
    GQE_LOG_DEBUG(
      "Ignoring AND expression during zone map filter transformation because both children are "
      "ignored");
    _transformation_map.emplace(expression, std::nullopt);
  }
}

// If any of the children of an OR expression are not supported, the entire expression has to be
// ignored, because the unsupported child could qualify partitions that are pruned by the supported
// child.
void gqe::zone_map_expression_transformer::visitOrExpression(const binary_op_expression* expression,
                                                             const gqe::expression* lhs,
                                                             const gqe::expression* rhs)
{
  lhs->accept(*this);
  rhs->accept(*this);
  auto& transformed_lhs = _transformation_map[lhs];
  auto& transformed_rhs = _transformation_map[rhs];
  if (transformed_lhs && transformed_rhs) {
    _transformation_map.emplace(expression,
                                arrow::compute::call("or", {*transformed_lhs, *transformed_rhs}));
  } else {
    GQE_LOG_DEBUG(
      "Ignoring OR expression during zone map filter transformation because it has an ignored "
      "child");
    _transformation_map.emplace(expression, std::nullopt);
  }
}

void gqe::zone_map_expression_transformer::visit(const binary_op_expression* expression)
{
  auto op       = expression->binary_operator();
  auto children = expression->children();
  auto lhs      = children[0];
  auto rhs      = children[1];
  switch (op) {
    case cudf::binary_operator::EQUAL: visitEqualExpression(expression, lhs, rhs); break;
    case cudf::binary_operator::NOT_EQUAL: visitNotEqualExpression(expression); break;
    case cudf::binary_operator::LESS:
    case cudf::binary_operator::LESS_EQUAL:
    case cudf::binary_operator::GREATER:
    case cudf::binary_operator::GREATER_EQUAL:
      visitComparisonExpression(expression, op, lhs, rhs);
      break;
    case cudf::binary_operator::NULL_LOGICAL_AND: visitAndExpression(expression, lhs, rhs); break;
    case cudf::binary_operator::NULL_LOGICAL_OR: visitOrExpression(expression, lhs, rhs); break;
    default: expression_visitor::visit(expression);
  }
}

// This function is called only once, but it's useful to keep the code that manipulates
// _current_comparison in one location.
void gqe::zone_map_expression_transformer::transform_comparisons(
  cudf::binary_operator op, const std::function<void()>& function)
{
  // Make sure there are no nested comparisons. SQL allows the following expression, but we cannot
  // handle it: SELECT * FROM foo WHERE bar <= (1 < 2);
  if (_current_comparison) { throw std::logic_error("Current comparison operator already set"); }
  _current_comparison = op;
  function();
  _current_comparison.reset();
}

void gqe::zone_map_expression_transformer::negate_comparisons(const std::function<void()>& function)
{
  _inside_negation = !_inside_negation;
  function();
  _inside_negation = !_inside_negation;
}

// The following properties of a comparison predicate influence whether the MIN or MAX column is
// used.
// (1) The comparison operator:
//     (a) column < value → MIN < value
//     (b) column > value → MAX > value
// (2) Whether the column is on the LHS or RHS:
//     (a) column < value → MIN < value
//     (b) value < column → value < MAX   (because it is equivalent to column > value)
// (3) Whether the expression is negated:
//     (a) column < value → MIN < value
//     (b) NOT (column < value) → NOT (MAX < value)   (because it is equivalent to column >= value)
// The code below computes the column indexes for MIN and MAX and then swaps them for each of the
// (b) cases above. In the end, the correct column index is the MIN column index.
//
void gqe::zone_map_expression_transformer::visit(const column_reference_expression* expression)
{
  if (!_current_comparison) { throw std::logic_error("Comparison operator not set"); }
  auto input_idx      = expression->column_idx();
  auto min_column_idx = 2 * input_idx;
  auto max_column_idx = 2 * input_idx + 1;
  if (*_current_comparison == cudf::binary_operator::GREATER ||
      *_current_comparison == cudf::binary_operator::GREATER_EQUAL) {
    std::swap(min_column_idx, max_column_idx);
  }
  if (_side == side::rhs) { std::swap(min_column_idx, max_column_idx); }
  if (_inside_negation) { std::swap(min_column_idx, max_column_idx); }
  _transformation_map.emplace(expression,
                              arrow::compute::field_ref(std::to_string(min_column_idx)));
}

void gqe::zone_map_expression_transformer::visit(const literal_expression<int8_t>* expression)
{
  _transformation_map.emplace(expression,
                              arrow::compute::literal(arrow::MakeScalar(expression->value())));
}

void gqe::zone_map_expression_transformer::visit(const literal_expression<int32_t>* expression)
{
  _transformation_map.emplace(expression,
                              arrow::compute::literal(arrow::MakeScalar(expression->value())));
}

void gqe::zone_map_expression_transformer::visit(const literal_expression<double>* expression)
{
  _transformation_map.emplace(expression,
                              arrow::compute::literal(arrow::MakeScalar(expression->value())));
}

void gqe::zone_map_expression_transformer::visit(
  const literal_expression<cudf::timestamp_D>* expression)
{
  cudf::timestamp_D const val = expression->value();
  int32_t const days          = static_cast<int32_t>(val.time_since_epoch().count());
  auto scalar                 = arrow::MakeScalar(arrow::date32(), days).ValueOrDie();
  _transformation_map.emplace(expression, arrow::compute::literal(scalar));
}

void gqe::zone_map_expression_transformer::visit(const literal_expression<std::string>* expression)
{
  _transformation_map.emplace(expression,
                              arrow::compute::literal(arrow::MakeScalar(expression->value())));
}

void gqe::zone_map_expression_transformer::visit(const scalar_function_expression* expression)
{
  switch (expression->fn_kind()) {
    case gqe::scalar_function_expression::function_kind::like:
      // TODO https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/148.
      // TPC-H queries contain LIKE expressions. A LIKE expression with a suffix can be be used for
      // pruning, e.g., `LIKE 'abc%'` can be transformed into `MIN >= 'abc' AND MAX < 'abd'`. TPC-H
      // Q14 and Q20 contain such expressions that filter 83% and 99% of the part table,
      // respectively. However, the part table is not sorted on the filtered columns, so pruning is
      // not effective. Therefore, we ignore LIKE expressions for now.
      GQE_LOG_DEBUG("Ignoring like expression during zone map filter transformation");
      _transformation_map.emplace(expression, std::nullopt);
      break;
    case gqe::scalar_function_expression::function_kind::substr: {
      const auto* substr_expr = dynamic_cast<const gqe::substr_expression*>(expression);
      const auto* child       = substr_expr->children()[0];
      child->accept(*this);
      auto& child_transformed = _transformation_map[child];
      if (!child_transformed) {
        _transformation_map.emplace(expression, std::nullopt);
        break;
      }
      auto options = std::make_shared<arrow::compute::SliceOptions>(
        substr_expr->start(), substr_expr->start() + substr_expr->length());
      _transformation_map.emplace(
        expression,
        arrow::compute::call("utf8_slice_codeunits", {*child_transformed}, std::move(options)));
    } break;
    default: expression_visitor::visit(expression);
  }
}

void gqe::zone_map_expression_transformer::visit(const unary_op_expression* expression)
{
  const auto child = expression->children()[0];
  switch (expression->unary_operator()) {
    case cudf::unary_operator::NOT: {
      negate_comparisons([&] { child->accept(*this); });
      auto& child_arrow = _transformation_map[child];
      if (child_arrow) {
        _transformation_map.emplace(expression, arrow::compute::call("invert", {*child_arrow}));
      } else {
        GQE_LOG_DEBUG("Ignoring unary_op_expression during zone map filter transformation");
        _transformation_map.emplace(expression, std::nullopt);
      }
    } break;
    default: expression_visitor::visit(expression);
  }
}

void gqe::zone_map_expression_transformer::visit(const literal_expression<int64_t>* expression)
{
  GQE_LOG_DEBUG("Ignoring literal_expression<int64_t> during zone map filter transformation");
  _transformation_map.emplace(expression, std::nullopt);
}

void gqe::zone_map_expression_transformer::visit(const literal_expression<float>* expression)
{
  GQE_LOG_DEBUG("Ignoring literal_expression<float> during zone map filter transformation");
  _transformation_map.emplace(expression, std::nullopt);
}

void gqe::zone_map_expression_transformer::visit(const subquery_expression* expression)
{
  GQE_LOG_DEBUG("Ignoring subquery_expression during zone map filter transformation");
  _transformation_map.emplace(expression, std::nullopt);
}

void gqe::zone_map_expression_transformer::visit(const if_then_else_expression* expression)
{
  GQE_LOG_DEBUG("Ignoring if_then_else_expression during zone map filter transformation");
  _transformation_map.emplace(expression, std::nullopt);
}

void gqe::zone_map_expression_transformer::visit(const cast_expression* expression)
{
  GQE_LOG_DEBUG("Ignoring cast_expression during zone map filter transformation");
  _transformation_map.emplace(expression, std::nullopt);
}

void gqe::zone_map_expression_transformer::visit(const is_null_expression* expression)
{
  GQE_LOG_DEBUG("Ignoring is_null_expression during zone map filter transformation");
  _transformation_map.emplace(expression, std::nullopt);
}

bool gqe::zone_map::partition::operator==(const partition& other) const
{
  return pruned == other.pruned && start == other.start && end == other.end &&
         null_counts == other.null_counts;
}

void check_partition_size(cudf::size_type partition_size)
{
  if (partition_size == 0) {
    throw std::logic_error("Partition size for partition pruning must be larger than 0");
  }
  if (constexpr int size_warning_threshold = 100'000; partition_size < size_warning_threshold) {
    GQE_LOG_WARN(
      "Partition size for partition pruning is {} rows per row group. This is very small and will "
      "result in large zone maps. Consider increasing it to {}.",
      partition_size,
      size_warning_threshold);
  }
}

gqe::zone_map::zone_map(const cudf::table_view& table,
                        cudf::size_type partition_size,
                        rmm::device_async_resource_ref mr)
  : _partition_size(partition_size), _num_rows(table.num_rows())
{
  _zone_map    = compute_arrow_zone_map(table, partition_size, mr);
  _null_counts = compute_null_counts(table, partition_size);

  check_partition_size(partition_size);
}

/// Compute min/max zone map on GPU and return as a cudf::table.
std::unique_ptr<cudf::table> compute_cudf_zone_map(const cudf::table_view& table,
                                                   cudf::size_type partition_size,
                                                   rmm::device_async_resource_ref mr)
{
  gqe::utility::nvtx_scoped_range range{"compute_zone_map"};

  const auto num_rows       = table.num_rows();
  const auto num_partitions = gqe::utility::divide_round_up(num_rows, partition_size);
  std::vector<cudf::size_type> offsets(num_partitions);
  std::generate(offsets.begin(), offsets.end(), [&, n = 0]() mutable {
    auto offset = n;
    n += partition_size;
    return offset;
  });
  // Add end offset of the last row group. The documentation of cudf::segmented_reduce is a bit
  // unclear, but it implies that the last offset is exclusive. If there are 99 rows and 20 rows
  // per row group, then we need to store ceil(99/20) + 1 = 6 offsets: 0, 20, 40, 60, 80, 99.
  offsets.push_back(num_rows);
  thrust::device_vector<cudf::size_type> device_offsets{offsets.begin(), offsets.end()};
  cudf::device_span<cudf::size_type const> offsets_span{
    thrust::raw_pointer_cast(device_offsets.data()), device_offsets.size()};

  // Compute min/max ranges for the columns of the input table
  std::vector<std::unique_ptr<cudf::column>> zone_map_columns;
  auto min_aggregation = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
  auto max_aggregation = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
  for (const auto& column : table) {
    auto min_values = cudf::segmented_reduce(column,
                                             offsets_span,
                                             *min_aggregation,
                                             column.type(),
                                             cudf::null_policy::EXCLUDE,
                                             cudf::get_default_stream(),
                                             mr);
    auto max_values = cudf::segmented_reduce(column,
                                             offsets_span,
                                             *max_aggregation,
                                             column.type(),
                                             cudf::null_policy::EXCLUDE,
                                             cudf::get_default_stream(),
                                             mr);
    zone_map_columns.push_back(std::move(min_values));
    zone_map_columns.push_back(std::move(max_values));
  }
  auto zone_map = std::make_unique<cudf::table>(std::move(zone_map_columns));

  GQE_LOG_DEBUG(
    "Created zone map: table.num_rows = {}, table.num_columns() = {}, partition_size = {}, "
    "num_partitions = {}, zone_map->num_rows() = {}, zone_map->num_columns() = {}",
    table.num_rows(),
    table.num_columns(),
    partition_size,
    num_partitions,
    zone_map->num_rows(),
    zone_map->num_columns());

  return zone_map;
}

std::shared_ptr<arrow::RecordBatch> gqe::zone_map::compute_arrow_zone_map(
  const cudf::table_view& table, cudf::size_type partition_size, rmm::device_async_resource_ref mr)
{
  auto cudf_zone_map      = compute_cudf_zone_map(table, partition_size, mr);
  auto cudf_zone_map_view = cudf_zone_map->view();
  return gqe::utility::cudf_table_to_arrow(
    std::span<cudf::table_view const>{&cudf_zone_map_view, 1});
}

std::vector<std::vector<cudf::size_type>> gqe::zone_map::compute_null_counts(
  const cudf::table_view& table, cudf::size_type partition_size)
{
  // Determine start offset (inclusive) and end offset (exclusive) of partition boundaries
  const auto num_rows       = table.num_rows();
  const auto num_partitions = gqe::utility::divide_round_up(num_rows, partition_size);
  std::vector<cudf::size_type> offsets{};
  offsets.reserve(2 * num_partitions);
  for (size_t i = 0; i < static_cast<decltype(i)>(num_partitions); ++i) {
    const cudf::size_type start_offset = i * partition_size;
    const cudf::size_type end_offset   = std::min(start_offset + partition_size, num_rows);
    offsets.push_back(start_offset);
    offsets.push_back(end_offset);
  }
  thrust::device_vector<cudf::size_type> device_offsets{offsets.begin(), offsets.end()};

  // Compute segmented null counts for all columns
  std::vector<std::vector<cudf::size_type>> null_counts{};
  null_counts.reserve(static_cast<size_t>(table.num_columns()));
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    const auto& column = table.column(i);
    if (column.has_nulls()) {
      GQE_LOG_DEBUG("Column {} has null values; computing null counts for each zone map partition",
                    i);
      const auto column_null_counts = cudf::segmented_null_count(
        column.null_mask(),
        cudf::host_span<cudf::size_type>(offsets.data(), offsets.size(), false),
        cudf::get_default_stream());
      null_counts.push_back(column_null_counts);
    } else {
      GQE_LOG_DEBUG("Column {} does not have null values", i);
      const auto& column_null_counts = std::vector<cudf::size_type>(num_partitions, 0);
      null_counts.push_back(column_null_counts);
    }
  }
  return null_counts;
}

std::vector<gqe::zone_map::partition> gqe::zone_map::evaluate(
  const arrow::compute::Expression& partial_filter)
{
  utility::nvtx_scoped_range range{"zone_map_evaluate"};

  auto bound_expr = partial_filter.Bind(*_zone_map->schema()).ValueOrDie();
  arrow::compute::ExecBatch exec_batch(*_zone_map);
  auto mask_datum = arrow::compute::ExecuteScalarExpression(bound_expr, exec_batch).ValueOrDie();
  arrow::BooleanArray mask_array(mask_datum.array());

  std::vector<partition> result{};
  size_t count = 0;
  for (cudf::size_type i = 0; i < mask_array.length(); ++i) {
    cudf::size_type start = i * _partition_size;
    cudf::size_type end   = std::min(start + _partition_size, _num_rows);
    std::vector<cudf::size_type> partition_null_counts(_null_counts.size());
    std::transform(_null_counts.begin(),
                   _null_counts.end(),
                   partition_null_counts.begin(),
                   [&](const auto& column_null_counts) { return column_null_counts[i]; });
    bool partition_matches = mask_array.Value(i);
    result.emplace_back(partition{.pruned      = !partition_matches,
                                  .start       = start,
                                  .end         = end,
                                  .null_counts = partition_null_counts});
    if (partition_matches) { count += 1; }
  }
  GQE_LOG_DEBUG("Returning matching partitions: count = {}, _zone_map.num_rows() = {}, ratio = {}",
                count,
                _zone_map->num_rows(),
                static_cast<double>(count) / static_cast<double>(_zone_map->num_rows()));
  return result;
}

gqe::zone_map::partition gqe::zone_map::consolidate_partitions(
  const std::vector<partition>::const_iterator begin,
  const std::vector<partition>::const_iterator end)
{
#ifndef NDEBUG
  // Verify that there are no gaps between the partitions, otherwise the null counts will be off.
  for (auto current = begin; current != end - 1; ++current) {
    if (current->end != (current + 1)->start) {
      throw std::logic_error("Partitions must be consecutive");
    }
  }
#endif
  // The null count vector of the initial element has to have the correct size, otherwise null
  // counts are not correctly accumulated. Therefore, we initialize with the accumulation with
  // the first partition and accumulate the remaining partitions.
  auto covering_partition = std::accumulate(
    begin + 1, end, *begin, [](const zone_map::partition& a, const zone_map::partition& b) {
      // Create a pairwise sum of the null counts. This assumes that
      // a.null_counts.size() == number of columns.
      std::vector<cudf::size_type> null_counts_sums(a.null_counts.size());
      std::transform(a.null_counts.begin(),
                     a.null_counts.end(),
                     b.null_counts.begin(),
                     null_counts_sums.begin(),
                     [](auto a, auto b) { return a + b; });
      // The returned partition starts at the beginning of 'a' and ends at the
      // beginning of 'b'.
      return zone_map::partition{
        .pruned = a.pruned, .start = a.start, .end = b.end, .null_counts = null_counts_sums};
    });
  return covering_partition;
}

gqe::zone_map::partition gqe::zone_map::consolidate_maximally_covering_partition(
  const std::vector<partition>& partitions)
{
  // Find first and last unpruned partition
  auto unpruned_p  = [](const partition& partition) { return !partition.pruned; };
  const auto begin = std::find_if(partitions.begin(), partitions.end(), unpruned_p);
  const auto end   = std::find_if(partitions.rbegin(), partitions.rend(), unpruned_p);
  return consolidate_partitions(begin, end.base());
}

std::vector<gqe::zone_map::partition> gqe::zone_map::consolidate_partitions(
  const std::vector<partition>& partitions)
{
  std::vector<partition> result;
  if (partitions.empty()) { return result; }
  bool pruned = partitions.front().pruned;
  auto begin  = partitions.begin();
  while (begin != partitions.end()) {
    auto end = std::find_if(begin + 1, partitions.end(), [&pruned](const auto& partition) {
      return partition.pruned != pruned;
    });
    auto aggregated_partition = consolidate_partitions(begin, end);
    result.push_back(aggregated_partition);
    pruned = !pruned;
    begin  = end;
  }
  return result;
}

#ifndef NDEBUG
void gqe::zone_map::write_to_parquet_file(const std::string_view filename) const
{
  auto table = arrow::Table::Make(_zone_map->schema(), _zone_map->columns(), _zone_map->num_rows());
  auto outfile = arrow::io::FileOutputStream::Open(std::string{filename}).ValueOrDie();
  parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile).ok();
}
#endif

gqe::shared_arrow_batch::shared_arrow_batch(const arrow::RecordBatch& batch,
                                            boost::interprocess::managed_shared_memory& segment)
  : _segment(&segment)
{
  auto sink     = arrow::io::BufferOutputStream::Create().ValueOrDie();
  auto writer   = arrow::ipc::MakeStreamWriter(sink, batch.schema()).ValueOrDie();
  auto write_st = writer->WriteRecordBatch(batch);
  if (!write_st.ok()) { throw std::runtime_error(write_st.ToString()); }
  auto close_st = writer->Close();
  if (!close_st.ok()) { throw std::runtime_error(close_st.ToString()); }
  auto buffer = sink->Finish().ValueOrDie();

  _size = buffer->size();
  _data = static_cast<std::byte*>(segment.allocate(static_cast<size_t>(_size)));
  std::memcpy(_data.get(), buffer->data(), static_cast<size_t>(_size));
}

gqe::shared_arrow_batch::~shared_arrow_batch()
{
  if (_data) {
    _segment->deallocate(_data.get());
    _data = nullptr;
  }
}

std::shared_ptr<arrow::RecordBatch> gqe::shared_arrow_batch::to_record_batch() const
{
  auto buffer =
    std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(_data.get()), _size);
  auto reader       = std::make_shared<arrow::io::BufferReader>(buffer);
  auto batch_reader = arrow::ipc::RecordBatchStreamReader::Open(reader).ValueOrDie();
  return batch_reader->Next().ValueOrDie();
}

gqe::shared_zone_map::shared_zone_map(cudf::size_type partition_size,
                                      std::string table_name,
                                      boost::interprocess::managed_shared_memory* segment)
  : zone_map(), _table_name(table_name), _segment(segment)
{
  // Initialize base class state not handled by its default constructor
  _partition_size = partition_size;

  check_partition_size(partition_size);
}

void gqe::shared_zone_map::load_from_shared_memory()
{
  auto shared_zone_map_table =
    gqe::utility::find_object<gqe::shared_zone_map_table>(_segment, _table_name);

  _zone_map = shared_zone_map_table->_zone_map->to_record_batch();
  _num_rows = shared_zone_map_table->_num_rows;

  // Need to convert boost::container::vector<boost::container::vector<cudf::size_type>> to
  // std::vector<std::vector<cudf::size_type>>
  auto null_counts = shared_zone_map_table->_null_counts;
  _null_counts.reserve(null_counts.size());
  for (const auto& each_null_count : null_counts) {
    std::vector<cudf::size_type> each_null_count_std(each_null_count.begin(),
                                                     each_null_count.end());
    _null_counts.push_back(each_null_count_std);
  }

  GQE_LOG_DEBUG("Loaded shared zone map from table name {}", _table_name);
}

std::vector<gqe::zone_map::partition> gqe::shared_zone_map::evaluate(
  const arrow::compute::Expression& partial_filter)
{
  // TODO: this should be done before in-memory-read-task call to avoid any overhead
  load_from_shared_memory();
  return zone_map::evaluate(partial_filter);
}

gqe::shared_zone_map::~shared_zone_map()
{
  if (gqe::utility::multi_process::nvshmem_rank_zero()) {
    _segment->destroy<gqe::shared_zone_map_table>(_table_name.c_str());
  }
}

gqe::shared_zone_map_table::shared_zone_map_table(
  const cudf::table_view& table,
  cudf::size_type partition_size,
  boost::interprocess::managed_shared_memory* segment)
  : _num_rows(table.num_rows()),
    _segment(segment),
    _null_counts(
      shared_zone_map_table::SharedVectorSizeTypeAllocator(segment->get_segment_manager()))
{
  auto cudf_zone_map =
    compute_cudf_zone_map(table, partition_size, rmm::mr::get_current_device_resource_ref());
  auto cudf_zone_map_view = cudf_zone_map->view();

  auto arrow_batch =
    gqe::utility::cudf_table_to_arrow(std::span<cudf::table_view const>{&cudf_zone_map_view, 1});
  auto std_null_counts = gqe::zone_map::compute_null_counts(table, partition_size);

  _null_counts.reserve(std_null_counts.size());
  for (const auto& each_null_count : std_null_counts) {
    auto inner_alloc =
      shared_zone_map_table::SharedSizeTypeAllocator(segment->get_segment_manager());
    boost::container::vector<cudf::size_type, shared_zone_map_table::SharedSizeTypeAllocator>
      each_null_count_boost(each_null_count.begin(), each_null_count.end(), inner_alloc);
    _null_counts.push_back(std::move(each_null_count_boost));
  }

  _num_rows = table.num_rows();

  _zone_map = segment->construct<gqe::shared_arrow_batch>(boost::interprocess::anonymous_instance)(
    *arrow_batch, *segment);
}

gqe::shared_zone_map_table::~shared_zone_map_table()
{
  if (_zone_map) {
    _segment->destroy_ptr(_zone_map.get());
    _zone_map = nullptr;
  }
}
