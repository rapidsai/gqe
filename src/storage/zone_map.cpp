/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/storage/zone_map.hpp>

#include <gqe/executor/eval.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/json_formatter.hpp>
#include <gqe/expression/literal.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <functional>
#include <numeric>

std::unique_ptr<gqe::expression> gqe::zone_map_expression_transformer::transform(
  const gqe::expression& input_expression)
{
  zone_map_expression_transformer transformer;
  input_expression.accept(transformer);
  auto result = transformer._transformation_map[&input_expression];
  return result ? result->clone() : nullptr;
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
  _transformation_map.emplace(expression, nullptr);
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
  // TODO Is there a way to reduce the code duplication using templates?
  switch (op) {
    case cudf::binary_operator::LESS:
      _transformation_map.emplace(
        expression,
        std::make_shared<less_expression>(_transformation_map[lhs], _transformation_map[rhs]));
      break;
    case cudf::binary_operator::LESS_EQUAL:
      _transformation_map.emplace(expression,
                                  std::make_shared<less_equal_expression>(
                                    _transformation_map[lhs], _transformation_map[rhs]));
      break;
    case cudf::binary_operator::GREATER_EQUAL:
      _transformation_map.emplace(expression,
                                  std::make_shared<greater_equal_expression>(
                                    _transformation_map[lhs], _transformation_map[rhs]));
      break;
    case cudf::binary_operator::GREATER:
      _transformation_map.emplace(
        expression,
        std::make_shared<greater_expression>(_transformation_map[lhs], _transformation_map[rhs]));
      break;
    // Disable the warning about unhandled cases in the switch statement.
    // We should never reach the default because the method should only be called for <, >, <=, >=.
    default: throw std::logic_error("Invalid comparison operator. Should never be here.");
  }
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
  auto transformed_lhs = _transformation_map[lhs];
  auto transformed_rhs = _transformation_map[rhs];
  if (transformed_lhs && transformed_rhs) {
    _transformation_map.emplace(
      expression, std::make_shared<logical_and_expression>(transformed_lhs, transformed_rhs));
  } else if (transformed_lhs) {
    _transformation_map.emplace(expression, transformed_lhs);
  } else if (transformed_rhs) {
    _transformation_map.emplace(expression, transformed_rhs);
  } else {
    _transformation_map.emplace(expression, nullptr);
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
  auto transformed_lhs = _transformation_map[lhs];
  auto transformed_rhs = _transformation_map[rhs];
  if (transformed_lhs && transformed_rhs) {
    _transformation_map.emplace(
      expression, std::make_shared<logical_or_expression>(transformed_lhs, transformed_rhs));
  } else {
    _transformation_map.emplace(expression, nullptr);
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
                              std::make_shared<column_reference_expression>(min_column_idx));
}

void gqe::zone_map_expression_transformer::visit(const literal_expression<int8_t>* expression)
{
  _transformation_map.emplace(expression, expression->clone());
}

void gqe::zone_map_expression_transformer::visit(const literal_expression<int32_t>* expression)
{
  _transformation_map.emplace(expression, expression->clone());
}

void gqe::zone_map_expression_transformer::visit(const literal_expression<double>* expression)
{
  _transformation_map.emplace(expression, expression->clone());
}

void gqe::zone_map_expression_transformer::visit(
  const literal_expression<cudf::timestamp_D>* expression)
{
  _transformation_map.emplace(expression, expression->clone());
}

void gqe::zone_map_expression_transformer::visit(const literal_expression<std::string>* expression)
{
  _transformation_map.emplace(expression, expression->clone());
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
      _transformation_map.emplace(expression, nullptr);
      break;
    case gqe::scalar_function_expression::function_kind::substr: {
      const auto* substr_expression = dynamic_cast<const gqe::substr_expression*>(expression);
      const auto* child             = substr_expression->children()[0];
      child->accept(*this);
      _transformation_map.emplace(
        substr_expression,
        std::make_shared<gqe::substr_expression>(
          _transformation_map[child], substr_expression->start(), substr_expression->length()));
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
      if (const auto transformed_child = _transformation_map[child]) {
        _transformation_map.emplace(expression,
                                    std::make_shared<gqe::not_expression>(transformed_child));
      } else {
        _transformation_map.emplace(expression, nullptr);
      }
    } break;
    default: expression_visitor::visit(expression);
  }
}

bool gqe::zone_map::partition::operator==(const partition& other) const
{
  return pruned == other.pruned && start == other.start && end == other.end &&
         null_counts == other.null_counts;
}

gqe::zone_map::zone_map(const cudf::table_view& table, cudf::size_type partition_size)
  : _partition_size(partition_size), _num_rows(table.num_rows())
{
  _memory_resource = std::make_unique<rmm::mr::cuda_memory_resource>();
  _zone_map        = compute_zone_map(table, partition_size, _memory_resource.get());
  _null_counts     = compute_null_counts(table, partition_size);
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

std::unique_ptr<cudf::table> gqe::zone_map::compute_zone_map(const cudf::table_view& table,
                                                             cudf::size_type partition_size,
                                                             rmm::mr::device_memory_resource* mr)
{
  gqe::utility::nvtx_scoped_range range{"compute_zone_map"};

  // Determine number of row groups and segment boundaries
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

  // Compute min/max ranges for the columns of the input table
  std::vector<std::unique_ptr<cudf::column>> zone_map_columns;
  auto min_aggregation = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
  auto max_aggregation = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
  for (const auto& column : table) {
    auto min_values = cudf::segmented_reduce(column,
                                             device_offsets,
                                             *min_aggregation,
                                             column.type(),
                                             cudf::null_policy::EXCLUDE,
                                             cudf::get_default_stream(),
                                             mr);
    auto max_values = cudf::segmented_reduce(column,
                                             device_offsets,
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
      const auto column_null_counts = cudf::detail::segmented_count_unset_bits(
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
  const gqe::expression& partial_filter) const
{
  std::vector<const gqe::expression*> expressions{&partial_filter};
  auto [mask, _]  = evaluate_expressions(_zone_map->view(), expressions);
  auto partitions = cudf::detail::make_host_vector_sync(
    cudf::device_span<bool const>(mask[0].data<bool>(), mask[0].size()),
    cudf::get_default_stream());
  std::vector<partition> result{};
  size_t count = 0;
  for (cudf::size_type i = 0; i < static_cast<decltype(i)>(partitions.size()); ++i) {
    cudf::size_type start = i * _partition_size;
    cudf::size_type end   = std::min(start + _partition_size, _num_rows);
    std::vector<cudf::size_type> partition_null_counts(_null_counts.size());
    std::transform(_null_counts.begin(),
                   _null_counts.end(),
                   partition_null_counts.begin(),
                   [&](const auto& column_null_counts) { return column_null_counts[i]; });
    result.emplace_back(partition{
      .pruned = !partitions[i], .start = start, .end = end, .null_counts = partition_null_counts});
    if (partitions[i]) { count += 1; }
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
  auto destination = cudf::io::sink_info{std::string{filename}};
  auto options     = cudf::io::chunked_parquet_writer_options::builder(destination);
  auto writer      = cudf::io::parquet_chunked_writer{options};
  writer.write(_zone_map->view());
  writer.close();
}
#endif