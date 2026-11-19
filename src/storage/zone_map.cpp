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
#include <gqe/memory_resource/system_memory_resource.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/cudf_to_arrow.hpp>
#include <gqe/utility/serialization.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/interop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/unary.hpp>
#include <parquet/arrow/writer.h>
#include <rmm/device_uvector.hpp>

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
#include <arrow/util/decimal.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <string>

namespace {

/// File name of a serialized zone map within its `zone_maps/` directory.
constexpr char k_zone_map_filename[] = "zone_map.bin";

constexpr int32_t k_zone_map_decimal_precision = 38;

template <typename DecimalT>
arrow::compute::Expression make_zone_map_decimal_literal(
  gqe::literal_expression<DecimalT> const* expression)
{
  auto const literal_type = expression->data_type({});
  auto const value        = expression->value();
  auto const rep          = static_cast<__int128_t>(value.value());
  auto const high_bits    = static_cast<int64_t>(rep >> 64);
  auto const low_bits     = static_cast<uint64_t>(rep);
  auto const arrow_value  = arrow::Decimal128(high_bits, low_bits);
  // cuDF fixed-point uses value = rep * 10^scale, while Arrow decimal uses
  // the SQL-style decimal scale convention.
  auto const arrow_scale = -static_cast<int32_t>(literal_type.scale());
  auto const type        = arrow::decimal128(k_zone_map_decimal_precision, arrow_scale);
  auto const scalar      = arrow::MakeScalar(type, arrow_value).ValueOrDie();
  return arrow::compute::literal(scalar);
}

}  // namespace

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
  // Pre-downcast to the smallest int type that fits. Arrow's binder otherwise picks
  // a decimal target precision from the downcast type but casts the original int32,
  // tripping `IntegerToDecimal`'s precision check.
  auto const value = expression->value();
  std::shared_ptr<arrow::Scalar> scalar;
  if (value >= std::numeric_limits<int8_t>::min() && value <= std::numeric_limits<int8_t>::max()) {
    scalar = arrow::MakeScalar(static_cast<int8_t>(value));
  } else if (value >= std::numeric_limits<int16_t>::min() &&
             value <= std::numeric_limits<int16_t>::max()) {
    scalar = arrow::MakeScalar(static_cast<int16_t>(value));
  } else {
    scalar = arrow::MakeScalar(value);
  }
  _transformation_map.emplace(expression, arrow::compute::literal(scalar));
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

void gqe::zone_map_expression_transformer::visit(
  const literal_expression<numeric::decimal32>* expression)
{
  _transformation_map.emplace(expression, make_zone_map_decimal_literal(expression));
}

void gqe::zone_map_expression_transformer::visit(
  const literal_expression<numeric::decimal64>* expression)
{
  _transformation_map.emplace(expression, make_zone_map_decimal_literal(expression));
}

void gqe::zone_map_expression_transformer::visit(
  const literal_expression<numeric::decimal128>* expression)
{
  _transformation_map.emplace(expression, make_zone_map_decimal_literal(expression));
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

namespace {

void check_partition_size(cudf::size_type partition_size)
{
  if (partition_size <= 0) {
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

/// Compute min/max zone map on GPU and return as a cudf::table.
std::unique_ptr<cudf::table> compute_cudf_zone_map(const cudf::table_view& table,
                                                   cudf::size_type partition_size,
                                                   cuda::stream_ref stream,
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
  rmm::device_uvector<cudf::size_type> device_offsets(offsets.size(), stream, mr);
  GQE_CUDA_TRY(cudaMemcpyAsync(device_offsets.data(),
                               offsets.data(),
                               offsets.size() * sizeof(cudf::size_type),
                               cudaMemcpyHostToDevice,
                               stream.get()));
  cudf::device_span<cudf::size_type const> offsets_span{device_offsets.data(),
                                                        device_offsets.size()};

  // Widen DECIMAL{32,64} min/max to DECIMAL128 before the cuDF -> Arrow handoff:
  // Arrow's compute kernel registry has no `DECIMAL{32,64} -> DECIMAL128` cast.
  std::vector<std::unique_ptr<cudf::column>> zone_map_columns;
  auto min_aggregation = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
  auto max_aggregation = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
  for (const auto& column : table) {
    auto min_values = cudf::segmented_reduce(column,
                                             offsets_span,
                                             *min_aggregation,
                                             column.type(),
                                             cudf::null_policy::EXCLUDE,
                                             stream,
                                             mr);
    auto max_values = cudf::segmented_reduce(column,
                                             offsets_span,
                                             *max_aggregation,
                                             column.type(),
                                             cudf::null_policy::EXCLUDE,
                                             stream,
                                             mr);
    if (column.type().id() == cudf::type_id::DECIMAL32 ||
        column.type().id() == cudf::type_id::DECIMAL64) {
      const cudf::data_type widened{cudf::type_id::DECIMAL128, column.type().scale()};
      min_values = cudf::cast(min_values->view(), widened, stream, mr);
      max_values = cudf::cast(max_values->view(), widened, stream, mr);
    }
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

/**
 * @brief Compute per-partition null counts for every column of @p table.
 *
 * Returned partition-major (i.e. `result[partition][column]`, the layout zone_map_storage
 * expects).
 *
 * @param table Table to compute null counts for.
 * @param partition_size Number of rows per partition.
 * @param stream CUDA stream used for the segmented null-count computation.
 */
std::vector<std::vector<cudf::size_type>> compute_null_counts(const cudf::table_view& table,
                                                              cudf::size_type partition_size,
                                                              cuda::stream_ref stream)
{
  // Start offset (inclusive) and end offset (exclusive) of partition boundaries
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
  // Segmented null counts for all columns
  std::vector<std::vector<cudf::size_type>> null_counts{};
  null_counts.reserve(static_cast<size_t>(table.num_columns()));
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    const auto& column = table.column(i);
    if (column.has_nulls()) {
      GQE_LOG_DEBUG("Column {} has null values; computing null counts for each zone map partition",
                    i);
      auto column_null_counts = cudf::segmented_null_count(
        column.null_mask(),
        cudf::host_span<cudf::size_type>(offsets.data(), offsets.size(), false),
        stream);
      null_counts.push_back(std::move(column_null_counts));
    } else {
      GQE_LOG_DEBUG("Column {} does not have null values", i);
      null_counts.emplace_back(num_partitions, 0);
    }
  }

  // Transpose from column-major to partition-major
  std::vector<std::vector<cudf::size_type>> partition_null_counts;
  partition_null_counts.reserve(num_partitions);
  for (cudf::size_type i = 0; i < num_partitions; ++i) {
    std::vector<cudf::size_type> current_partition_null_counts(null_counts.size());
    std::transform(
      null_counts.begin(),
      null_counts.end(),
      current_partition_null_counts.begin(),
      [&](const auto& current_column_null_counts) { return current_column_null_counts[i]; });
    partition_null_counts.push_back(std::move(current_partition_null_counts));
  }
  return partition_null_counts;
}

gqe::memory_resource::system_memory_resource& local_zone_map_storage_resource()
{
  static gqe::memory_resource::system_memory_resource resource;
  return resource;
}

}  // namespace

template <class Allocator>
gqe::zone_map_storage<Allocator>::zone_map_storage(cudf::table_view const& table,
                                                   cudf::size_type partition_size,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref compute_mr,
                                                   rmm::device_async_resource_ref storage_mr,
                                                   allocator_type const& alloc)
  : _partition_size(partition_size),
    _num_rows(table.num_rows()),
    _null_counts(gqe::storage_rebind_allocator_t<Allocator, null_count_vector>(alloc))
{
  check_partition_size(partition_size);

  auto cudf_zone_map      = compute_cudf_zone_map(table, partition_size, stream, compute_mr);
  auto cudf_zone_map_view = cudf_zone_map->view();
  auto arrow_batch        = gqe::utility::cudf_table_to_arrow(
    std::span<cudf::table_view const>{&cudf_zone_map_view, 1}, {}, stream, compute_mr);

  auto sink     = arrow::io::BufferOutputStream::Create().ValueOrDie();
  auto writer   = arrow::ipc::MakeStreamWriter(sink, arrow_batch->schema()).ValueOrDie();
  auto write_st = writer->WriteRecordBatch(*arrow_batch);
  if (!write_st.ok()) { throw std::runtime_error(write_st.ToString()); }
  auto close_st = writer->Close();
  if (!close_st.ok()) { throw std::runtime_error(close_st.ToString()); }
  auto buffer = sink->Finish().ValueOrDie();

  _ipc_record_batch.allocate(static_cast<std::size_t>(buffer->size()), stream, storage_mr);
  if (buffer->size() > 0) {
    std::memcpy(_ipc_record_batch.data(), buffer->data(), static_cast<std::size_t>(buffer->size()));
  }

  auto std_null_counts = compute_null_counts(table, partition_size, stream);
  _null_counts.reserve(std_null_counts.size());
  for (auto const& each_null_count : std_null_counts) {
    null_count_vector each_null_count_storage(
      each_null_count.begin(),
      each_null_count.end(),
      gqe::storage_rebind_allocator_t<Allocator, cudf::size_type>(alloc));
    _null_counts.push_back(std::move(each_null_count_storage));
  }
}

template <class Allocator>
gqe::zone_map_storage<Allocator>::zone_map_storage(
  cudf::size_type partition_size,
  cudf::size_type num_rows,
  std::vector<std::vector<cudf::size_type>> const& null_counts,
  std::span<std::byte const> ipc_record_batch,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref storage_mr,
  allocator_type const& alloc)
  : _partition_size(partition_size),
    _num_rows(num_rows),
    _null_counts(gqe::storage_rebind_allocator_t<Allocator, null_count_vector>(alloc))
{
  _ipc_record_batch.allocate(ipc_record_batch.size(), stream, storage_mr);
  if (!ipc_record_batch.empty()) {
    std::memcpy(_ipc_record_batch.data(), ipc_record_batch.data(), ipc_record_batch.size());
  }

  _null_counts.reserve(null_counts.size());
  for (auto const& column_counts : null_counts) {
    null_count_vector column_counts_storage(
      column_counts.begin(),
      column_counts.end(),
      gqe::storage_rebind_allocator_t<Allocator, cudf::size_type>(alloc));
    _null_counts.push_back(std::move(column_counts_storage));
  }
}

template <class Allocator>
void gqe::zone_map_storage<Allocator>::deallocate(cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  _ipc_record_batch.deallocate(stream, mr);
  _null_counts.clear();
  _partition_size = 0;
  _num_rows       = 0;
}

template <class StorageHandle>
template <class... Args>
gqe::basic_zone_map<StorageHandle>::basic_zone_map(Args&&... args)
  : _storage_handle(std::forward<Args>(args)...)
{
  auto const& storage         = _storage_handle.get();
  auto const ipc_record_batch = storage.ipc_record_batch();
  auto buffer                 = std::make_shared<arrow::Buffer>(
    reinterpret_cast<uint8_t const*>(ipc_record_batch.data()), ipc_record_batch.size());
  auto reader       = std::make_shared<arrow::io::BufferReader>(buffer);
  auto batch_reader = arrow::ipc::RecordBatchStreamReader::Open(reader).ValueOrDie();
  _zone_map_cache   = batch_reader->Next().ValueOrDie();
}

template <class StorageHandle>
std::vector<gqe::zone_map::partition> gqe::basic_zone_map<StorageHandle>::evaluate(
  const arrow::compute::Expression& partial_filter) const
{
  utility::nvtx_scoped_range range{"zone_map_evaluate"};

  auto const& storage = _storage_handle.get();
  auto bound_expr     = partial_filter.Bind(*_zone_map_cache->schema()).ValueOrDie();
  arrow::compute::ExecBatch exec_batch(*_zone_map_cache);
  auto mask_datum = arrow::compute::ExecuteScalarExpression(bound_expr, exec_batch).ValueOrDie();
  arrow::BooleanArray mask_array(mask_datum.array());

  std::vector<partition> result{};
  size_t count            = 0;
  auto const& null_counts = storage.null_counts();
  for (cudf::size_type i = 0; i < mask_array.length(); ++i) {
    cudf::size_type start  = i * storage.partition_size();
    cudf::size_type end    = std::min(start + storage.partition_size(), storage.num_rows());
    bool partition_matches = mask_array.Value(i);
    result.emplace_back(partition{
      .pruned      = !partition_matches,
      .start       = start,
      .end         = end,
      .null_counts = std::vector<cudf::size_type>(null_counts[i].begin(), null_counts[i].end())});
    if (partition_matches) { count += 1; }
  }
  GQE_LOG_DEBUG("Returning matching partitions: count = {}, _zone_map.num_rows() = {}, ratio = {}",
                count,
                _zone_map_cache->num_rows(),
                static_cast<double>(count) / static_cast<double>(_zone_map_cache->num_rows()));
  return result;
}

template <class StorageHandle>
void gqe::basic_zone_map<StorageHandle>::serialize_to_disk(
  std::filesystem::path const& zone_map_dir) const
{
  // The Arrow min/max batch is already stored as serialized IPC bytes and the null counts live in
  // host memory, so no device work is required.
  namespace u         = gqe::utility;
  auto const& storage = _storage_handle.get();

  std::error_code ec;
  std::filesystem::create_directories(zone_map_dir, ec);
  if (ec) {
    throw std::runtime_error("zone_map::serialize_to_disk: cannot create directory '" +
                             zone_map_dir.string() + "': " + ec.message());
  }

  auto const file_path = zone_map_dir / k_zone_map_filename;
  std::ofstream file(file_path, std::ios::binary | std::ios::trunc);
  if (!file) {
    throw std::runtime_error("zone_map::serialize_to_disk: cannot open '" + file_path.string() +
                             "' for writing");
  }

  // The storage keeps null counts partition-major: null_counts[partition][column].
  auto const& null_counts   = storage.null_counts();
  auto const num_partitions = static_cast<std::uint32_t>(null_counts.size());
  auto const num_columns =
    null_counts.empty() ? 0u : static_cast<std::uint32_t>(null_counts.front().size());

  gqe::proto::ZoneMapMetadata metadata;
  metadata.set_partition_size(static_cast<std::int32_t>(storage.partition_size()));
  metadata.set_num_rows(static_cast<std::int64_t>(storage.num_rows()));
  metadata.set_num_columns(static_cast<std::int32_t>(num_columns));
  metadata.set_num_partitions(static_cast<std::int32_t>(num_partitions));
  metadata.mutable_null_counts()->Reserve(
    static_cast<int>(static_cast<std::size_t>(num_columns) * num_partitions));
  for (auto const& partition_counts : null_counts) {
    for (cudf::size_type const value : partition_counts) {
      metadata.add_null_counts(static_cast<std::int32_t>(value));
    }
  }
  u::write_serialized_zone_map_metadata(file, metadata);

  auto const ipc      = storage.ipc_record_batch();
  auto const ipc_size = static_cast<std::uint64_t>(ipc.size());
  u::write_pod(file, ipc_size);
  if (ipc_size > 0) {
    file.write(reinterpret_cast<char const*>(ipc.data()), static_cast<std::streamsize>(ipc_size));
  }

  // close() flushes buffered data. Check afterwards for deferred write errors such as disk full.
  file.close();
  if (file.fail()) {
    throw std::runtime_error("zone_map::serialize_to_disk: write/close error for '" +
                             file_path.string() + "'");
  }
}

template <class StorageHandle>
void gqe::basic_zone_map<StorageHandle>::deallocate(cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  _zone_map_cache.reset();
  _storage_handle.deallocate(stream, mr);
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
  auto begin = partitions.begin();
  while (begin != partitions.end()) {
    auto last = std::adjacent_find(
      begin, partitions.end(), [](const partition& left, const partition& right) {
        return left.pruned != right.pruned || left.end != right.start;
      });
    auto end = (last != partitions.end()) ? std::next(last) : last;
    result.push_back(consolidate_partitions(begin, end));
    begin = end;
  }
  return result;
}

#ifndef NDEBUG
template <class StorageHandle>
void gqe::basic_zone_map<StorageHandle>::write_to_parquet_file(
  const std::string_view filename) const
{
  auto table = arrow::Table::Make(
    _zone_map_cache->schema(), _zone_map_cache->columns(), _zone_map_cache->num_rows());
  auto outfile  = arrow::io::FileOutputStream::Open(std::string{filename}).ValueOrDie();
  auto write_st = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile);
  if (!write_st.ok()) { throw std::runtime_error(write_st.ToString()); }
}
#endif

gqe::local_zone_map::local_zone_map(cudf::table_view const& table,
                                    cudf::size_type partition_size,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref compute_mr)
  : base_type(table,
              partition_size,
              stream,
              compute_mr,
              local_zone_map_storage_resource(),
              storage_type::allocator_type{})
{
}

gqe::local_zone_map::local_zone_map(cudf::size_type partition_size,
                                    cudf::size_type num_rows,
                                    std::vector<std::vector<cudf::size_type>> const& null_counts,
                                    std::span<std::byte const> ipc_record_batch,
                                    cuda::stream_ref stream)
  : base_type(partition_size,
              num_rows,
              null_counts,
              ipc_record_batch,
              stream,
              local_zone_map_storage_resource(),
              storage_type::allocator_type{})
{
}

namespace {

// Reconstruct arguments read from a serialized zone-map file, independent of whether the resulting
// storage is built process-locally or in the shared memory segment. Returns nullopt when the
// file is absent (the row group was serialized without a zone map).
struct parsed_zone_map {
  cudf::size_type partition_size;
  cudf::size_type num_rows;
  std::vector<std::vector<cudf::size_type>> null_counts;
  std::vector<std::byte> ipc_bytes;
};

std::optional<parsed_zone_map> parse_serialized_zone_map(std::filesystem::path const& zone_map_dir)
{
  namespace u          = gqe::utility;
  auto const file_path = zone_map_dir / k_zone_map_filename;
  std::error_code ec;
  bool const zone_map_file_exists = std::filesystem::exists(file_path, ec);
  if (ec) {
    // `Exists` reports a real filesystem error via `ec` while also returning false, so surface it
    // rather than silently treating it as an absent zone map.
    throw std::runtime_error("zone_map::deserialize_from_disk: cannot stat '" + file_path.string() +
                             "': " + ec.message());
  }
  if (!zone_map_file_exists) { return std::nullopt; }

  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("zone_map::deserialize_from_disk: Missing zone map file '" +
                             file_path.string());
  }

  // The reader works on a stream and does not know the path, so name the file here.
  auto const metadata = [&] {
    try {
      return u::read_serialized_zone_map_metadata(file);
    } catch (std::invalid_argument const& error) {
      throw std::invalid_argument("zone_map::" + std::string(error.what()) + " in '" +
                                  file_path.string() + "'");
    }
  }();

  std::int32_t const partition_size = metadata.partition_size();
  std::int64_t const num_rows       = metadata.num_rows();
  std::int32_t const num_columns    = metadata.num_columns();
  std::int32_t const num_partitions = metadata.num_partitions();

  // Validate the untrusted on-disk header before allocating or narrowing to cudf::size_type.
  if (partition_size <= 0) {
    throw std::invalid_argument("zone_map::deserialize_from_disk: non-positive partition_size " +
                                std::to_string(partition_size) + " in '" + file_path.string() +
                                "'");
  }
  if (num_rows < 0) {
    throw std::invalid_argument("zone_map::deserialize_from_disk: negative num_rows " +
                                std::to_string(num_rows) + " in '" + file_path.string() + "'");
  }
  if (num_columns < 0 || num_partitions < 0) {
    throw std::invalid_argument(
      "zone_map::deserialize_from_disk: negative num_columns/num_partitions "
      "in '" +
      file_path.string() + "'");
  }
  if (auto const expected_num_partitions =
        u::divide_round_up(num_rows, static_cast<std::int64_t>(partition_size));
      static_cast<std::int64_t>(num_partitions) != expected_num_partitions) {
    throw std::invalid_argument(
      "zone_map::deserialize_from_disk: partition count " + std::to_string(num_partitions) +
      " does not match ceil(num_rows / partition_size) = " +
      std::to_string(expected_num_partitions) + " in '" + file_path.string() + "'");
  }
  if (static_cast<std::size_t>(metadata.null_counts_size()) !=
      static_cast<std::size_t>(num_columns) * static_cast<std::size_t>(num_partitions)) {
    throw std::invalid_argument(
      "zone_map::deserialize_from_disk: null_counts length " +
      std::to_string(metadata.null_counts_size()) +
      " does not match num_columns * num_partitions = " +
      std::to_string(static_cast<std::size_t>(num_columns) * num_partitions) + " in '" +
      file_path.string() + "'");
  }

  // The flattened array is partition-major, matching the in-memory layout null_counts[partition]
  // [column].
  std::vector<std::vector<cudf::size_type>> null_counts(num_partitions);
  std::size_t flat_index = 0;
  for (std::int32_t p = 0; p < num_partitions; ++p) {
    null_counts[p].resize(num_columns);
    for (std::int32_t c = 0; c < num_columns; ++c) {
      null_counts[p][c] =
        static_cast<cudf::size_type>(metadata.null_counts(static_cast<int>(flat_index++)));
    }
  }

  std::uint64_t ipc_size = 0;
  u::read_pod(file, ipc_size);

  // The IPC payload is the file's last field, so its declared length must be exactly the number of
  // bytes left. This is checked before allocating.
  auto const payload_offset = static_cast<std::uint64_t>(file.tellg());
  auto const file_size      = static_cast<std::uint64_t>(std::filesystem::file_size(file_path, ec));
  if (ec) {
    throw std::runtime_error("zone_map::deserialize_from_disk: cannot size '" + file_path.string() +
                             "': " + ec.message());
  }
  if (payload_offset > file_size) {
    throw std::runtime_error("zone_map::deserialize_from_disk: cannot locate the IPC payload in '" +
                             file_path.string() + "'");
  }
  auto const remaining = file_size - payload_offset;
  if (ipc_size != remaining) {
    throw std::invalid_argument("zone_map::deserialize_from_disk: IPC payload length " +
                                std::to_string(ipc_size) + " does not match the " +
                                std::to_string(remaining) + " bytes remaining in '" +
                                file_path.string() + "'");
  }

  std::vector<std::byte> ipc_bytes(static_cast<size_t>(ipc_size));
  if (ipc_size > 0) {
    file.read(reinterpret_cast<char*>(ipc_bytes.data()), static_cast<std::streamsize>(ipc_size));
  }
  if (file.fail()) {
    throw std::runtime_error("zone_map::deserialize_from_disk: read error for '" +
                             file_path.string() + "'");
  }

  return parsed_zone_map{static_cast<cudf::size_type>(partition_size),
                         static_cast<cudf::size_type>(num_rows),
                         std::move(null_counts),
                         std::move(ipc_bytes)};
}

}  // namespace

std::unique_ptr<gqe::zone_map> gqe::local_zone_map::deserialize_from_disk(
  std::filesystem::path const& zone_map_dir, cuda::stream_ref stream)
{
  auto parsed = parse_serialized_zone_map(zone_map_dir);
  if (!parsed) { return nullptr; }
  return std::make_unique<local_zone_map>(
    parsed->partition_size,
    parsed->num_rows,
    parsed->null_counts,
    std::span<std::byte const>{parsed->ipc_bytes.data(), parsed->ipc_bytes.size()},
    stream);
}

std::optional<gqe::shared_memory_handle_t> gqe::deserialize_zone_map_into_segment(
  gqe::shared_segment& segment,
  gqe::storage::shared_memory_allocator const& alloc,
  std::filesystem::path const& zone_map_dir,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref storage_mr)
{
  auto parsed = parse_serialized_zone_map(zone_map_dir);
  if (!parsed) { return std::nullopt; }
  // Reconstruct the zone map directly in the segment via the reconstruct constructor. The
  // ipc/min-max buffers land in the segment through `storage_mr`.
  auto* storage =
    segment.construct<gqe::shared_zone_map_storage>(boost::interprocess::anonymous_instance)(
      parsed->partition_size,
      parsed->num_rows,
      parsed->null_counts,
      std::span<std::byte const>{parsed->ipc_bytes.data(), parsed->ipc_bytes.size()},
      stream,
      storage_mr,
      alloc);
  return segment.get_handle_from_address(storage);
}

void gqe::local_zone_map::deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  (void)mr;
  base_type::deallocate(stream, local_zone_map_storage_resource());
}

gqe::shared_zone_map::shared_zone_map(gqe::shared_segment& segment,
                                      typename storage_handle_type::handle_type handle)
  : base_type(segment, handle)
{
}

template class gqe::zone_map_storage<std::allocator<std::byte>>;
template class gqe::zone_map_storage<gqe::storage::shared_memory_allocator>;
template class gqe::basic_zone_map<gqe::storage::local_storage_handle<gqe::zone_map_storage<>>>;
template class gqe::basic_zone_map<
  gqe::storage::shared_storage_handle<gqe::shared_zone_map_storage>>;
