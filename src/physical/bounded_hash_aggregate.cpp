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

#include <gqe/physical/bounded_hash_aggregate.hpp>

#include <gqe/aggregation.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/utilities/traits.hpp>

#include <string>
#include <utility>

namespace gqe {
namespace physical {

bounded_hash_aggregate_relation::bounded_hash_aggregate_relation(
  std::shared_ptr<relation> input,
  std::vector<cudf::size_type> key_indices,
  std::vector<value_type> values,
  int64_t group_count_bound,
  std::optional<double> desired_load_factor)
  : relation({std::move(input)}, /*subqueries=*/{}),
    _key_indices(std::move(key_indices)),
    _values(std::move(values)),
    _group_count_bound(group_count_bound),
    _desired_load_factor(desired_load_factor)
{
  GQE_EXPECTS(children_unsafe()[0] != nullptr,
              "bounded_hash_aggregate_relation: input must not be null");
  GQE_EXPECTS(group_count_bound > 0,
              "bounded_hash_aggregate_relation: group_count_bound must be positive");
  GQE_EXPECTS(!desired_load_factor || (*desired_load_factor > 0.0 && *desired_load_factor <= 1.0),
              "bounded_hash_aggregate_relation: desired_load_factor must be in (0, 1]");

  auto const num_input_columns =
    static_cast<cudf::size_type>(children_unsafe()[0]->output_data_types().size());
  for (auto const key_index : _key_indices) {
    GQE_EXPECTS(key_index >= 0 && key_index < num_input_columns,
                "bounded_hash_aggregate_relation: key column index is outside the input schema");
  }
  for (auto const& [kind, value_index, value_nullability] : _values) {
    GQE_EXPECTS(value_index.has_value() == (kind != cudf::aggregation::COUNT_ALL),
                "bounded_hash_aggregate_relation: the value column index must be nullopt "
                "exactly for COUNT_ALL, which reads no column");
    GQE_EXPECTS(!value_index || (*value_index >= 0 && *value_index < num_input_columns),
                "bounded_hash_aggregate_relation: value column index is outside the input schema");
    GQE_EXPECTS(value_index.has_value() || value_nullability == nullability::nullable,
                "bounded_hash_aggregate_relation: COUNT_ALL reads no column, so it cannot "
                "declare one NOT NULL");
  }
}

[[nodiscard]] std::vector<cudf::data_type> bounded_hash_aggregate_relation::output_data_types()
  const
{
  auto const input_types = children_unsafe()[0]->output_data_types();

  std::vector<cudf::data_type> data_types;
  data_types.reserve(_key_indices.size() + _values.size());
  for (auto const key_index : _key_indices) {
    data_types.push_back(input_types[static_cast<std::size_t>(key_index)]);
  }
  for (auto const& [kind, value_index, value_nullability] : _values) {
    // Counts target a 64-bit integer: unlike a single cuDF call, the operator aggregates
    // unbounded pipelined rows into one map, so a 32-bit count could wrap. All other kinds
    // follow the cuDF target type of the aggregated column.
    if (kind == cudf::aggregation::COUNT_VALID || kind == cudf::aggregation::COUNT_ALL) {
      data_types.emplace_back(cudf::type_id::INT64);
    } else {
      // A decimal SUM widens to the DECIMAL128 accumulator type the operator requires; the
      // cuDF target type of every other combination keeps the operator's contract.
      auto const value_type = input_types[static_cast<std::size_t>(*value_index)];
      data_types.push_back(kind == cudf::aggregation::SUM && cudf::is_fixed_point(value_type)
                             ? gqe::detail::decimal_sum_target_type(value_type)
                             : gqe::detail::compute_target_type(value_type, kind));
    }
  }
  return data_types;
}

[[nodiscard]] std::string bounded_hash_aggregate_relation::to_string() const
{
  std::string relation_string = "{\"Bounded Hash Aggregate\" : {\n";
  std::string key_string      = "[";
  for (std::size_t i = 0; i < _key_indices.size(); ++i) {
    key_string += std::to_string(_key_indices[i]);
    if (i + 1 < _key_indices.size()) { key_string += ", "; }
  }
  key_string += "]";
  relation_string += "\t\"key column indices\" : " + key_string + ",\n";
  std::string value_string = "[";
  for (std::size_t i = 0; i < _values.size(); ++i) {
    auto const& [kind, value_index, value_nullability] = _values[i];
    value_string +=
      "{\"aggregation_kind\" : " + std::to_string(static_cast<int>(kind)) +
      ", \"value_index\" : " + (value_index ? std::to_string(*value_index) : "null") +
      ", \"value_not_null\" : " + (value_nullability == nullability::not_null ? "true" : "false") +
      "}";
    if (i + 1 < _values.size()) { value_string += ", "; }
  }
  value_string += "]";
  relation_string += "\t\"values\" : " + value_string + ",\n";
  relation_string += "\t\"group count bound\" : " + std::to_string(_group_count_bound) + ",\n";
  std::string load_factor_string =
    _desired_load_factor ? std::to_string(*_desired_load_factor) : "null";
  relation_string += "\t\"desired load factor\" : " + load_factor_string + "\n";
  relation_string += "}}";
  return relation_string;
}

}  // end of namespace physical
}  // end of namespace gqe
