/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/physical/aggregate.hpp>

#include <gqe/logical/utility.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

namespace gqe {

// Get the second aggregation kind from the first aggregation kind in apply-concat-apply
cudf::aggregation::Kind get_second_aggregation_kind(cudf::aggregation::Kind first_aggregation_kind);

namespace physical {

[[nodiscard]] std::vector<cudf::data_type> aggregate_relation_base::output_data_types() const
{
  auto input_relation = children_unsafe()[0];
  std::vector<cudf::data_type> data_types;
  for (auto const& key : _keys)
    data_types.push_back(key->data_type(input_relation->output_data_types()));

  for (auto [aggregation_kind, value] : values_unsafe()) {
    if (aggregation_kind == cudf::aggregation::MEAN) {
      // The `mean` aggregation needs to divide the sum by the count during post-processing, so we
      // treat it as a special case.
      data_types.emplace_back(cudf::type_id::FLOAT64);
    } else {
      // All other aggregations do not need post-processing, so the output type is the data type of
      // the second aggregation in apply-concat-apply.
      cudf::data_type output_type = cudf::detail::target_type(
        value->data_type(input_relation->output_data_types()), aggregation_kind);
      output_type =
        cudf::detail::target_type(output_type, get_second_aggregation_kind(aggregation_kind));
      data_types.push_back(output_type);
    }
  }
  return data_types;
}

[[nodiscard]] std::string aggregate_relation_base::print() const
{
  // only print the members
  // Aggregate keys
  std::string agg_mem_string =
    "\t\"key expressions\" : " + logical::utility::list_to_string(keys_unsafe()) + ",\n";
  // Aggregate measures
  auto measures = values_unsafe();
  agg_mem_string +=
    "\t\"measures\" : " + logical::utility::list_to_string(measures.begin(), measures.end()) +
    ",\n";
  auto condition            = condition_unsafe();
  std::string condition_str = condition ? condition->to_string() : "";
  // Condition
  agg_mem_string += "\t\"condition\" : \"" + condition_str + "\",\n";
  // Perfect hashing
  std::string perfect_hashing_str = perfect_hashing() ? "enabled" : "disabled";
  agg_mem_string += "\t\"perfect hashing\" : \"" + perfect_hashing_str + "\",\n";
  // Output data types
  agg_mem_string +=
    "\t\"output data types\" : " + logical::utility::list_to_string(output_data_types()) + ",\n";
  // Children
  agg_mem_string +=
    "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  return agg_mem_string;
}

[[nodiscard]] std::string concatenate_aggregate_relation::to_string() const
{
  std::string agg_relation_string = "{\"Concatenate Aggregate\" : {\n";
  agg_relation_string += print();
  agg_relation_string += "}}";
  return agg_relation_string;
}

}  // end of namespace physical
}  // end of namespace gqe
