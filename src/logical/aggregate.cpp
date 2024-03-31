/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/utility.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

namespace {
using measure_type = gqe::logical::aggregate_relation::measure_type;
bool compare_measure_vectors(const std::vector<measure_type>& v1,
                             const std::vector<measure_type>& v2)
{
  return equal(
    begin(v1), end(v1), begin(v2), end(v2), [](const measure_type lhs, measure_type rhs) {
      return (lhs.first == rhs.first) && (*lhs.second == *rhs.second);
    });
}
}  // namespace

namespace gqe {

// Get the second aggregation kind from the first aggregation kind in apply-concat-apply
cudf::aggregation::Kind get_second_aggregation_kind(cudf::aggregation::Kind first_aggregation_kind);

namespace logical {

aggregate_relation::aggregate_relation(
  std::shared_ptr<relation> input_relation,
  std::vector<std::shared_ptr<relation>> subquery_relations,
  std::vector<std::unique_ptr<expression>> keys,
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> measures)
  : relation({std::move(input_relation)}, std::move(subquery_relations)),
    _keys(std::move(keys)),
    _measures(std::move(measures))
{
}

std::vector<expression*> aggregate_relation::keys_unsafe() const noexcept
{
  return gqe::utility::to_raw_ptrs(_keys);
}

std::vector<aggregate_relation::measure_type> aggregate_relation::measures_unsafe() const noexcept
{
  std::vector<aggregate_relation::measure_type> measures_to_return;
  measures_to_return.reserve(_measures.size());
  for (auto const& [kind, expr] : _measures)
    measures_to_return.emplace_back(kind, expr.get());

  return measures_to_return;
}

[[nodiscard]] std::vector<cudf::data_type> aggregate_relation::data_types() const
{
  auto input_relation = children_unsafe()[0];
  std::vector<cudf::data_type> data_types;
  for (auto const& key : _keys)
    data_types.push_back(key->data_type(input_relation->data_types()));

  for (auto [aggregation_kind, value] : measures_unsafe()) {
    if (aggregation_kind == cudf::aggregation::MEAN) {
      // The `mean` aggregation needs to divide the sum by the count during post-processing, so we
      // treat it as a special case.
      data_types.emplace_back(cudf::type_id::FLOAT64);
    } else {
      // All other aggregations do not need post-processing, so the output type is the data type of
      // the second aggregation in apply-concat-apply.
      cudf::data_type output_type =
        cudf::detail::target_type(value->data_type(input_relation->data_types()), aggregation_kind);
      output_type =
        cudf::detail::target_type(output_type, get_second_aggregation_kind(aggregation_kind));
      data_types.push_back(output_type);
    }
  }
  return data_types;
}

[[nodiscard]] std::string aggregate_relation::to_string() const
{
  std::string agg_relation_string = "{\"Aggregate\" : {\n";
  // Aggregate keys
  agg_relation_string +=
    "\t\"key expressions\" : " + utility::list_to_string(keys_unsafe()) + ",\n";
  // Aggregate measures
  auto measures = measures_unsafe();
  agg_relation_string +=
    "\t\"measures\" : " + utility::list_to_string(measures.begin(), measures.end()) + ",\n";
  // Data types
  agg_relation_string += "\t\"data types\" : " + utility::list_to_string(data_types()) + ",\n";
  // Children
  agg_relation_string += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + "\n";
  agg_relation_string += "}}";
  return agg_relation_string;
}

bool aggregate_relation::operator==(const relation& other) const
{
  auto this_type = this->type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }
  auto other_agg_relation = dynamic_cast<const aggregate_relation*>(&other);
  // Compare attributes
  if (!gqe::utility::compare_pointer_vectors(this->keys_unsafe(),
                                             other_agg_relation->keys_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): keys mismatch");
    return false;
  }
  if (!compare_measure_vectors(this->measures_unsafe(), other_agg_relation->measures_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): measures mismatch");
    return false;
  }
  if (this->data_types() != other_agg_relation->data_types()) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  // Compare children
  if (!gqe::utility::compare_pointer_vectors(this->children_unsafe(),
                                             other_agg_relation->children_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): children mismatch");
    return false;
  }
  // Compare subquery_relations
  if (!gqe::utility::compare_pointer_vectors(this->subqueries_unsafe(),
                                             other_agg_relation->subqueries_unsafe())) {
    utility::log_relation_comparison_message(this_type,
                                             "operator==(): subquery relations mismatch");
    return false;
  }
  return true;
}

}  // namespace logical
}  // namespace gqe
