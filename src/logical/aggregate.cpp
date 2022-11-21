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

#include <cudf/detail/aggregation/aggregation.hpp>

namespace gqe {
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

void aggregate_relation::_init_data_types() const
{
  auto input_relation = children_unsafe()[0];
  _data_types         = std::vector<cudf::data_type>();
  for (auto const& key : _keys)
    _data_types.value().push_back(key->data_type(input_relation->data_types()));

  for (auto measure : measures_unsafe()) {
    auto aggregation_kind = measure.first;
    auto value            = measure.second;
    cudf::data_type output_type =
      cudf::detail::target_type(value->data_type(input_relation->data_types()), aggregation_kind);
    _data_types.value().push_back(output_type);
  }
}

std::vector<expression*> aggregate_relation::keys_unsafe() const noexcept
{
  return gqe::utility::to_raw_ptrs(_keys);
}

std::vector<std::pair<cudf::aggregation::Kind, expression*>> aggregate_relation::measures_unsafe()
  const noexcept
{
  std::vector<std::pair<cudf::aggregation::Kind, expression*>> measures_to_return;
  measures_to_return.reserve(_measures.size());
  for (auto const& [kind, expr] : _measures)
    measures_to_return.emplace_back(kind, expr.get());

  return measures_to_return;
}

[[nodiscard]] std::vector<cudf::data_type> aggregate_relation::data_types() const
{
  if (!this->_data_types.has_value()) { this->_init_data_types(); }
  return this->_data_types.value();
}

[[nodiscard]] std::string aggregate_relation::to_string() const
{
  // DEBUG. TODO: remove iostream import
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

}  // namespace logical
}  // namespace gqe