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

#include <gqe/logical/filter.hpp>
#include <gqe/logical/utility.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/types.hpp>

#include <vector>

namespace gqe {
namespace logical {

filter_relation::filter_relation(std::shared_ptr<relation> input_relation,
                                 std::vector<std::shared_ptr<relation>> subquery_relations,
                                 std::unique_ptr<expression> condition,
                                 std::vector<cudf::size_type> projection_indices)
  : relation({std::move(input_relation)}, std::move(subquery_relations)),
    _condition(std::move(condition)),
    _projection_indices(std::move(projection_indices))
{
  assert(this->children_size() == 1);
}

std::string filter_relation::to_string() const
{
  std::string filter_relation_string = "{\"Filter\" : {\n";
  // Sort expressions
  filter_relation_string += "\t\"condition\" : \"" + _condition->to_string() + "\",\n";
  // Data types
  filter_relation_string += "\t\"data types\" : " + utility::list_to_string(data_types()) + ",\n";
  // Project indices
  filter_relation_string +=
    "\t\"project indices\" : " +
    utility::list_to_string(_projection_indices.begin(), _projection_indices.end()) + ",\n";
  // Children
  filter_relation_string += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + "\n";
  filter_relation_string += "}}";
  return filter_relation_string;
}

bool filter_relation::operator==(const relation& other) const
{
  auto this_type = this->type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }
  auto other_filter_relation = dynamic_cast<const filter_relation*>(&other);
  // Compare attributes
  if (*this->condition() != *other_filter_relation->condition()) {
    utility::log_relation_comparison_message(this_type, "operator==(): condition mismatch");
    return false;
  }
  if (this->projection_indices() != other_filter_relation->projection_indices()) {
    utility::log_relation_comparison_message(this_type,
                                             "operator==(): projection indices mismatch");
    return false;
  }
  if (this->data_types() != other_filter_relation->data_types()) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  // Compare members defined in base class
  return relation::compare_relation_members(other);
}

std::vector<cudf::data_type> filter_relation::data_types() const
{
  // For each column materialized, the output data type is the same as the input data type.
  std::vector<cudf::data_type> input_data_types = this->children_unsafe()[0]->data_types();
  std::vector<cudf::data_type> data_types;
  data_types.reserve(_projection_indices.size());
  for (auto const& column_idx : _projection_indices)
    data_types.push_back(input_data_types[column_idx]);
  return data_types;
}

}  // namespace logical
}  // namespace gqe
