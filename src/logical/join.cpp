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

#include <gqe/logical/join.hpp>

#include <gqe/logical/utility.hpp>
#include <gqe/utility/helpers.hpp>

namespace gqe {
namespace logical {

join_relation::join_relation(std::shared_ptr<relation> left,
                             std::shared_ptr<relation> right,
                             std::vector<std::shared_ptr<relation>> subquery_relations,
                             std::unique_ptr<expression> condition,
                             join_type_type join_type,
                             std::vector<cudf::size_type> projection_indices)
  : relation({std::move(left), std::move(right)}, std::move(subquery_relations)),
    _condition(std::move(condition)),
    _join_type(join_type),
    _projection_indices(std::move(projection_indices))
{
}

std::vector<cudf::data_type> join_relation::data_types() const
{
  auto children = children_unsafe();
  auto left     = children[0];
  auto right    = children[1];

  // Initialize output column _data_types
  std::vector<cudf::data_type> full_data_types;  // Data types of all columns before the projection
  if (_join_type == join_type_type::inner || _join_type == join_type_type::full ||
      _join_type == join_type_type::left || _join_type == join_type_type::single) {
    for (auto const& column_type : left->data_types())
      full_data_types.push_back(column_type);
    for (auto const& column_type : right->data_types())
      full_data_types.push_back(column_type);
  } else if (_join_type == join_type_type::left_semi || _join_type == join_type_type::left_anti) {
    full_data_types = left->data_types();
  } else {
    throw std::runtime_error("JoinRelation: Unsupported join type");
  }

  std::vector<cudf::data_type> data_types;
  data_types.reserve(_projection_indices.size());
  for (auto const& column_idx : _projection_indices)
    data_types.push_back(full_data_types[column_idx]);
  return data_types;
}

std::string join_relation::to_string() const
{
  std::string join_relation_string = "{\"Join\" : {\n";
  // Join type
  join_relation_string += "\t\"type\" : \"" + utility::join_type_str(_join_type) + "\",\n";
  // Condition
  join_relation_string += "\t\"condition\" : \"" + _condition->to_string() + "\",\n";
  // Data types
  join_relation_string += "\t\"data types\" : " + utility::list_to_string(data_types()) + ",\n";
  // Projection indices
  join_relation_string +=
    "\t\"project indices\" : " +
    utility::list_to_string(_projection_indices.begin(), _projection_indices.end()) + ",\n";
  // Children
  join_relation_string += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + "\n";
  join_relation_string += "}}";
  return join_relation_string;
}

bool join_relation::operator==(const relation& other) const
{
  auto this_type = this->type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }
  auto other_join_relation = dynamic_cast<const join_relation*>(&other);
  // Compare attributes
  if (this->join_type() != other_join_relation->join_type()) {
    utility::log_relation_comparison_message(this_type, "operator==(): join type mismatch");
    return false;
  }
  if (*this->condition() != *other_join_relation->condition()) {
    utility::log_relation_comparison_message(this_type, "operator==(): condition mismatch");
    return false;
  }
  if (this->projection_indices() != other_join_relation->projection_indices()) {
    utility::log_relation_comparison_message(this_type,
                                             "operator==(): projection indices mismatch");
    return false;
  }
  if (this->data_types() != other_join_relation->data_types()) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  // Compare members defined in base class
  return relation::compare_relation_members(other);
}

}  // namespace logical
}  // namespace gqe
