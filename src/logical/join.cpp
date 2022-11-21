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

#include <gqe/logical/join.hpp>
#include <gqe/logical/utility.hpp>

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
  _init_data_types();
}

void join_relation::_init_data_types() const
{
  if (_data_types.size() > 0)
    throw std::runtime_error("Reinitialization of _data_types not allowed");

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

  _data_types.reserve(_projection_indices.size());
  for (auto const& column_idx : _projection_indices)
    _data_types.push_back(full_data_types[column_idx]);
}

std::vector<cudf::data_type> join_relation::data_types() const { return _data_types; }

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

}  // namespace logical
}  // namespace gqe