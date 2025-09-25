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

#include <gqe/physical/join.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {

std::vector<cudf::data_type> join_relation_base::output_data_types() const
{
  auto children = children_unsafe();
  auto left     = children[0];
  auto right    = children[1];

  // Initialize output column _data_types
  std::vector<cudf::data_type> full_data_types;  // Data types of all columns before the projection
  if (_join_type == join_type_type::inner || _join_type == join_type_type::full ||
      _join_type == join_type_type::left || _join_type == join_type_type::single) {
    for (auto const& column_type : left->output_data_types())
      full_data_types.push_back(column_type);
    for (auto const& column_type : right->output_data_types())
      full_data_types.push_back(column_type);
  } else if (_join_type == join_type_type::left_semi || _join_type == join_type_type::left_anti) {
    full_data_types = left->output_data_types();
  } else {
    throw std::runtime_error("JoinRelationBase: Unsupported join type");
  }

  std::vector<cudf::data_type> data_types;
  data_types.reserve(_projection_indices.size());
  for (auto const& column_idx : _projection_indices)
    data_types.push_back(full_data_types[column_idx]);
  return data_types;
}

std::string join_relation_base::print() const
{
  // Join type
  std::string join_mem_string =
    "\t\"type\" : \"" + logical::utility::join_type_str(_join_type) + "\",\n";
  // Condition
  join_mem_string += "\t\"condition\" : \"" + _condition->to_string() + "\",\n";
  // Output data types
  join_mem_string +=
    "\t\"output data types\" : " + logical::utility::list_to_string(output_data_types()) + ",\n";
  // Projection indices
  join_mem_string +=
    "\t\"project indices\" : " +
    logical::utility::list_to_string(_projection_indices.begin(), _projection_indices.end()) +
    ",\n";
  // Children
  join_mem_string +=
    "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + ",\n";
  return join_mem_string;
}

std::string broadcast_join_relation::to_string() const
{
  std::string join_relation_string = "{\"Broadcast Join\" : {\n";
  join_relation_string += print();
  // Unique keys policy
  join_relation_string += "\t\"unique keys policy\" : \"" +
                          logical::utility::unique_keys_policy_str(unique_keys_policy()) + "\",\n";
  // Broadcast policy
  std::string broadcast_policy_str = policy() == broadcast_policy::left ? "left" : "right";
  join_relation_string += "\t\"broadcast policy\" : \"" + broadcast_policy_str + "\",\n";
  // Perfect hashing
  std::string perfect_hashing_str = perfect_hashing() ? "enabled" : "disabled";
  join_relation_string += "\t\"perfect hashing\" : \"" + perfect_hashing_str + "\"\n";
  join_relation_string += "}}";
  return join_relation_string;
}

}  // end of namespace physical
}  // end of namespace gqe
