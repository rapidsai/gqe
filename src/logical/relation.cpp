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

#include <gqe/logical/relation.hpp>

#include <numeric>
#include <regex>
#include <stdexcept>
#include <vector>

namespace gqe {

namespace {

/**
 * @brief Return string representation of the list of output data types
 *
 * @param types Vector of cuDF data types
 * @return Output data types string repressentation
 */
std::string list_to_string(std::vector<cudf::data_type> const& types)
{
  std::string data_type_string = "[";
  bool first                   = true;
  for (auto dt : types) {
    if (!first) data_type_string += ", ";
    data_type_string += "\"" + cudf::type_dispatcher(dt, cudf::type_to_name{}) + "\"";
    first = false;
  }
  return data_type_string + "]";
}

/**
 * @brief Return string representation of relation list
 *
 * @param relation_list The list of relations to convert to string
 * @return Relation list string representation
 */
std::string list_to_string(std::vector<logical::relation*> relation_list)
{
  std::string relation_list_string = "[";
  bool first                       = true;
  for (auto relation : relation_list) {
    if (!first) relation_list_string += ", ";
    relation_list_string += relation->to_string();
    first = false;
  }
  relation_list_string += "]";
  return relation_list_string;
}

/**
 * @brief Return string representation of list of elements in input vector
 *
 * @tparam T Type of values in input vector
 * @param begin Input vector begin iterator
 * @param end Input vector end iterator
 * @return Input vector string representation
 */
template <typename T>
[[nodiscard]] std::string list_to_string(T begin, T end)
{
  std::stringstream ss;
  ss << "[";
  bool first = true;
  for (; begin != end; begin++) {
    if (!first) ss << ", ";
    ss << "\"";
    ss << *begin;
    ss << "\"";
    first = false;
  }
  ss << "]";
  return ss.str();
}

/**
 * @brief Convert `join_type` to its string representation
 *
 * @param join_type Join type to return string for
 * @return String representation on `join_type`
 */
std::string join_type_str(join_type_type join_type)
{
  switch (join_type) {
    case join_type_type::inner: return "inner"; break;
    case join_type_type::left: return "left"; break;
    case join_type_type::left_semi: return "left semi"; break;
    case join_type_type::full: return "full"; break;
    case join_type_type::left_anti: return "left anti"; break;
    case join_type_type::single: return "single"; break;
    default:
      throw std::runtime_error("Join type enum " + std::to_string(static_cast<int>(join_type)) +
                               " not supported");
  }
}

}  // namespace

namespace logical {

join_relation::join_relation(std::shared_ptr<relation> left,
                             std::shared_ptr<relation> right,
                             std::unique_ptr<expression> condition,
                             join_type_type join_type)
  : relation({left, right}), _condition(std::move(condition)), _join_type(join_type)
{
  _init_data_types();
  // Initialize _projection_indices
  // TODO: Configure projection indices from parent projection relation. For now,
  //       we'll return all columns and handle projection in a separate relation.
  this->_projection_indices.resize(this->_data_types.size());
  std::iota(_projection_indices.begin(), _projection_indices.end(), 0);
}

void join_relation::_init_data_types() const
{
  if (_data_types.size() > 0)
    throw std::runtime_error("Reinitialization of _data_types not allowed");

  auto children = children_unsafe();
  auto left     = children[0];
  auto right    = children[1];
  // Initialize output column _data_types
  if (_join_type == join_type_type::inner || _join_type == join_type_type::full ||
      _join_type == join_type_type::left || _join_type == join_type_type::single) {
    for (auto const& column_type : left->data_types())
      _data_types.push_back(column_type);
    for (auto const& column_type : right->data_types())
      _data_types.push_back(column_type);
  } else if (_join_type == join_type_type::left_semi || _join_type == join_type_type::left_anti) {
    _data_types = left->data_types();
  } else {
    throw std::runtime_error("JoinRelation: Unsupported join type");
  }
}

std::vector<cudf::data_type> join_relation::data_types() const { return _data_types; }

cudf::size_type join_relation::num_columns() const { return _data_types.size(); }

std::string join_relation::to_string() const
{
  std::string join_relation_string = "{\"Join\" : {\n";
  // Join type
  join_relation_string += "\t\"type\" : \"" + join_type_str(_join_type) + "\",\n";
  // Condition
  join_relation_string += "\t\"condition\" : \"" + _condition->to_string() + "\",\n";
  // Data types
  join_relation_string += "\t\"data types\" : " + gqe::list_to_string(data_types()) + ",\n";
  // Projection indices
  join_relation_string +=
    "\t\"project indices\" : " +
    gqe::list_to_string(_projection_indices.begin(), _projection_indices.end()) + ",\n";
  // Children
  join_relation_string += "\t\"children\" : " + gqe::list_to_string(children_unsafe()) + "\n";
  join_relation_string += "}}";
  return join_relation_string;
}

read_relation::read_relation(std::vector<std::string> column_names,
                             std::vector<cudf::data_type> column_types,
                             std::string table_name)
  : relation({}),
    _column_names(std::move(column_names)),
    _table_name(std::move(table_name)),
    _data_types(std::move(column_types))
{
}

std::string read_relation::to_string() const
{
  std::string read_relation_str = "{\"Read\" : {\n";
  // Table name
  read_relation_str += "\t\"table name\" : \"" + this->table_name() + "\",\n";
  // Data types
  read_relation_str += "\t\"data types\" : " + gqe::list_to_string(data_types()) + ",\n";
  // Column names
  read_relation_str +=
    "\t\"column names\" : " + gqe::list_to_string(_column_names.begin(), _column_names.end()) +
    ",\n";
  // Children
  read_relation_str += "\t\"children\" : " + gqe::list_to_string(children_unsafe()) + "\n";
  read_relation_str += "}}";
  return read_relation_str;
}

project_relation::project_relation(std::shared_ptr<relation> child,
                                   std::vector<std::unique_ptr<expression>> output_expressions)
  : relation({child}), _output_expressions(std::move(output_expressions))
{
}

void project_relation::_init_data_types() const
{
  assert(this->children_size() ==
         1);  // There should only be one input relation to a projection relation
  this->_data_types = std::vector<cudf::data_type>();
  for (auto const& output_expression : _output_expressions) {
    auto child_rel        = this->children_unsafe()[0];
    auto child_rel_dtypes = child_rel->data_types();
    auto exp_data_types   = output_expression->data_type(child_rel_dtypes);
    this->_data_types.value().push_back(exp_data_types);
  }
}

[[nodiscard]] std::vector<cudf::data_type> project_relation::data_types() const
{
  if (!this->_data_types.has_value()) { this->_init_data_types(); }
  return this->_data_types.value();
}

std::string project_relation::to_string() const
{
  std::string project_relation_str = "{\"Project\" : {\n";
  // Output expressions
  bool first = true;
  project_relation_str += "\t\"output expressions\" : [";
  for (auto expression : output_expressions_unsafe()) {
    if (!first) project_relation_str += ",\n";
    project_relation_str += "\"" + expression->to_string() + "\"";
    first = false;
  }
  project_relation_str += "\t],\n";
  // Data types
  project_relation_str += "\t\"data types\" : " + gqe::list_to_string(data_types()) + ",\n";
  // Children
  project_relation_str += "\t\"children\" : " + gqe::list_to_string(children_unsafe()) + "\n";
  project_relation_str += "}}";
  return project_relation_str;
}

}  // namespace logical
}  // namespace gqe
