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

#include <cudf/types.hpp>
#include <gqe/logical/relation.hpp>

#include <memory>
#include <numeric>
#include <ostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
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
 * @brief Return string representation of expression list
 *
 * @param expression_list The list of relations to convert to string
 * @return Expression list string representation
 */
std::string list_to_string(std::vector<expression*> expression_list)
{
  std::string expression_list_string = "[";
  bool first                         = true;
  for (auto expr : expression_list) {
    if (!first) expression_list_string += ", ";
    expression_list_string += "\"" + expr->to_string() + "\"";
    first = false;
  }
  expression_list_string += "]";
  return expression_list_string;
}

std::ostream& operator<<(std::ostream& os, cudf::order order)
{
  switch (order) {
    case cudf::order::ASCENDING: os << "ASC"; break;
    case cudf::order::DESCENDING: os << "DESC"; break;
    default:
      throw std::runtime_error("Invalid sort order enum: " +
                               std::to_string(static_cast<int>(order)));
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, cudf::null_order prec)
{
  switch (prec) {
    case cudf::null_order::BEFORE: os << "NULLS FIRST"; break;
    case cudf::null_order::AFTER: os << "NULLS LAST"; break;
    default:
      throw std::runtime_error("Invalid null order enum: " +
                               std::to_string(static_cast<int>(prec)));
  }
  return os;
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

fetch_relation::fetch_relation(std::shared_ptr<relation> input_relation,
                               int64_t offset,
                               int64_t count)
  : relation({std::move(input_relation)}), _offset(offset), _count(count)
{
  assert(this->children_size() == 1);
  // Output data types are the same as input data types
  _data_types = this->children_unsafe()[0]->data_types();
}

std::string fetch_relation::to_string() const
{
  std::string fetch_relation_string = "{\"Fetch\" : {\n";
  // Offset
  fetch_relation_string += "\t\"offset\" : \"" + std::to_string(_offset) + "\",\n";
  // Count
  fetch_relation_string += "\t\"count\" : \"" + std::to_string(_count) + "\",\n";
  // Data types
  fetch_relation_string += "\t\"data types\" : " + gqe::list_to_string(data_types()) + ",\n";
  // Children
  fetch_relation_string += "\t\"children\" : " + gqe::list_to_string(children_unsafe()) + "\n";
  fetch_relation_string += "}}";
  return fetch_relation_string;
}

sort_relation::sort_relation(std::shared_ptr<relation> input_relation,
                             std::vector<cudf::order> column_orders,
                             std::vector<cudf::null_order> null_precedences,
                             std::vector<std::unique_ptr<expression>> expressions)
  : relation({std::move(input_relation)}),
    _expressions(std::move(expressions)),
    _column_orders(std::move(column_orders)),
    _null_orders(std::move(null_precedences))
{
  assert(this->children_size() == 1);
  // Output data types are the same as input data types
  _data_types = this->children_unsafe()[0]->data_types();
}

std::string sort_relation::to_string() const
{
  std::string sort_relation_string = "{\"Sort\" : {\n";
  // Sort expressions
  sort_relation_string +=
    "\t\"sort expressions\" : " + list_to_string(expressions_unsafe()) + ",\n";
  // Column orders
  sort_relation_string +=
    "\t\"column orders\" : " + list_to_string(_column_orders.begin(), _column_orders.end()) + ",\n";
  // Null precedences
  sort_relation_string +=
    "\t\"null orders\" : " + list_to_string(_null_orders.begin(), _null_orders.end()) + ",\n";
  // Data types
  sort_relation_string += "\t\"data types\" : " + gqe::list_to_string(data_types()) + ",\n";
  // Children
  sort_relation_string += "\t\"children\" : " + gqe::list_to_string(children_unsafe()) + "\n";
  sort_relation_string += "}}";
  return sort_relation_string;
}

filter_relation::filter_relation(std::shared_ptr<relation> input_relation,
                                 std::unique_ptr<expression> condition)
  : relation({std::move(input_relation)}), _condition(std::move(condition))
{
  assert(this->children_size() == 1);
  // Output data types are the same as input data types
  _data_types = this->children_unsafe()[0]->data_types();
}

std::string filter_relation::to_string() const
{
  std::string filter_relation_string = "{\"Filter\" : {\n";
  // Sort expressions
  filter_relation_string += "\t\"condition\" : \"" + _condition->to_string() + "\",\n";
  // Data types
  filter_relation_string += "\t\"data types\" : " + gqe::list_to_string(data_types()) + ",\n";
  // Children
  filter_relation_string += "\t\"children\" : " + gqe::list_to_string(children_unsafe()) + "\n";
  filter_relation_string += "}}";
  return filter_relation_string;
}

join_relation::join_relation(std::shared_ptr<relation> left,
                             std::shared_ptr<relation> right,
                             std::unique_ptr<expression> condition,
                             join_type_type join_type)
  : relation({std::move(left), std::move(right)}),
    _condition(std::move(condition)),
    _join_type(join_type)
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
  project_relation_str +=
    "\t\"output expressions\" : " + gqe::list_to_string(output_expressions_unsafe()) + ",\n";
  // Data types
  project_relation_str += "\t\"data types\" : " + gqe::list_to_string(data_types()) + ",\n";
  // Children
  project_relation_str += "\t\"children\" : " + gqe::list_to_string(children_unsafe()) + "\n";
  project_relation_str += "}}";
  return project_relation_str;
}

}  // namespace logical
}  // namespace gqe
