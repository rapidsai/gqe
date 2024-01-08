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

#include <gqe/logical/read.hpp>
#include <gqe/logical/utility.hpp>
#include <gqe/utility/helpers.hpp>

namespace gqe {
namespace logical {

read_relation::read_relation(std::vector<std::shared_ptr<relation>> subquery_relations,
                             std::vector<std::string> column_names,
                             std::vector<cudf::data_type> column_types,
                             std::string table_name,
                             std::unique_ptr<expression> partial_filter)
  : relation({}, std::move(subquery_relations)),
    _column_names(std::move(column_names)),
    _data_types(std::move(column_types)),
    _table_name(std::move(table_name)),
    _partial_filter(std::move(partial_filter))
{
}

std::string read_relation::to_string() const
{
  std::string read_relation_str = "{\"Read\" : {\n";
  // Table name
  read_relation_str += "\t\"table name\" : \"" + this->table_name() + "\",\n";
  // Data types
  read_relation_str += "\t\"data types\" : " + utility::list_to_string(data_types()) + ",\n";
  // Column names
  read_relation_str +=
    "\t\"column names\" : " + utility::list_to_string(_column_names.begin(), _column_names.end()) +
    ",\n";
  // Data types
  std::string partial_filter_str =
    partial_filter_unsafe() ? partial_filter_unsafe()->to_string() : "NULL";
  read_relation_str += "\t\"partial filter\" : \"" + partial_filter_str + "\",\n";
  // Children
  read_relation_str += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + "\n";
  read_relation_str += "}}";
  return read_relation_str;
}

bool read_relation::operator==(const relation& other) const
{
  auto this_type = this->type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }
  auto other_read_relation = dynamic_cast<const read_relation*>(&other);
  // Compare attributes
  if (this->table_name() != other_read_relation->table_name()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==(): table name mismatch: " + this->table_name() + " vs. " +
        other_read_relation->table_name());
    return false;
  }
  if (this->column_names() != other_read_relation->column_names()) {
    utility::log_relation_comparison_message(this_type, "operator==(): column names mismatch");
    return false;
  }
  if (this->data_types() != other_read_relation->data_types()) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  if ((this->partial_filter_unsafe() && other_read_relation->partial_filter_unsafe()) &&
      !(*this->partial_filter_unsafe() == *other_read_relation->partial_filter_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  // Compare subquery_relations
  if (!gqe::utility::compare_pointer_vectors(this->subqueries_unsafe(),
                                             other_read_relation->subqueries_unsafe())) {
    utility::log_relation_comparison_message(this_type,
                                             "operator==(): subquery relations mismatch");
    return false;
  }
  return true;
}

}  // namespace logical
}  // namespace gqe
