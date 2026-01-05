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
  read_relation_str += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + ",\n";
  // Relation traits
  read_relation_str += "\t\"traits\" : {" + relation_traits().to_string() + "}\n";
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
  // Compare members defined in base class
  return relation::compare_relation_members(other);
}

}  // namespace logical
}  // namespace gqe
