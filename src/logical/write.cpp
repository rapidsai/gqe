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

#include <gqe/logical/write.hpp>

#include <gqe/logical/utility.hpp>
#include <gqe/utility/helpers.hpp>

#include <sstream>

namespace gqe {

namespace logical {

write_relation::write_relation(std::shared_ptr<relation> input_relation,
                               std::vector<std::string> column_names,
                               std::vector<cudf::data_type> column_types,
                               std::string table_name)
  : relation({std::move(input_relation)}, {}),
    _column_names(std::move(column_names)),
    _column_types(std::move(column_types)),
    _table_name(std::move(table_name))
{
}

relation::relation_type write_relation::type() const noexcept { return relation_type::write; }

std::vector<cudf::data_type> write_relation::data_types() const
{
  auto const child_relation = children_unsafe();
  assert(child_relation.size() == 1);
  auto const child_types = child_relation[0]->data_types();

  assert(_column_types == child_types);

  return _column_types;
}

std::string write_relation::to_string() const
{
  std::ostringstream ss;
  ss << "{\"Write\" : {\n"
     << "\t\"children\" : " << utility::list_to_string(children_unsafe()) << "\n"
     << "}}";

  return ss.str();
}

std::string write_relation::table_name() const { return _table_name; }

std::vector<std::string> write_relation::column_names() const { return _column_names; }

bool write_relation::operator==(const relation& other) const
{
  auto this_type = this->type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }
  auto other_write_relation = dynamic_cast<const write_relation*>(&other);
  // Compare attributes
  if (this->column_names() != other_write_relation->column_names()) {
    utility::log_relation_comparison_message(this_type, "operator==(): column names mismatch");
    return false;
  }
  if (this->table_name() != other_write_relation->table_name()) {
    utility::log_relation_comparison_message(this_type, "operator==(): table names mismatch");
    return false;
  }
  if (this->data_types() != other_write_relation->data_types()) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  // Compare members defined in base class
  return relation::compare_relation_members(other);
}

}  // namespace logical
}  // namespace gqe
