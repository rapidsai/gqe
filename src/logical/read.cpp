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

}  // namespace logical
}  // namespace gqe