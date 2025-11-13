/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/logical/utility.hpp>
#include <gqe/physical/read.hpp>

namespace gqe {
namespace physical {

std::string read_relation::to_string() const
{
  std::string read_relation_str = "{\"Read\" : {\n";
  // Table name
  read_relation_str += "\t\"table name\" : \"" + this->table_name() + "\",\n";
  // Output data types
  read_relation_str +=
    "\t\"output data types\" : " + logical::utility::list_to_string(output_data_types()) + ",\n";
  // Column names
  read_relation_str +=
    "\t\"column names\" : " +
    logical::utility::list_to_string(_column_names.begin(), _column_names.end()) + ",\n";
  // Partial filter
  std::string partial_filter_str =
    partial_filter_unsafe() ? partial_filter_unsafe()->to_string() : "NULL";
  read_relation_str += "\t\"partial filter\" : \"" + partial_filter_str + "\",\n";
  // Children
  read_relation_str += "\t\"children\" : " + logical::utility::list_to_string(children_unsafe());
  read_relation_str += "}}";
  return read_relation_str;
}

}  // end of namespace physical
}  // end of namespace gqe
