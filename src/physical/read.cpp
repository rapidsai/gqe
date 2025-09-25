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
