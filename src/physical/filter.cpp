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

#include <gqe/physical/filter.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {
std::vector<cudf::data_type> filter_relation::output_data_types() const
{
  // For each column materialized, the output data type is the same as the input data type.
  std::vector<cudf::data_type> input_data_types = this->children_unsafe()[0]->output_data_types();
  std::vector<cudf::data_type> data_types;
  data_types.reserve(_projection_indices.size());
  // It's possible for some handcoded query, they don't pass the column types
  // we need to deal with such case
  if (!input_data_types.empty()) {
    for (auto const& column_idx : _projection_indices)
      data_types.push_back(input_data_types[column_idx]);
  }
  return data_types;
}

std::string filter_relation::to_string() const
{
  std::string filter_relation_string = "{\"Filter\" : {\n";
  // Sort expressions
  filter_relation_string += "\t\"condition\" : \"" + _condition->to_string() + "\",\n";
  // Output data types
  filter_relation_string +=
    "\t\"output data types\" : " + logical::utility::list_to_string(output_data_types()) + ",\n";
  // Project indices
  filter_relation_string +=
    "\t\"project indices\" : " +
    logical::utility::list_to_string(_projection_indices.begin(), _projection_indices.end()) +
    ",\n";
  // Children
  filter_relation_string +=
    "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  filter_relation_string += "}}";
  return filter_relation_string;
}
}  // end of namespace physical
}  // end of namespace gqe
