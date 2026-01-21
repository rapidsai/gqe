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
    for (auto const& column_idx : _projection_indices) {
      // this is possible when the input is from a user_defined relation
      // which doesn't have any output data types, we have to throw exception here
      if (column_idx >= static_cast<cudf::size_type>(input_data_types.size())) {
        throw std::runtime_error("Filter relation: input data types is missing for column index " +
                                 std::to_string(column_idx));
      } else {
        data_types.push_back(input_data_types[column_idx]);
      }
    }
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
