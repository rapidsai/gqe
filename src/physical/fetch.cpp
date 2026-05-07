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

#include <gqe/physical/fetch.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {
std::vector<cudf::data_type> fetch_relation::output_data_types() const
{
  // Output data types are the same as input data types
  return this->children_unsafe()[0]->output_data_types();
}

std::string fetch_relation::to_string() const
{
  std::string fetch_relation_string = "{\"Fetch\" : {\n";
  // Offset
  fetch_relation_string += "\t\"offset\" : \"" + std::to_string(_offset) + "\",\n";
  // Count
  fetch_relation_string += "\t\"count\" : \"" + std::to_string(_count) + "\",\n";
  // Output data types
  fetch_relation_string +=
    "\t\"output data types\" : " + logical::utility::list_to_string(output_data_types()) + ",\n";
  // Children
  fetch_relation_string +=
    "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  fetch_relation_string += "}}";
  return fetch_relation_string;
}

}  // end of namespace physical
}  // end of namespace gqe
