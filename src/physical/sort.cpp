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

#include <gqe/physical/sort.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {

std::vector<cudf::data_type> sort_relation_base::output_data_types() const
{
  // Output data types are the same as input data types
  return this->children_unsafe()[0]->output_data_types();
}

std::string sort_relation_base::print() const
{
  // Sort expressions
  std::string sort_string =
    "\t\"sort expressions\" : " + logical::utility::list_to_string(keys_unsafe()) + ",\n";
  // Column orders
  sort_string += "\t\"column orders\" : " +
                 logical::utility::list_to_string(_column_orders.begin(), _column_orders.end()) +
                 ",\n";
  // Null precedences
  sort_string +=
    "\t\"null orders\" : " +
    logical::utility::list_to_string(_null_precedences.begin(), _null_precedences.end()) + ",\n";
  // Output data types
  sort_string +=
    "\t\"output data types\" : " + logical::utility::list_to_string(output_data_types()) + ",\n";
  // Children
  sort_string += "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  return sort_string;
}

std::string concatenate_sort_relation::to_string() const
{
  std::string sort_relation_string = "{\"Sort\" : {\n";
  sort_relation_string += print();
  sort_relation_string += "}}";
  return sort_relation_string;
}

}  // namespace physical
}  // namespace gqe
