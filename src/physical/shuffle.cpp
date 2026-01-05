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

#include <gqe/physical/shuffle.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {
std::vector<cudf::data_type> shuffle_relation::output_data_types() const
{
  // The output data type is the same as the input data type.
  return this->children_unsafe()[0]->output_data_types();
}

std::string shuffle_relation::to_string() const
{
  std::string shuffle_relation_string = "{\"Shuffle\" : {\n";
  shuffle_relation_string +=
    "\t\"shuffle columns\" : " + logical::utility::list_to_string(shuffle_cols_unsafe()) + ",\n";
  // Children
  shuffle_relation_string +=
    "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  shuffle_relation_string += "}}";
  return shuffle_relation_string;
}

}  // namespace physical
}  // namespace gqe
