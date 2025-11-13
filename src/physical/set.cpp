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

#include <gqe/physical/set.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {

std::vector<cudf::data_type> union_all_relation::output_data_types() const
{
  auto const child_relations = children_unsafe();
  assert(child_relations.size() == 2);

  auto const left_types                   = child_relations[0]->output_data_types();
  [[maybe_unused]] auto const right_types = child_relations[1]->output_data_types();

  assert(left_types == right_types);

  return left_types;
}

std::string union_all_relation::to_string() const
{
  std::string set_relation_string = "{\"Set\" : {\n";
  set_relation_string += "\t\"operation\" : \"UNION ALL\",\n";
  set_relation_string +=
    "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  set_relation_string += "}}";
  return set_relation_string;
}

}  // namespace physical
}  // namespace gqe
