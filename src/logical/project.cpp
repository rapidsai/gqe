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

#include <gqe/logical/project.hpp>

#include <gqe/logical/utility.hpp>
#include <gqe/utility/helpers.hpp>

namespace gqe {
namespace logical {

project_relation::project_relation(std::shared_ptr<relation> child,
                                   std::vector<std::shared_ptr<relation>> subquery_relations,
                                   std::vector<std::unique_ptr<expression>> output_expressions)
  : relation({std::move(child)}, std::move(subquery_relations)),
    _output_expressions(std::move(output_expressions))
{
}

[[nodiscard]] std::vector<cudf::data_type> project_relation::data_types() const
{
  assert(this->children_size() ==
         1);  // There should only be one input relation to a projection relation
  std::vector<cudf::data_type> data_types;
  for (auto const& output_expression : _output_expressions) {
    auto child_rel        = this->children_unsafe()[0];
    auto child_rel_dtypes = child_rel->data_types();
    auto exp_data_types   = output_expression->data_type(child_rel_dtypes);
    data_types.push_back(exp_data_types);
  }
  return data_types;
}

std::string project_relation::to_string() const
{
  std::string project_relation_str = "{\"Project\" : {\n";
  // Output expressions
  project_relation_str +=
    "\t\"output expressions\" : " + utility::list_to_string(output_expressions_unsafe()) + ",\n";
  // Data types
  project_relation_str += "\t\"data types\" : " + utility::list_to_string(data_types()) + ",\n";
  // Children
  project_relation_str += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + "\n";
  project_relation_str += "}}";
  return project_relation_str;
}

bool project_relation::operator==(const relation& other) const
{
  auto this_type = this->type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }
  auto other_project_relation = dynamic_cast<const project_relation*>(&other);
  // Compare attributes
  if (!gqe::utility::compare_pointer_vectors(this->output_expressions_unsafe(),
                                             other_project_relation->output_expressions_unsafe())) {
    utility::log_relation_comparison_message(this_type,
                                             "operator==(): output expressions mismatch");
    return false;
  }
  if (this->data_types() != other_project_relation->data_types()) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  // Compare children
  if (!gqe::utility::compare_pointer_vectors(this->children_unsafe(),
                                             other_project_relation->children_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): children mismatch");
    return false;
  }
  // Compare subquery_relations
  if (!gqe::utility::compare_pointer_vectors(this->subqueries_unsafe(),
                                             other_project_relation->subqueries_unsafe())) {
    utility::log_relation_comparison_message(this_type,
                                             "operator==(): subquery relations mismatch");
    return false;
  }
  return true;
}

}  // namespace logical
}  // namespace gqe
