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

#include <gqe/logical/project.hpp>
#include <gqe/logical/utility.hpp>

namespace gqe {
namespace logical {

project_relation::project_relation(std::shared_ptr<relation> child,
                                   std::vector<std::shared_ptr<relation>> subquery_relations,
                                   std::vector<std::unique_ptr<expression>> output_expressions)
  : relation({std::move(child)}, std::move(subquery_relations)),
    _output_expressions(std::move(output_expressions))
{
}

void project_relation::_init_data_types() const
{
  assert(this->children_size() ==
         1);  // There should only be one input relation to a projection relation
  this->_data_types = std::vector<cudf::data_type>();
  for (auto const& output_expression : _output_expressions) {
    auto child_rel        = this->children_unsafe()[0];
    auto child_rel_dtypes = child_rel->data_types();
    auto exp_data_types   = output_expression->data_type(child_rel_dtypes);
    this->_data_types.value().push_back(exp_data_types);
  }
}

[[nodiscard]] std::vector<cudf::data_type> project_relation::data_types() const
{
  if (!this->_data_types.has_value()) { this->_init_data_types(); }
  return this->_data_types.value();
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

}  // namespace logical
}  // namespace gqe