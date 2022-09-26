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

#include <gqe/logical/relation.hpp>

namespace gqe {
namespace logical {

read_relation::read_relation(std::vector<std::string> column_names,
                             std::vector<cudf::data_type> column_types,
                             std::string table_name)
  : relation({}),
    _column_names(std::move(column_names)),
    _table_name(std::move(table_name)),
    _data_types(std::move(column_types))
{
}

std::string read_relation::to_string() const
{
  std::string read_relation_str = "Relation: Read\n\tread location: ";
  read_relation_str += this->table_name();
  auto data_types = this->data_types();
  read_relation_str += "\n\tdata types:";
  for (cudf::data_type dt : data_types) {
    read_relation_str += "\t\t" + cudf::type_dispatcher(dt, cudf::type_to_name{}) + "\n";
  }
  return read_relation_str;
}

project_relation::project_relation(std::shared_ptr<relation> child,
                                   std::vector<std::shared_ptr<expression>> output_expressions)
  : relation({child}), output_expressions(std::move(output_expressions))
{
}

[[nodiscard]] std::vector<cudf::data_type> project_relation::data_types() const
{
  assert(this->children_size() ==
         1);  // There should only be one input relation to a projection relation
  if (!this->_data_types.has_value()) {
    this->_data_types = std::vector<cudf::data_type>();
    for (auto const& output_expression : output_expressions) {
      auto child_rel        = this->children_unsafe()[0];
      auto child_rel_dtypes = child_rel->data_types();
      auto exp_data_type    = output_expression->data_type(child_rel_dtypes);
      this->_data_types.value().push_back(exp_data_type);
    }
  }
  return this->_data_types.value();
}

std::string project_relation::to_string() const
{
  std::string project_relation_str = "Relation: Project\n\toutput expressions:\n";
  // Add information about output expressions
  for (auto expression : output_expressions) {
    std::string exp_str = expression->to_string();
    project_relation_str += "\t\t" + std::regex_replace(exp_str, std::regex("\n"), "\n\t") + "\n";
  }
  // Add information about output data types
  auto data_types = this->data_types();
  project_relation_str += "\tdata types:\n";
  for (cudf::data_type dt : data_types) {
    project_relation_str += "\t\t" + cudf::type_dispatcher(dt, cudf::type_to_name{}) + "\n";
  }
  return project_relation_str;
}

}  // namespace logical
}  // namespace gqe