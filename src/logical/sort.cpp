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

#include <gqe/logical/sort.hpp>
#include <gqe/logical/utility.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/types.hpp>

#include <vector>

namespace gqe {
namespace logical {

sort_relation::sort_relation(std::shared_ptr<relation> input_relation,
                             std::vector<std::shared_ptr<relation>> subquery_relations,
                             std::vector<cudf::order> column_orders,
                             std::vector<cudf::null_order> null_precedences,
                             std::vector<std::unique_ptr<expression>> expressions)
  : relation({std::move(input_relation)}, std::move(subquery_relations)),
    _expressions(std::move(expressions)),
    _column_orders(std::move(column_orders)),
    _null_orders(std::move(null_precedences))
{
  assert(this->children_size() == 1);
}

std::string sort_relation::to_string() const
{
  std::string sort_relation_string = "{\"Sort\" : {\n";
  // Sort expressions
  sort_relation_string +=
    "\t\"sort expressions\" : " + utility::list_to_string(expressions_unsafe()) + ",\n";
  // Column orders
  sort_relation_string += "\t\"column orders\" : " +
                          utility::list_to_string(_column_orders.begin(), _column_orders.end()) +
                          ",\n";
  // Null precedences
  sort_relation_string +=
    "\t\"null orders\" : " + utility::list_to_string(_null_orders.begin(), _null_orders.end()) +
    ",\n";
  // Data types
  sort_relation_string += "\t\"data types\" : " + utility::list_to_string(data_types()) + ",\n";
  // Children
  sort_relation_string += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + "\n";
  sort_relation_string += "}}";
  return sort_relation_string;
}

bool sort_relation::operator==(const relation& other) const
{
  auto this_type = this->type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }
  auto other_sort_relation = dynamic_cast<const sort_relation*>(&other);
  // Compare attributes
  if (this->column_orders() != other_sort_relation->column_orders()) {
    utility::log_relation_comparison_message(this_type, "operator==(): column orders mismatch");
    return false;
  }
  if (this->null_orders() != other_sort_relation->null_orders()) {
    utility::log_relation_comparison_message(this_type, "operator==(): null orders mismatch");
    return false;
  }
  if (this->data_types() != other_sort_relation->data_types()) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  if (!gqe::utility::compare_pointer_vectors(this->expressions_unsafe(),
                                             other_sort_relation->expressions_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): expressions mismatch");
    return false;
  }
  // Compare members defined in base class
  return relation::compare_relation_members(other);
}

std::vector<cudf::data_type> sort_relation::data_types() const
{
  // Output data types are the same as input data types
  return this->children_unsafe()[0]->data_types();
}

}  // namespace logical
}  // namespace gqe
