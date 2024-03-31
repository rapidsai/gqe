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
  // Compare children
  if (!gqe::utility::compare_pointer_vectors(this->children_unsafe(),
                                             other_sort_relation->children_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): children mismatch");
    return false;
  }
  // Compare subquery_relations
  if (!gqe::utility::compare_pointer_vectors(this->subqueries_unsafe(),
                                             other_sort_relation->subqueries_unsafe())) {
    utility::log_relation_comparison_message(this_type,
                                             "operator==(): subquery relations mismatch");
    return false;
  }
  return true;
}

std::vector<cudf::data_type> sort_relation::data_types() const
{
  // Output data types are the same as input data types
  return this->children_unsafe()[0]->data_types();
}

}  // namespace logical
}  // namespace gqe
