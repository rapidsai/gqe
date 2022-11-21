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
  // Output data types are the same as input data types
  _data_types = this->children_unsafe()[0]->data_types();
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

}  // namespace logical
}  // namespace gqe