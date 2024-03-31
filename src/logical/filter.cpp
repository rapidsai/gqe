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

#include <gqe/logical/filter.hpp>
#include <gqe/logical/utility.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/types.hpp>

#include <vector>

namespace gqe {
namespace logical {

filter_relation::filter_relation(std::shared_ptr<relation> input_relation,
                                 std::vector<std::shared_ptr<relation>> subquery_relations,
                                 std::unique_ptr<expression> condition)
  : relation({std::move(input_relation)}, std::move(subquery_relations)),
    _condition(std::move(condition))
{
  assert(this->children_size() == 1);
}

std::string filter_relation::to_string() const
{
  std::string filter_relation_string = "{\"Filter\" : {\n";
  // Sort expressions
  filter_relation_string += "\t\"condition\" : \"" + _condition->to_string() + "\",\n";
  // Data types
  filter_relation_string += "\t\"data types\" : " + utility::list_to_string(data_types()) + ",\n";
  // Children
  filter_relation_string += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + "\n";
  filter_relation_string += "}}";
  return filter_relation_string;
}

bool filter_relation::operator==(const relation& other) const
{
  auto this_type = this->type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }
  auto other_filter_relation = dynamic_cast<const filter_relation*>(&other);
  // Compare attributes
  if (*this->condition() != *other_filter_relation->condition()) {
    utility::log_relation_comparison_message(this_type, "operator==(): condition mismatch");
    return false;
  }
  if (this->data_types() != other_filter_relation->data_types()) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  // Compare children
  if (!gqe::utility::compare_pointer_vectors(this->children_unsafe(),
                                             other_filter_relation->children_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): children mismatch");
    return false;
  }
  // Compare subquery_relations
  if (!gqe::utility::compare_pointer_vectors(this->subqueries_unsafe(),
                                             other_filter_relation->subqueries_unsafe())) {
    utility::log_relation_comparison_message(this_type,
                                             "operator==(): subquery relations mismatch");
    return false;
  }

  return true;
}

std::vector<cudf::data_type> filter_relation::data_types() const
{
  // Output data types are the same as input data types
  return this->children_unsafe()[0]->data_types();
}

}  // namespace logical
}  // namespace gqe
