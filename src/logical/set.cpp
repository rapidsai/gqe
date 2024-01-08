/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/logical/set.hpp>
#include <gqe/logical/utility.hpp>
#include <gqe/utility/helpers.hpp>

namespace gqe {
namespace logical {

set_relation::set_relation(std::shared_ptr<relation> lhs,
                           std::shared_ptr<relation> rhs,
                           set_operator_type op)
  : relation({std::move(lhs), std::move(rhs)}, {}), _op(op)
{
}

std::vector<cudf::data_type> set_relation::data_types() const
{
  auto const child_relations = children_unsafe();
  assert(child_relations.size() == 2);

  auto const left_types                   = child_relations[0]->data_types();
  [[maybe_unused]] auto const right_types = child_relations[1]->data_types();

  assert(left_types == right_types);

  return left_types;
}

namespace {

std::string set_op_to_string(set_relation::set_operator_type op)
{
  switch (op) {
    case set_relation::set_union: return "UNION";
    case set_relation::set_union_all: return "UNION ALL";
    case set_relation::set_intersect: return "INTERSECT";
    case set_relation::set_minus: return "MINUS";
    default: throw std::logic_error("Cannot convert set operation to string");
  }
}

}  // namespace

std::string set_relation::to_string() const
{
  std::string set_relation_string = "{\"Set\" : {\n";
  set_relation_string += "\t\"operation\" : \"" + set_op_to_string(_op) + "\",\n";
  set_relation_string += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + "\n";
  set_relation_string += "}}";
  return set_relation_string;
}

bool set_relation::operator==(const relation& other) const
{
  auto this_type = this->type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }
  auto other_set_relation = dynamic_cast<const set_relation*>(&other);
  // Compare attributes
  if (this->set_operator() != other_set_relation->set_operator()) {
    utility::log_relation_comparison_message(this_type, "operator==(): operator mismatch");
    return false;
  }
  if (this->data_types() != other_set_relation->data_types()) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  // Compare children
  if (!gqe::utility::compare_pointer_vectors(this->children_unsafe(),
                                             other_set_relation->children_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): children mismatch");
    return false;
  }

  return true;
}

}  // namespace logical
}  // namespace gqe
