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
  // Compare members defined in base class
  return relation::compare_relation_members(other);
}

}  // namespace logical
}  // namespace gqe
