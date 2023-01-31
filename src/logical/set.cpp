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

}  // namespace logical
}  // namespace gqe
