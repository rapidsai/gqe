/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/physical/set.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {

std::vector<cudf::data_type> union_all_relation::output_data_types() const
{
  auto const child_relations = children_unsafe();
  assert(child_relations.size() == 2);

  auto const left_types                   = child_relations[0]->output_data_types();
  [[maybe_unused]] auto const right_types = child_relations[1]->output_data_types();

  assert(left_types == right_types);

  return left_types;
}

std::string union_all_relation::to_string() const
{
  std::string set_relation_string = "{\"Set\" : {\n";
  set_relation_string += "\t\"operation\" : \"UNION ALL\",\n";
  set_relation_string +=
    "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  set_relation_string += "}}";
  return set_relation_string;
}

}  // namespace physical
}  // namespace gqe
