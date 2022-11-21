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

namespace gqe {
namespace logical {

filter_relation::filter_relation(std::shared_ptr<relation> input_relation,
                                 std::vector<std::shared_ptr<relation>> subquery_relations,
                                 std::unique_ptr<expression> condition)
  : relation({std::move(input_relation)}, std::move(subquery_relations)),
    _condition(std::move(condition))
{
  assert(this->children_size() == 1);
  // Output data types are the same as input data types
  _data_types = this->children_unsafe()[0]->data_types();
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

}  // namespace logical
}  // namespace gqe