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

#include <gqe/logical/fetch.hpp>
#include <gqe/logical/utility.hpp>

#include <cudf/types.hpp>

#include <vector>

namespace gqe {
namespace logical {

fetch_relation::fetch_relation(std::shared_ptr<relation> input_relation,
                               int64_t offset,
                               int64_t count)
  : relation({std::move(input_relation)}, {}), _offset(offset), _count(count)
{
  assert(this->children_size() == 1);
}

std::string fetch_relation::to_string() const
{
  std::string fetch_relation_string = "{\"Fetch\" : {\n";
  // Offset
  fetch_relation_string += "\t\"offset\" : \"" + std::to_string(_offset) + "\",\n";
  // Count
  fetch_relation_string += "\t\"count\" : \"" + std::to_string(_count) + "\",\n";
  // Data types
  fetch_relation_string += "\t\"data types\" : " + utility::list_to_string(data_types()) + ",\n";
  // Children
  fetch_relation_string += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + "\n";
  fetch_relation_string += "}}";
  return fetch_relation_string;
}

bool fetch_relation::operator==(const relation& other) const
{
  auto this_type = this->type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }
  auto other_fetch_relation = dynamic_cast<const fetch_relation*>(&other);
  // Compare attributes
  if (this->offset() != other_fetch_relation->offset()) {
    utility::log_relation_comparison_message(this_type, "operator==(): offset mismatch");
    return false;
  }
  if (this->count() != other_fetch_relation->count()) {
    utility::log_relation_comparison_message(this_type, "operator==(): count mismatch");
    return false;
  }
  if (this->data_types() != other_fetch_relation->data_types()) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  // Compare members defined in base class
  return relation::compare_relation_members(other);
}

std::vector<cudf::data_type> fetch_relation::data_types() const
{
  // Output data types are the same as input data types
  return this->children_unsafe()[0]->data_types();
}

}  // namespace logical
}  // namespace gqe
