/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/logical/top_k.hpp>

#include <gqe/logical/utility.hpp>
#include <gqe/utility/error.hpp>

#include <sstream>
#include <stdexcept>
#include <utility>

namespace gqe::logical {

top_k_relation::top_k_relation(std::shared_ptr<relation> input_relation,
                               std::vector<std::shared_ptr<relation>> subquery_relations,
                               std::unique_ptr<expression> key,
                               cudf::order column_order,
                               cudf::null_order null_precedence,
                               int64_t offset,
                               int64_t count)
  : relation({std::move(input_relation)}, std::move(subquery_relations)),
    _key(std::move(key)),
    _column_order(column_order),
    _null_precedence(null_precedence),
    _offset(offset),
    _count(count)
{
  GQE_EXPECTS(children_unsafe().front() != nullptr,
              "logical::top_k_relation: input must not be null",
              std::invalid_argument);
  GQE_EXPECTS(
    _key != nullptr, "logical::top_k_relation: key must not be null", std::invalid_argument);
  GQE_EXPECTS(
    _offset >= 0, "logical::top_k_relation: offset must be non-negative", std::invalid_argument);
  GQE_EXPECTS(
    _count >= 0, "logical::top_k_relation: count must be non-negative", std::invalid_argument);
}

std::vector<cudf::data_type> top_k_relation::data_types() const
{
  return children_unsafe().front()->data_types();
}

std::string top_k_relation::to_string() const
{
  using utility::operator<<;
  std::ostringstream column_order;
  column_order << _column_order;
  std::ostringstream null_order;
  null_order << _null_precedence;

  std::string result = "{\"TopK\" : {\n";
  result += "\t\"key\" : \"" + _key->to_string() + "\",\n";
  result += "\t\"column order\" : \"" + column_order.str() + "\",\n";
  result += "\t\"null order\" : \"" + null_order.str() + "\",\n";
  result += "\t\"offset\" : \"" + std::to_string(_offset) + "\",\n";
  result += "\t\"count\" : \"" + std::to_string(_count) + "\",\n";
  result += "\t\"data types\" : " + utility::list_to_string(data_types()) + ",\n";
  result += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + "\n";
  result += "}}";
  return result;
}

bool top_k_relation::operator==(const relation& other) const
{
  auto const this_type = type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }

  auto const* other_top_k = dynamic_cast<top_k_relation const*>(&other);
  if (*key_unsafe() != *other_top_k->key_unsafe()) {
    utility::log_relation_comparison_message(this_type, "operator==(): key mismatch");
    return false;
  }
  if (column_order() != other_top_k->column_order()) {
    utility::log_relation_comparison_message(this_type, "operator==(): column order mismatch");
    return false;
  }
  if (null_precedence() != other_top_k->null_precedence()) {
    utility::log_relation_comparison_message(this_type, "operator==(): null order mismatch");
    return false;
  }
  if (offset() != other_top_k->offset()) {
    utility::log_relation_comparison_message(this_type, "operator==(): offset mismatch");
    return false;
  }
  if (count() != other_top_k->count()) {
    utility::log_relation_comparison_message(this_type, "operator==(): count mismatch");
    return false;
  }
  return compare_relation_members(other);
}

}  // namespace gqe::logical
