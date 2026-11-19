/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/physical/top_k.hpp>

#include <gqe/logical/utility.hpp>
#include <gqe/utility/error.hpp>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gqe::physical {

top_k_relation::top_k_relation(std::shared_ptr<relation> input,
                               cudf::size_type key_index,
                               cudf::order column_order,
                               cudf::null_order null_precedence,
                               int64_t offset,
                               int64_t count)
  : relation({std::move(input)}, {}),
    _key_index{key_index},
    _column_order{column_order},
    _null_precedence{null_precedence},
    _offset{offset},
    _count{count}
{
  GQE_EXPECTS(children_unsafe().front() != nullptr,
              "physical::top_k_relation: input must not be null",
              std::invalid_argument);
  GQE_EXPECTS(_key_index >= 0,
              "physical::top_k_relation: key_index must be non-negative",
              std::invalid_argument);
  GQE_EXPECTS(
    _offset >= 0, "physical::top_k_relation: offset must be non-negative", std::invalid_argument);
  GQE_EXPECTS(
    _count >= 0, "physical::top_k_relation: count must be non-negative", std::invalid_argument);
  auto constexpr max_size = static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max());
  GQE_EXPECTS(_offset <= max_size,
              "physical::top_k_relation: offset exceeds cudf::size_type",
              std::overflow_error);
  GQE_EXPECTS(_count <= max_size,
              "physical::top_k_relation: count exceeds cudf::size_type",
              std::overflow_error);
  GQE_EXPECTS(_offset <= max_size - _count,
              "physical::top_k_relation: offset + count overflows cudf::size_type",
              std::overflow_error);
  GQE_EXPECTS(_key_index <
                static_cast<cudf::size_type>(children_unsafe().front()->output_data_types().size()),
              "physical::top_k_relation: key_index is outside the input columns",
              std::out_of_range);
}

std::vector<cudf::data_type> top_k_relation::output_data_types() const
{
  return children_unsafe().front()->output_data_types();
}

std::string top_k_relation::to_string() const
{
  using logical::utility::operator<<;
  std::ostringstream column_order;
  column_order << _column_order;
  std::ostringstream null_order;
  null_order << _null_precedence;

  std::string result = "{\"TopK\" : {\n";
  result += "\t\"key index\" : \"" + std::to_string(_key_index) + "\",\n";
  result += "\t\"column order\" : \"" + column_order.str() + "\",\n";
  result += "\t\"null order\" : \"" + null_order.str() + "\",\n";
  result += "\t\"offset\" : \"" + std::to_string(_offset) + "\",\n";
  result += "\t\"count\" : \"" + std::to_string(_count) + "\",\n";
  result +=
    "\t\"output data types\" : " + logical::utility::list_to_string(output_data_types()) + ",\n";
  result += "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  result += "}}";
  return result;
}

}  // namespace gqe::physical
