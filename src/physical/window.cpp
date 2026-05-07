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

#include <gqe/physical/window.hpp>

#include <gqe/logical/utility.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

namespace gqe {
namespace physical {

std::vector<cudf::data_type> window_relation::output_data_types() const
{
  auto data_types = this->children_unsafe()[0]->output_data_types();
  auto args       = arguments_unsafe();
  if (args.size() > 0) {
    cudf::data_type output_type =
      cudf::detail::target_type(args[0]->data_type(data_types), _aggr_func);
    data_types.push_back(output_type);
  } else {
    // If no arguments, we can assume the aggregation function is RANK
    if (_aggr_func != cudf::aggregation::RANK) {
      throw std::runtime_error("Only RANK is supported for window relations with no arguments.");
    }
    data_types.emplace_back(cudf::data_type(cudf::type_to_id<cudf::size_type>()));
  }
  return data_types;
}

std::string window_relation::to_string() const
{
  std::string window_relation_string = "{\"Window\" : {\n";
  window_relation_string +=
    "\t\"cudf::aggregation::Kind\" : " +
    std::to_string(static_cast<std::underlying_type<cudf::aggregation::Kind>::type>(aggr_func())) +
    ",\n";
  window_relation_string +=
    "\t\"arguments\" : " + logical::utility::list_to_string(arguments_unsafe()) + ",\n";
  window_relation_string +=
    "\t\"order_by\" : " + logical::utility::list_to_string(order_by_unsafe()) + ",\n";
  window_relation_string +=
    "\t\"partition_by\" : " + logical::utility::list_to_string(partition_by_unsafe()) + ",\n";
  window_relation_string +=
    "\t\"data types\" : " + logical::utility::list_to_string(output_data_types()) + ",\n";
  // Children
  window_relation_string +=
    "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  window_relation_string += "}}";
  return window_relation_string;
}

}  // end of namespace physical
}  // end of namespace gqe
