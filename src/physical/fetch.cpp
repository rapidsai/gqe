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

#include <gqe/physical/fetch.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {
std::vector<cudf::data_type> fetch_relation::output_data_types() const
{
  // Output data types are the same as input data types
  return this->children_unsafe()[0]->output_data_types();
}

std::string fetch_relation::to_string() const
{
  std::string fetch_relation_string = "{\"Fetch\" : {\n";
  // Offset
  fetch_relation_string += "\t\"offset\" : \"" + std::to_string(_offset) + "\",\n";
  // Count
  fetch_relation_string += "\t\"count\" : \"" + std::to_string(_count) + "\",\n";
  // Output data types
  fetch_relation_string +=
    "\t\"output data types\" : " + logical::utility::list_to_string(output_data_types()) + ",\n";
  // Children
  fetch_relation_string +=
    "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  fetch_relation_string += "}}";
  return fetch_relation_string;
}

}  // end of namespace physical
}  // end of namespace gqe
