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

#include <gqe/physical/shuffle.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {
std::vector<cudf::data_type> shuffle_relation::output_data_types() const
{
  // The output data type is the same as the input data type.
  return this->children_unsafe()[0]->output_data_types();
}

std::string shuffle_relation::to_string() const
{
  std::string shuffle_relation_string = "{\"Shuffle\" : {\n";
  shuffle_relation_string +=
    "\t\"shuffle columns\" : " + logical::utility::list_to_string(shuffle_cols_unsafe()) + ",\n";
  // Children
  shuffle_relation_string +=
    "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  shuffle_relation_string += "}}";
  return shuffle_relation_string;
}

}  // namespace physical
}  // namespace gqe
