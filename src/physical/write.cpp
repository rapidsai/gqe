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

#include <gqe/physical/write.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {

std::vector<cudf::data_type> write_relation::output_data_types() const
{
  auto const child_relation = children_unsafe();
  assert(child_relation.size() == 1);
  return child_relation[0]->output_data_types();
}

std::string write_relation::to_string() const
{
  std::ostringstream ss;
  ss << "{\"Write\" : {\n"
     << "\t\"children\" : " << logical::utility::list_to_string(children_unsafe()) << "\n"
     << "}}";

  return ss.str();
}

}  // end of namespace physical
}  // end of namespace gqe
