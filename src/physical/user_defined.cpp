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

#include <gqe/physical/user_defined.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {

std::string user_defined_relation::to_string() const
{
  std::string udr_string = "{\"User defined relation\" : {\n";
  udr_string += "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  udr_string += "}}";
  return udr_string;
}

}  // end of namespace physical
}  // end of namespace gqe
