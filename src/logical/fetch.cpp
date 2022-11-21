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

#include <gqe/logical/fetch.hpp>
#include <gqe/logical/utility.hpp>

namespace gqe {
namespace logical {

fetch_relation::fetch_relation(std::shared_ptr<relation> input_relation,
                               int64_t offset,
                               int64_t count)
  : relation({std::move(input_relation)}, {}), _offset(offset), _count(count)
{
  assert(this->children_size() == 1);
  // Output data types are the same as input data types
  _data_types = this->children_unsafe()[0]->data_types();
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
}  // namespace logical
}  // namespace gqe