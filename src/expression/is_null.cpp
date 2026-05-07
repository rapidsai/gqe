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

#include <gqe/expression/is_null.hpp>

#include <cassert>
namespace gqe {
bool is_null_expression::operator==(const expression& other) const
{
  if (auto other_is_null_expr = dynamic_cast<const is_null_expression*>(&other);
      other_is_null_expr != nullptr) {
    auto this_children  = this->children();
    auto other_children = other_is_null_expr->children();
    assert(this_children.size() == 1 && other_children.size() == 1);
    return *this_children[0] == *other_children[0];
  }
  return false;
}
}  // namespace gqe
