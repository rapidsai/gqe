/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/expression/if_then_else.hpp>

#include <gqe/utility/helpers.hpp>

namespace gqe {

bool if_then_else_expression::operator==(const expression& other) const
{
  if (this->type() != other.type()) { return false; }
  auto other_if_then_expr = dynamic_cast<const if_then_else_expression&>(other);
  // Recuresively compare children
  return utility::compare_pointer_vectors(this->children(), other_if_then_expr.children());
}

}  // namespace gqe
