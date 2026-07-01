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

#include <gqe/qep/shapes/row_count.hpp>

#include <gqe/utility/error.hpp>

#include <optional>
#include <variant>

namespace gqe {
namespace qep {

state_container make_row_count_container(cudf::size_type row_count)
{
  return state_container_builder().add_state(state_kind::row_count{row_count}).build();
}

std::optional<cudf::size_type> try_row_count(state_container_view container)
{
  // Count-only: a lone `row_count` slot (what make_row_count_container produces).
  if (container.size() != 1) { return std::nullopt; }

  GQE_EXPECTS(container.front() != nullptr, "try_row_count: null state slot");

  if (auto const* rc = std::get_if<state_kind::row_count>(container.front().get())) {
    return rc->value;
  }

  return std::nullopt;
}

}  // namespace qep
}  // namespace gqe
