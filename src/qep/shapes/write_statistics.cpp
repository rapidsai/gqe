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

#include <gqe/qep/shapes/write_statistics.hpp>

#include <gqe/utility/error.hpp>

#include <optional>
#include <utility>
#include <variant>

namespace gqe::qep {

state_container make_write_statistics_container(table_statistics statistics)
{
  return state_container_builder()
    .add_state(state_kind::write_statistics{std::move(statistics)})
    .build();
}

std::optional<table_statistics> try_write_statistics(state_container_view container)
{
  if (container.size() != 1) { return std::nullopt; }

  GQE_EXPECTS(container.front() != nullptr, "try_write_statistics: null state slot");

  if (auto const* statistics = std::get_if<state_kind::write_statistics>(container.front().get())) {
    return statistics->value;
  }

  return std::nullopt;
}

}  // namespace gqe::qep
