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

#pragma once

#include <gqe/qep/state.hpp>

#include <cudf/types.hpp>

#include <optional>

namespace gqe {
namespace qep {

/**
 * @brief Build a count-only state container `[row_count{N}]`.
 *
 * Used by producers that have a row count but no columns (e.g. the chunk emitted for
 * `SELECT COUNT(*) FROM t`). Terminal — produces a complete container, not a starting point
 * for further `add_state` calls.
 *
 * @param[in] row_count The row count.
 *
 * @return A state container holding a single `row_count` slot.
 */
[[nodiscard]] state_container make_row_count_container(cudf::size_type row_count);

/**
 * @brief Detect the count-only shape and read its row count.
 *
 * A count-only container is a lone `row_count` slot — the shape `make_row_count_container`
 * produces.
 *
 * @param[in] container The state container to inspect.
 *
 * @return The row count if `container` is count-only, else `std::nullopt`.
 */
[[nodiscard]] std::optional<cudf::size_type> try_row_count(state_container_view container);

}  // namespace qep
}  // namespace gqe
