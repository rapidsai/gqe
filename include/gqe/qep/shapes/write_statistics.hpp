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
#include <gqe/types.hpp>

#include <optional>

namespace gqe::qep {

/**
 * @brief Build a terminal table-write statistics container `[write_statistics{statistics}]`.
 *
 * @param[in] statistics Statistics produced by the table write.
 * @return A state container holding one `write_statistics` slot.
 */
[[nodiscard]] state_container make_write_statistics_container(table_statistics statistics);

/**
 * @brief Detect the terminal table-write statistics shape and copy its value.
 *
 * @param[in] container The state container to inspect.
 *
 * @throws std::logic_error If `container` has exactly one null slot.
 *
 * @return The table statistics if `container` has the terminal write-statistics shape, otherwise
 *         `std::nullopt`.
 */
[[nodiscard]] std::optional<table_statistics> try_write_statistics(state_container_view container);

}  // namespace gqe::qep
