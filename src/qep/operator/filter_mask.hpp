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

#include <gqe/query_context.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace gqe::qep::detail {
/**
 * @brief Combine two selection masks into an owned, non-nullable BOOL8 mask.
 *
 * Only valid true entries in both inputs survive. False and null both reject rows.
 * Inputs are borrowed and remain unchanged. The caller retains them until stream work completes.
 * Work and output storage are O(n), where n is the common input size, including empty inputs.
 *
 * @param previous The upstream selection mask, possibly sliced or nullable.
 * @param predicate The evaluated predicate, possibly sliced or nullable.
 * @param stream The stream for allocation and execution.
 * @param mr The resource for all output and temporary allocations.
 * @return The non-nullable selection mask.
 * @throws std::invalid_argument If inputs have different sizes or are not BOOL8.
 */
[[nodiscard]] std::unique_ptr<cudf::column> compose_filter_masks(cudf::column_view previous,
                                                                 cudf::column_view predicate,
                                                                 query_stream const& stream_handle,
                                                                 rmm::device_async_resource_ref mr);
}  // namespace gqe::qep::detail
