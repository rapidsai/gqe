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

#include <gqe/qep/traits/splittable_source_trait.hpp>
#include <gqe/utility/error.hpp>

#include <stdexcept>
#include <vector>

namespace gqe::storage::detail {

/**
 * @brief Copy the descriptors in a validated source work range.
 *
 * @tparam T Copyable source descriptor, such as a file path or row-group pointer.
 * @param[in] work Stable sequence of source work descriptors.
 * @param[in] range Half-open range within `work`.
 * @return The selected descriptors in their original order.
 * @throws std::invalid_argument If the range is reversed or outside `work`.
 */
template <typename T>
[[nodiscard]] std::vector<T> select_source_work(std::vector<T> const& work,
                                                qep::source_work_range range)
{
  GQE_EXPECTS(range.begin <= range.end && range.end <= work.size(),
              "Source work range must be ordered and within the source work units",
              std::invalid_argument);
  return {work.begin() + range.begin, work.begin() + range.end};
}

}  // namespace gqe::storage::detail
