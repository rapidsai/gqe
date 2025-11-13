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

#pragma once

#include <cudf/types.hpp>

#include <cassert>
#include <vector>

namespace gqe {
namespace optimizer {
namespace utility {

/**
 * @brief This function determine whether each projection index is from the left or right child.
 * Then offset it accordingly assuming a left/right swap.
 *
 * @param projection_indices List of projection indices to offset
 * @param n_cols_left Number of columns from left child
 * @param n_cols_right Number of columns from right child
 */
inline void swap_projection_indices_inplace(std::vector<cudf::size_type>& projection_indices,
                                            cudf::size_type n_cols_left,
                                            cudf::size_type n_cols_right)
{
  for (auto& projection_index : projection_indices) {
    assert(projection_index < (n_cols_left + n_cols_right));
    projection_index = projection_index < n_cols_left ? projection_index + n_cols_right
                                                      : projection_index - n_cols_left;
  }
}

}  // namespace utility
}  // namespace optimizer
}  // namespace gqe
