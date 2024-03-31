/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
