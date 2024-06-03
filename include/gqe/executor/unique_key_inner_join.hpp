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

#include <rmm/device_uvector.hpp>

#include <memory>

namespace gqe {
/**
 * @brief Performs an optimized inner join for tables with
 * unique left keys and returns matched row indices.
 *
 * This function performs an inner join between two tables,
 * optimizing over cudf's inner join for cases where the
 * left keys contain unique values.
 *
 * The first returned vector contains the row indices from the left
 * keys that have a match in the right keys.
 * The corresponding values in the second returned vector are
 * the matched row indices from the right keys.
 *
 * If the keys are not numeric datatype, cudf's inner join is invoked.
 *
 * @param[in] build_keys The build keys
 * @param[in] probe_keys The probe keys
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param load_factor The load factor of the hash table
 * @param stream The cuda stream used for execution
 *
 * @return A pair of vectors [`build_indices`, `probe_indices`] that can be used to construct
 * the result of performing an inner join between two tables with `build_keys` and `probe_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
unique_key_inner_join(cudf::table_view build_keys,
                      cudf::table_view probe_keys,
                      cudf::null_equality compare_nulls,
                      float load_factor                   = 0.5,
                      rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                      rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace gqe
