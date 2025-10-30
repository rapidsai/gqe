/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <memory>

namespace gqe {

bool unique_key_join_supported(cudf::table_view const& keys);

namespace detail {

/**
 * @brief Forward declaration for our unique key join
 */
class unique_key_join;
}  // namespace detail

class unique_key_join {
 public:
  unique_key_join() = delete;
  ~unique_key_join();
  unique_key_join(unique_key_join const&)            = delete;
  unique_key_join(unique_key_join&&)                 = delete;
  unique_key_join& operator=(unique_key_join const&) = delete;
  unique_key_join& operator=(unique_key_join&&)      = delete;

  /**
   * @brief Constructs a unique key hash join object for subsequent probe calls
   *
   * @param build The build table that contains unique elements
   * @param build_mask An optional boolean mask that indicates which rows of the build table are
   * valid
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  unique_key_join(cudf::table_view const& build,
                  cudf::column_view const& build_mask = cudf::column_view(),
                  cudf::null_equality compare_nulls   = cudf::null_equality::EQUAL,
                  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                  rmm::device_async_resource_ref mr   = rmm::mr::get_current_device_resource_ref());

  /**
   * @brief Constructs a unique key hash join object for subsequent probe calls
   *
   * @param build The build table that contains unique elements
   * @param build_mask An optional boolean mask that indicates which rows of the build table are
   * valid
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param load_factor The load factor of the hash table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the device memory
   */
  unique_key_join(cudf::table_view const& build,
                  cudf::column_view const& build_mask = cudf::column_view(),
                  cudf::null_equality compare_nulls   = cudf::null_equality::EQUAL,
                  float load_factor                   = 0.5,
                  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                  rmm::device_async_resource_ref mr   = rmm::mr::get_current_device_resource_ref());

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * an inner join between two tables.
   *
   * @param probe The probe table, from which the keys are probed
   * @param probe_mask An optional boolean mask that indicates which rows of the probe table are
   * valid
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned indices' device memory.
   *
   * @return A pair of columns [`probe_indices`, `build_indices`] that can be used to
   * construct the result of performing an inner join between two tables
   * with `build` and `probe` as the join keys.
   *
   * The order of indices is to match output of cudf::hash_join::inner_join().
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  inner_join(cudf::table_view const& probe,
             cudf::column_view const& probe_mask,
             rmm::cuda_stream_view stream      = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  std::unique_ptr<gqe::detail::unique_key_join> _impl;
};

/**
 * @brief Performs an optimized inner join for tables with unique build keys and returns matched row
 * indices.
 *
 * This function performs an inner join between two tables where the build table keys contain unique
 * values. It returns a pair of vectors containing the row indices that can be used to construct the
 * result of performing an inner join between two tables with `build` and `probe` as the join keys.
 *
 * If the keys are not numeric datatype, cudf's inner join is invoked.
 *
 * @param[in] build The build table that contains unique elements
 * @param[in] probe The probe table, from which the keys are probed
 * @param[in] build_mask An optional boolean mask that indicates which rows of the build table are
 * valid
 * @param[in] probe_mask An optional boolean mask that indicates which rows of the probe table are
 * valid
 * @param[in] compare_nulls Controls whether null join-key values should match or not
 * @param[in] load_factor The load factor of the hash table
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned indices' device memory
 *
 * @return A pair of vectors [`build_indices`, `probe_indices`] that can be used to construct
 * the result of performing an inner join between two tables with `build` and `probe` as the join
 * keys
 */
std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
unique_key_inner_join(
  cudf::table_view const& build,
  cudf::table_view const& probe,
  cudf::column_view const& build_mask,
  cudf::column_view const& probe_mask,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
  float load_factor                 = 0.5,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace gqe
