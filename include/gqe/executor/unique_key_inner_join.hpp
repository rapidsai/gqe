/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>

namespace gqe {

/**
 * @brief Reports whether a key type is supported by `unique_key_join`.
 *
 * Supported types are integer, floating-point, Boolean, and fixed-point decimal types.
 *
 * @param[in] type Key type to check.
 * @return Whether the key type is supported.
 */
[[nodiscard]] bool unique_key_join_supported(cudf::data_type type);

/**
 * @brief Reports whether every key column has a supported type.
 *
 * This host-side type check does not inspect key values or establish uniqueness. An empty key
 * list passes the type check, but `unique_key_join` requires at least one key column.
 *
 * @param[in] keys Key columns whose types are checked.
 * @return Whether all key types are supported by `unique_key_join`.
 */
[[nodiscard]] bool unique_key_join_supported(cudf::table_view const& keys);

namespace detail {

/**
 * @brief Forward declaration for our unique key join
 */
class unique_key_join;
}  // namespace detail

/**
 * @brief Reusable hash join for build keys that are unique under the chosen null semantics.
 *
 * Build and probe keys must have matching integer, floating-point, Boolean, or fixed-point decimal
 * types. Callers must establish type support with `unique_key_join_supported` before construction
 * and probing.
 * The class does not check type support or scan for uniqueness, and it has no cuDF fallback.
 * Use `unique_key_inner_join` when the unsupported-type fallback described below is required.
 *
 * The caller retains the build columns for the join's lifetime and orders build completion before
 * probing on other streams. Const probes may run concurrently with independent output storage.
 * Hash-map, match-vector, and controllable scratch allocations use the supplied memory resource.
 * cuDF allocates device-side column-view metadata through its own resource. These metadata
 * allocations do not use `mr`.
 *
 * `inner_join` reads the match count on the host before returning exact-sized index vectors. This
 * synchronizes the supplied stream through the probe. It does not synchronize other streams.
 */
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
   * @param mr Device memory resource, subject to the metadata exception in the class contract.
   */
  unique_key_join(cudf::table_view const& build,
                  cudf::column_view const& build_mask = cudf::column_view(),
                  cudf::null_equality compare_nulls   = cudf::null_equality::EQUAL,
                  cuda::stream_ref stream             = cudf::get_default_stream(),
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
   * @param mr Device memory resource, subject to the metadata exception in the class contract.
   */
  unique_key_join(cudf::table_view const& build,
                  cudf::column_view const& build_mask = cudf::column_view(),
                  cudf::null_equality compare_nulls   = cudf::null_equality::EQUAL,
                  float load_factor                   = 0.5,
                  cuda::stream_ref stream             = cudf::get_default_stream(),
                  rmm::device_async_resource_ref mr   = rmm::mr::get_current_device_resource_ref());

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * an inner join between two tables.
   *
   * @param probe The probe table, from which the keys are probed
   * @param probe_mask An optional boolean mask that indicates which rows of the probe table are
   * valid
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource, subject to the metadata exception in the class contract.
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
             cuda::stream_ref stream           = cudf::get_default_stream(),
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
 * Key types accepted by `unique_key_join_supported` use `unique_key_join`. Other key types use
 * `cudf::distinct_hash_join`, which still requires unique build keys. Both masks must be empty
 * on that fallback path. `GQE_EXPECTS` rejects non-empty masks with `std::logic_error`.
 * The cuDF version currently used by GQE controls its distinct hash-map allocation because its
 * constructor has no memory-resource parameter. Returned match vectors use `mr`.
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
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace gqe
