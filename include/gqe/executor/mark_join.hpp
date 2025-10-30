/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>
#include <utility>

namespace gqe {

/**
 * @brief Forward declaration for mark join implementation, utilized for left {semi,anti} joins.
 * This algorithm always builds a hash table on the LHS, and then probes using the RHS. When
 * a match is found, a boolean is set to indicate the match. The output is then materialized
 * by scanning the boolean entries.
 * Reference: Neumann et al. "The Complete Story of Joins (in HyPer)" BTW 2017
 */
namespace detail {
class mark_join;
}  // namespace detail

class mark_join {
 public:
  mark_join() = delete;
  ~mark_join();
  mark_join(mark_join const&)            = delete;
  mark_join(mark_join&&)                 = delete;
  mark_join& operator=(mark_join const&) = delete;
  mark_join& operator=(mark_join&&)      = delete;

  /**
   * @brief Constructs a mark join object for subsequent probe calls
   *
   * @param build The build table that contains unique elements
   * @param build_mask The build table mask that indicates valid rows
   * @param is_cached If caching is expected, mark counts are delayed until all partitions are
   * probed.
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param load_factor used to initialize the hash table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory manager handle
   */
  mark_join(cudf::table_view const& build,
            cudf::column_view const& build_mask,
            bool is_cached,
            cudf::null_equality compare_nulls = cudf::null_equality::UNEQUAL,
            double load_factor                = 0.5,
            rmm::cuda_stream_view stream      = rmm::cuda_stream_default,
            rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

  /**
   * @brief Returns the row positions that can be used to construct the result of performing
   * an inner join between two tables.
   *
   * @param probe The probe table, from which the keys are probed
   * @param probe_mask The probe table mask that indicates valid rows
   * @param is_anti_join Determines if result is based on semi join or anti join.
   * @param left_conditional If this is a mixed join, contains the left column positions.
   * @param right_conditional If this is a mixed join, contains the right column positions.
   * @param binary_predicate If this is a mixed join, contains the expression. Null if equality-only
   * join.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned positions' device memory.
   *
   * @return A list of positions that should be present in the materialized join. This function does
   * not materialize the join directly. If the hash map cache is enabled, returns an empty list on
   * every invocation.
   *
   */
  [[nodiscard]]

  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  perform_mark_join(
    cudf::table_view const& probe,
    cudf::column_view const& probe_mask,
    bool is_anti_join,
    cudf::table_view const& left_conditional,
    cudf::table_view const& right_conditional,
    cudf::ast::expression const* binary_predicate,
    rmm::cuda_stream_view stream      = rmm::cuda_stream_default,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref()) const;

  /**
   * @brief Returns the row positions that can be used to construct the result of performing the
   * mark join. This is primarily useful for computing the hash map cache positions, although it can
   * be used on any mark join object to return the list of marked positions seen during the object
   * lifetime.
   *
   * @param is_anti_join Determines if result is based on semi join or anti join.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned positions' device memory.
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> compute_positions_list_from_cached_map(
    bool is_anti_join,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  std::unique_ptr<gqe::detail::mark_join> _impl;
};

std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_semi_mark_join(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  cudf::null_equality compare_nulls = cudf::null_equality::UNEQUAL,
  double load_factor                = 0.5,
  rmm::cuda_stream_view stream      = rmm::cuda_stream_default,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_anti_mark_join(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  cudf::null_equality compare_nulls = cudf::null_equality::UNEQUAL,
  double load_factor                = 0.5,
  rmm::cuda_stream_view stream      = rmm::cuda_stream_default,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mixed_left_semi_mark_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::table_view const& left_conditional,
  cudf::table_view const& right_conditional,
  cudf::ast::expression const& binary_predicate,
  cudf::null_equality compare_nulls = cudf::null_equality::UNEQUAL,
  double load_factor                = 0.5,
  rmm::cuda_stream_view stream      = rmm::cuda_stream_default,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mixed_left_anti_mark_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::table_view const& left_conditional,
  cudf::table_view const& right_conditional,
  cudf::ast::expression const& binary_predicate,
  cudf::null_equality compare_nulls = cudf::null_equality::UNEQUAL,
  double load_factor                = 0.5,
  rmm::cuda_stream_view stream      = rmm::cuda_stream_default,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace gqe
