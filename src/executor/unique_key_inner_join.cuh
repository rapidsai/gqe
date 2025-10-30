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

#include <gqe/utility/cuda.hpp>

#include <cuco/static_set.cuh>

#include <cudf/hashing.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <memory>

namespace gqe {

namespace detail {

struct element_comparator {
 public:
  template <typename T, std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
  __device__ bool operator()(cudf::column_device_view const& lhs,
                             cudf::size_type lhs_index,
                             cudf::column_device_view const& rhs,
                             cudf::size_type rhs_index)
  {
    if (_has_nulls) {
      bool const lhs_is_null{lhs.is_null(lhs_index)};
      bool const rhs_is_null{rhs.is_null(rhs_index)};
      if (lhs_is_null and rhs_is_null) {
        return _compare_nulls == cudf::null_equality::EQUAL;
      } else if (lhs_is_null != rhs_is_null) {
        return false;
      }
    }

    return lhs.element<T>(lhs_index) == rhs.element<T>(rhs_index);
  }

  template <typename T, std::enable_if_t<!cudf::is_numeric<T>()>* = nullptr>
  __device__ bool operator()(cudf::column_device_view const& lhs,
                             cudf::size_type lhs_index,
                             cudf::column_device_view const& rhs,
                             cudf::size_type rhs_index)
  {
    CUDF_UNREACHABLE("Unsupported datatype");
  }

  __device__ element_comparator(bool has_nulls, cudf::null_equality compare_nulls)
    : _has_nulls{has_nulls}, _compare_nulls{compare_nulls}
  {
  }

  cudf::null_equality _compare_nulls;
  bool _has_nulls;
};

/**
 * @brief A custom comparator used for the build table insertion
 */
struct always_not_equal {
  __device__ constexpr bool operator()(
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type> const&,
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type> const&)
    const noexcept
  {
    // All build table keys are distinct thus `false` no matter what
    return false;
  }
};

struct comparator_adapter {
  comparator_adapter(cudf::table_device_view const& lhs_table,
                     cudf::table_device_view const& rhs_table,
                     bool has_nulls,
                     cudf::null_equality compare_nulls)
    : _lhs_table{lhs_table},
      _rhs_table{rhs_table},
      _has_nulls{has_nulls},
      _compare_nulls{compare_nulls}
  {
  }

  __device__ constexpr auto operator()(
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::lhs_index_type> const& lhs,
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type> const& rhs)
    const noexcept
  {
    if (lhs.first != rhs.first) {
      return false;
    }

    else {
      cudf::size_type lhs_index = static_cast<cudf::size_type>(lhs.second);
      cudf::size_type rhs_index = static_cast<cudf::size_type>(rhs.second);

      for (cudf::size_type column_idx = 0; column_idx < _lhs_table.num_columns(); column_idx++) {
        if (!cudf::type_dispatcher(_lhs_table.column(column_idx).type(),
                                   element_comparator{_has_nulls, _compare_nulls},
                                   _lhs_table.column(column_idx),
                                   lhs_index,
                                   _rhs_table.column(column_idx),
                                   rhs_index)) {
          return false;
        }
      }
      return true;
    }
  }

 public:
  cudf::table_device_view const _lhs_table;
  cudf::table_device_view const _rhs_table;
  cudf::null_equality _compare_nulls;
  bool _has_nulls;
};

/* This is used for hash sets where each element is a pair,
and probing should be done based on only the first element of the pair */
template <typename Hasher>
struct hasher_adapter {
  hasher_adapter(Hasher const& d_hasher = {}) : _d_hasher{d_hasher} {}

  template <typename T>
  __device__ constexpr auto operator()(
    cuco::pair<cudf::hash_value_type, T> const& key) const noexcept
  {
    return _d_hasher(key.first);
  }

 private:
  Hasher _d_hasher;
};

class unique_key_join {
 public:
  unique_key_join()                                  = delete;
  ~unique_key_join()                                 = default;
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
   * @param load_factor The load factor of the hash map.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr The memory resource used for allocations.
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
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  inner_join(cudf::table_view const& probe,
             cudf::column_view const& probe_mask,
             rmm::cuda_stream_view stream      = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  using probing_scheme_type =
    cuco::linear_probing<1, gqe::detail::hasher_adapter<cuda::std::identity>>;
  using cuco_storage_type = cuco::storage<1>;
  using hash_table_type =
    cuco::static_set<cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type>,
                     cuco::extent<cudf::size_type>,
                     cuda::thread_scope_device,
                     always_not_equal,
                     probing_scheme_type,
                     rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<
                       cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type>>>,
                     cuco_storage_type>;

  cudf::table_view _build;
  cudf::column_view _build_mask;
  cudf::null_equality _nulls_equal;
  hash_table_type _build_set;
};

}  // namespace detail

}  // namespace gqe
