/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cooperative_groups.h>

#include <cudf/hashing.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/types.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/static_multiset.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <memory>

namespace gqe {

namespace detail {

template <typename HasherType>
class hasher_adapter {
 public:
  __device__ hasher_adapter() : _hasher({}) {}

  // Seed not passed on due to identity function.
  __device__ hasher_adapter(uint32_t seed) : _hasher({}) {}

  template <typename T>
  __device__ auto operator()(cuco::pair<cudf::hash_value_type, T> const& e) const noexcept
  {
    return _hasher(e.first);
  }

 private:
  HasherType _hasher;
};

template <template <typename> class hash_function>
struct element_hasher {
 public:
  __device__ element_hasher(uint32_t seed = cudf::DEFAULT_HASH_SEED) : _seed{seed} {}

  template <typename T, std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
  __device__ cudf::hash_value_type operator()(cudf::column_device_view const& col,
                                              cudf::size_type row_index)
  {
    return hash_function<T>{_seed}(col.element<T>(row_index));
  }

  template <typename T, std::enable_if_t<!cudf::is_numeric<T>()>* = nullptr>
  __device__ cudf::hash_value_type operator()(cudf::column_device_view const& col,
                                              cudf::size_type row_index)
  {
    CUDF_UNREACHABLE("Unsupported datatype");
  }

  uint32_t _seed;
};

template <template <typename> class hash_function>
class device_row_hasher {
 public:
  device_row_hasher(cudf::table_device_view const& table, uint32_t seed = cudf::DEFAULT_HASH_SEED)
    : _table{table}, _seed{seed}
  {
  }
  __device__ auto operator()(cudf::size_type row_index) const noexcept
  {
    constexpr auto bit_mask_31 = 0x7FFFFFFF;
    auto it                    = thrust::make_transform_iterator(
      _table.begin(), [=, this](cudf::column_device_view const& col) {
        return cudf::type_dispatcher(
          col.type(), element_hasher<hash_function>{_seed}, col, row_index);
      });

    // Hash each element and combine all the hash values together
    auto hash =
      cudf::detail::accumulate(it, it + _table.num_columns(), _seed, [](auto hash, auto h) {
        return cudf::hashing::detail::hash_combine(hash, h);
      });
    // This ignores the most significant bit so we can use it as storage for marks.
    return hash & bit_mask_31;
  }

  uint32_t _seed;
  cudf::table_device_view _table;
};

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
__host__ __device__ constexpr T mark_mask() noexcept
{
  size_t shift = sizeof(T) * size_t{8} - size_t{1};
  auto mask    = T{1} << shift;
  return mask;
}

// Returns a hashed value with the mark set.
template <typename T>
__host__ __device__ constexpr T set_mark(T value) noexcept
{
  return value | mark_mask<T>();
}

// Unsets the mark bit from a hash value.
template <typename T>
__host__ __device__ constexpr T unset_mark(T value) noexcept
{
  return value & ~mark_mask<T>();
}

// Checks if the mark bit is set in a hash.
template <typename T>
__host__ __device__ constexpr bool is_marked(T value) noexcept
{
  return value & mark_mask<T>();
}

template <typename Hasher, typename T>
class create_input_pair {
 public:
  __host__ __device__ create_input_pair(Hasher const& hash) : _hash{hash} {}

  __device__ __forceinline__ auto operator()(cudf::size_type i) const noexcept
  {
    return cuco::pair{_hash(i), T{i}};
  }

 private:
  Hasher _hash;
};

// We utilize this to instantiate pairs of hash, row indices from existing set of hashes.
template <typename HashType, typename IndexType>
class create_input_pair_from_column {
 public:
  __host__ __device__ create_input_pair_from_column(HashType const* hashes) : _hashes{hashes} {}

  __device__ __forceinline__ auto operator()(HashType i) const noexcept
  {
    return cuco::pair{_hashes[i], IndexType{i}};
  }

 private:
  HashType const* _hashes;
};

class equality_comparator {
 public:
  template <typename T, std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
  __device__ bool operator()(cudf::column_device_view const& lhs,
                             cudf::size_type lhs_index,
                             cudf::column_device_view const& rhs,
                             cudf::size_type rhs_index)
  {
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
};

class equality_comparator_adapter {
 public:
  equality_comparator_adapter(cudf::table_device_view const& build_equality_table,
                              cudf::table_device_view const& probe_equality_table)
    : _build_equality_table{build_equality_table}, _probe_equality_table{probe_equality_table}
  {
  }

  equality_comparator_adapter(equality_comparator_adapter const& base_adapter,
                              cudf::table_device_view const& probe_equality_table)
    : _build_equality_table{base_adapter._build_equality_table},
      _probe_equality_table{probe_equality_table}
  {
  }

  __device__ constexpr bool operator()(
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type> const& query_key,
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type> const& entry_key)
    const noexcept
  {
    return query_key.first == entry_key.first;
  }

  [[nodiscard]] __device__ constexpr bool operator()(
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::lhs_index_type> const& query_key,
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type> const& entry_key)
    const noexcept
  {
    bool ret = false;
    // If hashes match
    if (query_key.first == entry_key.first) {
      cudf::size_type query_index = static_cast<cudf::size_type>(query_key.second);
      cudf::size_type entry_index = static_cast<cudf::size_type>(entry_key.second);
      cudf::size_type column_idx  = 0;
      // Iterate every equality condition and compare their columns.
      do {
        ret = cudf::type_dispatcher(_probe_equality_table.column(column_idx).type(),
                                    equality_comparator{},
                                    _probe_equality_table.column(column_idx),
                                    query_index,
                                    _build_equality_table.column(column_idx),
                                    entry_index);
      } while (ret && ++column_idx < _build_equality_table.num_columns());
    }
    return ret;
  }

 private:
  cudf::table_device_view _build_equality_table;
  cudf::table_device_view _probe_equality_table;
};

// This operator represents the probe row prefilter operation.
template <typename IteratorType, typename FilterType, typename ElementType>
class mark_join_prefilter_operator {
 public:
  mark_join_prefilter_operator(IteratorType iterator,
                               FilterType filter,
                               cudf::size_type num_elements)
    : _iterator{iterator}, _filter{filter}, _num_elements{num_elements} {};

  // The predicate is true if the entry is in the bloom filter.
  __device__ __forceinline__ bool predicate(const ElementType& slot) const
  {
    return _filter.contains(slot.first);
  }

  // When we accept the item, the entire item is stored in the output buffer.
  __device__ constexpr __forceinline__ ElementType accept(const ElementType& slot) const
  {
    return slot;
  }

  // Indexing the iterator of probe rows.
  __device__ __forceinline__ auto get_bucket(const cudf::size_type index) const
  {
    return (_iterator + index);
  }

  __device__ constexpr __forceinline__ cudf::size_type elements_per_bucket() const { return 1; }

  __device__ constexpr __forceinline__ cudf::size_type num_buckets() const { return _num_elements; }

 private:
  const cudf::size_type _num_elements;
  const IteratorType _iterator;
  const FilterType _filter;
};

// This operator represents a mark scan operation. It generally iterates a hash table for
// scanned entries.
template <typename StructureType,
          typename IteratorType,
          typename InputType,
          uint32_t bucket_size,
          bool is_anti_join>
class mark_join_scan_operator {
 public:
  // mark scan operator relies on cuco types, so will fail if empty_key_sentinel or
  // erased_key_sentinel undefined.
  mark_join_scan_operator(StructureType data, IteratorType iterator, cudf::size_type num_elements)
    : _iterator{iterator},
      _num_elements{num_elements},
      _is_filled{cuco::detail::open_addressing_ns::slot_is_filled<false, InputType>{
        data.empty_key_sentinel(), data.erased_key_sentinel()}}
  {
  }
  // Predicate is true only if the slot contains a valid entry and is either marked, or
  // if this is an anti-join, not marked.
  __device__ __forceinline__ bool predicate(const InputType& slot) const
  {
    return _is_filled(slot) && (detail::is_marked(slot.first) ^ is_anti_join);
  }

  // When we accept an entry, we only write its row index to output and omit the hash.
  __device__ constexpr __forceinline__ auto accept(const InputType& slot) const
  {
    return static_cast<cudf::size_type>(slot.second);
  }

  // Underlying iterator must support bracket indexing.
  __device__ __forceinline__ auto get_bucket(const cudf::size_type index) const
  {
    return _iterator[index];
  }

  __device__ constexpr __forceinline__ cudf::size_type elements_per_bucket() const
  {
    return bucket_size;
  }

  __device__ constexpr __forceinline__ cudf::size_type num_buckets() const { return _num_elements; }

 private:
  const cudf::size_type _num_elements;
  const IteratorType _iterator;
  const cuco::detail::open_addressing_ns::slot_is_filled<false, InputType> _is_filled;
};

class mark_join {
  // Size of hash table storage bucket. We use one slot for now for performance. Can be
  // tweaked later if we find an efficient way to vectorize these operations.
  static constexpr uint32_t _slots_per_bucket = 1;
  using key_type = cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type>;
  using probing_scheme_type = cuco::linear_probing<1, hasher_adapter<cuda::std::identity>>;
  using hash_table_type     = cuco::static_multiset<
        key_type,
        cudf::size_type,
        cuda::thread_scope_device,
        equality_comparator_adapter,
        probing_scheme_type,
        rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<key_type>>,
        cuco::storage<_slots_per_bucket>>;

  using build_hasher_type = device_row_hasher<cudf::hashing::detail::default_hash>;

  // Usually, bloom filters need a good hash function to work. However, the key value here is
  // already a hash. Reusing this hash works fine and save about half cost on bloom filter
  // setup/evaluation.
  using bloom_filter_policy_type =
    cuco::default_filter_policy<cuco::detail::identity_hash<cudf::size_type>, cudf::size_type, 2U>;
  using bloom_filter_allocator_type =
    rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<cuda::std::byte>>;
  using bloom_filter_allocator_instance_type = rmm::mr::polymorphic_allocator<cuda::std::byte>;
  using bloom_filter_type                    = cuco::bloom_filter<cudf::size_type,
                                                                  cuco::extent<std::size_t>,
                                                                  cuda::thread_scope_device,
                                                                  bloom_filter_policy_type,
                                                                  bloom_filter_allocator_type>;

 public:
  mark_join()                            = delete;
  ~mark_join()                           = default;
  mark_join(mark_join const&)            = delete;
  mark_join(mark_join&&)                 = delete;
  mark_join& operator=(mark_join const&) = delete;
  mark_join& operator=(mark_join&&)      = delete;

  /**
   * @brief Constructs a mark join object for subsequent probe calls
   *
   * @param build The build table that contains unique elements
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param load_factor The load factor used for the underlying has htable
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource.
   */
  mark_join(cudf::table_view const& build,
            bool is_cached,
            cudf::null_equality compare_nulls = cudf::null_equality::UNEQUAL,
            double load_factor                = 0.5,
            rmm::cuda_stream_view stream      = rmm::cuda_stream_default,
            rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * the left semi or anti join.
   *
   * @param probe The probe table, from which the keys are probed
   * @param is_anti_join Determines if result is based on semi join or anti join.
   * @param left_conditional If this is a mixed join, contains the left column indices.
   * @param right_conditional If this is a mixed join, contains the right column indices.
   * @param binary_predicate If this is a mixed join, contains the expression. Null if equality-only
   * join.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned indices' device memory.
   *
   * @return A column of row indices that can be used to materialize the join.
   */

  [[nodiscard]]

  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  perform_mark_join(cudf::table_view const& probe,
                    bool is_anti_join,
                    cudf::table_view const& left_conditional,
                    cudf::table_view const& right_conditional,
                    cudf::ast::expression const* binary_predicate,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Returns the row indices that can be used to construct the result of performing the left
   * semi or anti join. This function is only valid if a cached hash map is being used. Otherwise,
   * returns an empty list.
   *
   * @param is_anti_join Determines if result is based on semi join or anti join.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned indices' device memory.
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> _compute_positions_list_from_map(
    bool is_anti_join,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  cudf::table_view _build;
  cudf::null_equality _nulls_equal;
  hash_table_type _mark_set;
  bloom_filter_type _bloom_filter;
  // We make num_marks atomic because it's mutable and could have race conditions in
  // multi-gpu or multi-worker scenarios.
  mutable std::atomic<cudf::size_type> _num_marks;
  bool _is_cached;

  /*
   * @brief This function estimates whether we believe the join is selective, for the purpose
   * of selectivity-specific optimizations.
   */
  bool is_low_selectivity() const;

  /*
   * @brief This function performs the mark_scan operation, counting the number of marked entries
   * in the map and returning the subsequent list of positions.
   * @param stream CUDA stream
   * @param mr Memory pool reference
   */
  template <bool is_anti_join>
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> _mark_scan(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref) const;

  /*
   * @brief This function performs the actual join, probing the hash table against input rows.
   * @param probe_equality_device_view Rows from the probe table used in equality conditions.
   * @param build_conditional_device_view A table representing the rows/columns used in mixed
   * condition evaluation from the build side.
   * @param probe_conditional_device_view A table representing the rows/columns used in mixed
   * condition evaluation from the probe side.
   * @param expr_device_view The device data respresenting the mixed conditions used by the AST
   * evaluator.
   * @param comparator_adapter The operator used for equality comparisons.
   * @param is_anti_join Determines if this is a semi or anti join.
   * @param shared_memory_per_thread Shared memory required by the cuDF AST evaluator, if any.
   * @param stream CUDA stream used for scheduling device operations.
   * @param mr Memory pool reference
   */
  template <typename ComparatorAdapterType, bool has_nulls, bool is_mixed>
  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  _perform_mark_join(cudf::table_device_view const& probe_equality_device_view,
                     cudf::table_device_view const& build_conditional_device_view,
                     cudf::table_device_view const& probe_conditional_device_view,
                     cudf::ast::detail::expression_device_view const& expr_device_view,
                     ComparatorAdapterType const& comparator_adapter,
                     bool is_anti_join,
                     uint32_t shared_memory_per_thread,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) const;
};

}  // namespace detail

}  // namespace gqe
