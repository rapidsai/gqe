/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "launchers.hpp"

#include "../../../executor/join_predicate.hpp"
#include "device_helpers.cuh"
#include "kernels.cuh"

#include <gqe/executor/eval.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cuco/operator.hpp>
#include <cuco/probing_scheme.cuh>
#include <cuco/static_multiset.cuh>
#include <cuda/atomic>
#include <cuda/std/functional>
#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cooperative_groups.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe::qep::mark_join_detail {
namespace {

using key_hasher = device_row_hasher<cudf::hashing::detail::default_hash, /*has_nulls=*/true>;

/** @brief Apply the configured null policy before comparing values with native equality. */
struct element_equal {
  /** @tparam T Matching numeric key type selected by cuDF dispatch after host validation. */
  template <typename T>
  __device__ bool operator()(cudf::column_device_view const& left,
                             cudf::size_type l,
                             cudf::column_device_view const& right,
                             cudf::size_type r,
                             cudf::null_equality nulls) const
  {
    if constexpr (cudf::is_numeric<T>()) {
      if (left.is_null(l) || right.is_null(r)) {
        return nulls == cudf::null_equality::EQUAL && left.is_null(l) && right.is_null(r);
      }
    }
    return equality_comparator{}.template operator()<T>(left, l, right, r);
  }
};

/** @brief Read a relation mask without conflating excluded rows with null join keys. */
struct active_row {
  bool const* values{};
  cudf::bitmask_type const* validity{};
  cudf::size_type offset{};

  explicit active_row(std::optional<cudf::column_view> mask)
  {
    if (mask) {
      values   = mask->data<bool>();
      validity = mask->null_mask();
      offset   = mask->offset();
    }
  }

  __device__ bool operator()(cudf::size_type row) const
  {
    return values == nullptr ||
           ((validity == nullptr || cudf::bit_is_set(validity, row + offset)) && values[row]);
  }
};

// ----- CuCo build, mark, and scan core -----

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

template <typename Hasher, typename T>
class create_input_pair {
 public:
  __host__ __device__ create_input_pair(Hasher const& hash) : _hash{hash} {}

  __device__ __forceinline__ cuco::pair<cudf::hash_value_type, T> operator()(
    cudf::size_type i) const noexcept
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

  __device__ __forceinline__ cuco::pair<HashType, IndexType> operator()(HashType i) const noexcept
  {
    return cuco::pair{_hashes[i], IndexType{i}};
  }

 private:
  HashType const* _hashes;
};

class equality_comparator_adapter {
 public:
  equality_comparator_adapter(cudf::table_device_view const& build_equality_table,
                              cudf::table_device_view const& probe_equality_table,
                              cudf::null_equality nulls)
    : _build_equality_table{build_equality_table},
      _probe_equality_table{probe_equality_table},
      _nulls{nulls}
  {
  }

  __device__ constexpr bool operator()(
    cuco::pair<cudf::hash_value_type, cudf::detail::row::rhs_index_type> const& query_key,
    cuco::pair<cudf::hash_value_type, cudf::detail::row::rhs_index_type> const& entry_key)
    const noexcept
  {
    return query_key.first == entry_key.first;
  }

  [[nodiscard]] __device__ constexpr bool operator()(
    cuco::pair<cudf::hash_value_type, cudf::detail::row::lhs_index_type> const& query_key,
    cuco::pair<cudf::hash_value_type, cudf::detail::row::rhs_index_type> const& entry_key)
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
                                    element_equal{},
                                    _probe_equality_table.column(column_idx),
                                    query_index,
                                    _build_equality_table.column(column_idx),
                                    entry_index,
                                    _nulls);
      } while (ret && ++column_idx < _build_equality_table.num_columns());
    }
    return ret;
  }

 private:
  cudf::table_device_view _build_equality_table;
  cudf::table_device_view _probe_equality_table;
  cudf::null_equality _nulls;
};

using key_type       = cuco::pair<cudf::hash_value_type, cudf::detail::row::rhs_index_type>;
using probing_scheme = cuco::linear_probing<1, hasher_adapter<cuda::std::identity>>;
using mark_set       = cuco::static_multiset<key_type,
                                             std::size_t,
                                             cuda::thread_scope_device,
                                             equality_comparator_adapter,
                                             probing_scheme,
                                             rmm::mr::polymorphic_allocator<key_type>,
                                             cuco::storage<1>>;
using mark_set_ref   = decltype(std::declval<mark_set const&>().ref(cuco::op::contains_tag{}));

/** @brief Reverse-order a CuCo allocation's stream after another stream's last use. */
void order_map_deallocation(cuda::stream_ref allocation, cuda::stream_ref use)
{
  if (allocation.get() == use.get()) { return; }
  cudaEvent_t event{};
  try {
    GQE_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    GQE_CUDA_TRY(cudaEventRecord(event, use.get()));
    GQE_CUDA_TRY(cudaStreamWaitEvent(allocation.get(), event));
  } catch (...) {
    if (event != nullptr) { GQE_CUDA_TRY_NO_THROW(cudaEventDestroy(event)); }
    throw;
  }
  GQE_CUDA_TRY(cudaEventDestroy(event));
}

}  // namespace

/**
 * @brief Own the core mark-join multiset, with the high hash bit reserved for marks.
 *
 * Each probe accumulator constructs and owns one mutable map from the retained build columns.
 * CuCo's allocator retains the construction stream. Explicit reverse stream edges protect that
 * allocation after cross-stream probe and scan uses.
 */
class hash_index {
 public:
  hash_index(cudf::size_type rows, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    : _stream{stream},
      _set{static_cast<std::size_t>(std::max(rows, 1)),
           0.5,
           cuco::empty_key{key_type{0x7fffffffU, cudf::detail::row::rhs_index_type{-1}}},
           // Storage comparisons only inspect hashes. Probe comparisons bind the input views.
           equality_comparator_adapter{*cudf::table_device_view::create({}, stream),
                                       *cudf::table_device_view::create({}, stream),
                                       cudf::null_equality::UNEQUAL},
           probing_scheme{},
           cuco::thread_scope_device,
           cuco::storage<1>{},
           rmm::mr::polymorphic_allocator<key_type>{mr},
           stream.get()}
  {
  }

  hash_index(cudf::table_view keys,
             std::optional<cudf::column_view> mask,
             cuda::stream_ref stream,
             rmm::device_async_resource_ref mr)
    : hash_index(mask ? static_cast<cudf::size_type>(
                          thrust::count_if(rmm::exec_policy(stream, mr),
                                           thrust::make_counting_iterator(0),
                                           thrust::make_counting_iterator(keys.num_rows()),
                                           active_row{mask}))
                      : keys.num_rows(),
                 stream,
                 mr)
  {
    auto table               = cudf::table_device_view::create(keys, stream);
    constexpr int block_size = 1024;
    auto const hash_rows =
      thrust::make_transform_iterator(thrust::make_counting_iterator(0), key_hasher{*table});
    rmm::device_uvector<cudf::hash_value_type> hashes(keys.num_rows(), stream, mr);
    auto kernel          = iter_to_column<decltype(hash_rows), cudf::hash_value_type, block_size>;
    auto const grid_size = utility::detect_launch_grid_size(kernel, block_size);
    kernel<<<grid_size, block_size, 0, stream.get()>>>(hash_rows, hashes.data(), keys.num_rows());
    GQE_CUDA_TRY(cudaPeekAtLastError());
    auto first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      create_input_pair_from_column<cudf::hash_value_type, cudf::detail::row::rhs_index_type>{
        hashes.data()});
    if (mask) {
      _set.insert_if_async(first,
                           first + keys.num_rows(),
                           thrust::make_counting_iterator(0),
                           active_row{mask},
                           stream.get());
    } else {
      _set.insert_async(first, first + keys.num_rows(), stream.get());
    }
  }

  [[nodiscard]] mark_set_ref ref() const { return _set.ref(cuco::op::contains_tag{}); }
  void add_marks(cudf::size_type count) { _num_marks.fetch_add(count); }
  void order_after(cuda::stream_ref stream) const { order_map_deallocation(_stream, stream); }

  /** @brief Scan this map into an exactly sized position list, then union its rows into marks. */
  void scan_into(uint8_t* marks, cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
  {
    constexpr int block_size         = 1024;
    auto mark_set_ref                = ref();
    using mark_set_key_type          = typename decltype(mark_set_ref)::key_type;
    cudf::size_type result_row_count = _num_marks.load();
    rmm::device_uvector<cudf::size_type> build_positions(result_row_count, stream, mr);
    rmm::device_scalar<cudf::size_type> mark_scan_offset(stream, mr);
    mark_scan_offset.set_value_to_zero_async(stream);
    mark_join_scan_operator<decltype(mark_set_ref),
                            decltype(mark_set_ref.storage_ref()),
                            mark_set_key_type,
                            1,
                            false>
      mark_scan_op{mark_set_ref,
                   mark_set_ref.storage_ref(),
                   static_cast<cudf::size_type>(mark_set_ref.storage_ref().num_buckets())};
    auto mark_scan_kernel = iterator_to_vector_if<block_size,
                                                  mark_set_key_type,
                                                  cudf::size_type,
                                                  decltype(mark_scan_op),
                                                  false>;
    int scan_grid_size    = utility::detect_launch_grid_size(mark_scan_kernel, block_size, 0);
    mark_scan_kernel<<<scan_grid_size, block_size, 0, stream.get()>>>(
      build_positions.data(), mark_scan_offset.data(), mark_scan_op, nullptr);
    GQE_CUDA_TRY(cudaPeekAtLastError());
    thrust::for_each_n(rmm::exec_policy_nosync(stream, mr),
                       build_positions.begin(),
                       result_row_count,
                       [=] __device__(auto row) { marks[row] = uint8_t{1}; });
    order_after(stream);
  }

 private:
  cuda::stream_ref _stream;
  mark_set _set;
  std::atomic<cudf::size_type> _num_marks{0};
};

namespace {

/** @brief Select chunk-local positions from completed marks using the compactor. */
struct marked_rows {
  uint8_t const* marks;
  cudf::size_type rows;
  active_row is_active;
  bool is_anti;

  __device__ auto get_bucket(cudf::size_type row) const
  {
    return thrust::make_counting_iterator(row);
  }
  __device__ bool predicate(cudf::size_type row) const
  {
    return is_active(row) && ((marks[row] == 0) == is_anti);
  }
  __device__ cudf::size_type accept(cudf::size_type row) const { return row; }
  __device__ constexpr cudf::size_type elements_per_bucket() const { return 1; }
  __device__ cudf::size_type num_buckets() const { return rows; }
};

}  // namespace

// ----- Mutable bitmap ownership -----

mark_bitmap::mark_bitmap(cudf::size_type rows,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
  : _stream{stream}, _marks{static_cast<std::size_t>(rows), stream, mr}
{
  if (rows > 0) {
    GQE_CUDA_TRY(cudaMemsetAsync(_marks.data(), 0, _marks.size() * sizeof(uint8_t), stream.get()));
  }
}

void mark_bitmap::order_after(cuda::stream_ref stream) const
{
  if (_marks.size() != 0) { order_map_deallocation(_stream, stream); }
}

// ----- Host-callable hash map operations -----

void hash_index_deleter::operator()(hash_index* index) const { delete index; }

hash_index_handle make_hash_index(cudf::table_view keys,
                                  std::optional<cudf::column_view> mask,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  return hash_index_handle{std::make_unique<hash_index>(keys, mask, stream, mr).release()};
}

void probe_hash_index(hash_index& index,
                      cudf::table_view left,
                      cudf::table_view right,
                      cudf::table_view left_payload_table,
                      cudf::table_view right_payload_table,
                      cudf::size_type left_column_count,
                      std::span<cudf::size_type const> left_columns,
                      std::span<cudf::size_type const> right_columns,
                      std::optional<cudf::column_view> probe_mask,
                      expression const* predicate,
                      cudf::null_equality nulls,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr)
{
  if (left.num_rows() == 0 || right.num_rows() == 0) { return; }
  auto left_keys  = cudf::table_device_view::create(left, stream);
  auto right_keys = cudf::table_device_view::create(right, stream);
  std::unique_ptr<gqe::detail::join_predicate> predicate_ast;
  std::unique_ptr<cudf::ast::detail::expression_parser> parser;
  cudf::ast::detail::expression_device_view expression_data{};
  bool has_nulls              = false;
  uint32_t scratch_per_thread = 0;
  if (predicate) {
    expression_binding const left_binding{left_columns};
    expression_binding const right_binding{right_columns};
    expression_input const left_input{left_payload_table, left_binding, left.num_rows()};
    expression_input const right_input{right_payload_table, right_binding, right.num_rows()};
    predicate_ast = std::make_unique<gqe::detail::join_predicate>(gqe::detail::make_join_predicate(
      *predicate, left_column_count, left_input, right_input, stream, mr));
    auto const& ast = predicate_ast->root();
    has_nulls       = ast.may_evaluate_null(left_payload_table, right_payload_table, stream);
    parser          = std::make_unique<cudf::ast::detail::expression_parser>(
      ast, left_payload_table, right_payload_table, has_nulls, stream, mr);
    GQE_EXPECTS(parser->output_type().id() == cudf::type_id::BOOL8,
                "mark join AST predicate must be Boolean",
                std::invalid_argument);
    expression_data    = parser->device_expression_data;
    scratch_per_thread = parser->shmem_per_thread;
  }

  // This cuDF factory does not expose mr. It owns only descriptor scratch, not table data.
  auto left_payload =
    cudf::table_device_view::create(predicate ? left_payload_table : cudf::table_view{}, stream);
  auto right_payload =
    cudf::table_device_view::create(predicate ? right_payload_table : cudf::table_view{}, stream);
  constexpr int block_size = 1024;
  auto const shared_bytes  = static_cast<std::size_t>(block_size) * scratch_per_thread;
  int shared_limit{};
  int device{};
  GQE_CUDA_TRY(cudaGetDevice(&device));
  GQE_CUDA_TRY(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlock, device));
  GQE_EXPECTS(shared_bytes <= static_cast<std::size_t>(shared_limit),
              "mark join AST exceeds per-block shared memory",
              std::invalid_argument);
  using probe_key_type = cuco::pair<cudf::hash_value_type, cudf::detail::row::lhs_index_type>;
  auto const rows      = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    create_input_pair<key_hasher, cudf::detail::row::lhs_index_type>{key_hasher{*right_keys}});
  rmm::device_uvector<probe_key_type> probe_rows(right.num_rows(), stream, mr);
  auto row_kernel     = iter_to_column<decltype(rows), probe_key_type, block_size>;
  auto const row_grid = utility::detect_launch_grid_size(row_kernel, block_size);
  row_kernel<<<row_grid, block_size, 0, stream.get()>>>(rows, probe_rows.data(), right.num_rows());
  GQE_CUDA_TRY(cudaPeekAtLastError());

  bool const* mask = probe_mask ? probe_mask->data<bool>() : nullptr;
  rmm::device_uvector<bool> mask_values(
    probe_mask && probe_mask->nullable() ? right.num_rows() : 0, stream, mr);
  if (probe_mask && probe_mask->nullable()) {
    // The device kernel consumes Boolean values. Fold nullable mask validity into those values.
    auto const active =
      thrust::make_transform_iterator(thrust::make_counting_iterator(0), active_row{probe_mask});
    auto mask_kernel     = iter_to_column<decltype(active), bool, block_size>;
    auto const mask_grid = utility::detect_launch_grid_size(mask_kernel, block_size);
    mask_kernel<<<mask_grid, block_size, 0, stream.get()>>>(
      active, mask_values.data(), right.num_rows());
    GQE_CUDA_TRY(cudaPeekAtLastError());
    mask = mask_values.data();
  }
  auto dispatch = [&]<bool is_mixed, bool nullable, bool has_probe_mask>() {
    auto kernel          = get_mark_probe_kernel<block_size,
                                                 equality_comparator_adapter,
                                                 mark_set_ref,
                                                 probe_key_type,
                                                 1,
                                                 nullable,
                                                 is_mixed,
                                                 has_probe_mask>(/*is_low_selectivity=*/false);
    auto const grid_size = utility::detect_launch_grid_size(kernel, block_size, shared_bytes);
    rmm::device_scalar<cudf::size_type> mark_counter(stream, mr);
    mark_counter.set_value_to_zero_async(stream);
    kernel<<<grid_size, block_size, shared_bytes, stream.get()>>>(
      *left_payload,
      *right_payload,
      expression_data,
      index.ref(),
      equality_comparator_adapter{*left_keys, *right_keys, nulls},
      probe_rows.data(),
      mask,
      right.num_rows(),
      mark_counter.data());
    GQE_CUDA_TRY(cudaPeekAtLastError());
    index.add_marks(mark_counter.value(stream));
  };
  auto dispatch_mask = [&]<bool is_mixed, bool nullable>() {
    if (probe_mask) {
      dispatch.template operator()<is_mixed, nullable, true>();
    } else {
      dispatch.template operator()<is_mixed, nullable, false>();
    }
  };
  if (!predicate) {
    dispatch_mask.template operator()<false, false>();
  } else if (has_nulls) {
    dispatch_mask.template operator()<true, true>();
  } else {
    dispatch_mask.template operator()<true, false>();
  }

  index.order_after(stream);
}

void scan_hash_index(hash_index const& index,
                     uint8_t* marks,
                     cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr)
{
  index.scan_into(marks, stream, mr);
}

rmm::device_uvector<cudf::size_type> select_marked_rows(uint8_t const* marks,
                                                        cudf::size_type rows,
                                                        std::optional<cudf::column_view> mask,
                                                        bool is_anti,
                                                        cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr)
{
  constexpr int block_size  = 1024;
  auto const op             = marked_rows{marks, rows, active_row{mask}, is_anti};
  auto const selected_count = static_cast<cudf::size_type>(
    thrust::count_if(rmm::exec_policy(stream, mr),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(rows),
                     [=] __device__(auto row) { return op.predicate(row); }));
  rmm::device_uvector<cudf::size_type> selected(selected_count, stream, mr);
  rmm::device_scalar<cudf::size_type> count{stream, mr};
  count.set_value_to_zero_async(stream);
  auto kernel =
    iterator_to_vector_if<block_size, cudf::size_type, cudf::size_type, marked_rows, false>;
  auto const grid_size = utility::detect_launch_grid_size(kernel, block_size, 0);
  kernel<<<grid_size, block_size, 0, stream.get()>>>(selected.data(), count.data(), op, nullptr);
  GQE_CUDA_TRY(cudaPeekAtLastError());
  return selected;
}

}  // namespace gqe::qep::mark_join_detail
