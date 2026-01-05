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

#include <gqe/executor/unique_key_inner_join.hpp>

#include "unique_key_inner_join.cuh"

#include <gqe/catalog.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/kernel_fusion_helpers.hpp>

#include <cuco/probing_scheme.cuh>
#include <cuco/static_set.cuh>
#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/pair.h>

#include <cstdlib>
#include <memory>
#include <utility>

namespace gqe {

namespace {

template <template <typename> class hash_function>
struct element_hasher {
 public:
  __device__ element_hasher(bool has_nulls = true, uint32_t seed = cudf::DEFAULT_HASH_SEED)
    : _has_nulls{has_nulls}, _seed{seed}
  {
  }

  template <typename T, std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
  __device__ cudf::hash_value_type operator()(cudf::column_device_view const& col,
                                              cudf::size_type row_index)
  {
    if (_has_nulls && col.is_null(row_index)) {
      return cuda::std::numeric_limits<cudf::hash_value_type>::max();
    }
    return hash_function<T>{_seed}(col.element<T>(row_index));
  }

  template <typename T, std::enable_if_t<!cudf::is_numeric<T>()>* = nullptr>
  __device__ cudf::hash_value_type operator()(cudf::column_device_view const& col,
                                              cudf::size_type row_index)
  {
    CUDF_UNREACHABLE("Unsupported datatype");
  }

  uint32_t _seed;
  bool _has_nulls;
};

template <template <typename> class hash_function>
class device_row_hasher {
 public:
  device_row_hasher(cudf::table_device_view const& table,
                    bool has_nulls = true,
                    uint32_t seed  = cudf::DEFAULT_HASH_SEED)
    : _table{table}, _has_nulls{has_nulls}, _seed{seed}
  {
  }

  __device__ auto operator()(cudf::size_type row_index) const noexcept
  {
    auto it = thrust::make_transform_iterator(
      _table.begin(), [this, row_index](cudf::column_device_view const& col) {
        return cudf::type_dispatcher(
          col.type(), element_hasher<hash_function>{_has_nulls, _seed}, col, row_index);
      });

    // Hash each element and combine all the hash values together
    return cudf::detail::accumulate(it, it + _table.num_columns(), _seed, [](auto hash, auto h) {
      return cudf::hashing::detail::hash_combine(hash, h);
    });
  }

  uint32_t _seed;
  cudf::table_device_view _table;
  bool _has_nulls;
};

template <typename Hasher, typename T>
class create_input_pair {
 public:
  GQE_HOST_DEVICE create_input_pair(Hasher const& hash) : _hash{hash} {}

  __device__ __forceinline__ auto operator()(cudf::size_type i) const noexcept
  {
    return cuco::pair{_hash(i), T{i}};
  }

 private:
  Hasher _hash;
};

template <int block_size>
__device__ void write_to_global(cudf::size_type* build_indices,
                                cudf::size_type const* buffer_build,
                                cudf::size_type* probe_indices,
                                cudf::size_type const* buffer_probe,
                                cudf::size_type* global_offset,
                                int16_t count)
{
  __shared__ cudf::size_type block_out_row_idx;

  if (threadIdx.x == 0) { block_out_row_idx = atomicAdd(global_offset, count); }
  __syncthreads();

  for (cudf::size_type thread_out_row_idx = threadIdx.x; thread_out_row_idx < count;
       thread_out_row_idx += block_size) {
    const auto global_out_row_idx     = block_out_row_idx + thread_out_row_idx;
    build_indices[global_out_row_idx] = buffer_build[thread_out_row_idx];
    probe_indices[global_out_row_idx] = buffer_probe[thread_out_row_idx];
  }
  __syncthreads();
}

/*
 * Probe kernel for join between two tables.
 *
 * Before running the kernel, we need to build hash set. Then,
 * block `idx` is assigned to part of the probe table with row indices in
 * [idx * in_rows_per_block, (idx + 1) * in_rows_per_block). The found matches
 * are stored in buffer till the buffer is full. The buffer is written to global memory
 * in a coalesced manner.The output location is calculated using
 * a global_offset variable, updated using atomic adds.
 */
template <int block_size, typename SetRef, typename ProbeIter, bool has_mask>
__global__ void probe_hash_set(SetRef build_set_ref,
                               cudf::size_type probe_num_rows,
                               ProbeIter probe_iter,
                               bool const* probe_mask,
                               cudf::size_type const in_rows_per_block,
                               cudf::size_type* build_indices,
                               cudf::size_type* probe_indices,
                               cudf::size_type* global_offset)
{
  __shared__ typename cub::BlockScan<int16_t, block_size>::TempStorage block_scan_temp_storage;

  int const buffer_capacity = block_size * 4;
  __shared__ cudf::size_type buffer_build[buffer_capacity];
  __shared__ cudf::size_type buffer_probe[buffer_capacity];
  int16_t filled_buffer = 0;

  // The current block is responsible for processing rows of the probe table in [start_idx,
  // end_idx)
  cudf::size_type const start_idx = blockIdx.x * in_rows_per_block;
  cudf::size_type const end_idx   = min((blockIdx.x + 1) * in_rows_per_block, probe_num_rows);

  for (cudf::size_type block_row_idx = start_idx; block_row_idx < end_idx;
       block_row_idx += block_size) {
    // Note: Since this code block contains whole-block scan, it is important that all threads in
    // the block reach here to avoid deadlock.

    // Row index in the probe table to be processed by the current thread
    auto const thread_row_idx = block_row_idx + threadIdx.x;
    int16_t num_matches       = 0;  // 0 for no matches, 1 for a match
    cudf::size_type tuple_idx = -1;

    if (thread_row_idx < end_idx && (!has_mask || probe_mask[thread_row_idx])) {
      auto const tuple = build_set_ref.find(*(probe_iter + thread_row_idx));
      if (tuple != build_set_ref.end()) {
        tuple_idx   = static_cast<cudf::size_type>(tuple->second);
        num_matches = 1;
      };
    }

    // Use a whole-block scan to calculate the output location
    int16_t out_offset;
    int16_t total_matches;
    cub::BlockScan<int16_t, block_size>(block_scan_temp_storage)
      .ExclusiveSum(num_matches, out_offset, total_matches);
    if (total_matches + filled_buffer > buffer_capacity) {
      write_to_global<block_size>(
        build_indices, buffer_build, probe_indices, buffer_probe, global_offset, filled_buffer);
      filled_buffer = 0;
    }
    if (num_matches == 1) {
      buffer_build[filled_buffer + out_offset] = tuple_idx;
      buffer_probe[filled_buffer + out_offset] = thread_row_idx;
    }
    filled_buffer += total_matches;
    __syncthreads();
  }

  if (filled_buffer > 0) {
    write_to_global<block_size>(
      build_indices, buffer_build, probe_indices, buffer_probe, global_offset, filled_buffer);
  }
}

template <typename Set>
cudf::size_type perform_join(Set const& build_set,
                             cudf::table_view const& build_keys,
                             cudf::size_type* build_indices,
                             cudf::table_view const& probe_keys,
                             cudf::column_view const& probe_mask,
                             cudf::size_type* probe_indices,
                             cudf::null_equality compare_nulls,
                             rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  auto probe_keys_view      = cudf::table_device_view::create(probe_keys, stream);
  auto build_keys_view      = cudf::table_device_view::create(build_keys, stream);
  bool probe_keys_has_nulls = cudf::has_nulls(probe_keys);
  bool build_keys_has_nulls = cudf::has_nulls(build_keys);

  auto comparator_adapter_obj =
    gqe::detail::comparator_adapter{*probe_keys_view,
                                    *build_keys_view,
                                    build_keys_has_nulls || probe_keys_has_nulls,
                                    compare_nulls};

  auto const d_probe_hasher =
    device_row_hasher<cudf::hashing::detail::default_hash>{*probe_keys_view, probe_keys_has_nulls};
  auto const probe_iter = cudf::detail::make_counting_transform_iterator(
    0,
    create_input_pair<decltype(d_probe_hasher), cudf::experimental::row::lhs_index_type>{
      d_probe_hasher});
  auto const probe_mask_iter = probe_mask.is_empty() ? nullptr : probe_mask.data<bool>();
  GQE_EXPECTS(probe_mask.is_empty() || probe_mask.type().id() == cudf::type_id::BOOL8,
              "The probe mask of unique key inner join is not a boolean column");

  // Thread block size of the "probe_hash_set" kernel, must be a multiple of `warp_size`
  constexpr int block_size = 128;

  auto build_set_base = build_set.ref(cuco::op::find);
  auto build_set_ref  = build_set_base.rebind_key_eq(comparator_adapter_obj);

  auto probe_hash_set_kernel =
    probe_mask_iter == nullptr
      ? probe_hash_set<block_size, decltype(build_set_ref), decltype(probe_iter), false>
      : probe_hash_set<block_size, decltype(build_set_ref), decltype(probe_iter), true>;
  int grid_size = utility::detect_launch_grid_size(probe_hash_set_kernel, block_size);

  auto const in_rows_per_block = gqe::utility::divide_round_up(probe_keys.num_rows(), grid_size);
  rmm::device_scalar<cudf::size_type> global_offset(0, stream);

  probe_hash_set_kernel<<<grid_size, block_size, 0, stream>>>(build_set_ref,
                                                              probe_keys.num_rows(),
                                                              probe_iter,
                                                              probe_mask_iter,
                                                              in_rows_per_block,
                                                              build_indices,
                                                              probe_indices,
                                                              global_offset.data());
  stream.synchronize();

  return global_offset.value(stream);
}

template <typename Set>
std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
unique_key_inner_join_impl(Set const& build_set,
                           cudf::table_view build_keys,
                           cudf::table_view probe_keys,
                           cudf::column_view const& probe_mask,
                           cudf::null_equality compare_nulls,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto const result_num_rows = probe_keys.num_rows();
  rmm::device_uvector<cudf::size_type> build_indices(result_num_rows, stream, mr);
  rmm::device_uvector<cudf::size_type> probe_indices(result_num_rows, stream, mr);

  cudf::size_type out_rows_total = perform_join(build_set,
                                                build_keys,
                                                build_indices.data(),
                                                probe_keys,
                                                probe_mask,
                                                probe_indices.data(),
                                                compare_nulls,
                                                stream);

  build_indices.resize(out_rows_total, stream);
  probe_indices.resize(out_rows_total, stream);

  stream.synchronize();
  return std::pair(
    std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(build_indices), stream, mr),
    std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(probe_indices), stream, mr));
}

}  // namespace

namespace detail {

unique_key_join::unique_key_join(cudf::table_view const& build,
                                 cudf::column_view const& build_mask,
                                 cudf::null_equality compare_nulls,
                                 float load_factor,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
  : _build{build},
    _nulls_equal{compare_nulls},
    _build_set{static_cast<size_t>(gqe::utility::get_num_active_keys(build.num_rows(), build_mask)),
               load_factor,
               cuco::empty_key{cuco::pair{std::numeric_limits<cudf::hash_value_type>::max(),
                                          cudf::experimental::row::rhs_index_type{-1}}},
               always_not_equal{},
               {},
               cuco::thread_scope_device,
               cuco_storage_type{},
               rmm::mr::stream_allocator_adaptor{
                 rmm::mr::polymorphic_allocator<
                   cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type>>{},
                 stream},
               stream.value()}
{
  GQE_EXPECTS(0 != build.num_columns(), "Hash join build table is empty");
  if (build.num_rows() == 0) { return; }

  auto build_keys_view      = cudf::table_device_view::create(build, stream);
  bool build_keys_has_nulls = cudf::has_nulls(build);
  auto const d_build_hasher =
    device_row_hasher<cudf::hashing::detail::default_hash>{*build_keys_view, build_keys_has_nulls};

  auto const build_iter = cudf::detail::make_counting_transform_iterator(
    0,
    create_input_pair<decltype(d_build_hasher), cudf::experimental::row::rhs_index_type>{
      d_build_hasher});
  if (build_mask.is_empty()) {
    _build_set.insert(build_iter, build_iter + build.num_rows(), stream.value());
  } else {
    GQE_EXPECTS(build_mask.type().id() == cudf::type_id::BOOL8,
                "The build mask of unique key inner join is not a boolean column");
    _build_set.insert_if(build_iter,
                         build_iter + build.num_rows(),
                         build_mask.data<bool>(),
                         gqe::utility::identity_pred{},
                         stream.value());
  }
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
unique_key_join::inner_join(cudf::table_view const& probe,
                            cudf::column_view const& probe_mask,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) const
{
  GQE_EXPECTS(0 != probe.num_columns(), "Hash join probe table is empty");

  GQE_EXPECTS(this->_build.num_columns() == probe.num_columns(),
              "Mismatch in number of columns to be joined on");

  // If either table is empty, return immediately
  if (this->_build.is_empty() || probe.is_empty() || 0 == this->_build.num_rows() ||
      0 == probe.num_rows()) {
    return std::pair(std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr));
  }

  GQE_EXPECTS(std::equal(std::cbegin(this->_build),
                         std::cend(this->_build),
                         std::cbegin(probe),
                         std::cend(probe),
                         [](auto const& b, auto const& p) { return b.type() == p.type(); }),
              "Mismatch in joining column data types");

  auto [build_indices, probe_indices] = unique_key_inner_join_impl(
    this->_build_set, this->_build, probe, probe_mask, this->_nulls_equal, stream, mr);

  return std::pair(std::move(probe_indices), std::move(build_indices));
}

}  // namespace detail

unique_key_join::unique_key_join(cudf::table_view const& build,
                                 cudf::column_view const& build_mask,
                                 cudf::null_equality compare_nulls,
                                 float load_factor,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  _impl = std::make_unique<gqe::detail::unique_key_join>(
    build, build_mask, compare_nulls, load_factor, stream, mr);
}

unique_key_join::unique_key_join(cudf::table_view const& build,
                                 cudf::column_view const& build_mask,
                                 cudf::null_equality compare_nulls,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
  : unique_key_join(build, build_mask, compare_nulls, 0.5, stream, mr)
{
}

unique_key_join::~unique_key_join() = default;

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
unique_key_join::inner_join(cudf::table_view const& probe,
                            cudf::column_view const& probe_mask,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(probe, probe_mask, stream, mr);
}

bool unique_key_join_supported(cudf::table_view const& keys)
{
  return thrust::all_of(thrust::host, keys.begin(), keys.end(), [](cudf::column_view col) {
    return cudf::is_numeric(col.type());
  });
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
unique_key_inner_join(cudf::table_view const& build,
                      cudf::table_view const& probe,
                      cudf::column_view const& build_mask,
                      cudf::column_view const& probe_mask,
                      cudf::null_equality compare_nulls,
                      float load_factor,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  if (!gqe::unique_key_join_supported(build) || !gqe::unique_key_join_supported(probe)) {
    GQE_LOG_WARN("Using cudf's distinct hash join, since keys are not numeric datatype");
    cudf::distinct_hash_join join_obj(build, compare_nulls, load_factor, stream);
    GQE_EXPECTS(build_mask.is_empty() && probe_mask.is_empty(),
                "Filter conditions are only supported in unique key inner join or perfect "
                "join with numeric join keys.");
    return join_obj.inner_join(probe, stream, mr);
  }

  auto join_obj = gqe::unique_key_join(build, build_mask, compare_nulls, load_factor, stream, mr);
  auto [probe_indices, build_indices] = join_obj.inner_join(probe, probe_mask, stream, mr);
  return std::pair(std::move(build_indices), std::move(probe_indices));
}
}  // namespace gqe
