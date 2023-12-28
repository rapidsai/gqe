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

#include <gqe/catalog.hpp>

#include <cuco/probing_scheme.cuh>
#include <cuco/static_map.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/pair.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace gqe {

template <typename type1, typename type2>
__device__ bool unequal_pair(cuco::pair<type1, type2> a, cuco::pair<type1, type2> b)
{
  if (a.first != b.first || a.second != b.second)
    return true;
  else
    return false;
}

template <int block_size>
__device__ void write_to_global(cudf::size_type* hash_map_indices,
                                cudf::size_type* buffer_hash_map,
                                cudf::size_type* probe_indices,
                                cudf::size_type* buffer_probe,
                                cudf::size_type* global_offset,
                                int16_t count)
{
  __shared__ cudf::size_type block_out_row_idx;
  if (threadIdx.x == 0) { block_out_row_idx = atomicAdd(global_offset, count); }
  __syncthreads();
  for (cudf::size_type thread_out_row_idx = threadIdx.x; thread_out_row_idx < count;
       thread_out_row_idx += block_size) {
    const auto global_out_row_idx        = block_out_row_idx + thread_out_row_idx;
    hash_map_indices[global_out_row_idx] = buffer_hash_map[thread_out_row_idx];
    probe_indices[global_out_row_idx]    = buffer_probe[thread_out_row_idx];
  }
  __syncthreads();
}

template <typename MapInsertRef, typename MapFindRef, typename T>
struct hash_map_view {
  hash_map_view(MapInsertRef map_insert_ref,
                MapFindRef map_find_ref,
                cuco::sentinel::empty_key<T> empty_key_sentinel,
                bool* empty_key_bit,
                cudf::size_type* empty_key_row,
                cudf::null_equality compare_nulls,
                bool* null_key_bit,
                cudf::size_type* null_key_row)
    : _map_insert_ref(map_insert_ref),
      _map_find_ref(map_find_ref),
      _empty_key_sentinel(empty_key_sentinel),
      _empty_key_bit(empty_key_bit),
      _empty_key_row(empty_key_row),
      _compare_nulls(compare_nulls),
      _null_key_bit(empty_key_bit),
      _null_key_row(empty_key_row)
  {
  }

  __device__ void insert_values(cudf::column_device_view* build_column, cudf::size_type row_idx)
  {
    if ((*build_column).is_valid(row_idx)) {
      if ((*build_column).element<T>(row_idx) == _empty_key_sentinel) {
        *_empty_key_bit = true;
        *_empty_key_row = row_idx;
      }
      _map_insert_ref.insert(
        thrust::pair<T, cudf::size_type>((*build_column).element<T>(row_idx), row_idx));
    } else {
      if (_compare_nulls == cudf::null_equality::EQUAL) {
        *_null_key_bit = true;
        *_null_key_row = row_idx;
      }
    }
  }

  __device__ void probe_find(cudf::column_device_view* probe_column,
                             cudf::size_type thread_row_idx,
                             cudf::size_type* tuple_idx,
                             int16_t* num_matches)
  {
    if ((*probe_column).is_valid(thread_row_idx)) {
      auto probe_key = (*probe_column).element<T>(thread_row_idx);
      if (*_empty_key_bit && (probe_key == _empty_key_sentinel)) {
        *tuple_idx   = *_empty_key_row;
        *num_matches = 1;
      } else {
        auto const tuple = _map_find_ref.find(probe_key);
        if (unequal_pair<T, cudf::size_type>(*tuple, *_map_find_ref.end())) {
          *tuple_idx   = (*tuple).second;
          *num_matches = 1;
        }
      }
    } else {
      if (_compare_nulls == cudf::null_equality::EQUAL && *_null_key_bit) {
        *tuple_idx   = *_null_key_row;
        *num_matches = 1;
      }
    }
  }

  MapInsertRef _map_insert_ref;
  MapFindRef _map_find_ref;

  cuco::sentinel::empty_key<T> _empty_key_sentinel;
  bool* _empty_key_bit;
  cudf::size_type* _empty_key_row;

  cudf::null_equality _compare_nulls;
  bool* _null_key_bit;
  cudf::size_type* _null_key_row;
};

/*
 * Probe kernel for join between two tables.
 *
 * Before running the kernel, we need to build hash map. Then,
 * block `idx` is assigned to part of the probe table with row indices in
 * [idx * in_rows_per_block, (idx + 1) * in_rows_per_block). The found matches
 * are stored in buffer till the buffer is full. The buffer is written to global memory
 * in a coalesced manner.The output location is calculated using
 * a global_offset variable, updated using atomic adds.
 */
template <int block_size, typename probe_type, typename MapView>
__global__ void probe_hash_maps(MapView build_map_view,
                                cudf::column_device_view probe_column,
                                cudf::size_type const in_rows_per_block,
                                cudf::size_type* hash_map_indices,
                                cudf::size_type* probe_indices,
                                cudf::size_type* global_offset)
{
  __shared__ typename cub::BlockScan<int16_t, block_size>::TempStorage block_scan_temp_storage;

  int const global_block_id = blockIdx.x;
  int const buffer_capacity = block_size * 4;
  __shared__ cudf::size_type buffer_hash_map[buffer_capacity];
  __shared__ cudf::size_type buffer_probe[buffer_capacity];
  int16_t filled_buffer = 0;

  // The current block is responsible for processing rows of the probe table in [start_idx,
  // end_idx)
  cudf::size_type const start_idx = global_block_id * in_rows_per_block;
  cudf::size_type const end_idx =
    min((global_block_id + 1) * in_rows_per_block, probe_column.size());

  for (cudf::size_type block_row_idx = start_idx; block_row_idx < end_idx;
       block_row_idx += block_size) {
    // Note: Since this code block contains whole-block scan, it is important that all threads in
    // the block reach here to avoid deadlock.

    // Row index in the probe table to be processed by the current thread
    auto const thread_row_idx = block_row_idx + threadIdx.x;
    int16_t num_matches       = 0;  // 0 for no matches, 1 for a match
    cudf::size_type tuple_idx = -1;

    if (thread_row_idx < end_idx) {
      build_map_view.probe_find(&probe_column, thread_row_idx, &tuple_idx, &num_matches);
    }

    // Use a whole-block scan to calculate the output location
    int16_t out_offset;
    int16_t total_matches;
    cub::BlockScan<int16_t, block_size>(block_scan_temp_storage)
      .ExclusiveSum(num_matches, out_offset, total_matches);
    if (total_matches + filled_buffer > buffer_capacity) {
      write_to_global<block_size>(hash_map_indices,
                                  buffer_hash_map,
                                  probe_indices,
                                  buffer_probe,
                                  global_offset,
                                  filled_buffer);
      filled_buffer = 0;
    }
    if (num_matches == 1) {
      buffer_hash_map[filled_buffer + out_offset] = tuple_idx;
      buffer_probe[filled_buffer + out_offset]    = thread_row_idx;
    }
    filled_buffer += total_matches;
    __syncthreads();
  }

  if (filled_buffer > 0) {
    write_to_global<block_size>(
      hash_map_indices, buffer_hash_map, probe_indices, buffer_probe, global_offset, filled_buffer);
  }
}

template <typename T>
using join_hash_map_type = cuco::experimental::static_map<
  T,
  cudf::size_type,
  cuco::experimental::extent<T>,
  cuda::thread_scope_device,
  thrust::equal_to<T>,
  cuco::experimental::
    double_hashing<1, cuco::detail::MurmurHash3_32<T>, cuco::detail::MurmurHash3_32<T>>,
  rmm::mr::stream_allocator_adaptor<
    rmm::mr::polymorphic_allocator<cuco::pair<T, cudf::size_type>>>>;

template <typename KernelType>
void set_grid_size(int* grid_size, int block_size, KernelType kernel)
{
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp device_props;
  cudaGetDeviceProperties(&device_props, device);
  int num_multiprocessors = device_props.multiProcessorCount;
  int max_active_blocks_per_multiprocessor;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks_per_multiprocessor, kernel, block_size, 0);
  *grid_size = max_active_blocks_per_multiprocessor * num_multiprocessors;
}

template <typename T>
std::optional<cudf::size_type> build_map(cudf::table_view build_keys,
                                         cudf::size_type* build_indices,
                                         cudf::table_view probe_keys,
                                         cudf::size_type* probe_indices,
                                         cudf::null_equality compare_nulls,
                                         float load_factor            = 0.5,
                                         rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  std::size_t const map_capacity = std::ceil(build_keys.num_rows() / load_factor);
  auto build_column              = cudf::column_device_view::create(build_keys.column(0));

  constexpr cuco::sentinel::empty_key<T> empty_key_sentinel(std::numeric_limits<T>::min());
  // Row indices are non-negative in nature.
  constexpr cuco::sentinel::empty_value<cudf::size_type> empty_value_sentinel(-1);

  rmm::mr::polymorphic_allocator<cuco::pair<T, cudf::size_type>> polly_alloc;
  auto stream_alloc = rmm::mr::make_stream_allocator_adaptor(polly_alloc, stream);
  join_hash_map_type<T> build_map(
    map_capacity, empty_key_sentinel, empty_value_sentinel, {}, {}, stream_alloc);

  rmm::device_scalar<bool> empty_key_bit(false, stream);
  rmm::device_scalar<cudf::size_type> empty_key_row(stream);

  rmm::device_scalar<bool> null_key_bit(false, stream);
  rmm::device_scalar<cudf::size_type> null_key_row(stream);

  hash_map_view build_map_view(build_map.ref(cuco::experimental::insert),
                               build_map.ref(cuco::experimental::find),
                               empty_key_sentinel,
                               empty_key_bit.data(),
                               empty_key_row.data(),
                               compare_nulls,
                               null_key_bit.data(),
                               null_key_row.data());

  thrust::for_each(
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(build_column->size()),
    [build_map_view, build_column_view = *build_column] __device__(auto row_idx) mutable {
      build_map_view.insert_values(&build_column_view, row_idx);
    });

  auto probe_column = cudf::column_device_view::create(probe_keys.column(0));

  // Thread block size of the "probe_hash_maps" kernel, must be a multiple of `warp_size`
  constexpr int block_size = 128;
  // Number of threadblocks for the "probe_hash_maps" kernel, chosen based on experiments
  int grid_size;
  set_grid_size(&grid_size, block_size, probe_hash_maps<block_size, T, decltype(build_map_view)>);
  auto const in_rows_per_block = gqe::utility::divide_round_up(probe_keys.num_rows(), grid_size);
  rmm::device_scalar<cudf::size_type> global_offset(0, stream);

  probe_hash_maps<block_size, T><<<grid_size, block_size>>>(build_map_view,
                                                            *probe_column,
                                                            in_rows_per_block,
                                                            build_indices,
                                                            probe_indices,
                                                            global_offset.data());
  stream.synchronize();

  return global_offset.value(stream);
}

struct build_map_functor {
  template <typename T>
  std::optional<cudf::size_type> operator()(cudf::table_view build_keys,
                                            cudf::size_type* build_indices,
                                            cudf::table_view probe_keys,
                                            cudf::size_type* probe_indices,
                                            cudf::null_equality compare_nulls,
                                            float load_factor,
                                            rmm::cuda_stream_view stream)
  {
    return std::nullopt;
  }
};
template <>
std::optional<cudf::size_type> build_map_functor::operator()<int32_t>(
  cudf::table_view build_keys,
  cudf::size_type* build_indices,
  cudf::table_view probe_keys,
  cudf::size_type* probe_indices,
  cudf::null_equality compare_nulls,
  float load_factor,
  rmm::cuda_stream_view stream)
{
  return build_map<int32_t>(
    build_keys, build_indices, probe_keys, probe_indices, compare_nulls, load_factor, stream);
}
template <>
std::optional<cudf::size_type> build_map_functor::operator()<int64_t>(
  cudf::table_view build_keys,
  cudf::size_type* build_indices,
  cudf::table_view probe_keys,
  cudf::size_type* probe_indices,
  cudf::null_equality compare_nulls,
  float load_factor,
  rmm::cuda_stream_view stream)
{
  return build_map<int64_t>(
    build_keys, build_indices, probe_keys, probe_indices, compare_nulls, load_factor, stream);
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
unique_key_inner_join(cudf::table_view left_keys,
                      cudf::table_view right_keys,
                      cudf::null_equality compare_nulls,
                      float load_factor            = 0.5,
                      rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  auto const result_num_rows = right_keys.num_rows();
  rmm::device_uvector<cudf::size_type> left_indices(result_num_rows, stream);
  rmm::device_uvector<cudf::size_type> right_indices(result_num_rows, stream);

  std::optional<cudf::size_type> out_rows_total = cudf::type_dispatcher(left_keys.column(0).type(),
                                                                        build_map_functor{},
                                                                        left_keys,
                                                                        left_indices.data(),
                                                                        right_keys,
                                                                        right_indices.data(),
                                                                        compare_nulls,
                                                                        load_factor,
                                                                        stream);

  if (!out_rows_total) return cudf::inner_join(left_keys, right_keys, compare_nulls);

  left_indices.resize(*out_rows_total, stream);
  right_indices.resize(*out_rows_total, stream);

  stream.synchronize();
  return std::pair(
    std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(left_indices)),
    std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(right_indices)));
}

}  // namespace gqe
