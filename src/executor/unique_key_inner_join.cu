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
#include <gqe/utility/error.hpp>

#include <cuco/probing_scheme.cuh>
#include <cuco/static_set.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join.hpp>
#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/equal.h>
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

namespace {

bool keys_not_supported(cudf::table_view keys)
{
  return thrust::any_of(thrust::host, keys.begin(), keys.end(), [](cudf::column_view col) {
    return !cudf::is_numeric(col.type());
  });
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
cudf_unique_key_inner_join(
  cudf::table_view build_keys,
  cudf::table_view probe_keys,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  if (cudf::detail::has_nested_columns(build_keys)) {
    cudf::distinct_hash_join<cudf::has_nested::YES> join_obj(
      build_keys, probe_keys, cudf::nullable_join::YES, compare_nulls, stream);
    return join_obj.inner_join(stream, mr);
  } else {
    cudf::distinct_hash_join<cudf::has_nested::NO> join_obj(
      build_keys, probe_keys, cudf::nullable_join::YES, compare_nulls, stream);
    return join_obj.inner_join(stream, mr);
  }
}

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
    auto it =
      thrust::make_transform_iterator(_table.begin(), [=](cudf::column_device_view const& col) {
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

struct comparator_adapter {
  comparator_adapter(cudf::table_device_view lhs_table,
                     cudf::table_device_view rhs_table,
                     bool has_nulls,
                     cudf::null_equality compare_nulls)
    : _lhs_table{lhs_table},
      _rhs_table{rhs_table},
      _has_nulls{has_nulls},
      _compare_nulls{compare_nulls}
  {
  }

  __device__ constexpr auto operator()(
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type> const&,
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type> const&)
    const noexcept
  {
    // All build table keys are distinct thus `false` no matter what
    return false;
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
  cudf::table_device_view _lhs_table;
  cudf::table_device_view _rhs_table;
  cudf::null_equality _compare_nulls;
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
template <int block_size, typename SetRef, typename ProbeIter>
__global__ void probe_hash_set(SetRef build_set_ref,
                               cudf::size_type probe_num_rows,
                               ProbeIter probe_iter,
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

    if (thread_row_idx < end_idx) {
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

template <int block_size, typename KernelType>
int find_grid_size(KernelType kernel)
{
  int dev_id{-1};
  GQE_CUDA_TRY(cudaGetDevice(&dev_id));

  int num_sms{-1};
  GQE_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  int max_active_blocks{-1};
  GQE_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0));

  int grid_size = max_active_blocks * num_sms;
  return grid_size;
}

cudf::size_type perform_join(cudf::table_view build_keys,
                             cudf::size_type* build_indices,
                             cudf::table_view probe_keys,
                             cudf::size_type* probe_indices,
                             cudf::null_equality compare_nulls,
                             float load_factor            = 0.5,
                             rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  std::size_t const set_capacity = std::ceil(build_keys.num_rows() / load_factor);

  int constexpr cg_size = 1;
  using probing_scheme_type =
    cuco::linear_probing<cg_size, hasher_adapter<thrust::identity<cudf::hash_value_type>>>;

  auto build_keys_view      = cudf::table_device_view::create(build_keys, stream);
  auto probe_keys_view      = cudf::table_device_view::create(probe_keys, stream);
  bool build_keys_has_nulls = cudf::has_nulls(build_keys);
  bool probe_keys_has_nulls = cudf::has_nulls(probe_keys);

  auto const d_build_hasher =
    device_row_hasher<cudf::hashing::detail::default_hash>{*build_keys_view, build_keys_has_nulls};

  rmm::mr::polymorphic_allocator<
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type>>
    polly_alloc;
  auto stream_alloc = rmm::mr::stream_allocator_adaptor(polly_alloc, stream);

  auto empty_key_sentinel =
    cuco::empty_key{cuco::pair{std::numeric_limits<cudf::hash_value_type>::max(),
                               cudf::experimental::row::rhs_index_type{-1}}};
  auto comparator_adapter_obj = comparator_adapter{*probe_keys_view,
                                                   *build_keys_view,
                                                   build_keys_has_nulls || probe_keys_has_nulls,
                                                   compare_nulls};

  auto build_set = cuco::static_set{build_keys.num_rows(),
                                    load_factor,
                                    empty_key_sentinel,
                                    comparator_adapter_obj,
                                    probing_scheme_type{},
                                    cuco::thread_scope_device,
                                    cuco::storage<1>{},
                                    stream_alloc,
                                    stream.value()};

  auto const build_iter = cudf::detail::make_counting_transform_iterator(
    0,
    create_input_pair<decltype(d_build_hasher), cudf::experimental::row::rhs_index_type>{
      d_build_hasher});
  build_set.insert(build_iter, build_iter + build_keys.num_rows(), stream.value());

  auto const d_probe_hasher =
    device_row_hasher<cudf::hashing::detail::default_hash>{*probe_keys_view, probe_keys_has_nulls};
  auto const probe_iter = cudf::detail::make_counting_transform_iterator(
    0,
    create_input_pair<decltype(d_probe_hasher), cudf::experimental::row::lhs_index_type>{
      d_probe_hasher});

  // Thread block size of the "probe_hash_set" kernel, must be a multiple of `warp_size`
  constexpr int block_size = 128;
  auto build_set_ref       = build_set.ref(cuco::op::find);

  auto probe_hash_set_kernel =
    probe_hash_set<block_size, decltype(build_set_ref), decltype(probe_iter)>;
  int grid_size = find_grid_size<block_size>(probe_hash_set_kernel);

  auto const in_rows_per_block = gqe::utility::divide_round_up(probe_keys.num_rows(), grid_size);
  rmm::device_scalar<cudf::size_type> global_offset(0, stream);

  probe_hash_set<block_size><<<grid_size, block_size, 0, stream>>>(build_set_ref,
                                                                   probe_keys.num_rows(),
                                                                   probe_iter,
                                                                   in_rows_per_block,
                                                                   build_indices,
                                                                   probe_indices,
                                                                   global_offset.data());
  stream.synchronize();

  return global_offset.value(stream);
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
gqe_unique_key_inner_join(
  cudf::table_view build_keys,
  cudf::table_view probe_keys,
  cudf::null_equality compare_nulls,
  float load_factor                   = 0.5,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const result_num_rows = probe_keys.num_rows();
  rmm::device_uvector<cudf::size_type> build_indices(result_num_rows, stream, mr);
  rmm::device_uvector<cudf::size_type> probe_indices(result_num_rows, stream, mr);

  cudf::size_type out_rows_total = perform_join(build_keys,
                                                build_indices.data(),
                                                probe_keys,
                                                probe_indices.data(),
                                                compare_nulls,
                                                load_factor,
                                                stream);

  build_indices.resize(out_rows_total, stream);
  probe_indices.resize(out_rows_total, stream);

  stream.synchronize();
  return std::pair(
    std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(build_indices), stream, mr),
    std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(probe_indices), stream, mr));
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
compute_unique_key_inner_join(
  cudf::table_view build_keys,
  cudf::table_view probe_keys,
  cudf::null_equality compare_nulls,
  float load_factor                   = 0.5,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  GQE_EXPECTS(0 != build_keys.num_columns(), "Hash join build table is empty");
  GQE_EXPECTS(0 != probe_keys.num_columns(), "Hash join probe table is empty");

  GQE_EXPECTS(build_keys.num_columns() == probe_keys.num_columns(),
              "Mismatch in number of columns to be joined on");

  // If either table is empty, return immediately
  if (build_keys.is_empty() || probe_keys.is_empty() || 0 == build_keys.num_rows() ||
      0 == probe_keys.num_rows()) {
    return std::pair(std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr));
  }

  GQE_EXPECTS(std::equal(std::cbegin(build_keys),
                         std::cend(build_keys),
                         std::cbegin(probe_keys),
                         std::cend(probe_keys),
                         [](auto const& b, auto const& p) { return b.type() == p.type(); }),
              "Mismatch in joining column data types");

  if (keys_not_supported(build_keys)) {
    GQE_LOG_WARN("Using cudf's distinct hash join, since keys are not numeric datatype");
    return cudf_unique_key_inner_join(build_keys, probe_keys, compare_nulls, stream, mr);
  }

  return gqe_unique_key_inner_join(build_keys, probe_keys, compare_nulls, load_factor, stream, mr);
}

}  // namespace

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
unique_key_inner_join(cudf::table_view build_keys,
                      cudf::table_view probe_keys,
                      cudf::null_equality compare_nulls,
                      float load_factor                   = 0.5,
                      rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                      rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  utility::nvtx_scoped_range unique_key_inner_join_range("unique_key_inner_join");
  return compute_unique_key_inner_join(
    build_keys, probe_keys, compare_nulls, load_factor, stream, mr);
}

}  // namespace gqe
