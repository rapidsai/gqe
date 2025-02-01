/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#include <cudf/strings/strings_column_view.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <rmm/exec_policy.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <numeric>
#include <stdexcept>

namespace {

// This kernel assumes the number of blocks is a multiple of the number of partitions. The behavior
// is undefined if this condition does not satisfy.
template <typename ColumnType>
__global__ void fill_key_column(ColumnType* column,
                                int64_t const* keys,
                                cudf::size_type const* offsets,
                                int num_partitions)
{
  auto const num_blocks_per_partition = gridDim.x / num_partitions;
  auto const partition_idx            = blockIdx.x / num_blocks_per_partition;
  auto const local_block_idx          = blockIdx.x % num_blocks_per_partition;
  auto const key                      = keys[partition_idx];
  auto const offset                   = offsets[partition_idx];
  auto const num_rows                 = offsets[partition_idx + 1] - offsets[partition_idx];

  for (cudf::size_type row_idx = local_block_idx * blockDim.x + threadIdx.x; row_idx < num_rows;
       row_idx += num_blocks_per_partition * blockDim.x) {
    column[offset + row_idx] = key;
  }
}

struct fill_key_column_functor {
  template <typename ColumnType, std::enable_if_t<std::is_integral_v<ColumnType>>* = nullptr>
  void operator()(cudf::mutable_column_view column,
                  int64_t const* keys,
                  cudf::size_type const* offsets,
                  int num_partitions)
  {
    int constexpr target_blocks = 4000;
    int constexpr block_size    = 128;
    int num_blocks = gqe::utility::divide_round_up(target_blocks, num_partitions) * num_partitions;

    fill_key_column<<<num_blocks, block_size>>>(
      column.data<ColumnType>(), keys, offsets, num_partitions);
  }

  template <typename ColumnType, std::enable_if_t<not std::is_integral_v<ColumnType>>* = nullptr>
  void operator()(cudf::mutable_column_view column,
                  int64_t const* keys,
                  cudf::size_type const* offsets,
                  int num_partitions)
  {
    throw std::logic_error("Partition key columns can only have integral types");
  }
};

// This expects the input to be cudf::data_type::STRING with each row a single (ASCII) character
std::unique_ptr<cudf::column> cast_string_to_int(cudf::column_view input,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  if (input.type() != cudf::data_type(cudf::type_id::STRING)) {
    throw std::invalid_argument("Expected input to have string data type");
  }

  auto const size = input.size();
  cudf::strings_column_view strings_view(input);
  auto const chars_size = strings_view.chars_size(stream);

  if (size != chars_size) {
    throw std::invalid_argument("Expected each input string to be a single (ASCII) character");
  }

  auto output = std::make_unique<cudf::column>(
    cudf::data_type(cudf::type_id::INT8),
    size,
    rmm::device_buffer{size * cudf::size_of(cudf::data_type(cudf::type_id::INT8)), stream, mr},
    cudf::copy_bitmask(input, stream, mr),
    input.null_count());

  cudf::mutable_column_view output_mutable = *output;

  thrust::transform(rmm::exec_policy(stream),
                    strings_view.chars_begin(stream),
                    strings_view.chars_begin(stream) + size,
                    output_mutable.begin<int8_t>(),
                    [] __device__(char element) { return static_cast<int8_t>(element); });

  return output;
}

}  // namespace

namespace gqe::storage {

std::unique_ptr<cudf::column> cast(cudf::column_view input,
                                   cudf::data_type type,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  if (input.type() == cudf::data_type(cudf::type_id::STRING) &&
      type == cudf::data_type(cudf::type_id::INT8)) {
    return cast_string_to_int(input, stream, mr);
  } else {
    return cudf::cast(input, type, stream, mr);
  }
}

std::unique_ptr<cudf::column> parquet_read_task::construct_partition_key_column(
  cudf::data_type dtype, std::vector<int64_t> keys, std::vector<cudf::size_type> num_rows)
{
  std::vector<cudf::size_type> offsets(num_rows.size() + 1);
  offsets[0] = 0;
  std::inclusive_scan(num_rows.begin(), num_rows.end(), offsets.begin() + 1);

  if (!cudf::is_numeric(dtype))
    throw std::logic_error("Only numeric types are supported for the partition key column");

  auto column = cudf::make_numeric_column(dtype, offsets.back());

  auto keys_dev = cudf::detail::make_device_uvector_async(
    keys, rmm::cuda_stream_default, rmm::mr::get_current_device_resource());

  auto offsets_dev = cudf::detail::make_device_uvector_async(
    offsets, rmm::cuda_stream_default, rmm::mr::get_current_device_resource());

  cudf::type_dispatcher(dtype,
                        fill_key_column_functor{},
                        column->mutable_view(),
                        keys_dev.data(),
                        offsets_dev.data(),
                        num_rows.size());

  return column;
}

}  // namespace gqe::storage
