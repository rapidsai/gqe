/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "masked_ast.hpp"

#include <gqe/utility/error.hpp>

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <stdexcept>

namespace gqe::detail {
namespace {
constexpr int masked_ast_max_threads_per_block = 128;
static_assert(masked_ast_max_threads_per_block % cudf::detail::warp_size == 0,
              "the maximum masked AST block size must contain whole warps");

enum class masked_ast_mode { filter, projection };

/** @brief Skip rejected rows before invoking the existing cuDF evaluator.
 * @tparam has_nulls Whether the expression can produce nulls.
 * @tparam has_complex_type Whether the plan references strings or fixed-point types.
 * @tparam selection_has_nulls Whether selection validity bits must be read.
 */
template <bool has_nulls, bool has_complex_type, bool selection_has_nulls>
__global__ void masked_filter_ast_kernel(cudf::table_device_view table,
                                         cudf::ast::detail::expression_device_view plan,
                                         bool const* selected,
                                         cudf::bitmask_type const* validity,
                                         cudf::size_type offset,
                                         bool* output)
{
  using intermediate = cudf::ast::detail::IntermediateDataType<has_nulls>;
  extern __shared__ __align__(16) unsigned char scratch[];
  auto* local = reinterpret_cast<intermediate*>(scratch) + threadIdx.x * plan.num_intermediates;
  cudf::ast::detail::expression_evaluator<has_nulls, has_complex_type> evaluator{table, plan};
  auto const row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= table.num_rows()) return;
  bool keep = false;
  if ((!selection_has_nulls || cudf::bit_is_set(validity, offset + row)) && selected[row]) {
    cudf::ast::detail::value_expression_result<bool, has_nulls> result;
    evaluator.evaluate(result, row, local);
    keep = result.is_valid() && result.value();
  }
  output[row] = keep;
}

/** @brief Capture cuDF's typed scalar output without reimplementing any operator.
 * @tparam has_nulls Whether the AST can produce nulls.
 */
template <bool has_nulls>
struct projection_result
  : cudf::ast::detail::expression_result<projection_result<has_nulls>, bool, has_nulls> {
  // Borrowed output storage, sized and aligned for the parser's output type for all rows.
  // The owning column keeps it alive through the kernel invocation.
  void* output_data;

  bool valid = false;
  __device__ explicit projection_result(void* output) : output_data(output) {}
  [[nodiscard]] __device__ bool is_valid() const { return valid; }
  [[nodiscard]] __device__ bool value() const
  {
    CUDF_UNREACHABLE("masked projection writes through output_data");
  }
  /** @brief Store the output type selected by cuDF's operator dispatch.
   * @tparam Element The scalar type emitted by cuDF.
   */
  template <typename Element>
  __device__ void set_value(
    cudf::size_type row, cudf::ast::detail::possibly_null_value_t<Element, has_nulls> const& value)
  {
    if constexpr (cudf::is_fixed_width<Element>()) {
      if constexpr (has_nulls) {
        valid = value.has_value();
        if (valid) reinterpret_cast<Element*>(output_data)[row] = *value;
      } else {
        valid                                        = true;
        reinterpret_cast<Element*>(output_data)[row] = value;
      }
    } else {
      CUDF_UNREACHABLE("masked projection requires fixed-width output");
    }
  }
};
/** @brief Evaluate selected projection rows and write their validity bits.
 * @tparam has_nulls Whether the AST can produce nulls.
 * @tparam has_complex_type Whether the plan references strings or fixed-point types.
 * @tparam selection_has_nulls Whether selection validity bits must be read.
 * @tparam is_sub_warp_block Whether to atomically update an initially all-null output mask.
 */
template <bool has_nulls, bool has_complex_type, bool selection_has_nulls, bool is_sub_warp_block>
__global__ void masked_projection_ast_kernel(cudf::table_device_view table,
                                             cudf::ast::detail::expression_device_view plan,
                                             bool const* selected,
                                             cudf::bitmask_type const* validity,
                                             cudf::size_type offset,
                                             void* output,
                                             cudf::bitmask_type* output_validity)
{
  using intermediate = cudf::ast::detail::IntermediateDataType<has_nulls>;
  extern __shared__ __align__(16) unsigned char scratch[];
  auto* local = reinterpret_cast<intermediate*>(scratch) + threadIdx.x * plan.num_intermediates;
  cudf::ast::detail::expression_evaluator<has_nulls, has_complex_type> evaluator{table, plan};
  auto const row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  projection_result<has_nulls> result{output};
  if (row < table.num_rows() &&
      (!selection_has_nulls || cudf::bit_is_set(validity, offset + row)) && selected[row]) {
    evaluator.evaluate(result, row, local);
  }
  // Sub-warp blocks may share an output word. The mask starts all-null for this path.
  if constexpr (is_sub_warp_block) {
    if (row < table.num_rows() && result.is_valid()) cudf::set_bit(output_validity, row);
    return;
  }
  // Every lane participates, including lanes beyond the final row. Only lane zero
  // writes each word, avoiding races between adjacent output validity bits.
  auto const bits = __ballot_sync(0xffffffffu, result.is_valid());
  if (threadIdx.x % cudf::detail::warp_size == 0 && row < table.num_rows()) {
    output_validity[cudf::word_index(row)] = bits;
  }
}
/** @brief Launch one specialization using the scratch-limited grid configuration.
 * @tparam mode Whether to produce a filter mask or a projection column.
 * @tparam has_nulls Whether the AST can produce nulls.
 * @tparam has_complex_type Whether the AST requires complex-type dispatch.
 * @tparam selection_has_nulls Whether selection validity bits must be read.
 */
template <masked_ast_mode mode, bool has_nulls, bool has_complex_type, bool selection_has_nulls>
void launch_masked_ast(cudf::table_device_view table,
                       cudf::ast::detail::expression_device_view plan,
                       cudf::column_view selection,
                       cudf::mutable_column_view output,
                       cudf::detail::grid_1d const& config,
                       size_t scratch_bytes,
                       bool is_sub_warp_block,
                       cuda::stream_ref stream)
{
  if constexpr (mode == masked_ast_mode::filter) {
    masked_filter_ast_kernel<has_nulls, has_complex_type, selection_has_nulls>
      <<<config.num_blocks, config.num_threads_per_block, scratch_bytes, stream.get()>>>(
        table,
        plan,
        selection.data<bool>(),
        selection.null_mask(),
        selection.offset(),
        output.data<bool>());
  } else {
    auto launch_projection = [&]<bool sub_warp>() {
      masked_projection_ast_kernel<has_nulls, has_complex_type, selection_has_nulls, sub_warp>
        <<<config.num_blocks, config.num_threads_per_block, scratch_bytes, stream.get()>>>(
          table,
          plan,
          selection.data<bool>(),
          selection.null_mask(),
          selection.offset(),
          output.head(),
          output.null_mask());
    };
    if (is_sub_warp_block)
      launch_projection.template operator()<true>();
    else
      launch_projection.template operator()<false>();
  }
  GQE_CUDA_TRY(cudaPeekAtLastError());
}
/** @brief Share parser preparation and allocation across the two output contracts.
 * @tparam mode Whether to produce a filter mask or a projection column.
 */
template <masked_ast_mode mode>
std::unique_ptr<cudf::column> masked_ast(cudf::table_view table,
                                         cudf::ast::expression const& expression,
                                         cudf::column_view selection,
                                         cudf::data_type expected_type,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  GQE_EXPECTS(selection.type().id() == cudf::type_id::BOOL8 && selection.size() == table.num_rows(),
              "masked_ast: selection must be BOOL8 with matching row count");
  if constexpr (mode == masked_ast_mode::filter) {
    GQE_EXPECTS(expected_type.id() == cudf::type_id::BOOL8,
                "masked_filter_ast: expected_type must be BOOL8");
  }
  if (!cudf::is_fixed_width(expected_type)) return nullptr;
  bool const has_nulls = expression.may_evaluate_null(table, stream);
  cudf::ast::detail::expression_parser parser{expression, table, has_nulls, stream, mr};
  if (parser.output_type() != expected_type || parser.device_expression_data.operators.empty())
    return nullptr;
  int block_size = masked_ast_max_threads_per_block;
  if (table.num_rows() != 0) {
    int device_id;
    GQE_CUDA_TRY(cudaGetDevice(&device_id));
    int shared_memory_limit;
    GQE_CUDA_TRY(
      cudaDeviceGetAttribute(&shared_memory_limit, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
    if (parser.shmem_per_thread != 0) {
      block_size = std::min(block_size, shared_memory_limit / parser.shmem_per_thread);
    }
    GQE_EXPECTS(block_size > 0,
                "masked_ast: one thread's scratch exceeds the device shared-memory limit",
                std::runtime_error);
    // Whole warps own distinct validity words. Smaller blocks use atomic bit updates.
    if (block_size >= cudf::detail::warp_size) {
      block_size -= block_size % cudf::detail::warp_size;
    }
  }
  // Use one decision for mask initialization and the kernel's validity-write strategy.
  bool const is_sub_warp_block = block_size < cudf::detail::warp_size;
  auto const mask_state =
    mode == masked_ast_mode::filter
      ? cudf::mask_state::UNALLOCATED
      : (is_sub_warp_block ? cudf::mask_state::ALL_NULL : cudf::mask_state::UNINITIALIZED);
  auto output =
    cudf::make_fixed_width_column(expected_type, table.num_rows(), mask_state, stream, mr);
  if (table.num_rows() == 0) return output;
  cudf::detail::grid_1d const config{table.num_rows(), block_size};
  auto const scratch_bytes =
    static_cast<size_t>(parser.shmem_per_thread) * config.num_threads_per_block;
  auto device_table = cudf::table_device_view::create(table, stream);
  auto launch       = [&]<bool expression_has_nulls, bool has_complex_type>() {
    if (selection.has_nulls()) {
      launch_masked_ast<mode, expression_has_nulls, has_complex_type, true>(
        *device_table,
        parser.device_expression_data,
        selection,
        output->mutable_view(),
        config,
        scratch_bytes,
        is_sub_warp_block,
        stream);
    } else {
      launch_masked_ast<mode, expression_has_nulls, has_complex_type, false>(
        *device_table,
        parser.device_expression_data,
        selection,
        output->mutable_view(),
        config,
        scratch_bytes,
        is_sub_warp_block,
        stream);
    }
  };
  if (has_nulls) {
    if (parser.has_complex_type())
      launch.template operator()<true, true>();
    else
      launch.template operator()<true, false>();
  } else {
    if (parser.has_complex_type())
      launch.template operator()<false, true>();
    else
      launch.template operator()<false, false>();
  }
  if constexpr (mode == masked_ast_mode::projection) {
    output->set_null_count(
      cudf::null_count(output->view().null_mask(), 0, table.num_rows(), stream));
  }
  return output;
}
}  // namespace

std::unique_ptr<cudf::column> masked_filter_ast(cudf::table_view table,
                                                cudf::ast::expression const& expression,
                                                cudf::column_view selection,
                                                cudf::data_type expected_type,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  return masked_ast<masked_ast_mode::filter>(
    table, expression, selection, expected_type, stream, mr);
}

std::unique_ptr<cudf::column> masked_projection_ast(cudf::table_view table,
                                                    cudf::ast::expression const& expression,
                                                    cudf::column_view selection,
                                                    cudf::data_type expected_type,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  return masked_ast<masked_ast_mode::projection>(
    table, expression, selection, expected_type, stream, mr);
}
}  // namespace gqe::detail
