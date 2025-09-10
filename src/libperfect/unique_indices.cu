#include <cudf/column/column_view.hpp>

#include "xor_hash_table.cuh"

#include "unique_indices.hpp"

namespace libperfect {

static std::vector<ConstCudaGpuBufferPointer> columns_to_buffers(
  std::vector<cudf::column_view> const& columns)
{
  std::vector<ConstCudaGpuBufferPointer> ret;
  for (uint column_index = 0; column_index < columns.size(); column_index++) {
    const auto& current_column = columns[column_index];
    ret.emplace_back(current_column.data<int>(), current_column.type().id());
  }
  return ret;
}

std::tuple<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
unique_indices(std::vector<cudf::column_view> const& key_columns, cudf::column_view const& mask)
{
  if (key_columns.empty()) { throw std::invalid_argument("key_columns is empty"); }
  if (key_columns[0].is_empty()) {
    return std::make_tuple(rmm::device_uvector<cudf::size_type>(0, rmm::cuda_stream_default),
                           rmm::device_uvector<cudf::size_type>(0, rmm::cuda_stream_default));
  }
  PUSH_RANGE("perfect unique indices", 0);
  PUSH_RANGE("make hash table", 1);
  auto key_buffers = columns_to_buffers(key_columns);
  auto keys_numel  = key_columns[0].size();
  std::optional<ConstCudaGpuBufferPointer> mask_buffer;
  if (!mask.is_empty()) { mask_buffer.emplace(mask.data<int>(), mask.type().id()); }

  auto hash_table = xor_hash_table::make_hash_table(key_buffers, keys_numel);
  POP_RANGE();
  auto ret = hash_table.template bulk_insert<xor_hash_table::CheckEquality::True,
                                             xor_hash_table::InsertOutput::True>(
    key_buffers, keys_numel, mask_buffer);
  auto& unique_element_indices = std::get<0>(ret).get_buffer();
  auto& group_indices          = std::get<1>(ret).get_buffer();
  POP_RANGE();
  return std::make_tuple(std::move(unique_element_indices), std::move(group_indices));
}

}  // namespace libperfect
