#include "xor_hash_table.cuh"

#include "unique_indices.hpp"

namespace libperfect {

std::tuple<CudaGpuArray<cudf::size_type>, CudaGpuArray<cudf::size_type>> unique_indices(
  const std::vector<ConstCudaGpuBufferPointer>& keys,
  const size_t keys_numel,
  const std::optional<ConstCudaGpuBufferPointer>& mask)
{
  auto hash_table = xor_hash_table::make_hash_table(keys, keys_numel);
  return hash_table
    .template bulk_insert<xor_hash_table::CheckEquality::True, xor_hash_table::InsertOutput::True>(
      keys, keys_numel, mask);
}

}  // namespace libperfect
