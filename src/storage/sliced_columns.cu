#include <gqe/storage/in_memory.hpp>

namespace gqe {

namespace storage {

template <typename offsets_type>
__global__ void adjust_offsets_kernel(offsets_type* offsets,
                                      size_t num_offsets,
                                      size_t partition_size,
                                      const offsets_type* partition_offsets,
                                      size_t char_array_size)
{
  // Thread ix -- I think there was some cool cub approach for this?
  int ix_thread = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_size = gridDim.x * blockDim.x;

  // We'll get a scan of the partition offsets and include it in the batched memcpy
  for (int ix = ix_thread; ix < num_offsets; ix += grid_size) {
    int ix_partition = ix / partition_size;

    if (ix == num_offsets - 1) {
      // Last thread will also fill in the final offset value (non-inclusive upper bound)
      offsets[num_offsets] = char_array_size;
    }
    if (ix_partition == 0) { continue; }

    offsets[ix] += partition_offsets[ix_partition];
  }
}

template <typename offsets_type>
void adjust_offsets_api(offsets_type* offsets,
                        size_t num_offsets,
                        size_t partition_size,
                        const offsets_type* partition_offsets,
                        size_t char_array_size,
                        rmm::cuda_stream_view stream)
{
  const int block_dim = 128;
  const int grid_dim  = gqe::utility::divide_round_up(num_offsets, block_dim);
  adjust_offsets_kernel<<<grid_dim, block_dim, 0, stream>>>(
    offsets, num_offsets, partition_size, partition_offsets, char_array_size);
}

// Explicit template instantiations
template void adjust_offsets_api<int32_t>(int32_t* offsets,
                                          size_t num_offsets,
                                          size_t partition_size,
                                          const int32_t* partition_offsets,
                                          size_t char_array_size,
                                          rmm::cuda_stream_view stream);

template void adjust_offsets_api<int64_t>(int64_t* offsets,
                                          size_t num_offsets,
                                          size_t partition_size,
                                          const int64_t* partition_offsets,
                                          size_t char_array_size,
                                          rmm::cuda_stream_view stream);

}  // namespace storage

}  // namespace gqe
