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

#include "nvcomp_adapter.hpp"

#include <gqe/storage/compression.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/utilities/pinned_memory.hpp>
#include <nvcomp/lz4_cpu.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>

namespace nvcomp {
// stubs
struct CPUHLIFManager {
  size_t get_compressed_output_size(const uint8_t*) { return size_t{}; }
  size_t get_max_compressed_output_size(const size_t) { return size_t{}; }
  void cpu_batch_compress(uint8_t** compressed_ptrs,
                          const uint8_t* const* uncomp_ptrs,
                          const size_t* uncomp_sizes,
                          size_t* compressed_sizes,
                          const size_t batch_count,
                          const int num_threads,
                          const size_t* max_comp_sizes,
                          const bool benchmark_mode = false,
                          const int iteration_count = 1)
  {
  }
};
}  // namespace nvcomp

namespace gqe {
namespace storage {

namespace {

nvcompDecompressBackend_t to_nvcomp_decompress_backend(gqe::decompression_backend backend)
{
  switch (backend) {
    case gqe::decompression_backend::de: return NVCOMP_DECOMPRESS_BACKEND_HARDWARE;
    case gqe::decompression_backend::sm: return NVCOMP_DECOMPRESS_BACKEND_CUDA;
    case gqe::decompression_backend::default_: return NVCOMP_DECOMPRESS_BACKEND_DEFAULT;
  }
  throw std::logic_error("Unknown decompress backend");
}

}  // namespace

std::unique_ptr<nvcomp_manager_adapter> nvcomp_manager_adapter::create_manager(
  compression_manager const& comp_manager,
  gqe::compression_format comp_format,
  nvcompType_t data_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::unique_ptr<nvcomp::nvcompManagerBase> manager;
  auto const decompress_backend =
    to_nvcomp_decompress_backend(comp_manager.get_decompress_backend());

  constexpr bool use_de_sort =
    false;  // This flag determins whether or not to sort the chunks before HW
            // decompression for load balancing purposes. Should be disabled when
            // chunks are approximately the same size.
  int algorithm = 0;

  switch (comp_format) {
    case gqe::compression_format::lz4: {
      GQE_LOG_TRACE("Creating LZ4 compression manager for column '{}'",
                    comp_manager.get_column_name());
      if (data_type == NVCOMP_TYPE_LONGLONG) {
        // LZ4 doesn't support LONGLONG, downcast to INT
        data_type = NVCOMP_TYPE_INT;
      }

      nvcompBitshuffleMode_t bitshuffle_mode = NVCOMP_BITSHUFFLE_NONE;
      manager                                = std::make_unique<nvcomp::LZ4Manager>(
        comp_manager.get_compression_chunk_size(),
        nvcompBatchedLZ4CompressOpts_t{data_type, bitshuffle_mode, {0}},
        nvcompBatchedLZ4DecompressOpts_t{
          decompress_backend, use_de_sort ? 1 : 0, data_type, bitshuffle_mode, {0}},
        stream,
        nvcomp::NoComputeNoVerify);
      break;
    }
    case gqe::compression_format::snappy:
      GQE_LOG_TRACE("Creating Snappy compression manager for column '{}'",
                    comp_manager.get_column_name());
      manager = std::make_unique<nvcomp::SnappyManager>(
        comp_manager.get_compression_chunk_size(),
        nvcompBatchedSnappyCompressOpts_t{{0}},
        nvcompBatchedSnappyDecompressOpts_t{decompress_backend, use_de_sort ? 1 : 0, {0}},
        stream,
        nvcomp::NoComputeNoVerify);
      break;
    case gqe::compression_format::ans:
      GQE_LOG_TRACE("Creating ANS compression manager for column '{}'",
                    comp_manager.get_column_name());
      manager = std::make_unique<nvcomp::ANSManager>(
        comp_manager.get_compression_chunk_size(),
        nvcompBatchedANSCompressOpts_t{nvcomp_rANS, nvcompType_t::NVCOMP_TYPE_CHAR, {0}},
        nvcompBatchedANSDecompressDefaultOpts,
        stream,
        nvcomp::NoComputeNoVerify);
      break;
    case gqe::compression_format::cascaded: {
      GQE_LOG_TRACE("Creating Cascaded compression manager for column '{}'",
                    comp_manager.get_column_name());
      nvcompBatchedCascadedCompressOpts_t cascaded_opts = nvcompBatchedCascadedCompressDefaultOpts;
      cascaded_opts.type                                = data_type;
      manager =
        std::make_unique<nvcomp::CascadedManager>(comp_manager.get_compression_chunk_size(),
                                                  cascaded_opts,
                                                  nvcompBatchedCascadedDecompressDefaultOpts,
                                                  stream,
                                                  nvcomp::NoComputeNoVerify);
      break;
    }
    case gqe::compression_format::gdeflate:
      GQE_LOG_TRACE("Creating Gdeflate compression manager for column '{}'",
                    comp_manager.get_column_name());
      manager = std::make_unique<nvcomp::GdeflateManager>(
        comp_manager.get_compression_chunk_size(),
        nvcompBatchedGdeflateCompressOpts_t{algorithm, {0}},
        nvcompBatchedGdeflateDecompressDefaultOpts,
        stream,
        nvcomp::NoComputeNoVerify);
      break;
    case gqe::compression_format::deflate:
      GQE_LOG_TRACE("Creating Deflate compression manager for column '{}'",
                    comp_manager.get_column_name());
      manager = std::make_unique<nvcomp::DeflateManager>(
        comp_manager.get_compression_chunk_size(),
        nvcompBatchedDeflateCompressOpts_t{algorithm, {0}},
        nvcompBatchedDeflateDecompressOpts_t{decompress_backend, use_de_sort ? 1 : 0, {0}},
        stream,
        nvcomp::NoComputeNoVerify);
      break;
    case gqe::compression_format::zstd:
      GQE_LOG_TRACE("Creating Zstd compression manager for column '{}'",
                    comp_manager.get_column_name());
      manager = std::make_unique<nvcomp::ZstdManager>(
        static_cast<size_t>(comp_manager.get_compression_chunk_size()),
        nvcompBatchedZstdCompressDefaultOpts,
        nvcompBatchedZstdDecompressDefaultOpts,
        stream,
        nvcomp::NoComputeNoVerify);
      break;
    case gqe::compression_format::bitcomp:
      GQE_LOG_TRACE("Creating Bitcomp compression manager for column '{}'",
                    comp_manager.get_column_name());
      manager = std::make_unique<nvcomp::BitcompManager>(
        comp_manager.get_compression_chunk_size(),
        nvcompBatchedBitcompCompressOpts_t{algorithm, data_type, {0}},
        nvcompBatchedBitcompDecompressDefaultOpts,
        stream,
        nvcomp::NoComputeNoVerify);
      break;
    default:
      GQE_LOG_ERROR("Unrecognized Compression Format '{}' for column '{}'",
                    gqe::compression_format_to_string(comp_format),
                    comp_manager.get_column_name());
      break;
  }

  auto alloc_fn = [mr, stream](std::size_t bytes) mutable {
    auto ptr = mr.allocate_async(bytes, alignof(std::max_align_t), stream);
    return ptr;
  };

  auto dealloc_fn = [mr, stream](void* ptr, std::size_t bytes) mutable {
    mr.deallocate_async(ptr, bytes, alignof(std::max_align_t), stream);
  };

  manager->set_scratch_allocators(alloc_fn, dealloc_fn);

  auto manager_adapter = std::make_unique<nvcomp_manager_adapter>();
  manager_adapter->set_manager(std::move(manager));

  return manager_adapter;
}

void nvcomp_manager_adapter::set_manager(std::unique_ptr<nvcomp::nvcompManagerBase> manager)
{
  _manager = std::move(manager);
}

nvcomp::CompressionConfig nvcomp_manager_adapter::configure_compression(
  const size_t uncomp_buffer_size)
{
  GQE_EXPECTS(_manager, "No nvCOMP manager set");
  return _manager->configure_compression(uncomp_buffer_size);
}

nvcomp::DecompressionConfig nvcomp_manager_adapter::configure_decompression(
  const uint8_t* comp_buffer, const size_t* comp_size)
{
  GQE_EXPECTS(_manager, "No nvCOMP manager set");
  return _manager->configure_decompression(comp_buffer, comp_size);
}

size_t nvcomp_manager_adapter::get_compressed_output_size_host(const uint8_t* comp_buffer)
{
  GQE_EXPECTS(_manager, "No nvCOMP manager set");
  return _manager->get_compressed_output_size(comp_buffer);
}

size_t nvcomp_manager_adapter::get_decompressed_output_size_host(const uint8_t* comp_buffer)
{
  GQE_EXPECTS(_manager, "No nvCOMP manager set");
  return _manager->get_decompressed_output_size(comp_buffer);
}

void nvcomp_manager_adapter::compress(const uint8_t* uncomp_buffer,
                                      uint8_t* comp_buffer,
                                      const nvcomp::CompressionConfig& comp_config,
                                      size_t* comp_size)
{
  GQE_EXPECTS(_manager, "No nvCOMP manager set");
  _manager->compress(uncomp_buffer, comp_buffer, comp_config, comp_size);
}

void nvcomp_manager_adapter::compress(const uint8_t* const* uncomp_buffers,
                                      uint8_t* const* comp_buffers,
                                      const std::vector<nvcomp::CompressionConfig>& comp_configs,
                                      size_t* comp_sizes)
{
  GQE_EXPECTS(_manager, "No nvCOMP manager set");
  _manager->compress(uncomp_buffers, comp_buffers, comp_configs, comp_sizes);
}

std::vector<std::unique_ptr<rmm::device_buffer>> nvcomp_manager_adapter::compress_batch(
  gqe::storage::compression_manager const& comp_manager,
  const std::vector<std::unique_ptr<rmm::device_buffer>>& device_uncompressed,
  const size_t total_uncompressed_size,
  uint8_t** uncompressed_ptrs,
  gqe::compression_format comp_format,
  nvcompType_t data_type,
  cudf::data_type cudf_type,
  size_t num_buffers,
  memory_kind::type memory_kind,
  uint8_t** compressed_ptrs,
  double& compression_ratio,
  size_t& total_compressed_size,
  std::vector<cudf::size_type>& compressed_sizes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  compressed_sizes.clear();
  compressed_sizes.reserve(num_buffers);
  // struct to enforce order of destruction, ensuring `CompressionConfig`s are destroyed before
  // `nvcomp_manager_adapter`
  struct compression_manager_with_configs {
    std::unique_ptr<nvcomp_manager_adapter> compression_manager;
    std::vector<nvcomp::CompressionConfig> compression_configs;
  };
  compression_manager_with_configs man_conf;
  bool compression_mr_is_host_accessible = memory_kind::is_cpu_accessible(memory_kind);
  man_conf.compression_manager = create_manager(comp_manager, comp_format, data_type, stream, mr);
  man_conf.compression_configs.reserve(num_buffers);
  std::vector<std::unique_ptr<rmm::device_buffer>> compressed_data_buffers;

  for (size_t ix = 0; ix < num_buffers; ix++) {
    auto& uncompressed = device_uncompressed[ix];
    auto config        = man_conf.compression_manager->configure_compression(uncompressed->size());
    man_conf.compression_configs.push_back(config);
    compressed_data_buffers.push_back(
      std::make_unique<rmm::device_buffer>(config.max_compressed_buffer_size, stream, mr));
    compressed_ptrs[ix] = static_cast<uint8_t*>(compressed_data_buffers.back()->data());
  }
  man_conf.compression_manager->compress(
    uncompressed_ptrs, compressed_ptrs, man_conf.compression_configs);

  total_compressed_size = 0;
  for (size_t ix = 0; ix < num_buffers; ix++) {
    size_t comp_size = 0;
    if (compression_mr_is_host_accessible) {
      comp_size =
        man_conf.compression_manager->get_compressed_output_size_host(compressed_ptrs[ix]);

    } else {
      comp_size = man_conf.compression_manager->get_compressed_output_size(compressed_ptrs[ix]);
    }

    assert(comp_size <= man_conf.compression_configs[ix].max_compressed_buffer_size);

    compressed_sizes.push_back(comp_size);
    total_compressed_size += comp_size;
  }
  compression_ratio = static_cast<double>(total_uncompressed_size) / total_compressed_size;
  return compressed_data_buffers;
}

void nvcomp_manager_adapter::decompress(uint8_t* decomp_buffer,
                                        const uint8_t* comp_buffer,
                                        const nvcomp::DecompressionConfig& decomp_config,
                                        size_t* comp_size)
{
  GQE_EXPECTS(_manager, "No nvCOMP manager set");
  _manager->decompress(decomp_buffer, comp_buffer, decomp_config, comp_size);
}

void nvcomp_manager_adapter::decompress(
  uint8_t* const* decomp_buffers,
  const uint8_t* const* device_comp_buffers,
  const std::vector<nvcomp::DecompressionConfig>& decomp_configs,
  const size_t batch_count,
  const size_t* comp_sizes,
  const uint8_t* const* host_comp_buffers)
{
  GQE_EXPECTS(_manager, "No nvCOMP manager set");

  _manager->decompress(decomp_buffers,
                       device_comp_buffers,
                       decomp_configs,
                       comp_sizes,
                       batch_count,
                       host_comp_buffers);
}

size_t nvcomp_manager_adapter::get_compressed_output_size(const uint8_t* comp_buffer)
{
  GQE_EXPECTS(_manager, "No nvCOMP manager set");
  return _manager->get_compressed_output_size(comp_buffer);
}

nvcomp_cpu_manager_adapter::nvcomp_cpu_manager_adapter(nvcomp_cpu_manager_adapter&&) = default;
nvcomp_cpu_manager_adapter& nvcomp_cpu_manager_adapter::operator=(nvcomp_cpu_manager_adapter&&) =
  default;
nvcomp_cpu_manager_adapter::~nvcomp_cpu_manager_adapter() = default;

std::unique_ptr<nvcomp_cpu_manager_adapter> nvcomp_cpu_manager_adapter::create_cpu_manager(
  compression_manager const& comp_manager,
  gqe::compression_format comp_format,
  int compression_level)
{
  std::unique_ptr<nvcomp_cpu_manager_adapter> cpu_manager_adapter;

  GQE_EXPECTS(comp_format == gqe::compression_format::lz4,
              "Unsupported compression format for CPU compression manager");
  std::unique_ptr<nvcomp::LZ4CPUManager> lz4_cpu_manager;
  lz4_cpu_manager = std::make_unique<nvcomp::LZ4CPUManager>(
    comp_manager.get_compression_chunk_size(),
    compression_level,
    0 /* num_threads = 0 uses maximum available hardware concurrency threads */);
  cpu_manager_adapter = std::make_unique<nvcomp_cpu_manager_adapter>();
  cpu_manager_adapter->set_lz4_cpu_manager(std::move(lz4_cpu_manager));

  return cpu_manager_adapter;
}

void nvcomp_cpu_manager_adapter::set_cpu_manager(
  std::unique_ptr<nvcomp::CPUHLIFManager> cpu_manager)
{
  throw std::runtime_error("`set_cpu_manager` is not available in nvcomp < 5.2");
  _cpu_manager = std::move(cpu_manager);
}

void nvcomp_cpu_manager_adapter::set_lz4_cpu_manager(
  std::unique_ptr<nvcomp::LZ4CPUManager> lz4_cpu_manager)
{
  _lz4_cpu_manager = std::move(lz4_cpu_manager);
}

size_t nvcomp_cpu_manager_adapter::get_compressed_output_size(const uint8_t* comp_buffer)
{
  GQE_EXPECTS(_lz4_cpu_manager, "No LZ4 CPU nvCOMP manager set");
  return _lz4_cpu_manager->get_compressed_output_size(comp_buffer);
}

size_t nvcomp_cpu_manager_adapter::get_max_compressed_output_size(const size_t input_size)
{
  GQE_EXPECTS(_lz4_cpu_manager, "No LZ4 CPU nvCOMP manager set");
  return _lz4_cpu_manager->configure_compression(input_size).max_compressed_buffer_size;
}

void nvcomp_cpu_manager_adapter::cpu_batch_compress(uint8_t** compressed_ptrs,
                                                    const uint8_t* const* uncomp_ptrs,
                                                    const size_t* uncomp_sizes,
                                                    size_t* compressed_sizes,
                                                    const size_t batch_count,
                                                    const int num_threads,
                                                    const size_t* max_comp_sizes,
                                                    const bool benchmark_mode,
                                                    const int iteration_count)
{
  GQE_EXPECTS(_lz4_cpu_manager, "No LZ4 CPU nvCOMP manager set");
  std::vector<size_t> uncomp_sizes_vec(uncomp_sizes, uncomp_sizes + batch_count);
  auto comp_configs = _lz4_cpu_manager->configure_compression(uncomp_sizes_vec);
  _lz4_cpu_manager->compress(uncomp_ptrs, compressed_ptrs, comp_configs, compressed_sizes);
}

std::vector<std::unique_ptr<rmm::device_buffer>> nvcomp_cpu_manager_adapter::compress_batch(
  gqe::storage::compression_manager const& comp_manager,
  const std::vector<std::unique_ptr<rmm::device_buffer>>& device_uncompressed,
  const size_t total_uncompressed_size,
  uint8_t** uncompressed_ptrs,
  gqe::compression_format comp_format,
  nvcompType_t data_type,
  cudf::data_type cudf_type,
  size_t num_buffers,
  memory_kind::type memory_kind,
  uint8_t** compressed_ptrs,
  double& compression_ratio,
  size_t& compressed_size,
  std::vector<cudf::size_type>& compressed_sizes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  assert(device_uncompressed.size() == num_buffers);

  compressed_sizes.clear();
  compressed_sizes.reserve(num_buffers);

  bool compression_mr_is_host_accessible = memory_kind::is_cpu_accessible(memory_kind);

  auto cpu_compression_manager =
    create_cpu_manager(comp_manager, comp_format, comp_manager.get_compression_level());

  std::vector<size_t> max_compressed_sizes;
  std::vector<size_t> uncompressed_sizes;
  std::vector<uint8_t*>
    host_compressed_ptrs;  // Points to the compressed data buffers in host-accessible memory
  std::vector<std::unique_ptr<rmm::device_buffer>>
    output_compressed_buffers;  // Points to the compressed data buffers in device memory (may be
                                // host-accessible or device-only)
  std::vector<std::vector<uint8_t>>
    host_uncompressed_buffers;  // Used only when mr is not host-accessible
  std::vector<std::vector<uint8_t>>
    host_compressed_buffers;  // Used only when mr is not host-accessible

  for (size_t ix = 0; ix < device_uncompressed.size(); ix++) {
    auto& uncompressed = device_uncompressed[ix];
    size_t max_compressed_size =
      cpu_compression_manager->get_max_compressed_output_size(uncompressed->size());
    max_compressed_sizes.push_back(max_compressed_size);
    uncompressed_sizes.push_back(uncompressed->size());

    output_compressed_buffers.push_back(
      std::make_unique<rmm::device_buffer>(max_compressed_size, stream, mr));
    compressed_ptrs[ix] = static_cast<uint8_t*>(output_compressed_buffers.back()->data());

    // Check if uncompressed data is host-accessible
    if (compression_mr_is_host_accessible) {
      // uncompressed_ptrs[ix] already points to host-accessible memory
      host_compressed_ptrs.push_back(compressed_ptrs[ix]);
      GQE_LOG_TRACE("compress_batch: Buffer is host-accessible, using pointer directly");
    } else {
      GQE_LOG_TRACE("compress_batch: Buffer is device-only memory, copying via cudaMemcpy D2H");
      // Data is device-only memory, need to copy via cudaMemcpy
      utility::nvtx_scoped_range nvtx_cpu_compress_dtoh("CPU_Compress_DtoH");
      host_uncompressed_buffers.push_back(std::vector<uint8_t>(uncompressed->size()));
      host_compressed_buffers.push_back(std::vector<uint8_t>(max_compressed_sizes[ix]));
      GQE_CUDA_TRY(cudaMemcpyAsync(host_uncompressed_buffers.back().data(),
                                   uncompressed->data(),
                                   uncompressed->size(),
                                   cudaMemcpyDeviceToHost,
                                   stream));
      // Update the uncompressed pointer to point to the host-accessible memory
      uncompressed_ptrs[ix] = host_uncompressed_buffers.back().data();
      host_compressed_ptrs.push_back(host_compressed_buffers.back().data());
    }
  }

  if (!compression_mr_is_host_accessible) {
    // Synchronize the stream to ensure the data is copied before we launch CPU compression (which
    // does not use CUDA stream).
    GQE_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  size_t total_compressed_size = 0;
  std::vector<size_t> batch_compressed_sizes(num_buffers);

  // Use lock_guard in a scope to serialize CPU batch compression
  {
    static std::mutex cpu_batch_compress_mutex;
    std::lock_guard<std::mutex> cpu_lock(cpu_batch_compress_mutex);
    // This object resets affinity to full-mask, enabling all threads to compress.
    scoped_cpu_affinity affinity_guard;

    utility::nvtx_scoped_range nvtx_cpu_compress("CPU_Compress");
    const int num_threads = std::thread::hardware_concurrency();
    GQE_LOG_TRACE(
      "Using CPU compression manager for batched compression of column '{}' with compression "
      "level {}",
      comp_manager.get_column_name(),
      comp_manager.get_compression_level());

    constexpr bool benchmark_mode = false;  // Disable benchmark mode for production
    constexpr int iteration_count = 1;      // Used only when benchmark_mode is true

    cpu_compression_manager->cpu_batch_compress(host_compressed_ptrs.data(),
                                                uncompressed_ptrs,
                                                uncompressed_sizes.data(),
                                                batch_compressed_sizes.data(),
                                                num_buffers,
                                                num_threads,
                                                max_compressed_sizes.data(),
                                                benchmark_mode,
                                                iteration_count);
  }

  if (!compression_mr_is_host_accessible) {
    // Copy the compressed data buffers back to the device memory
    utility::nvtx_scoped_range nvtx_cpu_compress_htod("CPU_Compress_HtoD");
    for (size_t ix = 0; ix < num_buffers; ix++) {
      GQE_CUDA_TRY(cudaMemcpyAsync(compressed_ptrs[ix],
                                   host_compressed_buffers[ix].data(),
                                   batch_compressed_sizes[ix],
                                   cudaMemcpyHostToDevice,
                                   stream));
    }
    GQE_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  for (size_t ix = 0; ix < num_buffers; ix++) {
    size_t batch_comp_size = batch_compressed_sizes[ix];
    assert(batch_comp_size <= max_compressed_sizes[ix]);
    compressed_sizes.push_back(batch_comp_size);
    total_compressed_size += batch_comp_size;
  }

  compressed_size   = total_compressed_size;
  compression_ratio = static_cast<double>(total_uncompressed_size) / total_compressed_size;
  return output_compressed_buffers;
}

}  // namespace storage
}  // namespace gqe
