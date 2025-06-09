/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdio>
#include <gqe/storage/compression.hpp>

compression_manager::compression_manager(gqe::compression_format comp_format,
                                         nvcompType_t data_format,
                                         int explicit_chunk_size)
  : _comp_format(comp_format), _data_type(data_format), _chunk_size(explicit_chunk_size)
{
}

std::unique_ptr<rmm::device_buffer> compression_manager::do_compress(
  rmm::device_buffer const* uncompressed,
  float& compression_ratio,
  bool& is_compressed,
  rmm::cuda_stream_view supplied_stream,
  rmm::device_async_resource_ref mr)
{
  is_compressed = true;
  if (uncompressed->size() <= 0) {
    return std::make_unique<rmm::device_buffer>(0, supplied_stream, mr);
  }

  auto manager           = create_manager(supplied_stream, mr);
  auto comp_config       = manager->configure_compression(uncompressed->size());
  auto compressed_buffer = std::make_unique<rmm::device_buffer>(
    comp_config.max_compressed_buffer_size, supplied_stream, mr);

  manager->compress(static_cast<uint8_t const*>(uncompressed->data()),
                    static_cast<uint8_t*>(compressed_buffer->data()),
                    comp_config);

  auto const comp_size =
    manager->get_compressed_output_size(static_cast<uint8_t*>(compressed_buffer->data()));
  compression_ratio = static_cast<float>(uncompressed->size()) / comp_size;

  compressed_buffer->resize(comp_size, supplied_stream);
  compressed_buffer->shrink_to_fit(supplied_stream);
  manager->deallocate_gpu_mem();

  if (comp_size > uncompressed->size()) {
    is_compressed = false;
    return std::make_unique<rmm::device_buffer>(*uncompressed, supplied_stream, mr);
  }

  return compressed_buffer;
}

std::unique_ptr<rmm::device_buffer> compression_manager::do_decompress(
  rmm::device_buffer const* compressed,
  rmm::cuda_stream_view supplied_stream,
  rmm::device_async_resource_ref mr)
{
  rmm::device_buffer device_memory_compressed{
    compressed->data(), compressed->size(), supplied_stream, mr};
  auto comp_buffer = static_cast<uint8_t const*>(device_memory_compressed.data());

  auto manager       = create_manager(supplied_stream, mr);
  auto decomp_config = manager->configure_decompression(comp_buffer);
  auto decompressed_buffer =
    std::make_unique<rmm::device_buffer>(decomp_config.decomp_data_size, supplied_stream, mr);

  manager->decompress(
    static_cast<uint8_t*>(decompressed_buffer->data()), comp_buffer, decomp_config);
  manager->deallocate_gpu_mem();

  return decompressed_buffer;
}

void compression_manager::print_usage() const
{
  GQE_LOG_ERROR(
    "Incorrect object initialization\nUsage: nvcompgqe object_name ([stream], "
    "[compression_format], [data_format], [chunk_size]) \n");
  GQE_LOG_ERROR("  %-60s One of < ans / cascaded / gdeflate / deflate / lz4 / snappy / zstd >\n",
                "[compression_format]");
  GQE_LOG_ERROR(
    "  %-60s Data format Options are < char / short / int / longlong / bits > (default value is "
    "'char')\n",
    "[data_format]");
  GQE_LOG_ERROR("  %-60s Chunk size (default value is 64 kB).\n", "[chunk_size]");
}

std::unique_ptr<nvcompManagerBase> compression_manager::create_manager(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref& mr) const
{
  std::unique_ptr<nvcompManagerBase> manager;

  switch (_comp_format) {
    case gqe::compression_format::lz4:
      manager = std::make_unique<LZ4Manager>(
        _chunk_size, nvcompBatchedLZ4Opts_t{_data_type}, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::snappy:
      manager = std::make_unique<SnappyManager>(
        _chunk_size, nvcompBatchedSnappyOpts_t{}, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::ans:
      manager = std::make_unique<ANSManager>(
        _chunk_size, nvcompBatchedANSDefaultOpts, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::cascaded: {
      nvcompBatchedCascadedOpts_t cascaded_opts = nvcompBatchedCascadedDefaultOpts;
      cascaded_opts.type                        = _data_type;
      cascaded_opts.internal_chunk_bytes        = _chunk_size;
      manager =
        std::make_unique<CascadedManager>(_chunk_size, cascaded_opts, stream, NoComputeNoVerify);
      break;
    }
    case gqe::compression_format::gdeflate:
      manager = std::make_unique<GdeflateManager>(
        _chunk_size, nvcompBatchedGdeflateOpts_t{0}, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::deflate:
      manager = std::make_unique<DeflateManager>(
        _chunk_size, nvcompBatchedDeflateOpts_t{5}, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::zstd:
      manager = std::make_unique<ZstdManager>(
        static_cast<size_t>(_chunk_size), nvcompBatchedZstdDefaultOpts, stream, NoComputeNoVerify);
      break;
    case gqe::compression_format::bitcomp:
      manager = std::make_unique<BitcompManager>(
        _chunk_size, nvcompBatchedBitcompFormatOpts{0, _data_type}, stream, NoComputeNoVerify);
      break;
    default: GQE_LOG_ERROR("Unrecognized Compression Format"); break;
  }

  auto alloc_fn = [&mr, stream](std::size_t bytes) {
    return mr.allocate_async(bytes, alignof(std::max_align_t), stream);
  };

  auto dealloc_fn = [&mr, stream](void* ptr, std::size_t bytes) {
    mr.deallocate_async(ptr, bytes, alignof(std::max_align_t), stream);
  };

  manager->set_scratch_allocators(alloc_fn, dealloc_fn);

  return manager;
}

gqe::compression_format compression_manager::get_comp_format() const { return _comp_format; }

nvcompType_t compression_manager::get_data_type() const { return _data_type; }

int compression_manager::get_chunk_size() const { return _chunk_size; }