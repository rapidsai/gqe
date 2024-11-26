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
#include <gqe/storage/nvcomp_gqe.hpp>

nvcomp_gqe_allocator::nvcomp_gqe_allocator(rmm::cuda_stream_view stream) : _stream(stream) {}

void* nvcomp_gqe_allocator::allocate(std::size_t bytes)
{
  return rmm::mr::get_current_device_resource()->allocate(bytes, _stream);
}

void nvcomp_gqe_allocator::deallocate(void* ptr, std::size_t bytes)
{
  rmm::mr::get_current_device_resource()->deallocate(ptr, bytes, _stream);
}

nvcomp_gqe::nvcomp_gqe(rmm::cuda_stream_view supplied_stream,
                       gqe::compression_format comp_format,
                       nvcompType_t data_format,
                       int explicit_chunk_size)
  : stream(supplied_stream),
    comp_format(comp_format),
    data_type(data_format),
    chunk_size(explicit_chunk_size),
    allocator(supplied_stream),
    is_compressed(true)
{
  int status = create_manager();
  if (status == 0) { GQE_LOG_ERROR("nvcomp object initialization failed"); }
}

std::unique_ptr<rmm::device_buffer> nvcomp_gqe::do_compress(rmm::device_buffer const* uncompressed,
                                                            bool& compression_viable,
                                                            rmm::device_async_resource_ref mr)
{
  compression_viable = true;
  if (uncompressed->size() <= 0) { return std::make_unique<rmm::device_buffer>(0, stream); }

  auto comp_config = manager->configure_compression(uncompressed->size());
  auto compressed_buffer =
    std::make_unique<rmm::device_buffer>(comp_config.max_compressed_buffer_size, stream, mr);

  manager->compress(static_cast<uint8_t const*>(uncompressed->data()),
                    static_cast<uint8_t*>(compressed_buffer->data()),
                    comp_config);

  auto const comp_size =
    manager->get_compressed_output_size(static_cast<uint8_t*>(compressed_buffer->data()));
  compression_ratio = static_cast<float>(uncompressed->size()) / comp_size;

  compressed_buffer->resize(comp_size, stream);
  compressed_buffer->shrink_to_fit(stream);
  manager->deallocate_gpu_mem();

  if (comp_size > uncompressed->size()) {
    compression_viable = false;
    compression_ratio  = 1.0;
    is_compressed      = false;
    return std::make_unique<rmm::device_buffer>(*uncompressed, stream, mr);
  }

  return compressed_buffer;
}

std::unique_ptr<rmm::device_buffer> nvcomp_gqe::do_decompress(
  rmm::device_buffer const* compressed, rmm::device_async_resource_ref mr) const
{
  if (is_compressed) {
    rmm::device_buffer device_memory_compressed{compressed->data(), compressed->size(), stream, mr};
    auto comp_buffer = static_cast<uint8_t const*>(device_memory_compressed.data());

    auto decomp_config = manager->configure_decompression(comp_buffer);
    auto decompressed_buffer =
      std::make_unique<rmm::device_buffer>(decomp_config.decomp_data_size, stream);

    manager->decompress(
      static_cast<uint8_t*>(decompressed_buffer->data()), comp_buffer, decomp_config);
    manager->deallocate_gpu_mem();

    return decompressed_buffer;
  } else {
    return std::make_unique<rmm::device_buffer>(compressed->data(), compressed->size(), stream, mr);
  }
}

void nvcomp_gqe::print_usage()
{
  printf(
    "Incorrect object initialization\nUsage: nvcompgqe object_name ([stream], "
    "[compression_format], [data_format], [chunk_size]) \n");
  printf("  %-60s One of < ans / cascaded / gdeflate / deflate / lz4 / snappy / zstd >\n",
         "[compression_format]");
  printf(
    "  %-60s Data format Options are < char / short / int / longlong / bits > (default value is "
    "'char')\n",
    "[data_format]");
  printf("  %-60s Chunk size (default value is 64 kB).\n", "[chunk_size]");
}

int nvcomp_gqe::create_manager()
{
  if (comp_format == gqe::compression_format::lz4) {
    manager = std::make_shared<LZ4Manager>(
      chunk_size, nvcompBatchedLZ4Opts_t{data_type}, stream, NoComputeNoVerify);
  } else if (comp_format == gqe::compression_format::snappy) {
    manager = std::make_shared<SnappyManager>(
      chunk_size, nvcompBatchedSnappyOpts_t{}, stream, NoComputeNoVerify);
  } else if (comp_format == gqe::compression_format::ans) {
    manager = std::make_shared<ANSManager>(
      chunk_size, nvcompBatchedANSDefaultOpts, stream, NoComputeNoVerify);
  } else if (comp_format == gqe::compression_format::cascaded) {
    nvcompBatchedCascadedOpts_t cascaded_opts = nvcompBatchedCascadedDefaultOpts;
    cascaded_opts.type                        = data_type;
    cascaded_opts.internal_chunk_bytes        = chunk_size;
    manager =
      std::make_shared<CascadedManager>(chunk_size, cascaded_opts, stream, NoComputeNoVerify);
  } else if (comp_format == gqe::compression_format::gdeflate) {
    manager = std::make_shared<GdeflateManager>(
      chunk_size, nvcompBatchedGdeflateOpts_t{0}, stream, NoComputeNoVerify);
  } else if (comp_format == gqe::compression_format::deflate) {
    manager = std::make_shared<DeflateManager>(
      chunk_size, nvcompBatchedDeflateDefaultOpts, stream, NoComputeNoVerify);
  } else if (comp_format == gqe::compression_format::zstd) {
    manager = std::make_shared<ZstdManager>(
      static_cast<size_t>(chunk_size), nvcompBatchedZstdDefaultOpts, stream, NoComputeNoVerify);
  } else if (comp_format == gqe::compression_format::bitcomp) {
    manager = std::make_shared<BitcompManager>(
      chunk_size, nvcompBatchedBitcompFormatOpts{0, data_type}, stream, NoComputeNoVerify);
  } else {
    print_usage();
    return 0;
  }

  auto alloc_fn   = std::bind(&nvcomp_gqe_allocator::allocate, &allocator, std::placeholders::_1);
  auto dealloc_fn = std::bind(
    &nvcomp_gqe_allocator::deallocate, &allocator, std::placeholders::_1, std::placeholders::_2);
  manager->set_scratch_allocators(alloc_fn, dealloc_fn);

  return 1;
}

gqe::compression_format nvcomp_gqe::get_comp_format() const { return comp_format; }

nvcompType_t nvcomp_gqe::get_data_type() const { return data_type; }

int nvcomp_gqe::get_chunk_size() const { return chunk_size; }

float nvcomp_gqe::get_compression_ratio() const { return compression_ratio; }

bool nvcomp_gqe::is_column_compressed() const { return is_compressed; }