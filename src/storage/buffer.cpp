/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/storage/buffer.hpp>

#include <gqe/utility/error.hpp>

#include <cuda_runtime_api.h>

#include <stdexcept>

namespace gqe {
namespace storage {

void buffer::allocate(std::size_t bytes, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  if (_data != nullptr) { throw std::logic_error("gqe::storage::buffer is already allocated"); }
  if (bytes == 0) {
    _size = 0;
    return;
  }

  _data = static_cast<std::byte*>(mr.allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT));
  _size = bytes;
}

buffer::buffer(rmm::device_buffer const& source,
               cuda::stream_ref stream,
               rmm::device_async_resource_ref mr)
{
  allocate(source.size(), stream, mr);
  if (source.size() == 0) { return; }

  GQE_CUDA_TRY(
    cudaMemcpyAsync(_data.get(), source.data(), source.size(), cudaMemcpyDefault, stream.get()));
}

void buffer::deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  if (_data != nullptr) {
    mr.deallocate(stream, _data.get(), _size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  _data = nullptr;
  _size = 0;
}

}  // namespace storage
}  // namespace gqe
