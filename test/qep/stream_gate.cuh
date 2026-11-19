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

#pragma once

#include <gqe/utility/error.hpp>

#include <cuda/stream>
#include <rmm/device_scalar.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace gqe {
namespace test {
namespace detail {

/** @brief Block a CUDA stream until another stream writes a nonzero gate value. */
static __global__ void wait_for_stream_gate(std::uint32_t const* value)
{
  auto const* gate = static_cast<std::uint32_t const volatile*>(value);
  while (*gate == 0) {
    __nanosleep(1000);
  }
}

}  // namespace detail

/**
 * @brief Device-side gate that leaves the host free while selected CUDA streams are blocked.
 *
 * The release stream and every stream passed to `wait` must remain valid until this object is
 * destroyed. Destruction opens a closed gate and synchronizes all participating streams so that a
 * failed assertion cannot leave a waiting kernel running.
 */
class stream_gate {
 public:
  explicit stream_gate(cuda::stream_ref release_stream)
    : _release_stream{release_stream}, _value{release_stream}
  {
    GQE_CUDA_TRY(cudaMemsetAsync(_value.data(), 0, sizeof(std::uint32_t), release_stream.get()));
    release_stream.sync();
  }

  ~stream_gate()
  {
    if (!_is_open) {
      GQE_CUDA_TRY_NO_THROW(
        cudaMemsetAsync(_value.data(), 1, sizeof(std::uint32_t), _release_stream.get()));
    }
    GQE_CUDA_TRY_NO_THROW(cudaStreamSynchronize(_release_stream.get()));
    for (auto const stream : _waiting_streams) {
      GQE_CUDA_TRY_NO_THROW(cudaStreamSynchronize(stream.get()));
    }
  }

  stream_gate(stream_gate const&)            = delete;
  stream_gate& operator=(stream_gate const&) = delete;

  /** @brief Enqueue a wait on `stream` without blocking the test thread. */
  void wait(cuda::stream_ref stream)
  {
    GQE_EXPECTS(stream.get() != _release_stream.get(),
                "stream_gate::wait() cannot wait on the release stream");
    if (std::ranges::none_of(_waiting_streams, [stream](auto const candidate) {
          return candidate.get() == stream.get();
        })) {
      _waiting_streams.push_back(stream);
    }
    detail::wait_for_stream_gate<<<1, 1, 0, stream.get()>>>(_value.data());
    GQE_CUDA_TRY(cudaPeekAtLastError());
  }

  /** @brief Open the gate asynchronously on the release stream. */
  void release()
  {
    if (_is_open) { return; }
    GQE_CUDA_TRY(cudaMemsetAsync(_value.data(), 1, sizeof(std::uint32_t), _release_stream.get()));
    _is_open = true;
  }

 private:
  cuda::stream_ref _release_stream;
  rmm::device_scalar<std::uint32_t> _value;
  std::vector<cuda::stream_ref> _waiting_streams;
  bool _is_open{};
};

}  // namespace test
}  // namespace gqe
