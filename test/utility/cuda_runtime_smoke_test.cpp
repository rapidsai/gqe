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

#include <cuda_runtime_api.h>

#include <iostream>

namespace {

bool check_cuda(cudaError_t status, char const* operation, int device)
{
  if (status == cudaSuccess) { return true; }

  std::cerr << "CUDA runtime smoke test failed for visible device " << device << " during "
            << operation << ": " << cudaGetErrorName(status) << " (" << cudaGetErrorString(status)
            << ")\n";
  return false;
}

}  // namespace

int main()
{
  int device_count{};
  if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", -1)) { return 1; }
  if (device_count == 0) {
    std::cerr << "CUDA runtime smoke test found no visible devices\n";
    return 1;
  }

  for (int device = 0; device < device_count; ++device) {
    if (!check_cuda(cudaSetDevice(device), "cudaSetDevice", device)) { return 1; }

    void* allocation{};
    if (!check_cuda(cudaMalloc(&allocation, 1), "cudaMalloc", device)) { return 1; }
    if (!check_cuda(cudaFree(allocation), "cudaFree", device)) { return 1; }
  }

  std::cout << "CUDA runtime smoke test passed for " << device_count << " visible device(s)\n";
  return 0;
}
