/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace libperfect {

// A thread-safe memory pool for CPU memory allocation.
class MemoryPool {
 public:
  explicit MemoryPool(size_t block_size);
  // Rule of 5.
  ~MemoryPool();
  MemoryPool(const MemoryPool&)                = delete;
  MemoryPool& operator=(const MemoryPool&)     = delete;
  MemoryPool(MemoryPool&&) noexcept            = default;
  MemoryPool& operator=(MemoryPool&&) noexcept = default;

  void* allocate(size_t size);
  void deallocate(void* ptr);

  void reset();

 private:
  mutable std::mutex mutex_;

  std::unordered_map<size_t, std::vector<void*>> unused_blocks_;
  std::unordered_map<void*, size_t> used_block_sizes_;

  size_t round_up_to_block_size(size_t size) const;
  size_t block_size_;
};

class GlobalMemoryPool {
 public:
  static MemoryPool& get();
  GlobalMemoryPool(const GlobalMemoryPool&)                = delete;
  GlobalMemoryPool& operator=(const GlobalMemoryPool&)     = delete;
  GlobalMemoryPool(GlobalMemoryPool&&) noexcept            = default;
  GlobalMemoryPool& operator=(GlobalMemoryPool&&) noexcept = default;

 private:
  GlobalMemoryPool() {}
};

}  // namespace libperfect
