#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
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
