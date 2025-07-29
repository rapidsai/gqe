#include "memory_pool.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess) {
    std::cerr << file << ":" << line << ": GPUassert: " << cudaGetErrorString(code) << "\n";
    if (abort) { exit(code); }
  }
}

#define gpu_assert(ans)                   \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }

inline void gpuThrow(cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess) {
    std::stringstream what;
    what << file << ":" << line << ": GPUassert: " << cudaGetErrorString(code);
    std::cout << what.str() << std::endl;
    throw std::logic_error(what.str());
  }
}

#define gpu_throw(ans)                   \
  do {                                   \
    gpuThrow((ans), __FILE__, __LINE__); \
  } while (0)

MemoryPool::MemoryPool(const size_t block_size) : block_size_(block_size) {}

MemoryPool::~MemoryPool() { reset(); }

void* MemoryPool::allocate(size_t size)
{
  if (size == 0) { return nullptr; }

  std::lock_guard<std::mutex> lock(mutex_);

  size_t block_size = round_up_to_block_size(size);

  // Find or create the list of unused blocks for this block size.
  auto& unused_blocks = unused_blocks_[block_size];

  if (unused_blocks.empty()) {
    void* buffer;
    gpu_throw(cudaMallocHost(reinterpret_cast<void**>(&buffer), block_size));
    unused_blocks.push_back(buffer);
  }

  // Get an unused block.
  void* block = unused_blocks.back();
  unused_blocks.pop_back();
  used_block_sizes_[block] = block_size;
  return block;
}

void MemoryPool::deallocate(void* ptr)
{
  if (!ptr) { return; }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = used_block_sizes_.find(ptr);
  if (it == used_block_sizes_.end()) {
    throw std::runtime_error("MemoryPool: Failed to deallocate memory");
  }

  size_t block_size = it->second;

  unused_blocks_[block_size].push_back(ptr);
  used_block_sizes_.erase(it);
}

void MemoryPool::reset()
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (used_block_sizes_.size() != 0) {
    throw std::runtime_error("MemoryPool: Can't free with used blocks");
  }

  // Free all unused_blocks
  for (auto& [block_size, unused_blocks] : unused_blocks_) {
    for (auto& block : unused_blocks) {
      gpu_throw(cudaFreeHost(block));
    }
    unused_blocks.clear();
  }
  unused_blocks_.clear();
}

size_t MemoryPool::round_up_to_block_size(size_t size) const
{
  if (size <= block_size_) { return block_size_; }

  // Round up to the nearest power of 2 block size
  size_t block_size = block_size_;
  while (block_size < size) {
    block_size *= 2;
  }

  return block_size;
}

// GlobalMemoryPool implementation

MemoryPool& GlobalMemoryPool::get()
{
  static MemoryPool pool(8);
  return pool;
}
