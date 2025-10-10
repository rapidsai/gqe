#pragma once

#include <optional>
#include <type_traits>
#include <variant>

#include "condense.cuh"
#include "index_select.cuh"
#include "query_common.cuh"
#include "reduce_and_or.cuh"

namespace libperfect {
namespace xor_hash_table {

enum class HasMask : bool { False = false, True = true };

enum class CheckEquality : bool { False = false, True = true };

enum class MultiSet : bool { False = false, True = true };

enum class PerfectHashing : bool { False = false, True = true };

enum class ReturnAll : bool { False = false, True = true };

enum class InsertOutput : bool { False = false, True = true };

constexpr uint THREADS_PER_BLOCK = 128;

template <typename T>
__device__ std::make_unsigned_t<T> make_unsigned(T val)
{
  return static_cast<std::make_unsigned_t<T>>(val);
}

class Shift1Instruction {
 public:
  Shift1Instruction(uint8_t left0) : left0(left0) {}
  __host__ __device__ uint8_t get_left0() const { return left0; }
  __host__ __device__ constexpr bool is_last() const { return true; }
  __device__ uint64_t apply(uint64_t value) const { return value << left0; }

 private:
  uint8_t left0;
};

class Shift3Instruction {
 public:
  Shift3Instruction(uint8_t left0, uint8_t right1, uint8_t left2, uint8_t last)
    : left0(left0), right1(right1), left2(left2), last(last)
  {
  }
  __host__ __device__ uint8_t get_left0() const { return left0; }
  __host__ __device__ uint8_t get_right1() const { return right1; }
  __host__ __device__ uint8_t get_left2() const { return left2; }
  __host__ __device__ bool is_last() const { return last != 0; }
  __device__ uint64_t apply(uint64_t value) const
  {
    return (((value << left0) >> right1) << left2);
  }

 private:
  uint8_t left0;
  uint8_t right1;
  uint8_t left2;
  uint8_t last;  // If this is non-zero, we can move to the next word.
};

template <class instruction_class>
class InstructionHasher;  // Forward declaration for `using` below.

template <class instruction_class>
class InstructionHasherView {
 public:
  using cpu_class = InstructionHasher<instruction_class>;
  template <typename hash_key_type>
  __device__ uint64_t hash(const ConstCudaGpuBufferPointer* tables,
                           const hash_key_type& key_index) const
  {
    uint64_t ret                  = 0;
    int current_instruction_index = 0;
    for (size_t column_index = 0; column_index < get_column_count(); column_index++) {
      const uint64_t val = tables[column_index][key_index];
      while (true) {
        const auto& current_instruction = instructions[current_instruction_index];
        current_instruction_index++;
        ret ^= current_instruction.apply(val);
        if (current_instruction.is_last()) { break; }
      }
    }
    return ret;
  }
  __device__ size_t get_column_count() const { return column_count; }
  InstructionHasherView() = default;

 private:
  friend class InstructionHasher<instruction_class>;
  InstructionHasherView(const instruction_class* instructions, const size_t column_count)
    : instructions(instructions), column_count(column_count)
  {
  }
  const instruction_class* instructions;
  size_t column_count;
};

template <class instruction_class>
class InstructionHasher {
 public:
  using view_class = InstructionHasherView<instruction_class>;
  InstructionHasher(CudaGpuArray<instruction_class>&& instructions_gpu,
                    size_t column_count,
                    int total_significant_bits)
    : instructions_gpu(std::move(instructions_gpu)),
      column_count(column_count),
      total_significant_bits(total_significant_bits)
  {
  }
  bool is_perfect() const { return total_significant_bits <= 64; }
  // This function is only called if `is_perfect()` returns true.
  uint64_t max() const
  {
    // Shifting by 64 is tricky.
    if (total_significant_bits == 64) {
      return 0xffffffffffffffffULL;
    } else {
      return (1ULL << total_significant_bits) - 1;
    }
  }
  size_t get_column_count() const { return column_count; }
  InstructionHasherView<instruction_class> view()
  {
    return InstructionHasherView<instruction_class>(instructions_gpu.get().data(), column_count);
  }
  const InstructionHasherView<instruction_class> view() const
  {
    return InstructionHasherView<instruction_class>(instructions_gpu.get().data(), column_count);
  }

 private:
  CudaGpuArray<instruction_class> instructions_gpu;
  size_t column_count;
  int total_significant_bits;
};

template <typename... Ts>
class IdentityHasher;

template <typename... Ts>
class IdentityHasherView;

template <>
class IdentityHasherView<> {
 public:
  using cpu_class = IdentityHasher<>;
  __device__ size_t get_column_count() const { return 0; }
  template <typename hash_key_type>
  __device__ uint64_t hash(const ConstCudaGpuBufferPointer* tables,
                           const hash_key_type& key_index) const
  {
    return 0;
  }
};

template <typename T, typename... others>
class IdentityHasherView<T, others...> : public IdentityHasherView<others...> {
 public:
  using cpu_class = IdentityHasher<T, others...>;
  __device__ size_t get_column_count() const
  {
    return 1 + IdentityHasherView<others...>::get_column_count();
  }
  template <typename hash_key_type>
  __device__ uint64_t hash(const ConstCudaGpuBufferPointer* tables,
                           const hash_key_type& key_index) const
  {
    return (tables[0].template at<T const>(key_index) ^
            (IdentityHasherView<others...>::hash(tables + 1, key_index) << ((sizeof(T) * 8) % 64)));
  }
};

template <>
class IdentityHasher<> {
 public:
  using view_class = IdentityHasherView<>;
  IdentityHasher(int total_significant_bits) : total_significant_bits(total_significant_bits) {}
  bool is_perfect() const { return total_significant_bits <= 64; }
  uint64_t max() const
  {
    // Shifting by 64 is tricky.
    if (total_significant_bits == 64) {
      return 0xffffffffffffffffULL;
    } else {
      return (1ULL << total_significant_bits) - 1;
    }
  }
  IdentityHasherView<> view() { return IdentityHasherView<>(); }
  const IdentityHasherView<> view() const { return IdentityHasherView<>(); }

 private:
  int total_significant_bits;
};

template <typename T, typename... others>
class IdentityHasher<T, others...> : public IdentityHasher<others...> {
 public:
  using view_class = IdentityHasherView<T, others...>;
  IdentityHasher(int total_significant_bits)
    : IdentityHasher<others...>(total_significant_bits - sizeof(T)),
      total_significant_bits(total_significant_bits)
  {
  }
  bool is_perfect() const { return total_significant_bits <= 64; }
  uint64_t max() const
  {
    // Shifting by 64 is tricky.
    if (total_significant_bits == 64) {
      return 0xffffffffffffffffULL;
    } else {
      return (1ULL << total_significant_bits) - 1;
    }
  }
  IdentityHasherView<T, others...> view() { return IdentityHasherView<T, others...>(); }
  const IdentityHasherView<T, others...> view() const { return IdentityHasherView<T, others...>(); }

 private:
  int total_significant_bits;
};

template <typename T>
std::optional<IdentityHasher<T>> make_identity_hasher(const std::vector<CudaPinnedBuffer>& and_ors)
{
  if (and_ors.size() != 1) { return std::nullopt; }
  if (and_ors[0].get().element_size() != sizeof(T)) { return std::nullopt; }
  auto or_value                   = static_cast<T const*>(and_ors[0].get())[1];
  uint64_t total_significant_bits = sizeof(or_value) * 8 - clz(or_value);
  if (total_significant_bits > 20) { return std::nullopt; }
  return {IdentityHasher<T>(total_significant_bits)};
}

template <typename T0, typename T1>
std::optional<IdentityHasher<T0, T1>> make_identity_hasher(
  const std::vector<CudaPinnedBuffer>& and_ors)
{
  if (and_ors.size() != 2) { return std::nullopt; }
  if (and_ors[0].get().element_size() != sizeof(T0)) { return std::nullopt; }
  if (and_ors[1].get().element_size() != sizeof(T1)) { return std::nullopt; }
  auto or_value                   = static_cast<T1 const*>(and_ors[1].get())[1];
  uint64_t total_significant_bits = (sizeof(or_value) * 8) - clz(or_value) + (sizeof(T0) * 8);
  if (total_significant_bits > 20) { return std::nullopt; }
  return {IdentityHasher<T0, T1>(total_significant_bits)};
}

static std::optional<InstructionHasher<Shift1Instruction>> make_shift1_hasher(
  const std::vector<CudaPinnedBuffer>& and_ors)
{
  // Can we just concatenate columns?
  uint64_t total_significant_bits = 0;
  std::vector<Shift1Instruction> instructions;
  for (uint i = 0; i < and_ors.size(); i++) {
    uint64_t or_value;
    auto column_element_size = and_ors[i].get().element_size();
    switch (column_element_size) {
      case 1: or_value = static_cast<uint8_t const*>(and_ors[i].get())[1]; break;
      case 2: or_value = static_cast<uint16_t const*>(and_ors[i].get())[1]; break;
      case 4: or_value = static_cast<uint32_t const*>(and_ors[i].get())[1]; break;
      case 8: or_value = static_cast<uint64_t const*>(and_ors[i].get())[1]; break;
      default: throw std::invalid_argument(annotate_line("Unknown element size"));
    }
    instructions.emplace_back(total_significant_bits);
    auto significant_bits = (sizeof(or_value) * 8) - clz(or_value);
    total_significant_bits += significant_bits;
  }
  if (total_significant_bits > 20) {
    return std::nullopt;  // Too many bits to use.
  }
  return {InstructionHasher<Shift1Instruction>(
    CudaGpuArray(instructions), and_ors.size(), total_significant_bits)};
}

static InstructionHasher<Shift3Instruction> make_shift3_hasher(
  const std::vector<CudaPinnedBuffer>& xors)
{
  uint64_t total_significant_bits = 0;
  std::vector<Shift3Instruction> instructions;
  for (uint i = 0; i < xors.size(); i++) {
    uint64_t xor_value;
    switch (xors[i].element_size()) {
      case 1: xor_value = *static_cast<uint8_t const*>(xors[i].get()); break;
      case 2: xor_value = *static_cast<uint16_t const*>(xors[i].get()); break;
      case 4: xor_value = *static_cast<uint32_t const*>(xors[i].get()); break;
      case 8: xor_value = *static_cast<uint64_t const*>(xors[i].get()); break;
      default: throw std::invalid_argument(annotate_line("Unknown element size"));
    }
    int bits_consumed = 0;
    do {
      // TODO: If the total_significant_bits is more than 64, the xors
      // should wrap around and put bits in the LSBs again.

      // NOTE: ffs can return 0 if there are no ones.
      if (xor_value == 0) {
        // Special case for 0, we want none of the bits.  Lots of left
        // shifting with no right shifting can do this.
        instructions.emplace_back(32, 0, 32, true);
        break;
      }
      if ((~xor_value) == 0) {
        // Special case for all 1s, we want all bits placed.
        instructions.emplace_back(0, 0, total_significant_bits, true);
        total_significant_bits += sizeof(xor_value) * 8;
        break;
      }
      // By this point, xor_value must have at least some ones and some zeros in it.
      auto lsb_zeros = ffs(xor_value) - 1;
      // If there is no shifting, then we have some zeros as above.
      // If there is some shifting, we'll have new zeros added.
      xor_value >>= lsb_zeros;  // lsb_zeros must be in range 0-63 so this shift is safe.
      // ffs(~xor_value) can return 0 if there are no zeros
      // in xor_value but that is already guaranteed above.
      auto lsb_ones = ffs(~xor_value) - 1;
      xor_value >>= lsb_ones;  // lsb_zeros must be in range 0-63 so this shift is safe.
      // How many bits are zeros to the left of the lsb_ones?
      auto num_bits_to_mask_on_right = bits_consumed + lsb_zeros;
      // How many bits are zeros to the right of the lsb_ones?
      auto num_bits_to_mask_on_left = sizeof(xor_value) * 8 - bits_consumed - lsb_zeros - lsb_ones;
      // How far to shift it left again to line it up with the output.
      auto alignment = total_significant_bits;
      // Now compute the shifts in the left/right/left pattern to align for the xor into the output.
      auto left0  = num_bits_to_mask_on_left;
      auto right1 = num_bits_to_mask_on_left + num_bits_to_mask_on_right;
      auto left2  = alignment;
      auto last   = (xor_value == 0);
      instructions.emplace_back(left0, right1, left2, last);
      total_significant_bits += lsb_ones;
      bits_consumed += lsb_zeros + lsb_ones;
    } while (xor_value != 0);
  }
  return InstructionHasher<Shift3Instruction>(
    CudaGpuArray<Shift3Instruction>(instructions), xors.size(), total_significant_bits);
}

template <typename hash_key_type, class hasher_class>
class HashTable;  // Forward declaration for `friend` below.

// Same as a HashTable but for calling from GPU kernels.
template <typename hash_key_type, class hasher_view_class>
class HashTableView {
 public:
  __device__ uint64_t hash(const ConstCudaGpuBufferPointer* tables,
                           const hash_key_type& key_index) const
  {
    return hasher.hash(tables, key_index);
  }
  __device__ size_t get_column_count() const { return hasher.get_column_count(); }
  // If the index is inserted, returns that index.  Otherwise, returns
  // the index of the element that was found in the hash that is
  // equivalent to the element that we are trying to insert.
  template <CheckEquality check_equality, PerfectHashing perfect_hashing>
  __device__ hash_key_type insert(const hash_key_type& key_index,
                                  const ConstCudaGpuBufferPointer* tables)
  {
    auto hash_slot  = hash(tables, key_index);
    auto start_slot = hash_slot;
    if constexpr (perfect_hashing == PerfectHashing::True) {
      hash_table[hash_slot] = key_index;
      return key_index;
    }
    // Only when hashing is not perfect do we need to do this
    // because perfect hashing implies that hash(x) is always less
    // than hash_table_size for all x.
    hash_slot %= hash_table_size;
    while (true) {
      hash_key_type orig;
      orig = hash_table[hash_slot];
      if (orig == -1) {
        orig = atomicCAS(&hash_table[hash_slot], static_cast<hash_key_type>(-1), key_index);
        if (orig == -1) {
          // Insert succeeded and we are done.
          return key_index;
        }
      }
      if constexpr (check_equality == CheckEquality::True) {
        // Only insert if the key that is currently in there is not
        // the same as the key that we are trying to insert.  TODO:
        // Maybe memoize the key of the key index to save time?
        if (equal_row(tables, key_index, tables, orig, get_column_count())) {
          // An equivalent key is already in the hash table so we can
          // give up and return what we found.
          return orig;
        }
      }
      hash_slot = (hash_slot + 1) % hash_table_size;
      if (hash_slot == start_slot) {
        return -1;  // Wrapped around and found nothing.
      }
    }
  }
  // Continue the search that was started by start_lookup.  Return >=
  // 0 if a match is found.  Otherwise, return -1 indicating that
  // there are no more matches.
  template <PerfectHashing perfect_hashing, typename lookup_key_type>
  __device__ hash_key_type continue_lookup(const lookup_key_type& key_index,
                                           const ConstCudaGpuBufferPointer* tables,
                                           const ConstCudaGpuBufferPointer* other_tables,
                                           uint64_t& hash_slot) const
  {
    hash_slot = (hash_slot + 1) % hash_table_size;
    while (true) {
      auto slot_content = hash_table[hash_slot];
      if constexpr (perfect_hashing == PerfectHashing::True) {
        // No need to check, this must be the right slot because the
        // hash is perfect.
        return slot_content;
      }
      // TODO: Memoize the lookup of the key_index?  This might be
      // faster but in the case of multiple keys, maybe we'd cause a
      // lookup of an element of the key that we never look at?
      if (slot_content == -1 ||
          equal_row(tables, key_index, other_tables, slot_content, get_column_count())) {
        return slot_content;
      }
      hash_slot = (hash_slot + 1) % hash_table_size;
    }
  }
  // Search the hash table until finding a match, which will return >=
  // 0, or finding no more matches, which will return -1.
  template <PerfectHashing perfect_hashing, typename lookup_key_type>
  __device__ hash_key_type start_lookup(const lookup_key_type& key_index,
                                        const ConstCudaGpuBufferPointer* tables,
                                        const ConstCudaGpuBufferPointer* other_tables,
                                        uint64_t& hash_slot) const
  {
    hash_slot = hash(tables, key_index) % hash_table_size;
    while (true) {
      auto slot_content = hash_table[hash_slot];
      if constexpr (perfect_hashing == PerfectHashing::True) {
        // No need to check, this must be the right slot because the
        // hash is perfect.
        return slot_content;
      }
      // TODO: Memoize the lookup of the key_index?  This might be
      // faster but in the case of multiple keys, maybe we'd cause a
      // lookup of an element of the key that we never look at?
      if (slot_content == -1 ||
          equal_row(tables, key_index, other_tables, slot_content, get_column_count())) {
        return slot_content;
      }
      hash_slot = (hash_slot + 1) % hash_table_size;
    }
  }
  __device__ const hash_key_type get_slot(const hash_key_type& index) const
  {
    return hash_table[index];
  }
  HashTableView() = default;
  // private: TODO: Why can't this be private but with friends?
  friend class HashTable<hash_key_type, typename hasher_view_class::cpu_class>;
  HashTableView(hash_key_type* hash_table, int64_t hash_table_size, const hasher_view_class hasher)
    : hash_table(hash_table), hash_table_size(hash_table_size), hasher(hasher)
  {
  }

 public:
  hash_key_type* hash_table;

 private:
  int64_t hash_table_size;
  hasher_view_class hasher;
};

template <HasMask has_mask,
          CheckEquality check_equality,
          PerfectHashing perfect_hashing,
          typename hash_key_type,
          class hasher_class,
          typename mask_type>
//__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
__launch_bounds__(THREADS_PER_BLOCK, 1) __global__
  void hash_insert_kernel(HashTableView<hash_key_type, hasher_class> hash_table,
                          const ConstCudaGpuBufferPointer* keys,
                          size_t keys_count,
                          mask_type mask)
{
  auto block_count           = gridDim.x;
  auto threads_per_block     = blockDim.x;
  auto block_index           = blockIdx.x;
  auto thread_in_block_index = threadIdx.x;
  auto thread_in_grid_index  = threads_per_block * block_index + thread_in_block_index;

  for (hash_key_type element_index = thread_in_grid_index; element_index < keys_count;
       element_index += threads_per_block * block_count) {
    if constexpr (has_mask == HasMask::True) {
      if (!mask[element_index]) { continue; }
    }
    hash_table.insert<check_equality, perfect_hashing>(element_index, keys);
  }
}

template <HasMask has_mask,
          CheckEquality check_equality,
          PerfectHashing perfect_hashing,
          typename hash_key_type,
          class hasher_class,
          typename mask_type>
//__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
__launch_bounds__(THREADS_PER_BLOCK, 1) __global__
  void hash_insert_kernel_with_write(HashTableView<hash_key_type, hasher_class> hash_table,
                                     const ConstCudaGpuBufferPointer* keys,
                                     size_t keys_count,
                                     mask_type mask,
                                     hash_key_type* num_unique_indices_ptr,
                                     hash_key_type* unique_indices_ptr,
                                     hash_key_type* indices_ptr,
                                     hash_key_type* reverse_indices_ptr)
{
  static_assert(perfect_hashing == PerfectHashing::False,
                "Perfect hashing is not compatible with writing outputs.");
  __shared__ hash_key_type shared_unique_index;
  __shared__ hash_key_type global_unique_index;
  // We'll need to write to outputs so the shared variables ready for it.
  if (threadIdx.x == 0) { shared_unique_index = 0; }
  __syncthreads();
  for (hash_key_type element_index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
       element_index < keys_count;
       element_index += gridDim.x * THREADS_PER_BLOCK) {
    if constexpr (has_mask == HasMask::True) {
      if (!mask[element_index]) { continue; }
    }
    // We need the return value so that we can fill in all the output ptrs.
    hash_key_type result_index =
      hash_table.template insert<check_equality, perfect_hashing>(element_index, keys);
    bool did_insert            = result_index == element_index;
    indices_ptr[element_index] = result_index;
    hash_key_type my_index;
    if (did_insert) {
      // It's much faster to do a sum into shared memory per block
      // and then put all those of those into global rather than sum
      // directly into global.  Measured at 2-5x times.
      my_index = atomicAdd(&shared_unique_index, 1);
    }
    __syncthreads();
    if (did_insert && my_index == 0) {
      global_unique_index = atomicAdd(num_unique_indices_ptr, shared_unique_index);
      shared_unique_index = 0;  // Reset for next time.
    }
    __syncthreads();
    if (did_insert) {
      unique_indices_ptr[global_unique_index + my_index] = result_index;
      reverse_indices_ptr[result_index]                  = global_unique_index + my_index;
    }
  }
}

template <HasMask has_mask,
          MultiSet multiset,
          ReturnAll return_all,
          PerfectHashing perfect_hashing,
          typename hash_key_type,
          class hasher_class,
          typename lookup_key_type,
          typename mask_type>
//__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
__launch_bounds__(THREADS_PER_BLOCK, 1) __global__
  void hash_lookup_kernel(const HashTableView<hash_key_type const, hasher_class> hash_table,
                          const ConstCudaGpuBufferPointer* keys,
                          const int64_t keys_count,
                          const mask_type mask,
                          const ConstCudaGpuBufferPointer* hash_keys,
                          hash_key_type* match_left,
                          lookup_key_type* match_right,
                          int64_t* global_match_count)
{
  // Prime the loop by doing the first lookup.
  __shared__ uint32_t shared_match_count;
  static_assert(THREADS_PER_BLOCK < (1ULL << (8 * sizeof(shared_match_count))));
  __shared__ uint64_t shared_global_index;
  if constexpr (return_all == ReturnAll::False) {
    if (threadIdx.x == 0) { shared_match_count = 0; }
    __syncthreads();
  }
  // signed so that we'll notice the wrap-around if it happens.
  int64_t element_index = static_cast<int64_t>(blockIdx.x) * THREADS_PER_BLOCK + threadIdx.x;
  uint64_t hash_slot;
  hash_key_type found_index;
  while (element_index < keys_count) {
    // At this spot, we have an element_index that is < keys_count but
    // might be masked away and we haven't even started the search on
    // it yet.
    if constexpr (has_mask == HasMask::True) {
      if (!mask[element_index]) {
        // masked away so go to a new element and try again.
        element_index += gridDim.x * THREADS_PER_BLOCK;
        continue;
      }
    }
    // At this spot, we have an element_index that is < keys_count and
    // also not masked away.
    found_index =
      hash_table.template start_lookup<perfect_hashing>(element_index, keys, hash_keys, hash_slot);
    // found_index is -1 if there was nothing found.
    uint64_t my_index;
    while (found_index >= 0) {
      // We have a valid element_index and found_index is an
      // unprocessed result.  We want to add this to the results.
      if constexpr (return_all == ReturnAll::True) {
        // We don't need to compact so we don't need atomics.  Just write into the corresponding
        // slot.
        match_left[element_index] = found_index;
      } else {
        my_index = atomicAdd(&shared_match_count, 1);
        __syncthreads();
        if (my_index == 0) {
          // If at least one thread exists that found and index then this
          // at least one of them will have my_index==0.  If no thread
          // exists then shared_match_count is already 0 and we don't need
          // to atomicAdd 0 so no need to enter this clause.
          shared_global_index =
            atomicAdd(global_match_count, static_cast<int64_t>(shared_match_count));
          shared_match_count = 0;
        }
        __syncthreads();
        match_left[shared_global_index + my_index]  = found_index;
        match_right[shared_global_index + my_index] = element_index;
      }

      // Now we advance.
      if constexpr (multiset == MultiSet::False || return_all == ReturnAll::True ||
                    perfect_hashing == PerfectHashing::True) {
        // Nothing more to look for, even if found_index isn't -1,
        // because we know that there cannot be anymore. If we're
        // returning all then we either found one or we didn't, but we
        // surely don't need to find a second one!
        found_index = -1;
      } else {
        // We might have more matches for this element_index.
        if (found_index >= 0) {
          // Look for another match and loop around.
          found_index = hash_table.template continue_lookup<perfect_hashing>(
            element_index, keys, hash_keys, hash_slot);
        }
      }
    }
    // At this point, either found_index is -1 indicating no more
    // matches or it isn't a multiset so we know that there are no
    // more matches, so we broke out of the loop.
    element_index += gridDim.x * THREADS_PER_BLOCK;
  }
}

//__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
template <typename hash_key_type, class hasher_class>
__launch_bounds__(THREADS_PER_BLOCK, 1) __global__
  void hashed_index_select_kernel(const HashTableView<hash_key_type, hasher_class> hash_table,
                                  const ConstCudaGpuBufferPointer* keys,
                                  const int64_t keys_count,
                                  hash_key_type* mapped_keys)
{
  for (auto element_index = int64_t(blockIdx.x) * THREADS_PER_BLOCK + threadIdx.x;
       element_index < keys_count;
       element_index += gridDim.x * THREADS_PER_BLOCK) {
    auto slot_index            = hash_table.hash(keys, element_index);
    mapped_keys[element_index] = hash_table.get_slot(slot_index);
  }
}

// (Indices of unique elements, map from inserted element to row in unique elements)
using InsertResult = std::tuple<CudaGpuArray<cudf::size_type>, CudaGpuArray<cudf::size_type>>;

// 8GB max hashtable size
constexpr int64_t max_hashtable_size = int64_t(1024) * 1024 * 1024 * 8;

// Return the smallest prime that is larger than 1<<value.
static int64_t nextprime_pow2(int64_t value)
{
  switch (value) {
    /* python code to generate this.
       import sympy
       for i in range(64):
         print(f"case {i}: return {sympy.nextprime(2**i)}; // 2**{i} == {2**i}")
    */
    case 0: return 2;                     // 2**0 == 1
    case 1: return 3;                     // 2**1 == 2
    case 2: return 5;                     // 2**2 == 4
    case 3: return 11;                    // 2**3 == 8
    case 4: return 17;                    // 2**4 == 16
    case 5: return 37;                    // 2**5 == 32
    case 6: return 67;                    // 2**6 == 64
    case 7: return 131;                   // 2**7 == 128
    case 8: return 257;                   // 2**8 == 256
    case 9: return 521;                   // 2**9 == 512
    case 10: return 1031;                 // 2**10 == 1024
    case 11: return 2053;                 // 2**11 == 2048
    case 12: return 4099;                 // 2**12 == 4096
    case 13: return 8209;                 // 2**13 == 8192
    case 14: return 16411;                // 2**14 == 16384
    case 15: return 32771;                // 2**15 == 32768
    case 16: return 65537;                // 2**16 == 65536
    case 17: return 131101;               // 2**17 == 131072
    case 18: return 262147;               // 2**18 == 262144
    case 19: return 524309;               // 2**19 == 524288
    case 20: return 1048583;              // 2**20 == 1048576
    case 21: return 2097169;              // 2**21 == 2097152
    case 22: return 4194319;              // 2**22 == 4194304
    case 23: return 8388617;              // 2**23 == 8388608
    case 24: return 16777259;             // 2**24 == 16777216
    case 25: return 33554467;             // 2**25 == 33554432
    case 26: return 67108879;             // 2**26 == 67108864
    case 27: return 134217757;            // 2**27 == 134217728
    case 28: return 268435459;            // 2**28 == 268435456
    case 29: return 536870923;            // 2**29 == 536870912
    case 30: return 1073741827;           // 2**30 == 1073741824
    case 31: return 2147483659;           // 2**31 == 2147483648
    case 32: return 4294967311;           // 2**32 == 4294967296
    case 33: return 8589934609;           // 2**33 == 8589934592
    case 34: return 17179869209;          // 2**34 == 17179869184
    case 35: return 34359738421;          // 2**35 == 34359738368
    case 36: return 68719476767;          // 2**36 == 68719476736
    case 37: return 137438953481;         // 2**37 == 137438953472
    case 38: return 274877906951;         // 2**38 == 274877906944
    case 39: return 549755813911;         // 2**39 == 549755813888
    case 40: return 1099511627791;        // 2**40 == 1099511627776
    case 41: return 2199023255579;        // 2**41 == 2199023255552
    case 42: return 4398046511119;        // 2**42 == 4398046511104
    case 43: return 8796093022237;        // 2**43 == 8796093022208
    case 44: return 17592186044423;       // 2**44 == 17592186044416
    case 45: return 35184372088891;       // 2**45 == 35184372088832
    case 46: return 70368744177679;       // 2**46 == 70368744177664
    case 47: return 140737488355333;      // 2**47 == 140737488355328
    case 48: return 281474976710677;      // 2**48 == 281474976710656
    case 49: return 562949953421381;      // 2**49 == 562949953421312
    case 50: return 1125899906842679;     // 2**50 == 1125899906842624
    case 51: return 2251799813685269;     // 2**51 == 2251799813685248
    case 52: return 4503599627370517;     // 2**52 == 4503599627370496
    case 53: return 9007199254740997;     // 2**53 == 9007199254740992
    case 54: return 18014398509482143;    // 2**54 == 18014398509481984
    case 55: return 36028797018963971;    // 2**55 == 36028797018963968
    case 56: return 72057594037928017;    // 2**56 == 72057594037927936
    case 57: return 144115188075855881;   // 2**57 == 144115188075855872
    case 58: return 288230376151711813;   // 2**58 == 288230376151711744
    case 59: return 576460752303423619;   // 2**59 == 576460752303423488
    case 60: return 1152921504606847009;  // 2**60 == 1152921504606846976
    case 61: return 2305843009213693967;  // 2**61 == 2305843009213693952
    case 62: return 4611686018427388039;  // 2**62 == 4611686018427387904
  }
  throw std::invalid_argument(annotate_line("can't find prime"));
}

template <typename hash_key_type, class hasher_class>
class HashTable {
 public:
  HashTable(int64_t keys_numel, hasher_class&& new_hasher)
    : perfect_hashing(new_hasher.is_perfect() && new_hasher.max() < max_hashtable_size),
      hash_table([&keys_numel, &new_hasher]() {
        // If the hash is perfect then we can just base the size off of that.
        auto hash_table_size = new_hasher.max() + 1;
        if (!(new_hasher.is_perfect() && new_hasher.max() < max_hashtable_size)) {
          // We can't use perfect hashing so pick a value that is big
          // enough for the input, using a prime number and with 25%
          // occupancy.
          auto bits_needed = sizeof(keys_numel) * 8 - clz(keys_numel);
          //// So that we have a little extra room.  Too much room is bad for
          //// caching.  Experimentally, 2 worked well.
          bits_needed += 2;
          hash_table_size = nextprime_pow2(bits_needed);
        }
        auto ret = CudaGpuArray<hash_key_type>(hash_table_size);
        ret.template fill_byte<-1>();
        return ret;
      }()),
      hasher(std::move(new_hasher))
  {
  }
  HashTableView<hash_key_type, typename hasher_class::view_class> view()
  {
    // TODO: Can we use CTAD to remove the template args here?
    return HashTableView<hash_key_type, typename hasher_class::view_class>(
      hash_table.get().data(), hash_table.numel(), hasher.view());
  }
  const HashTableView<hash_key_type const, typename hasher_class::view_class> view() const
  {
    // TODO: Can we use CTAD to remove the template args here?
    return HashTableView<hash_key_type const, typename hasher_class::view_class>(
      hash_table.get().data(), hash_table.numel(), hasher.view());
  }
  template <CheckEquality check_equality,
            InsertOutput insert_output,
            typename output_type =
              std::conditional_t<insert_output == InsertOutput::True, InsertResult, void>>
  output_type bulk_insert(const std::vector<ConstCudaGpuBufferPointer>& left_keys,
                          const size_t left_keys_numel,
                          const std::optional<ConstCudaGpuBufferPointer>& left_mask)
  {
    if (!left_mask.has_value()) {
      // Cast the nullptr because of a bug in nvcc.
      return bulk_insert<HasMask::False, check_equality, insert_output>(
        left_keys, left_keys_numel, static_cast<const int32_t*>(nullptr));
    } else if (left_mask->get_id() == cudf::type_id::BOOL8) {
      return bulk_insert<HasMask::True, check_equality, insert_output>(
        left_keys, left_keys_numel, static_cast<uint8_t const*>(*left_mask));
    } else if (left_mask->get_id() == cudf::type_id::INT32) {
      return bulk_insert<HasMask::True, check_equality, insert_output>(
        left_keys, left_keys_numel, static_cast<int32_t const*>(*left_mask));
    } else if (left_mask->get_id() == cudf::type_id::INT64) {
      return bulk_insert<HasMask::True, check_equality, insert_output>(
        left_keys, left_keys_numel, static_cast<int64_t const*>(*left_mask));
    } else {
      std::stringstream what;
      what << "Error: Yet unsupported mask type: " << size_of_id(left_mask->get_id()) << std::endl;
      what << "Error: Just open the file and add a new one, it's really easy.";
      throw std::invalid_argument(annotate_line(what.str()));
    }
  }
  std::tuple<CudaGpuArray<hash_key_type>,
             std::optional<CudaGpuArray<cudf::size_type>>,
             std::optional<CudaGpuBuffer>,
             std::optional<CudaGpuBuffer>>
  bulk_lookup(const std::vector<ConstCudaGpuBufferPointer>& right_keys,
              const size_t right_keys_numel,
              const std::optional<ConstCudaGpuBufferPointer>& right_mask,
              const std::vector<ConstCudaGpuBufferPointer>& left_keys,
              const size_t left_keys_numel,
              const bool& left_unique,
              const bool& right_unique,
              const bool& return_all) const
  {
    if (left_unique) {
      return bulk_lookup<xor_hash_table::MultiSet::False>(right_keys,
                                                          right_keys_numel,
                                                          right_mask,
                                                          left_keys,
                                                          left_keys_numel,
                                                          left_unique,
                                                          right_unique,
                                                          return_all);
    } else {
      return bulk_lookup<xor_hash_table::MultiSet::True>(right_keys,
                                                         right_keys_numel,
                                                         right_mask,
                                                         left_keys,
                                                         left_keys_numel,
                                                         left_unique,
                                                         right_unique,
                                                         return_all);
    }
  }

 private:
  template <HasMask has_mask,
            CheckEquality check_equality,
            InsertOutput insert_output,
            typename left_mask_type,
            typename output_type =
              std::conditional_t<insert_output == InsertOutput::True, InsertResult, void>>
  output_type bulk_insert(const std::vector<ConstCudaGpuBufferPointer>& left_keys,
                          const size_t left_keys_numel,
                          const left_mask_type& left_mask)
  {
    // assert that all elements of left_keys have the same number of elements?
    auto keys_tuple = CudaPinnedArray<ConstCudaGpuBufferPointer>(left_keys);
    gpu_throw(cudaDeviceSynchronize());
    if constexpr (insert_output == InsertOutput::True) {
      output_type ret = bulk_insert<has_mask, check_equality, insert_output>(
        keys_tuple, left_keys_numel, left_mask);
      // Don't let left_keys get destroyed before the kernel finishes.
      gpu_throw(cudaDeviceSynchronize());
      return ret;
    } else {
      bulk_insert<has_mask, check_equality, insert_output>(keys_tuple, left_keys_numel, left_mask);
      // Don't let left_keys get destroyed before the kernel finishes.
      gpu_throw(cudaDeviceSynchronize());
    }
  }

  template <HasMask has_mask,
            CheckEquality check_equality,
            InsertOutput insert_output,
            typename left_keys_type,
            typename left_mask_type,
            typename output_type =
              std::conditional_t<insert_output == InsertOutput::True, InsertResult, void>>
  output_type bulk_insert(const left_keys_type& left_keys,
                          int64_t left_keys_count,
                          const left_mask_type& left_mask)
  {
    const auto left_block_count =
      div_round_up(left_keys_count, static_cast<decltype(left_keys_count)>(THREADS_PER_BLOCK));

    if constexpr (insert_output == InsertOutput::True) {
      // TODO: If the hash table is much larger than the number of keys, avoid calling condense.
      if (perfect_hashing) {
        hash_insert_kernel<has_mask, check_equality, PerfectHashing::True>
          <<<left_block_count, THREADS_PER_BLOCK>>>(
            view(), left_keys.get().data(), left_keys_count, left_mask);
        auto narrowed_representatives =
          condense::condense<condense::WriteToInput::True>(hash_table);
        auto mapped_keys = CudaGpuArray<hash_key_type>(left_keys_count);
        hashed_index_select_kernel<<<left_block_count, THREADS_PER_BLOCK>>>(
          view(), left_keys.get().data(), left_keys_count, mapped_keys.get().data());
        return InsertResult(std::move(narrowed_representatives), std::move(mapped_keys));
      } else {
        auto num_unique_indices = CudaGpuArray<hash_key_type>(1);
        num_unique_indices.template fill_byte<0>();
        auto unique_indices  = CudaGpuArray<hash_key_type>(left_keys_count);
        auto indices         = CudaGpuArray<hash_key_type>(left_keys_count);
        auto reverse_indices = CudaGpuArray<hash_key_type>(left_keys_count);
        hash_insert_kernel_with_write<has_mask, check_equality, PerfectHashing::False>
          <<<left_block_count, THREADS_PER_BLOCK>>>(view(),
                                                    left_keys.get().data(),
                                                    left_keys_count,
                                                    left_mask,
                                                    num_unique_indices.get().data(),
                                                    unique_indices.get().data(),
                                                    indices.get().data(),
                                                    reverse_indices.get().data());
        CudaPinnedArray<hash_key_type> pinned_num_unique_indices(1);
        pinned_num_unique_indices.get().copy_from(
          num_unique_indices.get(), num_unique_indices.get() + 1, rmm::cuda_stream_default);
        gpu_throw(cudaDeviceSynchronize());
        auto output_size = pinned_num_unique_indices.get()[0];
        unique_indices.resize(output_size);
        auto narrowed_representatives = std::move(unique_indices);
        return InsertResult(std::move(narrowed_representatives),
                            std::move(index_select(reverse_indices, indices)));
      }
    } else {
      if (perfect_hashing) {
        hash_insert_kernel<has_mask, check_equality, PerfectHashing::True>
          <<<left_block_count, THREADS_PER_BLOCK>>>(
            view(), left_keys.get().data(), left_keys_count, left_mask);
      } else {
        hash_insert_kernel<has_mask, check_equality, PerfectHashing::False>
          <<<left_block_count, THREADS_PER_BLOCK>>>(
            view(), left_keys.get().data(), left_keys_count, left_mask);
      }
    }
  }
  template <MultiSet left_multiset>
  std::tuple<CudaGpuArray<cudf::size_type>,
             std::optional<CudaGpuArray<cudf::size_type>>,
             std::optional<CudaGpuBuffer>,
             std::optional<CudaGpuBuffer>>
  bulk_lookup(const std::vector<ConstCudaGpuBufferPointer>& right_keys,
              const size_t right_keys_numel,
              const std::optional<ConstCudaGpuBufferPointer>& right_mask,
              const std::vector<ConstCudaGpuBufferPointer>& left_keys,
              const size_t left_keys_numel,
              const bool& left_unique,
              const bool& right_unique,
              const bool& return_all) const
  {
    // The number of keys to lookup is on the right.
    // auto lookup_key_count = right_keys[0].numel();
    // auto lookup_count_fits_in_int32 = lookup_key_count < (1UL << 31);
    // Make the lookup_key_type always be cudf::size_type, for
    // compatibility with GQE.
    auto ret = bulk_lookup<left_multiset, cudf::size_type>(right_keys,
                                                           right_keys_numel,
                                                           right_mask,
                                                           left_keys,
                                                           left_keys_numel,
                                                           left_unique,
                                                           right_unique,
                                                           return_all);
    return ret;
  }
  template <MultiSet left_multiset, typename lookup_key_type>
  std::tuple<CudaGpuArray<hash_key_type>,
             std::optional<CudaGpuArray<lookup_key_type>>,
             std::optional<CudaGpuBuffer>,
             std::optional<CudaGpuBuffer>>
  bulk_lookup(const std::vector<ConstCudaGpuBufferPointer>& right_keys,
              const size_t right_keys_numel,
              const std::optional<ConstCudaGpuBufferPointer>& right_mask,
              const std::vector<ConstCudaGpuBufferPointer>& left_keys,
              const size_t left_keys_numel,
              const bool& left_unique,
              const bool& right_unique,
              const bool& return_all) const
  {
    if (!right_mask.has_value()) {
      // Cast the nullptr because of a bug in nvcc.
      return bulk_lookup<HasMask::False, left_multiset, lookup_key_type>(
        right_keys,
        right_keys_numel,
        static_cast<const int32_t*>(nullptr),
        left_keys,
        left_keys_numel,
        left_unique,
        right_unique,
        return_all);
    } else if (right_mask->get_id() == cudf::type_id::BOOL8) {
      return bulk_lookup<HasMask::True, left_multiset, lookup_key_type>(
        right_keys,
        right_keys_numel,
        static_cast<uint8_t const*>(*right_mask),
        left_keys,
        left_keys_numel,
        left_unique,
        right_unique,
        return_all);
    } else if (right_mask->get_id() == cudf::type_id::INT32) {
      return bulk_lookup<HasMask::True, left_multiset, lookup_key_type>(
        right_keys,
        right_keys_numel,
        static_cast<int32_t const*>(*right_mask),
        left_keys,
        left_keys_numel,
        left_unique,
        right_unique,
        return_all);
    } else if (right_mask->get_id() == cudf::type_id::INT64) {
      return bulk_lookup<HasMask::True, left_multiset, lookup_key_type>(
        right_keys,
        right_keys_numel,
        static_cast<int64_t const*>(*right_mask),
        left_keys,
        left_keys_numel,
        left_unique,
        right_unique,
        return_all);
    } else {
      std::stringstream what;
      what << "Error: Yet unsupported mask type: " << size_of_id(right_mask->get_id()) << std::endl;
      what << "Error: Just open the file and add a new one, it's really easy.";
      throw std::invalid_argument(annotate_line(what.str()));
    }
  }
  template <HasMask right_has_mask,
            MultiSet left_multiset,
            typename lookup_key_type,
            typename mask_type>
  std::tuple<CudaGpuArray<hash_key_type>,
             std::optional<CudaGpuArray<lookup_key_type>>,
             std::optional<CudaGpuBuffer>,
             std::optional<CudaGpuBuffer>>
  bulk_lookup(const std::vector<ConstCudaGpuBufferPointer>& right_keys,
              const size_t right_keys_numel,
              const mask_type right_mask,
              const std::vector<ConstCudaGpuBufferPointer>& left_keys,
              const size_t left_keys_numel,
              const bool& left_unique,
              const bool& right_unique,
              const bool& return_all) const
  {
    if (perfect_hashing) {
      return bulk_lookup<right_has_mask, left_multiset, PerfectHashing::True, lookup_key_type>(
        right_keys,
        right_keys_numel,
        right_mask,
        left_keys,
        left_keys_numel,
        left_unique,
        right_unique,
        return_all);
    } else {
      return bulk_lookup<right_has_mask, left_multiset, PerfectHashing::False, lookup_key_type>(
        right_keys,
        right_keys_numel,
        right_mask,
        left_keys,
        left_keys_numel,
        left_unique,
        right_unique,
        return_all);
    }
  }
  template <HasMask right_has_mask,
            MultiSet left_multiset,
            PerfectHashing perfect_hashing,
            typename lookup_key_type,
            typename mask_type>
  std::tuple<CudaGpuArray<cudf::size_type>,
             std::optional<CudaGpuArray<cudf::size_type>>,
             std::optional<CudaGpuBuffer>,
             std::optional<CudaGpuBuffer>>
  bulk_lookup(const std::vector<ConstCudaGpuBufferPointer>& right_keys,
              const size_t right_keys_numel,
              const mask_type right_mask,
              const std::vector<ConstCudaGpuBufferPointer>& left_keys,
              const size_t left_keys_numel,
              const bool& left_unique,
              const bool& right_unique,
              const bool& return_all) const
  {
    // assert that all elements of right_keys have the same number of elements?
    auto right_keys_count = right_keys_numel;
    // assert that all elements of left_keys have the same number of elements?
    auto left_keys_count = left_keys_numel;
    int64_t output_size  = return_all                      ? right_keys_count
                           : (left_unique && right_unique) ? min(left_keys_count, right_keys_count)
                           : left_unique                   ? right_keys_count
                           : right_unique                  ? left_keys_count
                                                           : right_keys_count * left_keys_count;

    auto match_left = CudaGpuArray<hash_key_type>(output_size);
    if (return_all) { match_left.template fill_byte<-1>(); }
    /*
    // For each element on the left, it might be unmatched on the right.
    // So the possible number of unmatched on the left is equal to the
    // number of elements on the left.  Same for right. TODO, support this.
    auto unmatched_left = torch::empty({left_keys_count}, cuda_tensor_options);
    auto unmatched_right = torch::empty({right_keys_count}, cuda_tensor_options);
    */

    // For each output above, we want to know how many we actually wrote in the end.
    // auto int64_tensor_options =
    //    torch::TensorOptions().dtype(torch::CppTypeToScalarType<int64_t>()).device(torch::kCUDA);

    // Why declare it up here, we can do it down below? TODO
    std::optional<CudaGpuArray<int64_t>> match_count;
    if (!return_all) {
      match_count = CudaGpuArray<int64_t>(1);
      match_count->fill_byte<0>();
    }
    // TODO: support unmatched
    // auto unmatched_left_count = torch::empty({}, cuda_tensor_options);
    // auto unmatched_right_count = torch::empty({}, cuda_tensor_options);

    const auto right_block_count =
      div_round_up(right_keys_count, static_cast<decltype(right_keys_count)>(THREADS_PER_BLOCK));
    std::optional<CudaGpuArray<lookup_key_type>> match_right;
    auto left_keys_gputensor  = CudaPinnedArray<ConstCudaGpuBufferPointer>(left_keys);
    auto right_keys_gputensor = CudaPinnedArray<ConstCudaGpuBufferPointer>(right_keys);
    gpu_throw(cudaDeviceSynchronize());
    if (return_all) {
      hash_lookup_kernel<right_has_mask, left_multiset, ReturnAll::True, perfect_hashing>
        <<<right_block_count, THREADS_PER_BLOCK>>>(view(),
                                                   right_keys_gputensor.get().data(),
                                                   right_keys_count,
                                                   right_mask,
                                                   left_keys_gputensor.get().data(),
                                                   match_left.get().data(),
                                                   static_cast<lookup_key_type*>(nullptr),
                                                   static_cast<int64_t*>(nullptr));
    } else {
      // We only need the right side if we are not returning all.  If
      // we are returning all then the right side would anyway just be
      // the range 0 to right_keys_count-1 so it wouldn't be useful to
      // even record it.
      // auto right_tensor_options = torch::TensorOptions()
      //                            .dtype(torch::CppTypeToScalarType<lookup_key_type>())
      //                            .device(torch::kCUDA);
      match_right = CudaGpuArray<lookup_key_type>(output_size);
      hash_lookup_kernel<right_has_mask, left_multiset, ReturnAll::False, perfect_hashing>
        <<<right_block_count, THREADS_PER_BLOCK>>>(view(),
                                                   right_keys_gputensor.get().data(),
                                                   right_keys_count,
                                                   right_mask,
                                                   left_keys_gputensor.get().data(),
                                                   match_left.get().data(),
                                                   match_right->get().data(),
                                                   match_count->get().data());
      CudaPinnedArray<int64_t> pinned_match_count(1);
      pinned_match_count.get().copy_from(
        match_count->get(), match_count->get() + 1, rmm::cuda_stream_default);
      gpu_throw(cudaDeviceSynchronize());
      auto local_match_count = pinned_match_count.get()[0];
      match_left.resize(local_match_count);
      match_right->resize(local_match_count);
    }
    return {std::move(match_left), std::move(match_right), std::nullopt, std::nullopt};
  }
  bool perfect_hashing;
  CudaGpuArray<hash_key_type> hash_table;
  hasher_class hasher;
};

using VariantHashTable =
  std::variant<HashTable<cudf::size_type, InstructionHasher<Shift1Instruction>>,
               HashTable<cudf::size_type, InstructionHasher<Shift3Instruction>>,
               HashTable<cudf::size_type, IdentityHasher<int8_t>>,
               HashTable<cudf::size_type, IdentityHasher<int16_t>>,
               HashTable<cudf::size_type, IdentityHasher<int32_t>>,
               HashTable<cudf::size_type, IdentityHasher<int64_t>>,
               HashTable<cudf::size_type, IdentityHasher<int8_t, int8_t>>>;

class DynamicHashTable {
 public:
  template <class hash_table_class>
  DynamicHashTable(hash_table_class&& hash_table) : hash_table(std::move(hash_table))
  {
  }
  template <CheckEquality check_equality,
            InsertOutput insert_output,
            typename output_type =
              std::conditional_t<insert_output == InsertOutput::True, InsertResult, void>>
  output_type bulk_insert(const std::vector<ConstCudaGpuBufferPointer>& left_keys,
                          const size_t left_keys_numel,
                          const std::optional<ConstCudaGpuBufferPointer>& left_mask)
  {
    return std::visit(
      [&](auto& hash_table) {
        return hash_table.template bulk_insert<check_equality, insert_output>(
          left_keys, left_keys_numel, left_mask);
      },
      hash_table);
  }

  std::tuple<CudaGpuArray<cudf::size_type>,
             std::optional<CudaGpuArray<cudf::size_type>>,
             std::optional<CudaGpuBuffer>,
             std::optional<CudaGpuBuffer>>
  bulk_lookup(const std::vector<ConstCudaGpuBufferPointer>& right_keys,
              const size_t right_keys_numel,
              const std::optional<ConstCudaGpuBufferPointer>& right_mask,
              const std::vector<ConstCudaGpuBufferPointer>& left_keys,
              const size_t left_keys_numel,
              const bool& left_unique,
              const bool& right_unique,
              const bool& return_all) const
  {
    return std::visit(
      [&](auto& hash_table) {
        auto ret = hash_table.bulk_lookup(right_keys,
                                          right_keys_numel,
                                          right_mask,
                                          left_keys,
                                          left_keys_numel,
                                          left_unique,
                                          right_unique,
                                          return_all);
        // Don't convert to buffer for compatibility with GQE.
        return ret;
      },
      hash_table);
  }

 private:
  VariantHashTable hash_table;
};

// TODO: Can we delete this function and use the constructor directly
// with CTAD?
template <typename hash_key_type>
DynamicHashTable make_hash_table(int64_t keys_numel, std::vector<CudaPinnedBuffer>&& and_ors)
{
  auto maybe_identity_hasher8 = make_identity_hasher<int8_t>(and_ors);
  if (maybe_identity_hasher8.has_value()) {
    return HashTable<hash_key_type, IdentityHasher<int8_t>>(keys_numel,
                                                            std::move(*maybe_identity_hasher8));
  }
  auto maybe_identity_hasher8_8 = make_identity_hasher<int8_t, int8_t>(and_ors);
  if (maybe_identity_hasher8_8.has_value()) {
    return HashTable<hash_key_type, IdentityHasher<int8_t, int8_t>>(
      keys_numel, std::move(*maybe_identity_hasher8_8));
  }
  auto maybe_identity_hasher16 = make_identity_hasher<int16_t>(and_ors);
  if (maybe_identity_hasher16.has_value()) {
    return HashTable<hash_key_type, IdentityHasher<int16_t>>(keys_numel,
                                                             std::move(*maybe_identity_hasher16));
  }
  auto maybe_identity_hasher32 = make_identity_hasher<int32_t>(and_ors);
  if (maybe_identity_hasher32.has_value()) {
    return HashTable<hash_key_type, IdentityHasher<int32_t>>(keys_numel,
                                                             std::move(*maybe_identity_hasher32));
  }
  auto maybe_identity_hasher64 = make_identity_hasher<int64_t>(and_ors);
  if (maybe_identity_hasher64.has_value()) {
    return HashTable<hash_key_type, IdentityHasher<int64_t>>(keys_numel,
                                                             std::move(*maybe_identity_hasher64));
  }
  auto maybe_shift1_hasher = make_shift1_hasher(and_ors);
  if (maybe_shift1_hasher.has_value()) {
    return HashTable<hash_key_type, InstructionHasher<Shift1Instruction>>(
      keys_numel, std::move(*maybe_shift1_hasher));
  }
  std::vector<CudaPinnedBuffer> xors;
  for (auto& and_or : and_ors) {
    and_or.get().set(0, and_or.get()[0] ^ and_or.get()[1]);
    and_or.resize(1);
    xors.push_back(std::move(and_or));
  }
  return HashTable<hash_key_type, InstructionHasher<Shift3Instruction>>(keys_numel,
                                                                        make_shift3_hasher(xors));
}

// Make a hash_table that will suit inserting the insert_keys and a
// subsequent lookup with the lookup_keys.  lookup_keys may be absent.
static DynamicHashTable make_hash_table(
  const std::vector<ConstCudaGpuBufferPointer>& insert_keys,
  const size_t insert_keys_numel,
  const std::optional<std::pair<std::vector<ConstCudaGpuBufferPointer>, size_t>> lookup_keys =
    std::nullopt)
{
  PUSH_RANGE("do reduce", 4);
  PUSH_RANGE("make the vector", 5);
  std::vector<CudaPinnedBuffer> insert_and_or_tensors;
  POP_RANGE();
  for (const auto& current_insert_keys : insert_keys) {
    PUSH_RANGE("run a reduce", 5);
    insert_and_or_tensors.push_back(
      reduce_and_or_cuda<true>(current_insert_keys, insert_keys_numel));
    POP_RANGE();
  }
  std::optional<std::vector<CudaPinnedBuffer>> lookup_and_or_tensors;
  if (lookup_keys.has_value()) {
    lookup_and_or_tensors.emplace();
    for (const auto& current_lookup_keys : lookup_keys->first) {
      PUSH_RANGE("push", 5);
      lookup_and_or_tensors->push_back(
        reduce_and_or_cuda<true>(current_lookup_keys, lookup_keys->second));
      POP_RANGE();
    }
  }
  gpu_throw(cudaDeviceSynchronize());
  POP_RANGE();
  PUSH_RANGE("combine reduce", 4);
  if (lookup_and_or_tensors.has_value()) {
    for (uint i = 0; i < insert_and_or_tensors.size(); i++) {
      auto& current_insert_and_or       = insert_and_or_tensors[i];
      const auto& current_lookup_and_or = lookup_and_or_tensors->at(i);

      auto const insert_and_value = current_insert_and_or.get()[0];
      auto const lookup_and_value = current_lookup_and_or.get()[0];
      current_insert_and_or.get().set(0, insert_and_value & lookup_and_value);

      auto const insert_or_value = current_insert_and_or.get()[1];
      auto const lookup_or_value = current_lookup_and_or.get()[1];
      current_insert_and_or.get().set(1, insert_or_value | lookup_or_value);
    }
  }
  POP_RANGE();
  PUSH_RANGE("make hash table", 4);
  // The hash stores indices into keys.  How many rows in keys?
  auto hash_key_count = insert_keys_numel;
  // Do we need hash slots words of int32_t or int64_t?
  auto key_count_fits_in_int32 = hash_key_count < (1UL << 31);
  // Make the hash_key_type always cudf::size_type for compatibility
  // with GQE.
  if (key_count_fits_in_int32) {
    auto ret =
      make_hash_table<cudf::size_type>(insert_keys_numel, std::move(insert_and_or_tensors));
    POP_RANGE();
    return ret;
  } else {
    auto ret =
      make_hash_table<cudf::size_type>(insert_keys_numel, std::move(insert_and_or_tensors));
    POP_RANGE();
    return ret;
  }
}

}  // namespace xor_hash_table
}  // namespace libperfect
