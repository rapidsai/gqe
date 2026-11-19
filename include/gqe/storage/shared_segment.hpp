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

#include <gqe/storage/segment_layout.hpp>
#include <gqe/storage/shared_memory_poison.hpp>
#include <gqe/types.hpp>

#include <boost/container/uses_allocator_construction.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/allocation_type.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/indexes/iset_index.hpp>
#include <boost/interprocess/managed_external_buffer.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/mem_algo/rbtree_best_fit.hpp>
#include <boost/interprocess/permissions.hpp>
#include <boost/interprocess/segment_manager.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_recursive_mutex.hpp>
#include <boost/interprocess/sync/mutex_family.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gqe {

/// @brief Tag selecting creation of a new shared memory segment.
struct create_only_t {};

/// @brief Tag selecting attachment to an existing shared memory segment.
struct open_only_t {};

/// @brief Select the constructor that creates a new segment.
inline constexpr create_only_t create_only{};

/// @brief Select the constructor that attaches to an existing segment.
inline constexpr open_only_t open_only{};

namespace detail {

class shared_segment_test_access;

/**
 * @brief Return what @p make returns, rethrowing a Boost error as `std::runtime_error`.
 *
 * A lock that fails, because a process died holding it, becomes a
 * @ref shared_memory_unrecoverable_error, and the segment is marked poisoned. A Boost lock failure
 * never carries an operating system error. Any other Boost error that carries one becomes a
 * `std::system_error` with that error, so callers can tell, for example, an existing object from
 * other failures.
 *
 * @tparam Make Callable of signature `R()`.
 * @param[in] action What @p make does, completing "Failed to …" in the message.
 * @param[in] make Operation to run.
 * @throw shared_memory_unrecoverable_error if @p make throws `boost::interprocess::lock_exception`.
 * @throw std::system_error if @p make throws `boost::interprocess::interprocess_exception` carrying
 * an operating system error.
 * @throw std::runtime_error if @p make throws any other
 * `boost::interprocess::interprocess_exception`.
 */
template <class Make>
auto rethrow_as_runtime_error(std::string_view action, Make make) -> decltype(make())
{
  try {
    return make();
  } catch (boost::interprocess::lock_exception const& e) {
    throw_shared_memory_unrecoverable(action, e);
  } catch (boost::interprocess::interprocess_exception const& e) {
    if (e.get_native_error() != 0) {
      throw std::system_error(
        e.get_native_error(), std::generic_category(), std::format("Failed to {}", action));
    }
    throw std::runtime_error(std::format("Failed to {}: {}", action, e.what()));
  }
}

}  // namespace detail

/**
 * @brief Arena layout of a shared memory segment, recorded in its mapping.
 *
 * Written once by the process that creates the segment and validated by each process that attaches.
 * Three metadata arrays follow this object in the mapping. Their sizes are determined at creation
 * from the arena count and the number of configured NUMA node ids.
 *
 *  - @ref arena_records records where each arena lies and how it was placed.
 *  - @ref fallback_order_lengths records how many arena choices each NUMA node has.
 *  - @ref fallback_orders stores those choices in one flat array. Each NUMA node has space for
 *    `arena_count` arena indices; its recorded length determines how many are used.
 *
 * See @ref segment_layout::fallback_order_by_node for the meaning of a fallback order and an
 * example. This header records that configuration so attaching processes use the same choices.
 */
class segment_header {
 public:
  using size_type = std::size_t;

  /**
   * @brief Integer type used for fallback-order lengths and arena indices.
   */
  using layout_entry_t = std::int32_t;

  /**
   * @brief Where one arena lies in the mapping and how its pages were placed.
   *
   * The node set is stored as the raw bits of a @ref gqe::cpu_set, because a `cpu_set` owns a heap
   * allocation another process cannot reach.
   */
  struct arena_record {
    /// Byte offset of the arena from the mapping base.
    size_type offset;

    /// Size of the arena in bytes.
    size_type bytes;

    /// How the arena's pages were placed on @ref nodes.
    arena_placement mode;

    /// Node set, laid out as `bits()`.
    std::array<unsigned long, cpu_set::qword_count> nodes;
  };

  segment_header(segment_header const&)            = delete;
  segment_header(segment_header&&)                 = delete;
  segment_header& operator=(segment_header const&) = delete;
  segment_header& operator=(segment_header&&)      = delete;

 private:
  friend class shared_segment_manager;

  /**
   * @brief Layout version, so an attaching process rejects a segment it cannot interpret.
   */
  static constexpr std::uint32_t layout_version = 1;

  /**
   * @brief Validate a header read out of a mapping.
   *
   * Checks the layout version, that the recorded size matches the mapping, and that the fields the
   * accessors compute addresses from describe a mapping of that size. The tables must fit between
   * this object and the first arena, every recorded arena must lie after them and within the
   * mapping without overlapping another. At least one fallback order must be configured, and
   * every arena index in a fallback order must refer to a recorded arena.
   *
   * @param[in] mapping_base Base of the mapping this header lies in.
   * @param[in] mapped_bytes Size of the mapping.
   * @throw std::runtime_error if any of those does not hold.
   */
  void validate(void const* mapping_base, size_type mapped_bytes) const;

  /**
   * @brief Return the policy the segment was created with.
   */
  [[nodiscard]] segment_layout_policy_type policy() const noexcept { return _policy; }

  /**
   * @brief Return the number of arenas.
   */
  [[nodiscard]] std::size_t arena_count() const noexcept { return _arena_count; }

  /**
   * @brief Return the number of fallback orders.
   */
  [[nodiscard]] std::size_t fallback_order_count() const noexcept { return _fallback_order_count; }

  /**
   * @brief Return the size of the whole mapping in bytes.
   */
  [[nodiscard]] size_type total_bytes() const noexcept { return _total_bytes; }

  /**
   * @brief Return the byte offset from the mapping base to the first arena.
   */
  [[nodiscard]] size_type data_offset() const noexcept { return _data_offset; }

  /**
   * @brief Return the table recording where each arena lies and how it was placed.
   */
  [[nodiscard]] std::span<arena_record const> arena_records() const noexcept;

  /**
   * @brief Return the arena order to try for a thread running on @p numa_node.
   *
   * A NUMA node id without a configured fallback order uses the order at index 0.
   */
  [[nodiscard]] std::span<layout_entry_t const> fallback_order(int numa_node) const noexcept;

  /**
   * @brief Return the bytes the tables occupy after this object.
   *
   * @pre The product of the two counts does not overflow `size_type`, which @ref layout_tables_fit
   * establishes for counts read out of a mapping. Not checked.
   */
  [[nodiscard]] static size_type layout_table_bytes(std::size_t arena_count,
                                                    std::size_t fallback_order_count) noexcept;

  /**
   * @brief Write the fields and the tables that follow this object.
   *
   * @pre The mapping holds @ref layout_table_bytes after this object, which
   * @ref shared_segment_manager::header_bytes reserves. Not checked.
   *
   * @param[in] total_bytes Size of the whole mapping.
   * @param[in] data_offset Byte offset of the first arena.
   * @param[in] policy Policy the layout came from.
   * @param[in] layout Arenas and fallback orders to record.
   * @param[in] arena_offsets Byte offset of each arena from the mapping base, in arena order.
   */
  segment_header(size_type total_bytes,
                 size_type data_offset,
                 segment_layout_policy_type policy,
                 segment_layout const& layout,
                 std::span<size_type const> arena_offsets);

  /**
   * @brief Check that the arena records and fallback orders describe this mapping.
   *
   * @throw std::runtime_error if an arena lies outside the space after the header or overlaps the
   * one before it, if there are no fallback orders, or if a fallback order is longer than there are
   * arenas or names an arena that does not exist.
   */
  void check_layout_tables() const;

  /**
   * @brief Return whether the layout tables fit in @p available_bytes.
   *
   * Cannot overflow for any counts, including ones read out of a mapping this build did not write.
   *
   * @pre @p arena_count is non-zero. Not checked.
   */
  [[nodiscard]] static bool layout_tables_fit(size_type available_bytes,
                                              std::uint32_t arena_count,
                                              std::uint32_t fallback_order_count) noexcept;

  /**
   * @copydoc gqe::segment_header::arena_records
   */
  [[nodiscard]] std::span<arena_record> arena_records() noexcept;

  /**
   * @brief Return the number of arena choices configured for each NUMA node, indexed by node id.
   */
  [[nodiscard]] std::span<layout_entry_t> fallback_order_lengths() noexcept;

  /**
   * @copydoc gqe::segment_header::fallback_order_lengths
   */
  [[nodiscard]] std::span<layout_entry_t const> fallback_order_lengths() const noexcept;

  /**
   * @brief Return the arena choices for all NUMA nodes as one flat array.
   *
   * The fallback order for node `n` starts at `n * _arena_count`. Its arena indices appear in
   * preference order, and `fallback_order_lengths()[n]` determines how many are valid. The
   * remaining entries reserved for that node are unused.
   */
  [[nodiscard]] std::span<layout_entry_t> fallback_orders() noexcept;

  /**
   * @copydoc gqe::segment_header::fallback_orders
   */
  [[nodiscard]] std::span<layout_entry_t const> fallback_orders() const noexcept;

  /// Layout version the creating process wrote.
  std::uint32_t _layout_version;

  /// Policy the segment was created with. Declared as the enum so a debugger prints its name.
  segment_layout_policy_type _policy;

  /// Number of arenas.
  std::uint32_t _arena_count;

  /// Number of fallback orders.
  std::uint32_t _fallback_order_count;

  /// Size of the whole mapping in bytes, including the segment manager.
  size_type _total_bytes;

  /// Byte offset from the mapping base to the first arena.
  size_type _data_offset;
};

static_assert(alignof(segment_header) % alignof(segment_header::arena_record) == 0,
              "The arena records start at the end of the header, so the header's alignment has to "
              "satisfy theirs");

static_assert(sizeof(segment_header::arena_record) % alignof(segment_header::layout_entry_t) == 0,
              "The fallback tables start at the end of the arena records, so a record's size has "
              "to keep them aligned");

/**
 * @brief Segment manager that serves a shared memory mapping's allocations from arenas.
 *
 * # Design
 *
 * This type implements the segment manager interface that `boost::interprocess::allocator` and
 * shared storage are written against, and serves it from *arenas*. An arena is a contiguous range
 * of the mapping with its own free list and lock, and may be placed on a set of NUMA nodes. Code
 * using the API, including every `boost::interprocess::allocator` built from this manager,
 * allocates from the arenas without being aware of them.
 *
 * Each arena has its own allocator, free list, and lock. The segment manager routes requests
 * between these allocators using the configured fallback order. Separate arena locks allow
 * allocations in different arenas to proceed independently.
 *
 * Implements the subset of Boost's segment-manager interface needed by GQE: allocation and
 * deallocation, single-object construction and destruction, named lookup, address/handle
 * conversion, and free-space queries. Array construction and `find_or_construct` are not supported.
 *
 * The number, sizes, and placement of the arenas are decided by the
 * @ref gqe::segment_layout_policy_type the creating process chose, and recorded in the mapping.
 * This type routes each request to an arena.
 *
 * New allocations try arenas in the configured fallback order for the NUMA node on which the
 * calling thread is running. See @ref segment_layout::fallback_order_by_node for the definition
 * and an example.
 *
 *  - `deallocate`, `destroy_ptr`, and `size` go to the arena that owns the block, and
 *    `allocation_command` on an existing block asks that arena first.
 *  - An allocation larger than every arena cannot be served.
 *
 * An allocation tries at most `R` arenas, where `R` is the number of arena choices in the selected
 * fallback order. Each try is a best-fit search of that arena's free blocks. Finding the arena that
 * owns an address scans all `A` arenas, so `deallocate`, `size`, `destroy_ptr`, and expansion
 * through `allocation_command` cost `O(A)` before the arena's own work. A refused request costs all
 * `R` tries, then throws `std::bad_alloc` or, for a nothrow request, returns null.
 *
 * Every `std::bad_alloc` thrown here is preceded by one error log line giving the policy, the
 * largest arena size, the arena count, and the summed free bytes.
 *
 * `construct` tries arenas in the fallback order and asks each arena's manager to construct the
 * object.
 * That manager constructs, names, finds, and destroys its objects itself. The one record kept here
 * maps each name to the arena holding it, so a name is unique across arenas and `find` asks one
 * arena. The map lives in the segment, created when the first name is registered and released when
 * the last is removed.
 *
 * ## Layout
 *
 * The mapping stores the fixed-size manager object, trailing layout tables, and arenas in order.
 * Here, "manager fields" means the members preceding @ref segment_header, which is the manager's
 * last member.
 *
 * @code{.text}
 * [ manager fields | segment_header ]
 * [ arena records ][ order lengths ][ fallback orders ][ padding ]
 * [ arenas ]
 * @endcode
 *
 * The three tables live beyond `sizeof(shared_segment_manager)`, rather than inside
 * `sizeof(segment_header)`. Padding aligns the first arena to a page boundary.
 *
 * This object lives at offset 0 of the mapping, which each process maps at a different address.
 * It is therefore standard layout, has no virtual functions, and stores offsets rather than
 * pointers. Arenas are reached by offsetting from `this`, and allocators stored in the segment
 * reach this manager through a self-relative `boost::interprocess::offset_ptr`. Handles are byte
 * offsets from the mapping base, so no object has handle 0, which
 * @ref gqe::storage::shared_storage_handle treats as invalid.
 *
 * @ref segment_header, this object's last member, records where each arena lies and the order
 * allocations try them in. Its tables follow it, so they begin where this object ends.
 * @ref header_bytes computes the page-rounded region the two occupy before the first arena.
 *
 * ## Thread Safety
 *
 * All members are safe to call concurrently from any number of threads and processes. Allocation
 * takes no manager-wide lock. Each arena has its own interprocess mutex, so allocations on
 * different arenas do not contend, and losing a race for an arena's last block only moves the
 * request to the next arena in the fallback order.
 *
 * The name map has its own interprocess mutex, held across named construction, the destruction of
 * a named object, and @ref find. Allocating, and constructing or destroying an anonymous object,
 * never take it. The mutex is recursive, so a constructor or destructor that creates or destroys
 * another named object does not deadlock.
 *
 * A lock that cannot be taken, because a process died holding it, surfaces as
 * @ref shared_memory_unrecoverable_error, a `std::runtime_error`, from the member that needed it,
 * and marks the segment poisoned.
 *
 * The layout fields and tables are written once, by the creating process, before the segment is
 * published, and are read-only afterwards.
 */
class shared_segment_manager {
 private:
  friend class shared_segment;
  friend class detail::shared_segment_test_access;

  /**
   * @brief One arena of the mapping, locked with an interprocess mutex so processes allocating from
   * it concurrently are serialized.
   */
  using arena_type = boost::interprocess::basic_managed_external_buffer<
    char,
    boost::interprocess::rbtree_best_fit<boost::interprocess::mutex_family>,
    boost::interprocess::iset_index>;

  /**
   * @brief The manager owning one arena's free list and lock.
   */
  using arena_manager = arena_type::segment_manager;

 public:
  using void_pointer          = arena_manager::void_pointer;
  using size_type             = arena_manager::size_type;
  using difference_type       = arena_manager::difference_type;
  using multiallocation_chain = arena_manager::multiallocation_chain;

  /**
   * @brief Numeric handle for an object in the segment, as a byte offset from the mapping base.
   */
  using handle_t = difference_type;

  ~shared_segment_manager()                                        = delete;
  shared_segment_manager(shared_segment_manager const&)            = delete;
  shared_segment_manager(shared_segment_manager&&)                 = delete;
  shared_segment_manager& operator=(shared_segment_manager const&) = delete;
  shared_segment_manager& operator=(shared_segment_manager&&)      = delete;

  // ----- Allocation interface `boost::interprocess::allocator` calls -----

  /**
   * @brief Allocate @p bytes with the given alignment, trying arenas in the configured fallback
   * order for the NUMA node on which the calling thread is running.
   *
   * @param[in] bytes Number of bytes to allocate.
   * @param[in] alignment Required alignment of the returned address.
   * @return Address of the allocation.
   * @throw std::bad_alloc if no arena can satisfy the request.
   * @throw std::runtime_error if an arena's lock cannot be taken.
   */
  void* allocate_aligned(size_type bytes, size_type alignment);

  /**
   * @brief Allocate @p bytes with the given alignment, returning null instead of throwing if no
   * arena can satisfy the request.
   *
   * @param[in] bytes Number of bytes to allocate.
   * @param[in] alignment Required alignment of the returned address.
   * @param[in] tag Selects this overload.
   * @return Address of the allocation, or null if no arena can satisfy the request.
   * @throw std::runtime_error if an arena's lock cannot be taken.
   */
  void* allocate_aligned(size_type bytes, size_type alignment, std::nothrow_t const& tag);

  /**
   * @brief Release an allocation back to the arena that owns it.
   *
   * @pre @p ptr is null or was allocated from this segment. Checked by an assertion.
   *
   * @param[in] ptr Allocation to release.
   * @throw std::runtime_error if the owning arena's lock cannot be taken.
   */
  void deallocate(void* ptr);

  /**
   * @brief Expand, shrink, or replace an existing allocation.
   *
   * The arena owning @p reuse is asked first, with the whole command. If that arena refuses and
   * @p command permits a replacement, the request tries the remaining arenas in the configured
   * fallback order for the NUMA node on which the calling thread is running.
   *
   * Sets @p reuse to null when it replaces the block.
   *
   * @pre @p reuse is null or was allocated from this segment. Checked by an assertion.
   *
   * @tparam T Element type of the allocation.
   * @param[in] command Allocation command flags.
   * @param[in] limit_size Smallest acceptable size, in elements.
   * @param[in,out] prefer_in_recvd_out_size Preferred size on entry, size granted on return.
   * @param[in,out] reuse Block to operate on. Null on return if a new block was allocated.
   * @return The allocated block, or null if @p command is a nothrow allocation that no arena could
   * serve.
   * @throw std::bad_alloc if no arena could serve a throwing allocation.
   * @throw std::runtime_error if an arena's lock cannot be taken.
   */
  template <class T>
  T* allocation_command(boost::interprocess::allocation_type command,
                        size_type limit_size,
                        size_type& prefer_in_recvd_out_size,
                        T*& reuse)
  {
    void* raw_reuse = reuse;
    auto* result    = allocation_command(
      command, limit_size, prefer_in_recvd_out_size, raw_reuse, sizeof(T), alignof(T));
    reuse = static_cast<T*>(raw_reuse);
    return static_cast<T*>(result);
  }

  /**
   * @copydoc gqe::shared_segment_manager::allocation_command
   *
   * @param[in] sizeof_object Size of one element in bytes.
   * @param[in] alignof_object Alignment of one element in bytes.
   */
  void* allocation_command(boost::interprocess::allocation_type command,
                           size_type limit_size,
                           size_type& prefer_in_recvd_out_size,
                           void*& reuse,
                           size_type sizeof_object,
                           size_type alignof_object);

  /**
   * @brief Release every block in @p chain, routing each to the arena that owns it.
   *
   * @param[in,out] chain Blocks to release. Empty on return.
   * @throw std::runtime_error if the owning arena's lock cannot be taken.
   */
  void deallocate_many(multiallocation_chain& chain);

  /**
   * @brief Return the usable size of an allocation.
   *
   * @pre @p ptr was allocated from this segment. Checked by an assertion.
   *
   * @param[in] ptr Allocation to measure.
   */
  [[nodiscard]] size_type size(void const* ptr) const noexcept;

  /**
   * @brief Return the total size of the mapping, including this manager.
   */
  [[nodiscard]] size_type get_size() const noexcept { return _header.total_bytes(); }

  // ----- Single-object construct and destroy_ptr -----

  /**
   * @brief Deferred construction of a single object, returned by @ref construct.
   *
   * Exists so a call reads `construct<T>(anonymous_instance)(args...)`.
   *
   * @tparam T Type to construct. May be any object type, including a scalar.
   */
  template <class T>
  class construct_proxy {
   public:
    /**
     * @brief Bind a proxy to its manager and instance name.
     *
     * @param[in] manager Manager that will serve the allocation.
     * @param[in] name Instance name, or null for an anonymous instance.
     */
    construct_proxy(shared_segment_manager* manager, char const* name) noexcept
      : _manager(manager), _name(name)
    {
    }

    /**
     * @brief Construct the object in the first arena in the fallback order that can hold it.
     *
     * A named instance's name is checked before any arena is asked, so a name already in use never
     * runs the constructor or consumes @p args.
     *
     * An allocator-aware `T` whose allocator can be built from this manager receives one, as the
     * last or the leading constructor argument.
     *
     * If construction fails, its name is removed before the exception is rethrown. If that
     * cleanup fails, the cleanup exception propagates instead of the construction exception.
     *
     * @tparam Args Constructor argument types.
     * @param[in] args Constructor arguments, forwarded exactly once, by the arena that holds the
     * object.
     * @return Address of the constructed object, which is also the address its handle names.
     * @throw std::bad_alloc if no arena can hold the object, or if a named instance's map entry
     * cannot be allocated.
     * @throw std::runtime_error if a named instance's name is already registered, or if a lock
     * cannot be taken.
     */
    template <class... Args>
    T* operator()(Args&&... args) const
    {
      ctor_args<T, Args...> arguments(_manager, args...);
      auto const construct = [&](arena_manager* arena) {
        return arena->generic_construct(arguments, _name, 1, false, false);
      };
      if (_name == nullptr) { return _manager->construct_in_arena<T>(construct); }

      auto const guard = _manager->lock_name_index();
      if (_manager->arena_holding(_name) != nullptr) {
        throw std::runtime_error(
          std::format("Shared memory instance name '{}' is already registered", _name));
      }
      auto& holder = _manager->record_name(_name);
      try {
        return _manager->construct_in_arena<T>([&](arena_manager* arena) {
          holder = _manager->get_handle_from_address(arena);
          return construct(arena);
        });
      } catch (...) {
        _manager->forget_name(_name);
        throw;
      }
    }

   private:
    shared_segment_manager* _manager;
    char const* _name;
  };

  /**
   * @brief Construct one named object, trying arenas in the configured fallback order for the
   * NUMA node on which the calling thread is running.
   *
   * The object and the allocations its constructor makes through an allocator built from this
   * manager are allocated independently using the fallback order, so they may land on different
   * arenas.
   *
   * @tparam T Type to construct.
   * @param[in] name Instance name.
   * @return Proxy whose call operator takes the constructor arguments.
   */
  template <class T>
  [[nodiscard]] construct_proxy<T> construct(char const* name)
  {
    return construct_proxy<T>(this, name);
  }

  /**
   * @brief Construct one anonymous object, called with `boost::interprocess::anonymous_instance`.
   *
   * @tparam T Type to construct.
   * @return Proxy whose call operator takes the constructor arguments.
   */
  template <class T>
  [[nodiscard]] construct_proxy<T> construct(decltype(boost::interprocess::anonymous_instance))
  {
    return construct_proxy<T>(this, nullptr);
  }

  /**
   * @brief Destroy an object and release its storage to the arena that owns it.
   *
   * @pre @p ptr is null or was returned by @ref construct. Checked by an assertion.
   *
   * @tparam T Type of the object, which must be the type it was constructed as.
   * @param[in] ptr Object to destroy.
   * @throw std::runtime_error if the name map's or the arena's lock cannot be taken.
   */
  template <class T>
  void destroy_ptr(T const* ptr)
  {
    if (ptr == nullptr) { return; }

    if (arena_manager::get_instance_type(ptr) != boost::interprocess::named_type) {
      destroy_in_arena(ptr);
      return;
    }
    auto const guard = lock_name_index();
    forget_name(arena_manager::get_instance_name(ptr));
    destroy_in_arena(ptr);
  }

  // ----- Named lookup -----

  /**
   * @brief Find a named object.
   *
   * @pre @p name is not null. Checked by an assertion.
   *
   * @tparam T Type the name was constructed with.
   * @param[in] name Instance name.
   * @return The object and a count of one, or `{nullptr, 0}` if the name is not registered.
   * @throw std::runtime_error if the name map's or the arena's lock cannot be taken.
   */
  template <class T>
  [[nodiscard]] std::pair<T*, size_type> find(char const* name)
  {
    assert(name != nullptr && "An anonymous object cannot be found by name");
    auto const guard = lock_name_index();
    auto* arena      = arena_holding(name);
    if (arena == nullptr) { return {nullptr, size_type{0}}; }
    return detail::rethrow_as_runtime_error("find an object in a shared memory arena",
                                            [&] { return arena->template find<T>(name); });
  }

  // ----- Conversion between addresses and handles -----

  /**
   * @brief Return the handle identifying @p ptr, as a byte offset from the mapping base.
   *
   * @param[in] ptr Address in the mapping.
   */
  [[nodiscard]] handle_t get_handle_from_address(void const* ptr) const noexcept;

  /**
   * @brief Return the address a handle refers to.
   *
   * @param[in] handle Handle returned by @ref get_handle_from_address.
   */
  [[nodiscard]] void* get_address_from_handle(handle_t handle) noexcept;

  // ----- Free-space queries -----

  /**
   * @brief Return the free bytes summed over every arena.
   */
  [[nodiscard]] size_type get_free_memory() const noexcept;

#ifdef GQE_ENABLE_FAULT_INJECTION
  /**
   * @brief Return the allocator lock of arena @p index.
   *
   * Only for tests that poison the lock deliberately, by holding it in a process that dies.
   *
   * Declared only in a build with the CMake option `GQE_ENABLE_FAULT_INJECTION`, which is off by
   * default, so that a regular build does not hand out the allocator's lock.
   *
   * @pre @p index is less than the number of arenas. Checked by an assertion.
   */
  [[nodiscard]] boost::interprocess::interprocess_mutex& arena_lock_for_testing(std::size_t index);
#endif

  /**
   * @brief Return the segment's poisoned flag, which is nonzero once its contents are known to be
   * unrecoverable.
   *
   * The flag sits at a fixed place in the manager rather than under a name, so a process reaches it
   * without taking any lock in the segment. That matters because the lock that would guard a lookup
   * may be the one that became unrecoverable. See @ref gqe::attach_shared_memory_poison_flag.
   */
  [[nodiscard]] std::atomic<std::uint32_t>& poisoned_flag() noexcept { return _poisoned; }

  /**
   * @brief Return whether every allocation made from this segment has been released.
   *
   * A segment whose objects have all been destroyed reports true whether or not any of them were
   * named.
   *
   * @throw std::runtime_error if an arena's lock cannot be taken.
   */
  [[nodiscard]] bool all_memory_deallocated();

  // ----- Header sizing for segment layout planning -----

  /**
   * @brief Return the byte offset of the first arena for a mapping with this manager at its head.
   *
   * The region covers the manager and the header's tables, rounded up so the arenas begin on a page
   * boundary.
   *
   * @param[in] page_bytes System page size.
   * @param[in] arena_count Number of arenas the mapping will hold.
   * @param[in] fallback_order_count Number of fallback orders.
   * @return Byte offset of the first arena, a whole number of pages.
   */
  [[nodiscard]] static size_type header_bytes(size_type page_bytes,
                                              std::size_t arena_count,
                                              std::size_t fallback_order_count) noexcept;

 private:
  /**
   * @brief Handle value that names no object.
   *
   * This manager occupies offset 0 of the mapping, so no allocation can ever start there and zero
   * is free to mean "absent".
   */
  static constexpr handle_t invalid_handle = 0;

  /**
   * @brief Value stamped at the head of a fully initialized segment.
   */
  static constexpr std::uint64_t magic_number = 0x5347'5F4D'4147'4943ULL;  // "SG_MAGIC"

  /**
   * @brief Return the number of arenas.
   */
  [[nodiscard]] std::size_t arena_count() const noexcept { return _header.arena_count(); }

  /**
   * @brief Return the size of the largest arena in bytes.
   */
  [[nodiscard]] size_type largest_arena_bytes() const noexcept;

  /**
   * @brief Return the policy the segment was created with.
   */
  [[nodiscard]] segment_layout_policy_type policy() const noexcept { return _header.policy(); }

  /**
   * @brief Construct each arena's manager, then this manager, over an already-mapped,
   * already-placed region.
   *
   * Called once by the creating process after every arena has been placed and populated.
   *
   * @param[in] mapping_base Base of the mapping, which the manager occupies from offset 0.
   * @param[in] total_bytes Size of the whole mapping.
   * @param[in] data_offset Byte offset of the first arena.
   * @param[in] policy Policy the layout came from.
   * @param[in] layout Arenas and fallback orders to record.
   * @param[in] arena_offsets Byte offset of each arena from the mapping base, in arena order.
   * @throw std::runtime_error if an arena's manager or this manager cannot be built.
   */
  static shared_segment_manager* initialize(void* mapping_base,
                                            size_type total_bytes,
                                            size_type data_offset,
                                            segment_layout_policy_type policy,
                                            segment_layout const& layout,
                                            std::span<size_type const> arena_offsets);

  /**
   * @brief Return whether the creator has finished @ref initialize.
   */
  [[nodiscard]] static bool is_initialized(void* mapping_base) noexcept;

  /**
   * @brief Validate a mapping written by @ref initialize and return its manager.
   *
   * Checks the magic number, then validates the header with @ref segment_header::validate.
   *
   * @param[in] mapping_base Base of the mapping, which the manager occupies from offset 0.
   * @param[in] mapped_bytes Size of the mapping.
   * @throw std::runtime_error if the magic number is absent or the header does not validate.
   */
  [[nodiscard]] static shared_segment_manager* attach(void* mapping_base, size_type mapped_bytes);

  /**
   * @brief Write the manager and its header over an already-placed mapping.
   *
   * Invoked only through @ref initialize, which placement-news the manager at the mapping base.
   *
   * @param[in] total_bytes Size of the whole mapping.
   * @param[in] data_offset Byte offset of the first arena.
   * @param[in] policy Policy the layout came from.
   * @param[in] layout Arenas and fallback orders to record.
   * @param[in] arena_offsets Byte offset of each arena from the mapping base, in arena order.
   */
  shared_segment_manager(size_type total_bytes,
                         size_type data_offset,
                         segment_layout_policy_type policy,
                         segment_layout const& layout,
                         std::span<size_type const> arena_offsets);

  /**
   * @brief Return the arena that owns @p ptr.
   */
  [[nodiscard]] arena_manager* arena_of(void const* ptr) noexcept;

  /**
   * @copydoc gqe::shared_segment_manager::arena_of
   */
  [[nodiscard]] arena_manager const* arena_of(void const* ptr) const noexcept;

  /**
   * @brief Return the arena at @p index.
   */
  [[nodiscard]] arena_manager* arena_at(std::size_t index) noexcept;

  /**
   * @copydoc gqe::shared_segment_manager::arena_at
   */
  [[nodiscard]] arena_manager const* arena_at(std::size_t index) const noexcept;

  /**
   * @brief Return the index of the arena that owns @p ptr, or @ref arena_count if no arena does.
   */
  [[nodiscard]] std::size_t arena_index_of(void const* ptr) const noexcept;

  /**
   * @brief Lock guarding the name map.
   */
  using name_index_lock =
    boost::interprocess::scoped_lock<boost::interprocess::interprocess_recursive_mutex>;

  /**
   * @brief Constructor arguments that build an object with this manager as its allocator source.
   *
   * An arena's manager allocates the object and calls @ref construct_n to build it. The arguments
   * are held by reference and forwarded once. Uses-allocator construction gives an allocator-aware
   * `T` an allocator built from this manager, not from the arena's.
   *
   * @tparam T Type to construct.
   * @tparam Args Constructor argument types, as deduced by @ref construct_proxy.
   */
  template <class T, class... Args>
  class ctor_args {
   public:
    using object_type = T;

    /**
     * @brief Bind the arguments to the manager the object's allocator is built from.
     *
     * @param[in] manager Manager the object's allocator is built from.
     * @param[in] args Constructor arguments, which must outlive this object.
     */
    explicit ctor_args(shared_segment_manager* manager, Args&... args) noexcept
      : _manager(manager), _args(args...)
    {
    }

    /**
     * @brief Construct the object at @p memory.
     *
     * @pre @p count is one. Checked by an assertion.
     *
     * @tparam ArenaManager Type of the arena's manager, which is not used.
     * @param[in] memory Storage for the object.
     * @param[in] count Number of objects to construct.
     */
    template <class ArenaManager>
    void construct_n(void* memory, ArenaManager*, [[maybe_unused]] std::size_t count)
    {
      assert(count == 1 && "Only single objects are constructed");
      std::apply(
        [&](Args&... values) {
          boost::container::uninitialized_construct_using_allocator(
            static_cast<T*>(memory),
            boost::interprocess::uses_segment_manager<shared_segment_manager>(_manager),
            std::forward<Args>(values)...);
        },
        _args);
    }

   private:
    shared_segment_manager* _manager;
    std::tuple<Args&...> _args;
  };

  /**
   * @brief Construct an object in the first arena in the fallback order whose manager accepts it.
   *
   * @tparam T Type of the object.
   * @tparam Construct Callable of signature `T*(arena_manager*)` that asks one arena's manager to
   * construct the object and returns null if that arena cannot hold it.
   * @param[in] construct Construction to try on each arena.
   * @return Address of the constructed object.
   * @throw std::bad_alloc if no arena can hold the object.
   * @throw std::runtime_error if an arena's lock cannot be taken.
   */
  template <class T, class Construct>
  T* construct_in_arena(Construct construct)
  {
    auto* object = try_arenas([&](arena_manager* arena) -> T* {
      return detail::rethrow_as_runtime_error("construct an object in a shared memory arena",
                                              [&] { return construct(arena); });
    });
    if (object == nullptr) {
      log_refused_allocation(sizeof(T));
      throw std::bad_alloc();
    }
    return object;
  }

  /**
   * @brief Destroy an object through the manager of the arena that holds it.
   *
   * @tparam T Type of the object, which must be the type it was constructed as.
   * @pre @p object was returned by @ref construct. Checked by an assertion.
   * @throw std::runtime_error if the arena's lock cannot be taken.
   */
  template <class T>
  void destroy_in_arena(T const* object)
  {
    auto* arena = arena_of(object);
    assert(arena != nullptr && "Address does not belong to the shared memory segment");
    detail::rethrow_as_runtime_error("destroy an object in a shared memory arena",
                                     [&] { arena->destroy_ptr(object); });
  }

  /**
   * @brief Log an allocation of @p bytes that no arena could serve.
   */
  void log_refused_allocation(size_type bytes) const;

  /**
   * @brief Ask one arena to serve @p command, reporting refusal as a null return.
   *
   * @param[in] arena Arena to ask.
   * @param[in] command Allocation command flags.
   * @param[in] limit_size Smallest acceptable size, in elements.
   * @param[in,out] prefer_in_recvd_out_size Preferred size on entry, size granted on return.
   * @param[in,out] reuse Block to operate on. Null on return if a new block was allocated.
   * @param[in] sizeof_object Size of one element in bytes.
   * @param[in] alignof_object Alignment of one element in bytes.
   * @return The allocated block, or null if this arena could not serve the request.
   * @throw std::runtime_error if the arena's lock cannot be taken.
   */
  static void* attempt_allocation_command(arena_manager* arena,
                                          boost::interprocess::allocation_type command,
                                          size_type limit_size,
                                          size_type& prefer_in_recvd_out_size,
                                          void*& reuse,
                                          size_type sizeof_object,
                                          size_type alignof_object);

  /**
   * @brief Report an allocation command that no arena could serve.
   *
   * Returns null only for a `nothrow_allocation` command, because callers of a throwing command use
   * the result without checking it. A throwing command is logged before it throws.
   *
   * @param[in] command Allocation command flags, as the caller gave them.
   * @param[in] refused_bytes Smallest size the caller would have accepted, in bytes.
   * @return Null, if and only if @p command is a nothrow allocation.
   * @throw std::bad_alloc otherwise.
   */
  void* refuse_allocation_command(boost::interprocess::allocation_type command,
                                  size_type refused_bytes) const;

  /**
   * @brief Try @p attempt on each arena in the calling thread's fallback order until one succeeds.
   *
   * @tparam Attempt Callable of signature `R(arena_manager*)`, where `R` is contextually
   * convertible to `bool` and value-initializes to a false value. A failed attempt reports failure
   * by returning a false value, which lets the request continue to the next arena in the fallback
   * order. An exception from an attempt stops further attempts and propagates.
   * @param[in] attempt Operation to try on each arena.
   * @return The first true result, or a value-initialized `R` if every arena refused.
   */
  template <class Attempt>
  auto try_arenas(Attempt attempt) -> decltype(attempt(std::declval<arena_manager*>()))
  {
    // With one configured fallback order, every NUMA node uses the order at index 0. Looking up
    // the calling thread's node cannot change the arena choices.
    auto const numa_node = _header.fallback_order_count() == 1 ? 0 : current_numa_node();
    for (auto const arena_index : _header.fallback_order(numa_node)) {
      auto result = attempt(arena_at(static_cast<std::size_t>(arena_index)));
      if (result) { return result; }
    }
    return decltype(attempt(std::declval<arena_manager*>())){};
  }

  /**
   * @brief Lock the name map.
   *
   * @throw std::runtime_error if the lock cannot be taken.
   */
  [[nodiscard]] name_index_lock lock_name_index();

  /**
   * @brief Return the arena holding the object named @p name, or null if no object has that name.
   *
   * @pre The caller holds @ref lock_name_index, and @p name is not null. Not checked.
   */
  [[nodiscard]] arena_manager* arena_holding(char const* name);

  /**
   * @brief Add @p name to the map, creating the map if this is the first name.
   *
   * A map created here is released again if the insert fails.
   *
   * @pre The caller holds @ref lock_name_index, and @p name is not in the map. Not checked.
   * @return The handle of the arena the name maps to, @ref invalid_handle until the caller sets it.
   * @throw std::bad_alloc if the map entry cannot be allocated.
   */
  [[nodiscard]] handle_t& record_name(char const* name);

  /**
   * @brief Drop @p name from the map if it is there, releasing the map if that empties it.
   *
   * @pre The caller holds @ref lock_name_index. Not checked.
   */
  void forget_name(char const* name);

  /**
   * @brief Return the NUMA node the calling thread is running on.
   */
  [[nodiscard]] static int current_numa_node();

  /**
   * Must remain first, because attaching processes read this before interpreting anything else.
   *
   * Accessed through `std::atomic_ref`, because a process attaching during creation reads it while
   * the creator writes it.
   */
  std::uint64_t _magic_number;

  /// Guards the name map.
  boost::interprocess::interprocess_recursive_mutex _name_index_latch;

  /// Handle of the name map, or @ref invalid_handle while no named instance exists.
  handle_t _name_index_handle;

  /// See @ref poisoned_flag.
  std::atomic<std::uint32_t> _poisoned;

  /// Arena layout. Must remain last, because its tables follow it in the mapping.
  segment_header _header;
};

static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free,
              "The magic number lives in a mapping shared between processes, so it must not be "
              "published through a lock the other process cannot see");

static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
              "The poisoned flag lives in a mapping shared between processes, so it must not be "
              "guarded by a lock the other process cannot see");

static_assert(std::is_standard_layout_v<shared_segment_manager>,
              "The segment manager is read from a mapping shared between processes, so it must be "
              "standard layout with no virtual functions");

/**
 * @brief An inter-process shared memory segment whose allocations are served from arenas.
 *
 * # Design
 *
 * This type owns a process's mapping of a shared memory object. Allocation, construction, lookup,
 * and handle operations forward to @ref shared_segment_manager. See that class for allocation
 * routing, shared layout, and synchronization. `get_segment_manager()` exposes the manager for
 * building a `boost::interprocess::allocator`.
 *
 * Creation sizes the object without reserving its pages, places each arena, and then populates
 * it.
 *
 * One process creates the segment, naming a @ref gqe::segment_layout_policy_type, and the
 * arrangement is recorded in the mapping's header. Any number of processes then attach with the
 * open constructor, which maps and validates the segment and reads the arrangement from the
 * header. Attaching populates this process's page tables and registers its mapping with CUDA by
 * default. CPU-only attachments can disable registration. Creation never registers, and attachment
 * never changes placement. A registered mapping is unregistered before it is unmapped.
 *
 * The shared memory object's name belongs to the caller. Neither constructor removes it, and
 * neither does the destructor. A segment that fails to come up leaves its object behind for the
 * caller to unlink.
 *
 * ## Thread Safety
 *
 * Construction and destruction are not thread-safe. Every other member is, with the guarantees
 * described on @ref shared_segment_manager.
 */
class shared_segment {
 public:
  using segment_manager = shared_segment_manager;
  using handle_t        = segment_manager::handle_t;
  using size_type       = segment_manager::size_type;

  /**
   * @brief Whether an attaching process registers its mapping with CUDA after population.
   */
  enum class registration_policy {
    none,  ///< Populate without CUDA registration.
    cuda   ///< Populate and register the mapping with CUDA.
  };

  /**
   * @brief Create, place, and populate a new segment.
   *
   * Fails if a segment of the same name already exists.
   *
   * If creation fails once the object exists, the object is truncated to one byte, which tells a
   * process waiting to open it that creation failed, and is left for the caller to unlink.
   *
   * @param[in] name Shared memory object name. A leading slash is added if it has none.
   * @param[in] bytes Requested total size. The policy lays its arenas out within it.
   * @param[in] policy Arrangement to build. Recorded in the mapping's header, so a process that
   * later attaches reads the arrangement from there.
   * @param[in] perm Permissions for the shared memory object, applied regardless of the umask.
   * @throw std::runtime_error if the object cannot be created, placed, or populated.
   * @throw std::system_error carrying the operating system error when a system call fails, which
   * is `std::errc::file_exists` if the object already exists.
   * @throw std::invalid_argument if @p policy names no policy.
   * @throw std::logic_error if the policy's layout is malformed.
   */
  shared_segment(create_only_t,
                 std::string const& name,
                 size_type bytes,
                 segment_layout_policy_type policy,
                 boost::interprocess::permissions const& perm = boost::interprocess::permissions());

  /**
   * @brief Attach to and populate an existing segment, registering with CUDA by default.
   *
   * Attachment waits up to 300 seconds for the creator to size the object, then up to another
   * 300 seconds for initialization to finish. A detected creation failure ends either wait
   * immediately.
   *
   * @param[in] name Shared memory object name. A leading slash is added if it has none.
   * @param[in] registration Whether to register this process's mapping with CUDA.
   * @throw std::runtime_error if the object's creation failed or did not finish in time, or it
   * does not validate.
   * @throw std::system_error carrying the operating system error when a system call fails, which
   * is `std::errc::no_such_file_or_directory` if the object does not exist.
   * @throw gqe::cuda_error if CUDA registration fails.
   */
  shared_segment(open_only_t,
                 std::string const& name,
                 registration_policy registration = registration_policy::cuda);

  ~shared_segment() noexcept;

  shared_segment(shared_segment const&)            = delete;
  shared_segment(shared_segment&&)                 = delete;
  shared_segment& operator=(shared_segment const&) = delete;
  shared_segment& operator=(shared_segment&&)      = delete;

  /**
   * @brief Return the manager that owns allocation for this segment.
   */
  [[nodiscard]] segment_manager* get_segment_manager() const noexcept { return _manager; }

  /**
   * @brief Return the base address of the mapping in this process.
   */
  [[nodiscard]] void* get_address() const noexcept { return _region.get_address(); }

  /**
   * @brief Return the size of the mapping in bytes.
   */
  [[nodiscard]] size_type get_size() const noexcept { return _region.get_size(); }

  /**
   * @copydoc gqe::shared_segment_manager::allocate_aligned
   */
  void* allocate_aligned(size_type bytes, size_type alignment)
  {
    return _manager->allocate_aligned(bytes, alignment);
  }

  /**
   * @copydoc gqe::shared_segment_manager::deallocate
   */
  void deallocate(void* ptr) { _manager->deallocate(ptr); }

  /**
   * @brief Construct one named object. See @ref gqe::shared_segment_manager::construct.
   *
   * @tparam T Type to construct.
   * @param[in] name Instance name.
   * @return Proxy whose call operator takes the constructor arguments.
   */
  template <class T>
  [[nodiscard]] segment_manager::construct_proxy<T> construct(char const* name)
  {
    return _manager->template construct<T>(name);
  }

  /**
   * @brief Construct one anonymous object. See @ref gqe::shared_segment_manager::construct.
   *
   * @tparam T Type to construct.
   * @param[in] anonymous `boost::interprocess::anonymous_instance`.
   * @return Proxy whose call operator takes the constructor arguments.
   */
  template <class T>
  [[nodiscard]] segment_manager::construct_proxy<T> construct(
    decltype(boost::interprocess::anonymous_instance) anonymous)
  {
    return _manager->template construct<T>(anonymous);
  }

  /**
   * @copydoc gqe::shared_segment_manager::destroy_ptr
   */
  template <class T>
  void destroy_ptr(T const* ptr)
  {
    _manager->template destroy_ptr<T>(ptr);
  }

  /**
   * @copydoc gqe::shared_segment_manager::find
   */
  template <class T>
  [[nodiscard]] std::pair<T*, size_type> find(char const* name)
  {
    return _manager->template find<T>(name);
  }

  /**
   * @copydoc gqe::shared_segment_manager::get_handle_from_address
   */
  [[nodiscard]] handle_t get_handle_from_address(void const* ptr) const noexcept
  {
    return _manager->get_handle_from_address(ptr);
  }

  /**
   * @copydoc gqe::shared_segment_manager::get_address_from_handle
   */
  [[nodiscard]] void* get_address_from_handle(handle_t handle) const noexcept
  {
    return _manager->get_address_from_handle(handle);
  }

  /**
   * @copydoc gqe::shared_segment_manager::get_free_memory
   *
   * This is a sum over arenas and is therefore an upper bound on what any single allocation can
   * obtain, not a size any single request is guaranteed to be granted.
   */
  [[nodiscard]] size_type get_free_memory() const noexcept { return _manager->get_free_memory(); }

  /**
   * @copydoc gqe::shared_segment_manager::all_memory_deallocated
   */
  [[nodiscard]] bool all_memory_deallocated() const { return _manager->all_memory_deallocated(); }

 private:
  boost::interprocess::mapped_region _region;
  segment_manager* _manager{};

  /// Whether this process's mapping was successfully registered with CUDA.
  bool _is_cuda_registered{};

  /**
   * Policy the segment was created with, copied out of the mapping.
   *
   * Kept in process memory because a core dump omits the mapping, which is file-backed and shared.
   */
  segment_layout_policy_type _policy{};
};

}  // namespace gqe
