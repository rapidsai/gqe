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

#include <gqe/storage/shared_segment.hpp>

#include <gqe/storage/segment_layout.hpp>
#include <gqe/storage/shared_memory.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/version.hpp>
#include <rmm/aligned.hpp>

#include <cuda_runtime_api.h>

#include <linux/mman.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

// The conda toolchain's sysroot predates this (kernel 5.14 / glibc 2.35). The value is stable UAPI.
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <new>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace gqe {

namespace {

/// Number of threads used to fault in the mapping. Chosen by benchmark, because population is
/// bandwidth bound and stops scaling shortly after this point.
constexpr std::size_t population_thread_count = 4;

/// How long opening waits for the process creating the segment.
constexpr std::chrono::seconds initialize_timeout{300};

/// Pause between checks while opening waits for the process creating the segment.
constexpr std::chrono::milliseconds initialize_poll_interval{1};

/// Size a failed creation leaves the shared memory object at, which opening reads as that failure.
constexpr std::size_t failed_creation_bytes = 1;

using detail::rethrow_as_runtime_error;

// ----- Shared memory object -----

/**
 * @brief Set the size of the object open on @p descriptor, retrying while the call is interrupted
 * by a signal.
 *
 * @return 0, or -1 with `errno` set.
 */
int eintr_aware_ftruncate(int descriptor, std::size_t bytes) noexcept
{
  int result = -1;
  do {
    result = ::ftruncate(descriptor, static_cast<off_t>(bytes));
  } while (result == -1 && errno == EINTR);
  return result;
}

/**
 * @brief Return the size of @p object.
 *
 * @throw std::system_error if the size cannot be read.
 */
std::size_t object_bytes(boost::interprocess::shared_memory_object const& object,
                         std::string const& name)
{
  boost::interprocess::offset_t bytes = 0;
  if (!object.get_size(bytes)) {
    auto const error = errno;
    throw std::system_error(error,
                            std::generic_category(),
                            std::format("Failed to size shared memory object '{}'", name));
  }
  return static_cast<std::size_t>(bytes);
}

/**
 * @brief Poll until @p is_ready returns true.
 *
 * @tparam Predicate Callable of signature `bool()`. It may throw to end the wait early.
 * @param[in] name Shared memory object name, for the message.
 * @param[in] is_ready Condition to wait for.
 * @throw std::runtime_error if @p is_ready is still false after @ref initialize_timeout.
 */
template <class Predicate>
void wait_for_creator(std::string const& name, Predicate is_ready)
{
  auto const deadline = std::chrono::steady_clock::now() + initialize_timeout;
  while (!is_ready()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
        std::format("Timed out waiting for shared memory segment '{}' to be created", name));
    }
    std::this_thread::sleep_for(initialize_poll_interval);
  }
}

/**
 * @brief Return the error opening reports for a segment whose creation failed.
 */
std::runtime_error creation_failed(std::string const& name)
{
  return std::runtime_error(
    std::format("Shared memory segment '{}' failed to be created by its creating process", name));
}

// ----- Arena placement and population -----

/**
 * @brief Return the ids in @p nodes as a comma-separated list, for messages.
 */
std::string node_list(cpu_set const& nodes)
{
  std::string list;
  for (int node = 0; node < cpu_set::max_count; ++node) {
    if (!nodes.contains(node)) { continue; }
    if (!list.empty()) { list += ","; }
    list += std::to_string(node);
  }
  return list;
}

/**
 * @brief Place a page-aligned range on a set of NUMA nodes.
 *
 * The placement governs pages that have yet to be faulted in, so it must be applied before the
 * range is populated or otherwise touched.
 *
 * @param[in] range Page-aligned range to place.
 * @param[in] mode How to place the range. `arena_placement::none` leaves it to first touch.
 * @param[in] nodes NUMA nodes the range is placed on.
 * @throw std::runtime_error if the system reports more NUMA nodes than a CPU set can describe.
 * @throw std::system_error if the range cannot be placed.
 */
void apply_placement(std::span<std::byte> range, arena_placement mode, cpu_set const& nodes)
{
  int memory_policy = MPOL_DEFAULT;
  switch (mode) {
    case arena_placement::none: return;
    case arena_placement::bind: memory_policy = MPOL_BIND; break;
    case arena_placement::interleave: memory_policy = MPOL_INTERLEAVE; break;
  }

  // `mbind` sizes the mask in node bits, so the bound comes from the kernel's node capacity rather
  // than from the CPU capacity `cpu_set` is otherwise dimensioned for. `cpu_set` still supplies the
  // storage, and is wide enough for every node the kernel can report.
  auto const node_bits = ::numa_num_possible_nodes();
  if (node_bits <= 0 || node_bits > cpu_set::max_count) {
    throw std::runtime_error("The system reports more NUMA nodes than a CPU set can describe");
  }

  // The kernel reads one bit fewer than `maxnode`, so `maxnode` is one more than the number of node
  // bits.
  if (::mbind(range.data(),
              range.size(),
              memory_policy,
              nodes.bits(),
              static_cast<unsigned long>(node_bits) + 1,
              MPOL_MF_STRICT) == -1) {
    auto const error = errno;
    throw std::system_error(
      error,
      std::generic_category(),
      std::format("Failed to place shared memory arena on NUMA nodes {}", node_list(nodes)));
  }
}

/**
 * @brief Advise the kernel which page kind to back a range with.
 *
 * The advice governs pages that have yet to be faulted in, so it must be applied before the range
 * is populated.
 *
 * @param[in] range Page-aligned range to advise.
 * @param[in] pages Page kind to back the range with.
 * @throw std::runtime_error if @p pages is a HugeTLB page kind.
 * @throw std::system_error if the advice is refused.
 */
void apply_page_kind(std::span<std::byte> range, page_kind::type pages)
{
  int advice = 0;
  // The advice matches what `numa_memory_resource` gives each page kind it supports.
  switch (pages) {
    case page_kind::type::system_default: return;
    case page_kind::type::small: advice = MADV_NOHUGEPAGE; break;
    case page_kind::type::transparent_huge: advice = MADV_HUGEPAGE; break;
    case page_kind::type::huge2mb:
    case page_kind::type::huge1gb:
      throw std::runtime_error("Shared memory segment cannot be backed with HugeTLB pages");
  }

  if (::madvise(range.data(), range.size(), advice) == -1) {
    throw std::system_error(
      errno, std::generic_category(), "Failed to advise shared memory page kind");
  }
}

/**
 * @brief Return the byte offset of each arena from the mapping base, laid end to end.
 *
 * @param[in] layout Arrangement whose arenas are laid out.
 * @param[in] data_offset Byte offset of the first arena.
 * @param[in] page_bytes Size of the pages the arenas are backed with.
 * @throw std::logic_error if an arena is empty, or if an arena other than the last is not a whole
 * number of pages, which would leave the arena after it unaligned for `mbind`.
 */
std::vector<shared_segment::size_type> arena_offsets(segment_layout const& layout,
                                                     shared_segment::size_type data_offset,
                                                     shared_segment::size_type page_bytes)
{
  std::vector<shared_segment::size_type> offsets;
  offsets.reserve(layout.arenas.size());
  auto offset = data_offset;
  for (std::size_t index = 0; index < layout.arenas.size(); ++index) {
    auto const& arena  = layout.arenas[index];
    auto const is_last = index + 1 == layout.arenas.size();
    GQE_EXPECTS(arena.bytes > 0 && (is_last || arena.bytes % page_bytes == 0),
                "Segment layout gives an arena a size that is not a whole number of pages");
    offsets.push_back(offset);
    offset += arena.bytes;
  }
  return offsets;
}

/**
 * @brief Check that every fallback order names only arenas the layout has.
 *
 * @throw std::logic_error if a fallback order is longer than there are arenas or names an arena
 * that does not exist.
 */
void check_fallback_order_by_node(segment_layout const& layout)
{
  GQE_EXPECTS(!layout.fallback_order_by_node.empty(),
              "Segment layout describes no fallback orders");
  auto const arena_count = static_cast<int>(layout.arenas.size());
  for (auto const& order : layout.fallback_order_by_node) {
    GQE_EXPECTS(order.size() <= layout.arenas.size(),
                "Segment layout gives a fallback order more entries than there are arenas");
    GQE_EXPECTS(
      std::ranges::all_of(order, [&](int arena) { return arena >= 0 && arena < arena_count; }),
      "Segment layout gives a fallback order an arena that does not exist");
  }
}

/**
 * @brief Fault in every page of a range, writable, using several threads.
 *
 * @param[in] range Range to populate.
 * @param[in] page_bytes System page size, used to split the work on page boundaries.
 * @throw std::system_error if any part of the range cannot be populated.
 */
void populate(std::span<std::byte> range, std::size_t page_bytes)
{
  if (range.empty()) { return; }

  auto const chunk_bytes =
    rmm::align_up(utility::divide_round_up(range.size(), population_thread_count), page_bytes);

  std::array<int, population_thread_count> errors{};
  {
    std::vector<std::jthread> populators;
    populators.reserve(population_thread_count);
    for (std::size_t i = 0; i < population_thread_count; ++i) {
      auto const offset = i * chunk_bytes;
      if (offset >= range.size()) { break; }
      auto const bytes = std::min(chunk_bytes, range.size() - offset);

      populators.emplace_back(
        [chunk = range.subspan(offset, bytes), &error = errors[i]]() noexcept {
          // The kernel performs the write faults, so the pages are instantiated under the binding
          // already applied to the range without this thread writing to them.
          if (::madvise(chunk.data(), chunk.size(), MADV_POPULATE_WRITE) == -1) { error = errno; }
        });
    }
  }

  auto const failure = std::ranges::find_if(errors, [](int error) { return error != 0; });
  if (failure != errors.end()) {
    throw std::system_error(
      *failure, std::generic_category(), "Failed to populate shared memory segment");
  }
}

/**
 * @brief Place each arena and populate the whole newly sized mapping.
 *
 * @param[in] base Base of the mapping.
 * @param[in] layout Arrangement to place.
 * @param[in] offsets Byte offset of each arena from @p base, in arena order.
 * @param[in] mapping Whole mapping, including its manager and layout tables.
 * @param[in] page_bytes Size of the pages the arenas are backed with.
 * @throw std::runtime_error if the layout asks for an unsupported page kind or more NUMA nodes than
 * a CPU set can describe.
 * @throw std::system_error if an arena cannot be placed or populated.
 */
void build_arenas(std::byte* base,
                  segment_layout const& layout,
                  std::span<shared_segment::size_type const> offsets,
                  std::span<std::byte> mapping,
                  shared_segment::size_type page_bytes)
{
  for (std::size_t arena = 0; arena < layout.arenas.size(); ++arena) {
    auto const& spec  = layout.arenas[arena];
    auto const extent = std::span{base + offsets[arena], spec.bytes};
    apply_page_kind(extent, layout.pages);
    apply_placement(extent, spec.mode, spec.nodes);
  }

  populate(mapping, page_bytes);
}

}  // namespace

segment_header::size_type segment_header::layout_table_bytes(
  std::size_t arena_count, std::size_t fallback_order_count) noexcept
{
  auto const entries = fallback_order_count + fallback_order_count * arena_count;
  return static_cast<size_type>(arena_count) * sizeof(arena_record) +
         static_cast<size_type>(entries) * sizeof(layout_entry_t);
}

bool segment_header::layout_tables_fit(size_type available_bytes,
                                       std::uint32_t arena_count,
                                       std::uint32_t fallback_order_count) noexcept
{
  // Phrased as divisions rather than a multiplication compared against a bound, so that counts
  // read out of a mapping this build did not write cannot overflow the check.
  if (arena_count > available_bytes / sizeof(arena_record)) { return false; }
  auto const capacity =
    (available_bytes - arena_count * sizeof(arena_record)) / sizeof(layout_entry_t);
  // Each fallback order needs one length entry and `arena_count` reserved arena entries.
  return fallback_order_count <= capacity / (std::size_t{arena_count} + 1);
}

std::span<segment_header::arena_record> segment_header::arena_records() noexcept
{
  auto* tail = reinterpret_cast<char*>(this) + sizeof(segment_header);
  return {reinterpret_cast<arena_record*>(tail), _arena_count};
}

std::span<segment_header::arena_record const> segment_header::arena_records() const noexcept
{
  auto const* tail = reinterpret_cast<char const*>(this) + sizeof(segment_header);
  return {reinterpret_cast<arena_record const*>(tail), _arena_count};
}

std::span<segment_header::layout_entry_t> segment_header::fallback_order_lengths() noexcept
{
  auto const records = arena_records();
  return {reinterpret_cast<layout_entry_t*>(records.data() + records.size()),
          _fallback_order_count};
}

std::span<segment_header::layout_entry_t const> segment_header::fallback_order_lengths()
  const noexcept
{
  auto const records = arena_records();
  return {reinterpret_cast<layout_entry_t const*>(records.data() + records.size()),
          _fallback_order_count};
}

std::span<segment_header::layout_entry_t> segment_header::fallback_orders() noexcept
{
  auto const lengths = fallback_order_lengths();
  return {lengths.data() + lengths.size(), std::size_t{_fallback_order_count} * _arena_count};
}

std::span<segment_header::layout_entry_t const> segment_header::fallback_orders() const noexcept
{
  auto const lengths = fallback_order_lengths();
  return {lengths.data() + lengths.size(), std::size_t{_fallback_order_count} * _arena_count};
}

segment_header::segment_header(size_type total_bytes,
                               size_type data_offset,
                               segment_layout_policy_type policy,
                               segment_layout const& layout,
                               std::span<size_type const> arena_offsets)
  : _layout_version(layout_version),
    _policy(policy),
    _arena_count(static_cast<std::uint32_t>(layout.arenas.size())),
    _fallback_order_count(static_cast<std::uint32_t>(layout.fallback_order_by_node.size())),
    _total_bytes(total_bytes),
    _data_offset(data_offset)
{
  std::ranges::transform(
    layout.arenas,
    arena_offsets,
    arena_records().begin(),
    [](arena_spec const& spec, size_type offset) {
      arena_record record{.offset = offset, .bytes = spec.bytes, .mode = spec.mode, .nodes = {}};
      std::copy_n(spec.nodes.bits(), record.nodes.size(), record.nodes.begin());
      return record;
    });

  auto const lengths = fallback_order_lengths();
  auto const orders  = fallback_orders();
  for (std::size_t node_index = 0; node_index < layout.fallback_order_by_node.size();
       ++node_index) {
    auto const& order   = layout.fallback_order_by_node[node_index];
    lengths[node_index] = static_cast<layout_entry_t>(order.size());
    std::ranges::copy(order, orders.subspan(node_index * _arena_count).begin());
  }
}

shared_segment_manager::size_type shared_segment_manager::header_bytes(
  size_type page_bytes, std::size_t arena_count, std::size_t fallback_order_count) noexcept
{
  return rmm::align_up(sizeof(shared_segment_manager) +
                         segment_header::layout_table_bytes(arena_count, fallback_order_count),
                       page_bytes);
}

shared_segment_manager::shared_segment_manager(size_type total_bytes,
                                               size_type data_offset,
                                               segment_layout_policy_type policy,
                                               segment_layout const& layout,
                                               std::span<size_type const> arena_offsets)
  : _name_index_handle(invalid_handle),
    _poisoned(0),
    _header(total_bytes, data_offset, policy, layout, arena_offsets)
{
  // Stored last, so a segment abandoned mid-initialization never validates. Release ordering pairs
  // with the acquire in `is_initialized`, so a process opening the segment while it is created sees
  // every field written above once it sees this.
  std::atomic_ref<std::uint64_t>(_magic_number).store(magic_number, std::memory_order_release);
}

shared_segment_manager* shared_segment_manager::initialize(void* mapping_base,
                                                           size_type total_bytes,
                                                           size_type data_offset,
                                                           segment_layout_policy_type policy,
                                                           segment_layout const& layout,
                                                           std::span<size_type const> arena_offsets)
{
  static_assert(
    offsetof(shared_segment_manager, _header) + sizeof(segment_header) ==
      sizeof(shared_segment_manager),
    "The header's tables follow it in the mapping, so it must be the manager's last member");

  auto* const base = static_cast<std::byte*>(mapping_base);
  for (std::size_t arena = 0; arena < layout.arenas.size(); ++arena) {
    // Constructing the buffer places the arena's manager at the start of the arena. The wrapper
    // owns nothing and its destructor only drops its own pointer, so it is not kept. Every later
    // access reaches the manager by offsetting from the mapping base.
    [[maybe_unused]] auto const arena_buffer = rethrow_as_runtime_error(
      std::format("build arena {} of the shared memory segment", arena), [&] {
        return arena_type(boost::interprocess::create_only,
                          base + arena_offsets[arena],
                          layout.arenas[arena].bytes);
      });
  }

  return rethrow_as_runtime_error("initialize the shared memory segment manager", [&] {
    return ::new (mapping_base)
      shared_segment_manager(total_bytes, data_offset, policy, layout, arena_offsets);
  });
}

bool shared_segment_manager::is_initialized(void* mapping_base) noexcept
{
  auto* manager = static_cast<shared_segment_manager*>(mapping_base);
  return std::atomic_ref<std::uint64_t>(manager->_magic_number).load(std::memory_order_acquire) ==
         magic_number;
}

shared_segment_manager* shared_segment_manager::attach(void* mapping_base, size_type mapped_bytes)
{
  auto* manager = static_cast<shared_segment_manager*>(mapping_base);

  if (!is_initialized(mapping_base)) {
    throw std::runtime_error("Shared memory segment is not a GQE segment");
  }
  manager->_header.validate(mapping_base, mapped_bytes);
  return manager;
}

void segment_header::validate(void const* mapping_base, size_type mapped_bytes) const
{
  if (_layout_version != layout_version) {
    throw std::runtime_error(
      std::format("Shared memory segment has layout version {}, this build understands {}",
                  _layout_version,
                  layout_version));
  }
  if (_total_bytes != mapped_bytes) {
    throw std::runtime_error(
      std::format("Shared memory segment records a size of {} bytes but maps {} bytes",
                  _total_bytes,
                  mapped_bytes));
  }

  // The remaining fields are the ones every accessor computes an address from, so they are checked
  // once here rather than at each use.
  if (_arena_count == 0) { throw std::runtime_error("Shared memory segment records no arenas"); }
  auto const tables_offset =
    static_cast<size_type>(reinterpret_cast<char const*>(this) + sizeof(segment_header) -
                           static_cast<char const*>(mapping_base));
  if (_data_offset < tables_offset || _data_offset > mapped_bytes) {
    throw std::runtime_error(
      std::format("Shared memory segment places its first arena at offset {} of {} bytes",
                  _data_offset,
                  mapped_bytes));
  }

  auto const header_tail_bytes = _data_offset - tables_offset;
  if (!layout_tables_fit(header_tail_bytes, _arena_count, _fallback_order_count)) {
    throw std::runtime_error(
      std::format("Shared memory segment reserves {} header bytes for layout tables describing {} "
                  "arenas and {} fallback orders",
                  header_tail_bytes,
                  _arena_count,
                  _fallback_order_count));
  }

  check_layout_tables();
}

void segment_header::check_layout_tables() const
{
  auto arenas_end = _data_offset;
  for (auto const& record : arena_records()) {
    if (record.offset < arenas_end || record.bytes == 0 || record.offset > _total_bytes ||
        record.bytes > _total_bytes - record.offset) {
      throw std::runtime_error(
        std::format("Shared memory segment records an arena of {} bytes at offset {}, which is not "
                    "within the {} bytes following its header after the arena before it",
                    record.bytes,
                    record.offset,
                    _total_bytes - _data_offset));
    }
    arenas_end = record.offset + record.bytes;
  }

  if (_fallback_order_count == 0) {
    throw std::runtime_error("Shared memory segment records no fallback orders");
  }

  auto const lengths = fallback_order_lengths();
  auto const orders  = fallback_orders();
  for (std::size_t node_index = 0; node_index < lengths.size(); ++node_index) {
    auto const length = lengths[node_index];
    if (length < 0 || static_cast<std::uint32_t>(length) > _arena_count) {
      throw std::runtime_error(
        std::format("Shared memory segment records a fallback order of {} entries over {} arenas",
                    length,
                    _arena_count));
    }

    auto const entries =
      orders.subspan(node_index * _arena_count, static_cast<std::size_t>(length));
    if (!std::ranges::all_of(entries, [this](layout_entry_t arena) {
          return arena >= 0 && static_cast<std::uint32_t>(arena) < _arena_count;
        })) {
      throw std::runtime_error(
        std::format("Shared memory segment records a fallback order for NUMA node {} naming an "
                    "arena outside its {} "
                    "arenas",
                    node_index,
                    _arena_count));
    }
  }
}

int shared_segment_manager::current_numa_node()
{
  // `numa_node_of_cpu` allocates a bitmask and scans the machine's nodes on every call, which is
  // far more than an allocation can afford to spend deciding which arena to try first. The mapping
  // does not change while the process runs, so it is built once and read afterwards. It is never
  // destroyed, so an allocation made during program exit still finds it.
  static std::vector<int> const& cpu_to_node = *new std::vector<int>([] {
    auto const cpu_count = ::numa_num_configured_cpus();
    std::vector<int> mapping(cpu_count > 0 ? static_cast<std::size_t>(cpu_count) : 0, 0);
    std::ranges::transform(
      std::views::iota(0, static_cast<int>(mapping.size())), mapping.begin(), [](int cpu) {
        auto const node = ::numa_node_of_cpu(cpu);
        return node < 0 ? 0 : node;
      });
    return mapping;
  }());

  auto const cpu = ::sched_getcpu();
  if (cpu < 0 || static_cast<std::size_t>(cpu) >= cpu_to_node.size()) { return 0; }
  return cpu_to_node[static_cast<std::size_t>(cpu)];
}

std::span<segment_header::layout_entry_t const> segment_header::fallback_order(
  int numa_node) const noexcept
{
  auto const node_index =
    (numa_node < 0 || static_cast<std::uint32_t>(numa_node) >= _fallback_order_count)
      ? std::size_t{0}
      : static_cast<std::size_t>(numa_node);
  auto const length = static_cast<std::size_t>(fallback_order_lengths()[node_index]);
  return fallback_orders().subspan(node_index * _arena_count, length);
}

shared_segment_manager::arena_manager* shared_segment_manager::arena_at(std::size_t index) noexcept
{
  return reinterpret_cast<arena_manager*>(reinterpret_cast<char*>(this) +
                                          _header.arena_records()[index].offset);
}

#ifdef GQE_ENABLE_FAULT_INJECTION
boost::interprocess::interprocess_mutex& shared_segment_manager::arena_lock_for_testing(
  std::size_t index)
{
  assert(index < arena_count() && "Arena index out of range");
  // Boost has no accessor for the lock. In `rbtree_best_fit`, the only data member is a header that
  // derives from the mutex first, and the memory algorithm is the segment manager's base, so the
  // mutex is at the memory algorithm's address. That layout is private to Boost, so the static
  // assertion pins the version it was checked against.
  static_assert(BOOST_VERSION == 109200,
                "Re-check the rbtree_best_fit layout that arena_lock_for_testing relies on");
  auto& algorithm = arena_at(index)->get_memory_algorithm();
  return *reinterpret_cast<boost::interprocess::interprocess_mutex*>(&algorithm);
}
#endif

shared_segment_manager::arena_manager const* shared_segment_manager::arena_at(
  std::size_t index) const noexcept
{
  return reinterpret_cast<arena_manager const*>(reinterpret_cast<char const*>(this) +
                                                _header.arena_records()[index].offset);
}

std::size_t shared_segment_manager::arena_index_of(void const* ptr) const noexcept
{
  if (ptr == nullptr) { return arena_count(); }

  auto const offset = reinterpret_cast<char const*>(ptr) - reinterpret_cast<char const*>(this);
  if (offset < static_cast<std::ptrdiff_t>(_header.data_offset()) ||
      offset >= static_cast<std::ptrdiff_t>(_header.total_bytes())) {
    return arena_count();
  }

  // Arena sizes may differ, so the owner is found by scanning rather than by dividing. Arena counts
  // are small enough that the scan is cheap.
  auto const position = static_cast<size_type>(offset);
  auto const records  = _header.arena_records();
  auto const owner =
    std::ranges::find_if(records, [position](segment_header::arena_record const& record) {
      return position >= record.offset && position - record.offset < record.bytes;
    });
  return static_cast<std::size_t>(owner - records.begin());
}

shared_segment_manager::arena_manager* shared_segment_manager::arena_of(void const* ptr) noexcept
{
  auto const index = arena_index_of(ptr);
  return index < arena_count() ? arena_at(index) : nullptr;
}

shared_segment_manager::arena_manager const* shared_segment_manager::arena_of(
  void const* ptr) const noexcept
{
  auto const index = arena_index_of(ptr);
  return index < arena_count() ? arena_at(index) : nullptr;
}

void* shared_segment_manager::allocate_aligned(size_type bytes,
                                               size_type alignment,
                                               std::nothrow_t const&)
{
  // No free-space check precedes the attempt. An arena reports total free bytes rather than its
  // largest contiguous block, so a check could pass on an arena that then refuses the request and
  // the walk would stop on it while a sibling still had room.
  return try_arenas([&](arena_manager* arena) -> void* {
    return rethrow_as_runtime_error("allocate from a shared memory arena", [&] {
      return arena->allocate_aligned(bytes, alignment, std::nothrow);
    });
  });
}

void shared_segment_manager::log_refused_allocation(size_type bytes) const
{
  // `bad_alloc` carries no message, and the only free figure a caller sees is summed across arenas,
  // so a request larger than any one of them looks like failure on a roomy segment.
  GQE_LOG_ERROR(
    "Shared memory allocation of {} bytes failed: no arena of the {} segment could serve it. "
    "The largest arena is {} bytes, across {} arenas totaling {} free bytes.",
    bytes,
    to_string(policy()),
    largest_arena_bytes(),
    arena_count(),
    get_free_memory());
}

void* shared_segment_manager::allocate_aligned(size_type bytes, size_type alignment)
{
  auto* ptr = allocate_aligned(bytes, alignment, std::nothrow);
  if (ptr == nullptr) {
    log_refused_allocation(bytes);
    throw std::bad_alloc();
  }
  return ptr;
}

void shared_segment_manager::deallocate(void* ptr)
{
  if (ptr == nullptr) { return; }

  auto* arena = arena_of(ptr);
  assert(arena != nullptr && "Address does not belong to the shared memory segment");
  rethrow_as_runtime_error("release shared memory to its arena", [&] { arena->deallocate(ptr); });
}

void shared_segment_manager::deallocate_many(multiallocation_chain& chain)
{
  while (!chain.empty()) {
    deallocate(boost::interprocess::ipcdetail::to_raw_pointer(chain.pop_front()));
  }
}

shared_segment_manager::size_type shared_segment_manager::size(void const* ptr) const noexcept
{
  auto* arena = arena_of(ptr);
  assert(arena != nullptr && "Address does not belong to the shared memory segment");
  return arena->size(ptr);
}

void* shared_segment_manager::attempt_allocation_command(
  arena_manager* arena,
  boost::interprocess::allocation_type command,
  size_type limit_size,
  size_type& prefer_in_recvd_out_size,
  void*& reuse,
  size_type sizeof_object,
  size_type alignof_object)
{
  // Refusal returns null rather than throwing. Walking the arenas on exceptions would cost a throw
  // per arena and would mistake a failure to take the arena's lock for a full arena.
  return rethrow_as_runtime_error("run an allocation command on a shared memory arena", [&] {
    return arena->allocation_command(command | boost::interprocess::nothrow_allocation,
                                     limit_size,
                                     prefer_in_recvd_out_size,
                                     reuse,
                                     sizeof_object,
                                     alignof_object);
  });
}

void* shared_segment_manager::allocation_command(boost::interprocess::allocation_type command,
                                                 size_type limit_size,
                                                 size_type& prefer_in_recvd_out_size,
                                                 void*& reuse,
                                                 size_type sizeof_object,
                                                 size_type alignof_object)
{
  // The first arena asked works on the caller's outputs, so a refusal reports what that arena
  // left, as a one-arena command does. Every later arena starts from the caller's values on
  // copies.
  auto const requested  = prefer_in_recvd_out_size;
  bool is_first_attempt = true;
  auto const attempt    = [&](arena_manager* arena, void* block) -> void* {
    if (std::exchange(is_first_attempt, false)) {
      reuse = block;
      return attempt_allocation_command(
        arena, command, limit_size, prefer_in_recvd_out_size, reuse, sizeof_object, alignof_object);
    }
    auto granted = requested;
    auto* reused = block;
    auto* result = attempt_allocation_command(
      arena, command, limit_size, granted, reused, sizeof_object, alignof_object);
    if (result != nullptr) {
      prefer_in_recvd_out_size = granted;
      reuse                    = reused;
    }
    return result;
  };

  arena_manager* owner = nullptr;
  if (reuse != nullptr) {
    owner = arena_of(reuse);
    assert(owner != nullptr && "Address does not belong to the shared memory segment");
    if (auto* expanded = attempt(owner, reuse)) { return expanded; }
    if ((command & boost::interprocess::allocate_new) == 0) {
      return refuse_allocation_command(command, limit_size * sizeof_object);
    }
  }

  auto* allocated = try_arenas(
    [&](arena_manager* arena) { return arena == owner ? nullptr : attempt(arena, nullptr); });
  return allocated != nullptr ? allocated
                              : refuse_allocation_command(command, limit_size * sizeof_object);
}

void* shared_segment_manager::refuse_allocation_command(
  boost::interprocess::allocation_type command, size_type refused_bytes) const
{
  if ((command & boost::interprocess::nothrow_allocation) != 0) { return nullptr; }
  log_refused_allocation(refused_bytes);
  throw std::bad_alloc();
}

shared_segment_manager::handle_t shared_segment_manager::get_handle_from_address(
  void const* ptr) const noexcept
{
  return static_cast<handle_t>(reinterpret_cast<char const*>(ptr) -
                               reinterpret_cast<char const*>(this));
}

void* shared_segment_manager::get_address_from_handle(handle_t handle) noexcept
{
  return reinterpret_cast<char*>(this) + handle;
}

namespace {

/**
 * @brief Types backing the segment's map from each name to the handle of the arena holding it.
 *
 * They live here rather than in the header because they are parameterized on the manager, which is
 * incomplete inside its own definition, and because the map is an implementation detail no caller
 * needs to name.
 *
 * The allocators do not throw when freeing. Erasing an entry frees its name from the string's
 * destructor, and an exception there would terminate the process. If an arena's lock is
 * unrecoverable, the block is leaked and the shared memory is marked poisoned instead.
 */
using name_char_allocator = shared_memory_container_allocator<char>;
using name_string =
  boost::interprocess::basic_string<char, std::char_traits<char>, name_char_allocator>;
using name_index_allocator =
  shared_memory_container_allocator<std::pair<name_string const, shared_segment_manager::handle_t>>;
using name_index = boost::interprocess::
  map<name_string, shared_segment_manager::handle_t, std::less<>, name_index_allocator>;

}  // namespace

shared_segment_manager::name_index_lock shared_segment_manager::lock_name_index()
{
  return rethrow_as_runtime_error("lock the shared memory name map",
                                  [&] { return name_index_lock(_name_index_latch); });
}

shared_segment_manager::arena_manager* shared_segment_manager::arena_holding(char const* name)
{
  if (_name_index_handle == invalid_handle) { return nullptr; }

  auto* index      = static_cast<name_index*>(get_address_from_handle(_name_index_handle));
  auto const entry = index->find(name);
  return entry == index->end()
           ? nullptr
           : static_cast<arena_manager*>(get_address_from_handle(entry->second));
}

shared_segment_manager::handle_t& shared_segment_manager::record_name(char const* name)
{
  auto const insert = [&](name_index& index) -> handle_t& {
    return index.emplace(name_string(name, name_char_allocator(this)), invalid_handle)
      .first->second;
  };
  if (_name_index_handle != invalid_handle) {
    return insert(*static_cast<name_index*>(get_address_from_handle(_name_index_handle)));
  }

  // A failed insert leaves an existing map unchanged, so only a map created here needs releasing.
  auto* index = construct<name_index>(boost::interprocess::anonymous_instance)(
    std::less<>(), name_index_allocator(this));
  try {
    auto& holder       = insert(*index);
    _name_index_handle = get_handle_from_address(index);
    return holder;
  } catch (...) {
    destroy_ptr(index);
    throw;
  }
}

void shared_segment_manager::forget_name(char const* name)
{
  if (_name_index_handle == invalid_handle) { return; }

  auto* index      = static_cast<name_index*>(get_address_from_handle(_name_index_handle));
  auto const entry = index->find(name);
  if (entry == index->end()) { return; }
  index->erase(entry);

  // An emptied map is released, so `all_memory_deallocated` holds once every object is destroyed.
  if (!index->empty()) { return; }
  destroy_ptr(index);
  _name_index_handle = invalid_handle;
}

shared_segment_manager::size_type shared_segment_manager::get_free_memory() const noexcept
{
  auto const arenas = std::views::iota(std::size_t{0}, arena_count());
  return std::transform_reduce(
    arenas.begin(), arenas.end(), size_type{0}, std::plus<>(), [this](std::size_t arena) {
      return arena_at(arena)->get_free_memory();
    });
}

bool shared_segment_manager::all_memory_deallocated()
{
  return std::ranges::all_of(
    std::views::iota(std::size_t{0}, arena_count()), [this](std::size_t arena) {
      return rethrow_as_runtime_error("check whether a shared memory arena is empty",
                                      [&] { return arena_at(arena)->all_memory_deallocated(); });
    });
}

shared_segment_manager::size_type shared_segment_manager::largest_arena_bytes() const noexcept
{
  // Every segment has at least one arena, which creation and `validate` both check.
  return std::ranges::max(_header.arena_records(), {}, &segment_header::arena_record::bytes).bytes;
}

shared_segment::shared_segment(create_only_t,
                               std::string const& name,
                               size_type bytes,
                               segment_layout_policy_type policy,
                               boost::interprocess::permissions const& perm)
  : _policy(policy)
{
  auto const layout     = plan_segment_layout(policy, bytes);
  auto const page_bytes = page_kind{layout.pages}.size();
  GQE_EXPECTS(!layout.arenas.empty(), "Segment layout describes no arenas");
  check_fallback_order_by_node(layout);

  auto const data_offset = segment_manager::header_bytes(
    page_bytes, layout.arenas.size(), layout.fallback_order_by_node.size());
  auto const offsets     = arena_offsets(layout, data_offset, page_bytes);
  auto const total_bytes = offsets.back() + layout.arenas.back().bytes;

  auto object =
    rethrow_as_runtime_error(std::format("create shared memory object '{}'", name), [&] {
      return boost::interprocess::shared_memory_object(
        boost::interprocess::create_only, name.c_str(), boost::interprocess::read_write, perm);
    });
  auto const descriptor = object.get_mapping_handle().handle;

  try {
    // `ftruncate` only sets the object's size. Reserving its blocks here, with `posix_fallocate`,
    // would instantiate the pages before they can be bound to a node and leave nothing for `mbind`
    // to place.
    if (eintr_aware_ftruncate(descriptor, total_bytes) != 0) {
      auto const error = errno;
      throw std::system_error(
        error,
        std::generic_category(),
        std::format("Failed to size shared memory object '{}' to {} bytes", name, total_bytes));
    }

    _region = rethrow_as_runtime_error(std::format("map shared memory object '{}'", name), [&] {
      return boost::interprocess::mapped_region(
        object, boost::interprocess::read_write, 0, total_bytes);
    });

    auto* const base = static_cast<std::byte*>(_region.get_address());
    build_arenas(base, layout, offsets, std::span{base, total_bytes}, page_bytes);

    _manager = segment_manager::initialize(
      _region.get_address(), total_bytes, data_offset, policy, layout, offsets);
  } catch (...) {
    // The object is shrunk to one byte, which a process waiting to open it reads as a failed
    // creation instead of waiting out its timeout.
    std::ignore = eintr_aware_ftruncate(descriptor, failed_creation_bytes);
    throw;
  }
}

shared_segment::shared_segment(open_only_t,
                               std::string const& name,
                               registration_policy registration)
{
  auto const object =
    rethrow_as_runtime_error(std::format("open shared memory object '{}'", name), [&] {
      return boost::interprocess::shared_memory_object(
        boost::interprocess::open_only, name.c_str(), boost::interprocess::read_write);
    });

  // Wait for a creator that has not sized the object yet, then for it to finish writing the
  // segment. A failed creation shrinks the object to one byte, which ends either wait at once.
  size_type mapped_bytes = 0;
  wait_for_creator(name, [&] {
    mapped_bytes = object_bytes(object, name);
    return mapped_bytes != 0;
  });
  if (mapped_bytes == failed_creation_bytes) { throw creation_failed(name); }

  _region = rethrow_as_runtime_error(std::format("map shared memory object '{}'", name), [&] {
    return boost::interprocess::mapped_region(
      object, boost::interprocess::read_write, 0, mapped_bytes);
  });

  wait_for_creator(name, [&] {
    if (object_bytes(object, name) == failed_creation_bytes) { throw creation_failed(name); }
    return segment_manager::is_initialized(_region.get_address());
  });

  _manager = segment_manager::attach(_region.get_address(), _region.get_size());
  _policy  = _manager->policy();

  populate(std::span{static_cast<std::byte*>(get_address()), get_size()},
           page_kind{page_kind::system_default}.size());
  if (registration == registration_policy::cuda) {
    GQE_CUDA_TRY(cudaHostRegister(get_address(), get_size(), cudaHostRegisterDefault));
    _is_cuda_registered = true;
  }
}

shared_segment::~shared_segment() noexcept
{
  if (_is_cuda_registered) {
    auto const status = cudaHostUnregister(get_address());
    if (status != cudaSuccess) {
      GQE_LOG_ERROR("cudaHostUnregister failed with: {}", cudaGetErrorString(status));
    }
  }
}

}  // namespace gqe
