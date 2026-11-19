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

#include <gqe/storage/segment_layout.hpp>

#include <gqe/storage/shared_segment.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/linux.hpp>

#include <rmm/aligned.hpp>

#include <numa.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gqe {

namespace {

// ----- Policy planners -----

/**
 * @brief Return the size of each of @p arena_count arenas that fill @p segment_bytes after its
 * header.
 *
 * Every arena but the last is the same whole number of pages, because each arena starts where the
 * one before it ends and `mbind` places a range only from a page-aligned address. The last takes
 * the remainder, so the header and arenas add up to exactly @p segment_bytes.
 *
 * @pre @p arena_count is non-zero. Checked by `GQE_EXPECTS`, which throws `std::logic_error`.
 *
 * @throw std::runtime_error if @p segment_bytes cannot hold the header and a page for every arena.
 */
[[nodiscard]] std::vector<std::size_t> equal_arena_bytes(std::size_t segment_bytes,
                                                         page_kind::type pages,
                                                         std::size_t arena_count,
                                                         std::size_t fallback_row_count)
{
  GQE_EXPECTS(arena_count > 0, "Segment layout needs at least one arena");
  auto const page_bytes = page_kind{pages}.size();
  auto const data_offset =
    shared_segment_manager::header_bytes(page_bytes, arena_count, fallback_row_count);
  if (segment_bytes <= data_offset) {
    throw std::runtime_error("Shared memory segment is too small to hold its manager");
  }
  auto const data_bytes  = segment_bytes - data_offset;
  auto const arena_bytes = rmm::align_down(data_bytes / arena_count, page_bytes);
  if (arena_bytes == 0) {
    throw std::runtime_error("Shared memory segment is too small to give every arena a page");
  }
  std::vector<std::size_t> sizes(arena_count, arena_bytes);
  sizes.back() = data_bytes - (arena_count - 1) * arena_bytes;
  return sizes;
}

/**
 * @brief Return the @ref segment_layout_policy_type::single_arena_unplaced arrangement.
 *
 * The one arena takes everything after the header, so the object is exactly @p segment_bytes.
 *
 * @throw std::runtime_error if @p segment_bytes cannot hold the header and a page.
 */
segment_layout plan_single_arena_unplaced(std::size_t segment_bytes)
{
  // No topology query, because one unbound arena is the same arrangement on every machine.
  segment_layout layout{
    .arenas = {}, .fallback_order_by_node = {{0}}, .pages = page_kind::system_default};
  layout.arenas.push_back(
    arena_spec{.nodes = cpu_set{},
               .mode  = arena_placement::none,
               .bytes = equal_arena_bytes(segment_bytes, layout.pages, 1, 1).front()});
  return layout;
}

/**
 * @brief Return one row per node id, listing every arena in increasing NUMA distance from it.
 *
 * The requesting node comes first. Remote ties break on arena index, so the same machine
 * always yields the same rows.
 *
 * Complexity: `O(R · A log A)` calls to the distance function, where `R` is @p row_count and `A` is
 * the number of arenas. The sort falls back to `O(R · A log² A)` if it cannot allocate its buffer.
 *
 * @param[in] arena_nodes NUMA node backing each arena, in arena order.
 * @param[in] row_count Number of node ids an allocation can come from.
 */
[[nodiscard]] std::vector<std::vector<int>> nearest_first_rows(std::span<int const> arena_nodes,
                                                               std::size_t row_count,
                                                               detail::node_distance distance)
{
  std::vector<std::vector<int>> rows(row_count);
  std::ranges::transform(
    std::views::iota(0, static_cast<int>(row_count)), rows.begin(), [&](int node) {
      std::vector<int> row(arena_nodes.size());
      std::iota(row.begin(), row.end(), 0);
      std::ranges::stable_sort(row, std::ranges::less{}, [&](int arena) {
        auto const target = arena_nodes[static_cast<std::size_t>(arena)];
        return std::pair{target != node, distance(node, target)};
      });
      return row;
    });
  return rows;
}

/**
 * @brief Return the @ref segment_layout_policy_type::per_node_arenas_bound_overflow arrangement.
 *
 * @throw std::runtime_error if no permitted NUMA node has CPUs, or if @p segment_bytes cannot hold
 * the header and a page for every arena.
 * @throw std::system_error if NUMA memory policy discovery fails.
 */
[[nodiscard]] segment_layout plan_per_node_arenas_bound_overflow(std::size_t segment_bytes)
{
  auto const max_numa_node = ::numa_max_node();
  auto const usable_nodes  = utility::cpu_bearing_numa_nodes();

  std::vector<int> arena_nodes;
  std::ranges::copy_if(std::views::iota(0, max_numa_node + 1),
                       std::back_inserter(arena_nodes),
                       [&](int node) { return usable_nodes.contains(node); });
  return detail::plan_bound_arenas(
    segment_bytes, arena_nodes, static_cast<std::size_t>(max_numa_node) + 1, ::numa_distance);
}

// ----- Policy table -----

/**
 * @brief A row of `policies`, describing one segment layout policy.
 */
struct policy_entry {
  /// Policy this row describes.
  segment_layout_policy_type type;

  /// Name the policy goes by in logs.
  std::string_view name;

  /// Function that plans the policy's arrangement for a segment of the given size.
  segment_layout (*plan)(std::size_t segment_bytes);
};

/// Builds a policy's row from its enumerator, so its name and planner cannot drift from it.
#define GQE_POLICY_ENTRY(policy) \
  policy_entry { segment_layout_policy_type::policy, #policy, &plan_##policy }

/// Every policy, one row each. Row `i` is the policy whose value is `i`.
constexpr std::array policies{GQE_POLICY_ENTRY(single_arena_unplaced),
                              GQE_POLICY_ENTRY(per_node_arenas_bound_overflow)};

#undef GQE_POLICY_ENTRY

/**
 * @brief Return whether every policy has exactly one row, at the index of its value.
 */
constexpr bool lists_every_policy()
{
  return policies.size() == static_cast<std::size_t>(segment_layout_policy_type::num_policies) &&
         std::ranges::all_of(std::views::iota(std::size_t{0}, policies.size()),
                             [](std::size_t index) {
                               return static_cast<std::size_t>(policies[index].type) == index;
                             });
}

static_assert(lists_every_policy(),
              "Every segment layout policy needs one row in policies, at the index of its value");

/**
 * @brief Return the row for @p policy.
 *
 * @param[in] policy Policy to look up.
 * @return The row, or null for a value that names no policy.
 */
policy_entry const* find_policy(segment_layout_policy_type policy) noexcept
{
  auto const index = static_cast<std::size_t>(policy);
  return index < policies.size() ? &policies[index] : nullptr;
}

}  // namespace

segment_layout detail::plan_bound_arenas(std::size_t segment_bytes,
                                         std::span<int const> arena_nodes,
                                         std::size_t row_count,
                                         detail::node_distance distance)
{
  if (arena_nodes.empty()) {
    throw std::runtime_error("No NUMA node is available to hold shared memory");
  }

  segment_layout layout{
    .arenas                 = {},
    .fallback_order_by_node = nearest_first_rows(arena_nodes, row_count, distance),
    .pages                  = page_kind::system_default};

  auto const arena_sizes =
    equal_arena_bytes(segment_bytes, layout.pages, arena_nodes.size(), row_count);
  layout.arenas.reserve(arena_nodes.size());
  std::ranges::transform(
    arena_nodes, arena_sizes, std::back_inserter(layout.arenas), [](int node, std::size_t bytes) {
      return arena_spec{.nodes = cpu_set{node}, .mode = arena_placement::bind, .bytes = bytes};
    });
  return layout;
}

std::string_view to_string(segment_layout_policy_type type) noexcept
{
  auto const* entry = find_policy(type);
  return entry != nullptr ? entry->name : std::string_view{"unknown"};
}

template <>
segment_layout_policy_type from_string<segment_layout_policy_type>(std::string_view s)
{
  auto const name = utility::to_lower(s);
  auto const match =
    std::ranges::find_if(policies, [&](policy_entry const& entry) { return entry.name == name; });
  if (match != policies.end()) { return match->type; }

  std::string valid_names;
  for (auto const& entry : policies) {
    if (!valid_names.empty()) { valid_names += ", "; }
    valid_names += entry.name;
  }
  throw std::invalid_argument(
    std::format("Unrecognized segment layout policy: {}. Valid policies: {}", s, valid_names));
}

segment_layout plan_segment_layout(segment_layout_policy_type policy, std::size_t segment_bytes)
{
  auto const* entry = find_policy(policy);
  if (entry == nullptr) {
    throw std::invalid_argument(std::format("Segment layout policy value {} names no policy",
                                            static_cast<std::uint32_t>(policy)));
  }
  return entry->plan(segment_bytes);
}

}  // namespace gqe
