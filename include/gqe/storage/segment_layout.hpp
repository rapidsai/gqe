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

#include <gqe/types.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gqe {

/**
 * @brief How the pages of an arena are placed on NUMA nodes.
 */
enum class arena_placement : std::uint32_t {
  none,       ///< No memory policy. Pages land on the node of the thread that first touches them.
  bind,       ///< `MPOL_BIND` over the arena's node set.
  interleave  ///< `MPOL_INTERLEAVE` over the arena's node set.
};

/**
 * @brief One arena of a shared memory segment.
 */
struct arena_spec {
  /// Nodes the arena's pages are placed on. Empty for `arena_placement::none`.
  cpu_set nodes;

  /// How the arena's pages are placed on `nodes`.
  arena_placement mode;

  /// Size of the arena in bytes. A whole number of the segment's pages, except for the last arena,
  /// which takes the rest of the mapping.
  std::size_t bytes;
};

/**
 * @brief How a shared memory mapping is divided into arenas.
 *
 * Describes the arrangement only. Byte offsets are derived from the arena sizes by whoever creates
 * the mapping.
 */
struct segment_layout {
  /// Arenas in mapping order.
  std::vector<arena_spec> arenas;

  /**
   * Arena choices for allocations requested from each NUMA node. Index the outer vector with the
   * calling thread's NUMA node id to obtain its fallback order. Each entry in that order is an
   * arena index. The first entry is the preferred arena; later entries are alternatives to try in
   * order.
   *
   * @code{.text}
   * Requesting NUMA node   Arena indices in fallback order
   * 0                     [0, 1]  Try arena 0, then arena 1
   * 1                     [1, 0]  Try arena 1, then arena 0
   * @endcode
   *
   * If an arena cannot satisfy a new allocation, the manager tries the next arena in the selected
   * fallback order. Requests from a node beyond the outer vector's size use the fallback order at
   * index 0. Arenas absent from the selected fallback order are never tried.
   */
  std::vector<std::vector<int>> fallback_order_by_node;

  /// Page kind applied to every arena.
  page_kind::type pages;
};

/**
 * @brief Segment layout policy, which decides how a shared memory mapping is divided into arenas.
 *
 * Chosen by the process that creates the segment and recorded in its header, so a process that
 * attaches reads the arrangement from the header rather than choosing a policy.
 */
enum class segment_layout_policy_type : std::uint32_t {
  /**
   * One arena spanning the mapping, bound to nothing.
   *
   * Placement is left to first touch. A single allocation is limited to the mapping after its
   * header.
   */
  single_arena_unplaced,

  /**
   * One arena per permitted CPU-bearing NUMA node, each bound to its node.
   *
   * An allocation is served from the requesting thread's own node where that node's arena can
   * supply it, and overflows to the next-nearest node where it cannot. The arenas are the same
   * size, and the last also takes the rest of the mapping. A single allocation is limited to one
   * arena.
   */
  per_node_arenas_bound_overflow,

  num_policies  ///< Number of policies. Names no policy.
};

/**
 * @brief Policy a node manager creates its segment with when none is named.
 */
inline constexpr segment_layout_policy_type default_shmem_policy =
  segment_layout_policy_type::single_arena_unplaced;

/**
 * @brief Return the name of a segment layout policy, or "unknown" for a value that names none.
 */
[[nodiscard]] std::string_view to_string(segment_layout_policy_type type) noexcept;

/**
 * @brief Return the segment layout policy with the given name, ignoring case.
 *
 * @throw std::invalid_argument listing the valid names if @p s names no policy.
 */
template <>
[[nodiscard]] segment_layout_policy_type from_string<segment_layout_policy_type>(
  std::string_view s);

/**
 * @brief Return the arrangement @p policy divides a mapping of @p segment_bytes into.
 *
 * @param[in] policy Policy whose arrangement to build.
 * @param[in] segment_bytes Requested size of the whole mapping, including its header.
 * @throw std::runtime_error if the machine cannot support it or @p segment_bytes cannot hold it.
 * @throw std::invalid_argument if @p policy names no policy.
 * @throw std::system_error if NUMA memory policy discovery fails for the bound policy.
 */
[[nodiscard]] segment_layout plan_segment_layout(segment_layout_policy_type policy,
                                                 std::size_t segment_bytes);

namespace detail {

/// Function returning the distance between two node IDs.
using node_distance = int (*)(int, int);

/**
 * @brief Plan bound arenas using an explicit topology, independently of host discovery.
 *
 * @param[in] segment_bytes Size of the complete mapping, including its header.
 * @param[in] arena_nodes Distinct usable node IDs in arena order.
 * @param[in] row_count Number of node IDs represented by fallback rows.
 * @param[in] distance Callable returning the distance between two node IDs.
 * @pre Every node ID is nonnegative and below both `row_count` and `cpu_set::max_count`.
 * The topology discovery caller guarantees these bounds. Not checked here.
 * @throw std::runtime_error if no nodes are supplied or the segment cannot hold the arenas.
 */
[[nodiscard]] segment_layout plan_bound_arenas(std::size_t segment_bytes,
                                               std::span<int const> arena_nodes,
                                               std::size_t row_count,
                                               node_distance distance);

}  // namespace detail

}  // namespace gqe
