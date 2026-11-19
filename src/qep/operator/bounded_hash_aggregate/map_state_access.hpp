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

#include "map_layout.hpp"

#include <gqe/qep/shapes/keyed_accumulator.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cuda/stream>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <format>
#include <vector>

namespace gqe {
namespace qep {
namespace bounded_hash_aggregate::detail {

/**
 * @brief Host-side access to the map state shared by the build and scan tasks.
 */

/** @brief Extract the map handle from a state slot, or throw naming `who` as the caller. */
[[nodiscard]] inline keyed_accumulator_handle const& get_map_handle(shared_state const& slot,
                                                                    char const* who)
{
  GQE_EXPECTS(slot != nullptr, std::format("{}: null state slot", who));
  auto const* wrapper = std::get_if<state_kind::keyed_accumulator>(slot.get());
  GQE_EXPECTS(wrapper != nullptr, std::format("{}: the state is not a keyed_accumulator", who));
  return wrapper->state();
}

/** @brief Extract the map state from the build task's single-slot accumulator container. */
[[nodiscard]] inline keyed_accumulator_state& get_accumulator(state_container_view accumulator)
{
  GQE_EXPECTS(accumulator.size() == 1,
              "bounded_hash_aggregate_build_task: the accumulator must contain exactly the map "
              "state");
  return *get_map_handle(accumulator.front(), "bounded_hash_aggregate_build_task");
}

/** @brief Host copy of the control block, read after all inserts quiesced. */
[[nodiscard]] inline control_block read_control_block(keyed_accumulator_state const& map,
                                                      cuda::stream_ref stream)
{
  control_block control{};
  GQE_CUDA_TRY(cudaMemcpyAsync(
    &control, map.control(), sizeof(control_block), cudaMemcpyDefault, stream.get()));
  stream.sync();
  return control;
}

/** @brief Host copy of the full control region: the control block plus the occupancy histogram. */
struct control_region {
  control_block control;
  std::vector<int32_t> segment_counts;
};

/** @brief Read the control region in one transfer under one synchronize (the map is final). */
[[nodiscard]] inline control_region read_control_region(keyed_accumulator_state const& map,
                                                        cuda::stream_ref stream)
{
  auto const bytes = control_region_bytes(map.capacity());
  std::vector<char> host(bytes);
  GQE_CUDA_TRY(cudaMemcpyAsync(host.data(), map.control(), bytes, cudaMemcpyDefault, stream.get()));
  stream.sync();

  control_region region{};
  std::memcpy(&region.control, host.data(), sizeof(control_block));
  region.segment_counts.resize(histogram_segments(map.capacity()));
  std::memcpy(region.segment_counts.data(),
              host.data() + sizeof(control_block),
              region.segment_counts.size() * sizeof(int32_t));
  return region;
}

}  // namespace bounded_hash_aggregate::detail
}  // namespace qep
}  // namespace gqe
