/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cstdint>
#include <string>

namespace gqe {

/**
 * @brief Parameters indicating which optimizations are enabled and their settings.
 *
 * Implementation note: Add a comment for each parameter that indicates the task subtype it affects.
 * Each paramter must have a default setting.
 */
struct optimization_parameters {
  /**
   * @brief Build a new instance with default parameters and configured parameters.
   *
   * @note Currently collects environment variables to set the optimization parameters. In future,
   * this will be changed to load from a file.
   *
   * @param[in] only_defaults Disables loading of parameters from files or environment variables.
   * Used for testing.
   */
  explicit optimization_parameters(bool only_defaults = false);

  std::size_t max_num_workers = 1;  ///< Maximum number of worker threads per stage.
  int32_t max_num_partitions =
    8;  ///< The maximum number of read tasks that can be generated for a single table.
  std::string log_level = "info";  ///< Enable log messages for this level or higher.
  bool join_use_hash_map_cache =
    false;  ///< Allow multiple join tasks to reuse the same hash map. Enabling this option may
            ///< increase device-memory usage in some circumstances. If `max_num_workers` is set to
            ///< more than 1, this option is disabled.
  bool read_zero_copy_enable =
    true;  ///< Enable zero-copy reads for in-memory table. When disabled, read tasks copy input
           ///< data to a temporary output buffer.
};

}  // namespace gqe
