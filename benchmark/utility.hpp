/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/utility.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace gqe {
namespace benchmark {

// Return all parquet files in `path` including its subdirectories
inline std::vector<std::string> get_file_paths(std::string path)
{
  std::vector<std::string> parquet_files;
  for (auto const& entry : std::filesystem::recursive_directory_iterator(path)) {
    if (entry.is_regular_file() && entry.path().extension().string() == ".parquet")
      parquet_files.push_back(entry.path());
  }
  return parquet_files;
}

// Wrapper for logging the execution time
template <typename function_type, typename... argument_type>
inline void time_function(function_type func, argument_type&&... args)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  func(std::forward<argument_type>(args)...);
  auto stop_time = std::chrono::high_resolution_clock::now();
  auto duration  = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
  GQE_LOG_INFO("Query execution time: {} ms.", duration.count());
}

}  // namespace benchmark
}  // namespace gqe
