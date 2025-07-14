/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <deque>
#include <string>
#include <tuple>

namespace gqe::utility {

class bandwidth_timer {
 public:
  bandwidth_timer(std::string name);

  bandwidth_timer(bandwidth_timer&& other);
  bandwidth_timer& operator=(bandwidth_timer&& other);

  bandwidth_timer(const bandwidth_timer&)            = delete;
  bandwidth_timer& operator=(const bandwidth_timer&) = delete;

  void start();
  void end();
  void add(uint64_t bytes, uint64_t units = 0);
  std::string to_string() const;

 private:
  std::chrono::microseconds elapsed_time, total_elapsed_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  std::atomic_uint64_t input_bytes, total_input_bytes = 0, units = 0, total_units = 0;
  std::string name;
  std::deque<std::tuple<uint64_t, uint64_t, uint64_t>> data_history;  // bytes, units, elapsed_time
};

}  // namespace gqe::utility
