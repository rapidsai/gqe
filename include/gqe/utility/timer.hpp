/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
