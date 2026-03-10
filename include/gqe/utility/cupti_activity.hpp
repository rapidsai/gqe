/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <rmm/cuda_device.hpp>

#include <cupti_activity.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gqe {
namespace utility {

namespace detail {
class activity_profiler_impl;
}  // namespace detail

// Interval of an event.
struct event_interval {
  uint64_t start_time;
  uint64_t end_time;
};

// Kernel event.
struct kernel_event {
  event_interval interval;
  std::string name;
};

// Memcpy event.
struct memcpy_event {
  event_interval interval;
  // @TODO: Refer to the enum.
  uint8_t kind;  // CUpti_ActivityMemcpyKind
};

// NVTX marker event.
struct nvtx_event {
  event_interval interval;
  std::string name;
};

// Mem decompress event. Could only be observed on GB200 so far.
struct mem_decompress_event {
  event_interval interval;
};

// The activity records captured.
struct activity_records {
  std::vector<kernel_event> kernels;
  std::vector<memcpy_event> memcopies;
  std::vector<nvtx_event> markers;
  std::vector<mem_decompress_event> mem_decompress;
};

// The activity time breakdown.
struct time_breakdown {
  double in_memory_read_task_s;
  double compute_kernel_s;
  double io_kernel_s;
  double memcpy_s;
  double mem_decompress_s;
  double merged_io_activity_s;
};

/**
 * @brief CUPTI Activity API session.
 *
 * Captures GPU execution events (kernels, memcpy, NVTX markers) as
 * timestamped records within a user-defined measurement window.
 *
 * Only one activity_session may exist at a time (process-global callbacks).
 *
 * Usage:
 *   activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
 *                               CUPTI_ACTIVITY_KIND_MARKER});
 *   profiler.start();
 *   // ... GPU work ...
 *   auto records = profiler.stop();
 *   auto time_breakdown = profiler.get_time_breakdown(records);
 */
class activity_profiler {
 public:
  /**
   * @brief Construct and enable the session.
   *
   * Registers CUPTI buffer callbacks and enables the requested activity kinds.
   * Equivalent to setup() in user_range_profiler.
   */
  explicit activity_profiler(std::vector<CUpti_ActivityKind> kinds = {
                               CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                               CUPTI_ACTIVITY_KIND_MEMCPY,
                               CUPTI_ACTIVITY_KIND_MARKER,
                               CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS});

  /**
   * @brief Destruct and disable the session.
   *
   * Disables all activity kinds and unregisters callbacks.
   * Equivalent to teardown() in user_range_profiler.
   */
  ~activity_profiler() noexcept(false);

  activity_profiler(activity_profiler const&)            = delete;
  activity_profiler& operator=(activity_profiler const&) = delete;

  /**
   * @brief Begin a measurement window.
   *
   * Flushes and discards any records accumulated before this call,
   * giving a clean baseline. Equivalent to start() in user_range_profiler.
   */
  void start();

  /**
   * @brief End a measurement window and return collected records.
   *
   * Flushes all pending CUPTI buffers and returns every event captured
   * since start(). Equivalent to stop() in user_range_profiler.
   */
  [[nodiscard]] activity_records stop();

  /**
   * @brief Get the activity time breakdown during the measurement window.
   *
   * Captured activity intervals would be filtered and merged to avoid double counting.
   * (E.g for multiple workers, different in_memory_read_task nvtx markers would overlap, and we
   * only want to calculate the merged duration).
   * The time would then be summed for each type of activity.
   *
   * @param records The activity records captured since the last start().
   * @return The activity time breakdown.
   */
  [[nodiscard]] static time_breakdown get_time_breakdown(
    activity_records& records,
    const std::vector<std::string>& io_kernel_filter_list = {
      "adjust_offsets_kernel", "fused_concatenate", "decompression_kernel"});

 private:
  std::unique_ptr<detail::activity_profiler_impl> _impl;
};

}  // namespace utility
}  // namespace gqe
