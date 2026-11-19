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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gqe {
namespace utility {

namespace impl {
class pm_sampler_impl;
}

/**
 * @brief CUPTI PM (Performance Monitoring) sampler
 *
 * Lifecycle: `setup()` -> `start()` -> `stop()` -> `teardown()`. The object is
 * reusable: a full cycle may be repeated on the same instance (each `setup()`
 * resets the per-session state and drops the previous session's samples).
 * `start()` must follow a `setup()`, and `stop()` must follow a `start()`.
 * Calling `setup()` again before `stop()` throws.
 *
 * `set_sample_callback()` (optional) must be installed before `start()`.
 * The destructor joins the decode thread and releases any still-held CUPTI
 * resources, so an explicit `teardown()` is not required for correctness.
 */
class pm_sampler {
 public:
  struct sample {
    size_t range_index;
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    std::unordered_map<std::string, double> metric_values;
  };

  struct configuration {
    int deviceIndex           = 0;
    int queryBaseMetrics      = 0;
    int queryMetricProperties = 0;
    std::string chipName      = "";
    uint64_t samplingInterval = 100000;             // (in nanoseconds) 100us
    size_t hardwareBufferSize = 512 * 1024 * 1024;  // (in bytes) 512MB
    uint64_t maxSamples       = 10000;
    // Owned by value so the strings outlive the sampler (incl. the decode
    // thread) regardless of how the caller built them. Passing `const char*`
    // here would dangle if the caller's strings were temporaries or mutated.
    std::vector<std::string> metrics = {
      "gr__cycles_active.avg",                            // Active Cycles
      "gr__cycles_elapsed.max",                           // Elapsed Cycles
      "gpu__time_duration.sum",                           // Duration
      "sm__inst_executed_realtime.avg.per_cycle_active",  // Inst Executed per Active Cycle
      "sm__cycles_active.avg"                             // SM Active Cycles
    };
  };

  pm_sampler(configuration const& args);
  ~pm_sampler();

  pm_sampler(pm_sampler&)  = delete;
  pm_sampler(pm_sampler&&) = delete;

  pm_sampler& operator=(pm_sampler&)  = delete;
  pm_sampler& operator=(pm_sampler&&) = delete;

  /// Callback invoked from the decode thread for each newly-decoded sample.
  /// Useful for live monitoring (e.g. SmIdleMonitor) where waiting for stop()
  /// is not acceptable. Must be lockfree / thread-safe.
  using sample_callback = std::function<void(sample const&)>;

  /// Install a per-sample callback. Must be called before start().
  /// Pass an empty function to disable.
  void set_sample_callback(sample_callback cb);

  void setup();
  void start();
  [[nodiscard]] std::vector<sample> stop();
  void teardown();

  [[nodiscard]] int query_metrics();

 private:
  std::unique_ptr<impl::pm_sampler_impl> _impl;
};

}  // namespace utility
}  // namespace gqe
