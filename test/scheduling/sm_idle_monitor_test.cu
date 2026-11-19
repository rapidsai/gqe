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
 *
 * Smoke test for the CUPTI PM sampler integration + SmIdleMonitor.
 *
 * Verifies:
 *   1. SmIdleMonitor can construct + start CUPTI sampling without throwing
 *      (i.e. the cupti_pm_sampler callback wiring + the pm_sampler::setup/start
 *      both work).
 *   2. Sampling actually produces samples (samples_received > 0 after some
 *      GPU activity).
 *   3. SM utilization atomic gets populated and tracks idle / busy state.
 *
 * NOTE: this is a smoke test, not a perf test. It only checks that the
 * `get_current_sm_util()` atomic moves in the right direction across known
 * idle / busy windows.
 */

#include <gqe/scheduling/sm_idle_monitor.hpp>

#include "../utility.hpp"

#include <gqe/utility/error.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <thread>

namespace {

// A trivial GPU workload that keeps SM busy for a few hundred ms. Used to
// drive the SM utilization metric toward 1.
__global__ void busy_kernel(int* dummy, int iters)
{
  int x = 0;
  for (int i = 0; i < iters; ++i) {
    x ^= i * blockIdx.x + threadIdx.x;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) { *dummy = x; }
}

int get_sm_count()
{
  int device = 0;
  GQE_CUDA_TRY(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  GQE_CUDA_TRY(cudaGetDeviceProperties(&prop, device));
  return prop.multiProcessorCount;
}

}  // namespace

TEST(SmIdleMonitorTest, ConfigFromEnvironmentOverridesDefaults)
{
  gqe_test::scoped_environment_variable sample_interval("GQE_SM_MONITOR_SAMPLE_INTERVAL_NS",
                                                        "1000000");
  gqe_test::scoped_environment_variable trace_json_file("GQE_SM_MONITOR_TRACE_JSON_FILE",
                                                        "sm_monitor_trace.json");
  gqe_test::scoped_environment_variable trace_max_entries("GQE_SM_MONITOR_TRACE_MAX_ENTRIES",
                                                          "1234");

  auto cfg = gqe::scheduling::SmIdleMonitor::config_from_environment();

  EXPECT_EQ(cfg.sample_interval_ns, 1'000'000u);
  EXPECT_EQ(cfg.trace_json_file, "sm_monitor_trace.json");
  EXPECT_EQ(cfg.trace_max_entries, 1234);
}

TEST(SmIdleMonitorTest, ConstructsAndProducesSamples)
{
  gqe::scheduling::SmIdleMonitor::Config cfg{};
  cfg.device_id          = 0;
  cfg.sample_interval_ns = 1'000'000;  // 1 ms

  gqe::scheduling::SmIdleMonitor monitor(cfg);

  // Wait long enough that several samples should accrue. CUPTI sampling at
  // 1 ms × ≥ 100 ms = at least dozens of samples.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // The monitor should have received samples.
  uint64_t samples = monitor.get_samples_received();
  EXPECT_GT(samples, 0u) << "no CUPTI samples received in 100 ms";
}

TEST(SmIdleMonitorTest, UsesConfiguredSmoothingWindow)
{
  constexpr size_t window_samples = 4;

  gqe::scheduling::SmIdleMonitor::Config cfg{};
  cfg.device_id          = 0;
  cfg.sample_interval_ns = 1'000'000;
  cfg.window_samples     = window_samples;

  gqe::scheduling::SmIdleMonitor monitor(cfg);

  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (monitor.get_samples_received() < window_samples &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_GE(monitor.get_samples_received(), window_samples);
  EXPECT_EQ(monitor.get_window_size(), window_samples);

  double const smoothed_sm_util = monitor.get_smoothed_sm_util();
  EXPECT_TRUE(std::isfinite(smoothed_sm_util));
  EXPECT_GE(smoothed_sm_util, 0.0);
  EXPECT_LE(smoothed_sm_util, 1.0);
}

TEST(SmIdleMonitorTest, StopsTraceAtConfiguredEntryLimit)
{
  constexpr size_t trace_max_entries = 2;
  std::filesystem::path const trace_path =
    std::filesystem::temp_directory_path() /
    ("gqe_sm_monitor_trace_" +
     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");

  {
    gqe::scheduling::SmIdleMonitor::Config cfg{};
    cfg.device_id          = 0;
    cfg.sample_interval_ns = 1'000'000;
    cfg.trace_json_file    = trace_path.string();
    cfg.trace_max_entries  = trace_max_entries;

    gqe::scheduling::SmIdleMonitor monitor(cfg);

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (monitor.get_samples_received() <= trace_max_entries &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_GT(monitor.get_samples_received(), trace_max_entries);
  }

  std::ifstream trace_file(trace_path);
  size_t trace_entries = 0;
  std::string trace_entry;
  while (std::getline(trace_file, trace_entry)) {
    ++trace_entries;
  }
  EXPECT_EQ(trace_entries, trace_max_entries);

  trace_file.close();
  std::filesystem::remove(trace_path);
}

TEST(SmIdleMonitorTest, DetectsBusyVsIdle)
{
  gqe::scheduling::SmIdleMonitor::Config cfg{};
  cfg.device_id          = 0;
  cfg.sample_interval_ns = 1'000'000;

  gqe::scheduling::SmIdleMonitor monitor(cfg);

  // Initially the GPU is idle (no workload running). get_current_sm_util should
  // settle near 0 after a few samples. This assertion may fail if another
  // process is using the GPU concurrently.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_LT(monitor.get_current_sm_util(), 0.3)
    << "GPU should be idle, sm_util=" << monitor.get_current_sm_util();

  // Launch a long-running busy kernel with enough blocks to keep several
  // CTAs resident per SM. occupancy-style metrics like
  // sm__warps_active.avg.pct_of_peak_sustained_elapsed need enough blocks
  // to register a meaningful share of warp slots.
  int* d_dummy = nullptr;
  GQE_CUDA_TRY(cudaMalloc(&d_dummy, sizeof(int)));
  auto const blocks = get_sm_count() * 8;
  busy_kernel<<<blocks, 256>>>(d_dummy, 100'000'000);
  GQE_CUDA_TRY(cudaPeekAtLastError());
  // Do NOT synchronize. Poll the monitor while the kernel runs.

  // Wait long enough for sample to land.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  // Poll up to 500 ms looking for elevated sm_util while the kernel is in
  // flight. busy_kernel keeps all SMs occupied, so we expect sm_util > 0.5.
  bool busy_detected = false;
  double max_sm_util = 0.0;
  for (int poll_idx = 0; poll_idx < 50; ++poll_idx) {
    double const sm_util = monitor.get_current_sm_util();
    max_sm_util          = std::max(max_sm_util, sm_util);
    if (sm_util > 0.5) {
      busy_detected = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(busy_detected) << "monitor did not detect busy SM in 500ms, "
                             << "sm_util=" << monitor.get_current_sm_util()
                             << " max_sm_util=" << max_sm_util
                             << " samples=" << monitor.get_samples_received();

  GQE_CUDA_TRY(cudaDeviceSynchronize());
  GQE_CUDA_TRY(cudaFree(d_dummy));

  // After the kernel finishes, sm_util should fall back near 0.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_LT(monitor.get_current_sm_util(), 0.3)
    << "monitor did not return to idle, sm_util=" << monitor.get_current_sm_util();
}

TEST(SmIdleMonitorTest, DecisionCountersTrack)
{
  gqe::scheduling::SmIdleMonitor::Config cfg{};
  gqe::scheduling::SmIdleMonitor monitor(cfg);

  EXPECT_EQ(monitor.get_total_decisions(), 0u);
  EXPECT_EQ(monitor.get_offload_decisions(), 0u);

  monitor.record_decision(true);
  monitor.record_decision(false);
  monitor.record_decision(true);

  EXPECT_EQ(monitor.get_total_decisions(), 3u);
  EXPECT_EQ(monitor.get_offload_decisions(), 2u);
}
