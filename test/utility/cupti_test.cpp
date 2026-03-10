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

#include "../utility.hpp"

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/cupti_activity.hpp>
#include <gqe/utility/cupti_range.hpp>

#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ── Shared base fixture ────────────────────────────────────────────────────────

/**
 * @brief Base fixture shared by UserRangeProfilerTest and ActivityProfilerTest.
 *
 * Provides a small GPU dataset, a stream, a memory resource, and two GPU
 * workload helpers so that individual fixtures do not repeat setup logic.
 */
class CuptiTestBase : public ::testing::Test {
 public:
  CuptiTestBase() : stream(cudf::get_default_stream()) {}

  void SetUp() override
  {
    size_t constexpr num_rows   = 100;
    int32_t constexpr min_value = 0;
    int32_t constexpr max_value = 1000;

    data = generate_fixed_width_column(num_rows, /* null_rate = */ 0.0, min_value, max_value);
  }

  /**
   * @brief A helper that launches a CUDA kernel.
   *
   * What the kernel does doesn't really matter, it just needs to run on the GPU so that tests can
   * profile it.
   */
  void run_cuda_fn()
  {
    auto agg_type  = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    auto data_type = cudf::data_type(cudf::type_to_id<int32_t>());
    auto result    = cudf::reduce(data->view(), *agg_type, data_type, stream, mr);

    stream.synchronize();
  }

  /**
   * @brief Performs a small host-to-device memcpy to produce observable memory
   * transfer activity for profiling.
   */
  void run_memcpy()
  {
    constexpr size_t num_bytes = 1024;
    std::vector<uint8_t> host_buf(num_bytes, 0);
    void* device_buf = nullptr;
    cudaMalloc(&device_buf, num_bytes);
    cudaMemcpy(device_buf, host_buf.data(), num_bytes, cudaMemcpyHostToDevice);
    cudaFree(device_buf);
  }

  rmm::mr::cuda_memory_resource mr;
  rmm::cuda_stream_view stream;
  std::unique_ptr<cudf::column> data;
};

// ── UserRangeProfilerTest ─────────────────────────────────────────────────────

class UserRangeProfilerTest : public CuptiTestBase {
 public:
  static constexpr auto metric = "sm__inst_executed.sum";
};

/**
 * @brief A simple use of the profiler.
 */
TEST_F(UserRangeProfilerTest, Simple)
{
  gqe::utility::user_range_profiler::configuration ur_profiler_config;
  ur_profiler_config.device_id = rmm::get_current_cuda_device();
  ur_profiler_config.metrics   = {metric};

  gqe::utility::user_range_profiler ur_profiler(ur_profiler_config);

  ur_profiler.start();

  run_cuda_fn();

  auto profile = ur_profiler.stop();

  EXPECT_TRUE(profile.metric_values.contains(metric));
  EXPECT_GT(profile.metric_values[metric], 0);
}

/**
 * @brief Use the profiler multiple times between setup and teardown.
 */
TEST_F(UserRangeProfilerTest, MultiUse)
{
  int32_t constexpr runs = 5;

  gqe::utility::user_range_profiler::configuration ur_profiler_config;
  ur_profiler_config.device_id = rmm::get_current_cuda_device();
  ur_profiler_config.metrics   = {metric};

  gqe::utility::user_range_profiler ur_profiler(ur_profiler_config);

  double first_profile = 0.0;

  for (int32_t run = 0; run < runs; ++run) {
    ur_profiler.start();

    run_cuda_fn();

    auto profile = ur_profiler.stop();

    EXPECT_TRUE(profile.metric_values.contains(metric));
    EXPECT_GT(profile.metric_values[metric], 0);

    // SM instructions executed should be nearly identical. When testing on L40S on 10/7/2025, the
    // profiled values were exactly identical over 5 runs. But we give it some tolerance in case
    // other GPU models have variance, so that the test isn't flakey.
    if (run == 0) {
      first_profile = profile.metric_values[metric];
    } else {
      EXPECT_NEAR(profile.metric_values[metric], first_profile, /* abs_error = */ 10.0);
    }
  }
}

/**
 * @brief Profile multiple metrics in a single run.
 */
TEST_F(UserRangeProfilerTest, MultiMetric)
{
  const std::vector<std::string> metrics = {"sm__inst_executed.sum", "smsp__warps_launched.sum"};

  gqe::utility::user_range_profiler::configuration ur_profiler_config;
  ur_profiler_config.device_id = rmm::get_current_cuda_device();
  ur_profiler_config.metrics   = metrics;

  gqe::utility::user_range_profiler ur_profiler(ur_profiler_config);

  ur_profiler.start();

  run_cuda_fn();

  auto profile = ur_profiler.stop();

  for (auto const& metric : metrics) {
    EXPECT_TRUE(profile.metric_values.contains(metric));
    EXPECT_GT(profile.metric_values[metric], 0);
  }
}

/**
 * @brief Recover from a thrown exception.
 *
 * In gqe-python, exceptions thrown by GQE due to, e.g., faulty parameters,
 * cause `CUPTI_ERROR_UNKNOWN` or `CUPTI_ERROR_INVALID_PARAMETER` in the next
 * iteration, which crashes the program.
 *
 * This test reproduces this scenario and tests that the profiler recovers.
 */
TEST_F(UserRangeProfilerTest, ExceptionRecovery)
{
  bool exception_was_caught = false;

  // Simulate a first benchmark run that fails with an exception.
  try {
    gqe::utility::user_range_profiler::configuration ur_profiler_config;
    ur_profiler_config.device_id = rmm::get_current_cuda_device();
    ur_profiler_config.metrics   = {metric};

    gqe::utility::user_range_profiler ur_profiler(ur_profiler_config);

    ur_profiler.start();

    throw std::runtime_error("An exception occurred.");

    auto profile = ur_profiler.stop();
  } catch (std::runtime_error const& e) {
    exception_was_caught = true;
  }

  EXPECT_TRUE(exception_was_caught);

  // Setup profiler again, simulating next benchmark run.
  gqe::utility::user_range_profiler::configuration ur_profiler_config;
  ur_profiler_config.metrics = {metric};

  gqe::utility::user_range_profiler ur_profiler(ur_profiler_config);

  ur_profiler.start();

  run_cuda_fn();

  auto profile = ur_profiler.stop();

  EXPECT_TRUE(profile.metric_values.contains(metric));
  EXPECT_GT(profile.metric_values[metric], 0);
}

// ── ActivityProfilerTest ───────────────────────────────────────────────────────

class ActivityProfilerTest : public CuptiTestBase {};

/**
 * @brief A kernel launched between start() and stop() produces at least one kernel record.
 */
TEST_F(ActivityProfilerTest, KernelCaptured)
{
  gqe::utility::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});

  profiler.start();
  run_cuda_fn();
  auto records = profiler.stop();

  EXPECT_FALSE(records.kernels.empty());
  for (auto const& k : records.kernels) {
    EXPECT_FALSE(k.name.empty());
    EXPECT_GT(k.interval.end_time, k.interval.start_time);
  }

  auto time_breakdown = gqe::utility::activity_profiler::get_time_breakdown(records);
  EXPECT_GT(time_breakdown.compute_kernel_s, 0);
  EXPECT_EQ(time_breakdown.io_kernel_s, 0);
  EXPECT_EQ(time_breakdown.memcpy_s, 0);
  EXPECT_EQ(time_breakdown.mem_decompress_s, 0);
  EXPECT_EQ(time_breakdown.merged_io_activity_s, 0);
}

/**
 * @brief Records only contain events from within the start()/stop() window.
 *
 * start() flushes and discards any pre-existing buffered records, so a kernel
 * run before start() must not appear in stop() return value.
 */
TEST_F(ActivityProfilerTest, WindowIsolation)
{
  gqe::utility::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});

  run_cuda_fn();  // before start() — flushed and discarded by start()

  profiler.start();
  auto records = profiler.stop();

  EXPECT_TRUE(records.kernels.empty());
  EXPECT_TRUE(records.memcopies.empty());
  EXPECT_TRUE(records.markers.empty());
  EXPECT_TRUE(records.mem_decompress.empty());
}

/**
 * @brief The profiler can be reused across multiple start()/stop() cycles.
 */
TEST_F(ActivityProfilerTest, MultiUse)
{
  constexpr int32_t runs = 3;

  gqe::utility::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});

  for (int32_t run = 0; run < runs; ++run) {
    profiler.start();
    run_cuda_fn();
    auto records = profiler.stop();

    EXPECT_FALSE(records.kernels.empty()) << "Failed on run " << run;
  }
}

/**
 * @brief An NVTX scoped range pushed inside the window appears as an nvtx_event
 * with the correct name and valid start/end timestamps.
 */
TEST_F(ActivityProfilerTest, NvtxMarkerCaptured)
{
  constexpr auto range_name = "test_nvtx_range";

  gqe::utility::activity_profiler profiler(
    {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MARKER});

  profiler.start();
  {
    gqe::utility::nvtx_scoped_range range(range_name);
    run_cuda_fn();
  }  // nvtxRangePop fires here
  auto records = profiler.stop();

  auto it = std::find_if(
    records.markers.begin(), records.markers.end(), [](gqe::utility::nvtx_event const& e) {
      return e.name.find(range_name) != std::string::npos;
    });

  EXPECT_NE(it, records.markers.end()) << "Expected nvtx_event named '" << range_name << "'";
  if (it != records.markers.end()) { EXPECT_GE(it->interval.end_time, it->interval.start_time); }
}

/**
 * @brief A host-to-device memcpy inside the window produces at least one memcpy record
 * with valid timestamps.
 */
TEST_F(ActivityProfilerTest, MemcpyCaptured)
{
  gqe::utility::activity_profiler profiler({CUPTI_ACTIVITY_KIND_MEMCPY});

  profiler.start();
  run_memcpy();
  auto records = profiler.stop();

  EXPECT_FALSE(records.memcopies.empty());
  for (auto const& m : records.memcopies) {
    EXPECT_GT(m.interval.end_time, m.interval.start_time);
  }

  auto time_breakdown = gqe::utility::activity_profiler::get_time_breakdown(records);
  EXPECT_EQ(time_breakdown.compute_kernel_s, 0);
  EXPECT_EQ(time_breakdown.io_kernel_s, 0);
  EXPECT_GT(time_breakdown.memcpy_s, 0);
  EXPECT_EQ(time_breakdown.mem_decompress_s, 0);
  EXPECT_GT(time_breakdown.merged_io_activity_s, 0);
}

/**
 * @brief An exception thrown inside a start()/stop() bracket does not corrupt
 * subsequent uses via a freshly-constructed profiler.
 *
 * The destructor calls void stop(), which flushes CUPTI and clears
 * g_active_profiler, so the next construction succeeds.
 */
TEST_F(ActivityProfilerTest, ExceptionRecovery)
{
  bool exception_was_caught = false;

  try {
    gqe::utility::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});
    profiler.start();
    throw std::runtime_error("Simulated exception.");
    auto records = profiler.stop();
  } catch (std::runtime_error const&) {
    exception_was_caught = true;
  }

  EXPECT_TRUE(exception_was_caught);

  // A new profiler constructed after the failed one must work correctly.
  gqe::utility::activity_profiler profiler({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});
  profiler.start();
  run_cuda_fn();
  auto records = profiler.stop();

  EXPECT_FALSE(records.kernels.empty());
}

/**
 * @brief Constructing a second activity_profiler while one already exists throws a logic error,
 * because the CUPTI Activity API uses process-global buffer callbacks.
 */
TEST_F(ActivityProfilerTest, SingleInstanceEnforced)
{
  gqe::utility::activity_profiler first({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL});

  EXPECT_THROW(
    { gqe::utility::activity_profiler second({CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL}); },
    std::logic_error);
}
