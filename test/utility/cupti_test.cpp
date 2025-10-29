/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/utility/cupti.hpp>

#include "../utility.hpp"

#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class UserRangeProfilerTest : public ::testing::Test {
 public:
  UserRangeProfilerTest() : stream(rmm::cuda_stream_default) {}

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

  static constexpr auto metric = "sm__inst_executed.sum";

  rmm::mr::cuda_memory_resource mr;
  rmm::cuda_stream_view stream;
  std::unique_ptr<cudf::column> data;
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
