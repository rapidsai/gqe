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

#include <gqe/utility/error.hpp>

#include <cuda.h>

#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <cupti_result.h>
#include <cupti_target.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define GQE_CUPTI_TRY(api_function_call)                                                          \
  do {                                                                                            \
    CUptiResult status = api_function_call;                                                       \
    if (status != CUPTI_SUCCESS) {                                                                \
      const char* error_string;                                                                   \
      cuptiGetResultString(status, &error_string);                                                \
                                                                                                  \
      auto error_message = std::string("CUPTI error: ") + __FILE__ + ":" +                        \
                           std::to_string(__LINE__) + ": Function " + #api_function_call +        \
                           " failed with error(" + std::to_string(status) + "): " + error_string; \
                                                                                                  \
      throw gqe::cuda_error{error_message};                                                       \
    }                                                                                             \
  } while (0)

// Redefine CUPTI_API_CALL to throw a GQE exception instead of exiting the program on error.
//
// Used by `range_profiling.h`. Redefinition avoids heavy modifications of this file, to ease
// upgrading to new CUPTI versions.
#define CUPTI_API_CALL(api_function_call) GQE_CUPTI_TRY(api_function_call)
#include <cupti/range_profiling.h>

namespace gqe {
namespace utility {

namespace detail {

/**
 * @brief User range profiler implementation.
 *
 * See `user_range_profiler` for API documentation.
 */
class user_range_profiler_impl {
 public:
  user_range_profiler_impl(user_range_profiler::configuration config)
    : _is_setup(false), _is_running(false), _has_result(false), _configuration(std::move(config))
  {
    // Convert vector contents from `std::string` to `char const*`.
    _char_metrics.reserve(_configuration.metrics.size());
    for (auto const& metric : _configuration.metrics) {
      _char_metrics.push_back(metric.c_str());
    }
  }

  ~user_range_profiler_impl() noexcept(false)
  {
    if (_is_running) { stop(); }

    if (_is_setup) { teardown(); }
  }

  void setup()
  {
    CUcontext cuda_context;
    CUresult cuda_code = cuCtxGetCurrent(&cuda_context);

    CUdevice cuda_device;
    if (cuda_code == CUDA_SUCCESS) {
      cuda_code = cuDeviceGet(&cuda_device, _configuration.device_id.value());
    }

    if (cuda_code != CUDA_SUCCESS) {
      const char* error;
      cuGetErrorString(cuda_code, &error);

      auto error_message =
        std::string("Failed to initialize CUPTI user range profiler due to CUDA error: ") + error;
      throw gqe::cuda_error{error_message};
    }

    _profiler_host = std::make_unique<CuptiProfilerHost>();

    RangeProfilerConfig profiler_target_config = {
      /* maxNumOfRanges = */ 1, /* numOfNestingLevel = */ 1, /* minNestingLevel = */ 1};
    _profiler_target = std::make_unique<RangeProfilerTarget>(cuda_context, profiler_target_config);

    std::string chip_name;
    GQE_CUPTI_TRY(RangeProfilerTarget::GetChipName(cuda_device, chip_name));

    if (!do_counter_availability_image_workaround()) {
      GQE_CUPTI_TRY(RangeProfilerTarget::GetCounterAvailabilityImage(cuda_context,
                                                                     _counter_availability_image));
    }

    _profiler_host->SetUp(chip_name, _counter_availability_image);

    std::vector<uint8_t> config_image;
    size_t num_passes = 0;
    GQE_CUPTI_TRY(_profiler_host->CreateConfigImage(_char_metrics, config_image, num_passes));

    // Ensure that only a single pass is needed.
    if (num_passes > 1) {
      throw std::invalid_argument("The configured profiling metrics require " +
                                  std::to_string(num_passes) +
                                  " passes. Reduce or change your metrics to use only one pass.");
    }

    GQE_CUPTI_TRY(_profiler_target->EnableRangeProfiler());

    GQE_CUPTI_TRY(_profiler_target->CreateCounterDataImage(_char_metrics, _counter_data_image));

    GQE_CUPTI_TRY(_profiler_target->SetConfig(
      CUPTI_UserRange, CUPTI_UserReplay, config_image, _counter_data_image));

    _is_setup = true;
  }

  void teardown()
  {
    if (!_is_setup) { throw std::logic_error("Teardown called on profiler without being setup."); }

    if (_is_running) {
      throw std::logic_error(
        "Teardown called on profiler that is still running. Profiler should be stopped first.");
    }

    _is_setup = false;

    // Disable Range profiler
    GQE_CUPTI_TRY(_profiler_target->DisableRangeProfiler());
    _profiler_host->TearDown();

    _profiler_target.reset();
    _profiler_host.reset();
  }

  void start()
  {
    if (!_is_setup) { throw std::logic_error("Start called on profiler without being setup."); }

    if (_is_running) {
      throw std::logic_error("Start called on profiler that is already running.");
    }

    // Start the profiler.
    GQE_CUPTI_TRY(_profiler_target->StartRangeProfiler());

    // Push a range.
    GQE_CUPTI_TRY(_profiler_target->PushRange("gqe_profiling_run"));

    _is_running = true;
  }

  void stop()
  {
    if (!_is_running) { throw std::logic_error("Stop called on profiler without being started."); }

    _is_running = false;

    // Pop the range.
    GQE_CUPTI_TRY(_profiler_target->PopRange());

    // Stop the profiler.
    GQE_CUPTI_TRY(_profiler_target->StopRangeProfiler());

    _has_result = true;
  }

  [[nodiscard]] user_range_profiler::profile decode_result()
  {
    if (!_has_result) {
      throw std::logic_error("Decode result called on profiler that doesn't have ready result.");
    }

    _has_result = false;

    // Ensure that profiler is done.
    if (!_profiler_target->IsAllPassSubmitted()) {
      throw std::runtime_error("Failed to submit all profiler passes.");
    }

    // Decode the profile returned by the hardware.
    GQE_CUPTI_TRY(_profiler_target->DecodeCounterData());

    // Ensure that one range was profiled.
    size_t num_ranges = 0;
    GQE_CUPTI_TRY(_profiler_host->GetNumOfRanges(_counter_data_image, num_ranges));
    if (num_ranges != 1) {
      throw std::runtime_error(
        "Failed to profile the correct amount of ranges. Expected 1 range, but got " +
        std::to_string(num_ranges) + ".");
    }

    // Convert the binary profile to C++ types.
    GQE_CUPTI_TRY(_profiler_host->EvaluateCounterData(
      /* rangeIndex = */ 0, _char_metrics, _counter_data_image));

    // Retrieve the profile.
    ProfilerRange range = _profiler_host->GetProfilerRange()[0];
    user_range_profiler::profile profile;
    profile.metric_values = std::move(range.metricValues);

    return profile;
  }

 private:
  /**
   * @brief Check if workaround for NVLink and C2C device counters is required.
   *
   * FIXME: CUPTI fixed this issue upstream in v13.0, according to Slack. Can remove this workaround
   * when GQE upgrades.
   *
   * The `counter_availability_image` vector must be empty when profiling these device-level metrics
   * (i.e., non-SM metrics). However, other device-level metrics such as `pcie__read_bytes.sum`
   * don't need this workaround.
   *
   * References:
   * https://forums.developer.nvidia.com/t/cupti-12-8-profiling-of-nvlink-metrics-using-profiler-host-api-and-range-profiler-api/325173/2
   * https://nvidia.slack.com/archives/C02DNK3FUF9/p1759205846281239?thread_ts=1758925217.137659&cid=C02DNK3FUF9
   */
  bool do_counter_availability_image_workaround()
  {
    std::regex needle("^ctc__.*");
    bool do_workaround = false;

    for (auto it = _configuration.metrics.begin();
         do_workaround == false && it != _configuration.metrics.end();
         ++it) {
      do_workaround = std::regex_match(*it, needle);
    }

    return do_workaround;
  }

  bool _is_setup;
  bool _is_running;
  bool _has_result;

  user_range_profiler::configuration _configuration;
  std::vector<char const*> _char_metrics;

  std::unique_ptr<CuptiProfilerHost> _profiler_host;
  std::unique_ptr<RangeProfilerTarget> _profiler_target;
  std::vector<uint8_t> _counter_availability_image;
  std::vector<uint8_t> _counter_data_image;
};

}  // namespace detail

user_range_profiler::user_range_profiler(configuration config)
  : _impl(std::make_unique<detail::user_range_profiler_impl>(std::move(config)))
{
  _impl->setup();
}

user_range_profiler::~user_range_profiler() noexcept(false)
{
  // Rely on `_impl` destructor to stop and teardown in correct order, as it tracks the stop and
  // setup states.
}

user_range_profiler::user_range_profiler(user_range_profiler&&) = default;

user_range_profiler& user_range_profiler::operator=(user_range_profiler&&) = default;

void user_range_profiler::start() { _impl->start(); }

user_range_profiler::profile user_range_profiler::stop()
{
  _impl->stop();
  return _impl->decode_result();
}

}  // namespace utility
}  // namespace gqe
