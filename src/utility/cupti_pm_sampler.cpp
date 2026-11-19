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

#include <gqe/utility/cupti_pm_sampler.hpp>

#include <gqe/utility/cupti_common.hpp>
#include <gqe/utility/error.hpp>

#include <cuda.h>

// Redefine the vendored sample macros to throw GQE exceptions instead of
// exit()ing the process. Used by `pm_sampling.h`. Redefinition avoids heavy
// modifications of that file, to ease upgrading to new CUPTI versions.
#define CUPTI_API_CALL(api_function_call) GQE_CUPTI_TRY(api_function_call)
#define DRIVER_API_CALL(api_function_call)                                                \
  do {                                                                                    \
    CUresult _status = (api_function_call);                                               \
    if (_status != CUDA_SUCCESS) {                                                        \
      const char* _err_str = nullptr;                                                     \
      cuGetErrorString(_status, &_err_str);                                               \
      throw gqe::cu_error(std::string("CUDA driver error: ") + __FILE__ + ":" +           \
                          std::to_string(__LINE__) + ": Function " + #api_function_call + \
                          " failed with error(" + std::to_string(_status) +               \
                          "): " + (_err_str ? _err_str : "(unknown)"));                   \
    }                                                                                     \
  } while (0)

#include <cupti/pm_sampling.h>
#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace gqe {
namespace utility {

namespace impl {

void decode_counter_data(std::vector<uint8_t>& counterDataImage,
                         std::vector<const char*> metricsList,
                         std::atomic<bool>& startDecodeThread,
                         std::atomic<bool>& stopDecodeThread,
                         std::atomic<bool>& is_decode_thread_running,
                         CuptiPmSampling& cuptiPmSamplingTarget,
                         CuptiProfilerHost& pmSamplingHost,
                         std::atomic<CUptiResult>& result,
                         std::exception_ptr& decode_exception,
                         pm_sampler::sample_callback const& callback)
{
  // Wait for start() — but also honour stopDecodeThread so the dtor can
  // wake us up if it runs before start() was ever called (otherwise this
  // thread spins forever and the std::thread dtor calls std::terminate).
  while (!startDecodeThread && !stopDecodeThread) {
    std::this_thread::yield();
  }
  if (stopDecodeThread) { return; }

  is_decode_thread_running = true;

  // Track how many samples we have already forwarded to the callback so that
  // each new evaluation invokes the callback exactly once.
  size_t already_emitted = 0;

  // Any exception thrown inside the decode loop — a CUPTI failure from
  // EvaluateCounterData (CUPTI_API_CALL now throws instead of exit()ing) or a
  // throwing user callback — would otherwise escape this thread's top-level
  // function and call std::terminate. Capture it, stop decoding, and surface it
  // from stop() so the caller can handle the failure and clean up.
  try {
    while (!stopDecodeThread) {
      const char* errstr;
      result = cuptiPmSamplingTarget.DecodePmSamplingData(counterDataImage);
      if (result != CUPTI_SUCCESS) {
        cuptiGetResultString(result, &errstr);
        std::cerr << "DecodePmSamplingData failed with error " << errstr << std::endl;
        return;
      }

      CUpti_PmSampling_GetCounterDataInfo_Params counterDataInfo{
        CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
      counterDataInfo.pCounterDataImage    = counterDataImage.data();
      counterDataInfo.counterDataImageSize = counterDataImage.size();
      result                               = cuptiPmSamplingGetCounterDataInfo(&counterDataInfo);
      if (result != CUPTI_SUCCESS) {
        cuptiGetResultString(result, &errstr);
        std::cerr << "cuptiPmSamplingGetCounterDataInfo failed with error " << errstr << std::endl;
        return;
      }

      for (size_t sampleIndex = 0; sampleIndex < counterDataInfo.numCompletedSamples;
           ++sampleIndex) {
        pmSamplingHost.EvaluateCounterData(
          cuptiPmSamplingTarget.GetPmSamplerObject(), sampleIndex, metricsList, counterDataImage);
      }

      if (callback) {
        auto& range = pmSamplingHost.GetSamplerRange();
        for (size_t i = already_emitted; i < range.size(); ++i) {
          pm_sampler::sample s;
          s.range_index     = range[i].rangeIndex;
          s.start_timestamp = range[i].startTimestamp;
          s.end_timestamp   = range[i].endTimestamp;
          s.metric_values   = range[i].metricValues;
          callback(s);
        }
        already_emitted = range.size();
      }

      result = cuptiPmSamplingTarget.ResetCounterDataImage(counterDataImage);
      if (result != CUPTI_SUCCESS) {
        cuptiGetResultString(result, &errstr);
        std::cerr << "ResetCounterDataImage failed with error " << errstr << std::endl;
        return;
      }
    }
  } catch (...) {
    decode_exception = std::current_exception();
    stopDecodeThread = true;
    return;
  }
}

class pm_sampler_impl {
 public:
  pm_sampler_impl(pm_sampler::configuration const& args)
    : args(args),
      is_decode_thread_running(false),
      startDecodeThread(false),
      stopDecodeThread(false),
      decode_thread_result(CUPTI_SUCCESS)
  {
    CuptiPmSampling::GetChipName(args.deviceIndex, chipName);
    CuptiPmSampling::GetCounterAvailabilityImage(args.deviceIndex, counterAvailibilityImage);
  }

  // Destructors are implicitly noexcept — throwing here would call
  // std::terminate. Pass may_throw=false so the dtor swallows + logs CUPTI
  // failures instead. stop()/teardown() use the throwing path for normal flow.
  // release_resources() disables sampling and tears down CUPTI even if the
  // caller never reached teardown() (e.g. setup() threw partway).
  ~pm_sampler_impl()
  {
    joinDecodeThread(/*may_throw=*/false);
    release_resources(/*may_throw=*/false);
  }

  void joinDecodeThread(bool may_throw = true)
  {
    // Idempotent — second call is a no-op.
    if (!decode_thread.joinable()) { return; }

    // Both flags get set: stopDecodeThread tells a *running* thread to
    // exit its outer loop, and startDecodeThread releases a thread that
    // is still spinning in the pre-start wait (start() was never called).
    stopDecodeThread  = true;
    startDecodeThread = true;
    decode_thread.join();
    is_decode_thread_running = false;

    // Surface a user-callback exception first: it is the caller's own error and
    // more actionable than a CUPTI status. Swallow on the dtor path (may_throw
    // == false) since destructors must not throw.
    if (decode_exception) {
      auto ex          = decode_exception;
      decode_exception = nullptr;
      if (may_throw) {
        std::rethrow_exception(ex);
      } else {
        std::cerr << "pm_sampler: sample callback threw during decode (suppressed in dtor)\n";
      }
    }

    auto result = decode_thread_result.load();
    if (result != CUPTI_SUCCESS) {
      const char* errstr = nullptr;
      cuptiGetResultString(result, &errstr);
      std::string const msg = std::string("DecodeCounterData Thread failed with error ") +
                              (errstr ? errstr : "(unknown)");
      if (may_throw) {
        throw std::runtime_error(msg);
      } else {
        std::cerr << msg << '\n';
      }
    }
  };

  void setup()
  {
    // Reuse guard: a prior session must be stop()'d (which joins the thread)
    // before setup() runs again, otherwise the std::thread move-assign below
    // would std::terminate on a still-joinable thread.
    if (decode_thread.joinable()) {
      throw std::logic_error(
        "pm_sampler::setup() called while a previous session "
        "is still active; call stop() first");
    }

    // Reset the per-session state so a reused sampler starts clean (the flags
    // are left set after the previous stop()).
    startDecodeThread        = false;
    stopDecodeThread         = false;
    is_decode_thread_running = false;
    decode_thread_result     = CUPTI_SUCCESS;
    decode_exception         = nullptr;
    // The host accumulates samples across sessions (SetUp does not clear them);
    // drop the previous session's samples so they are not re-emitted/returned.
    pmSamplingHost.GetSamplerRange().clear();

    CuptiPmSampling::GetChipName(args.deviceIndex, chipName);

    pmSamplingHost.SetUp(chipName, counterAvailibilityImage);
    host_setup_ = true;

    // CUPTI wants `const char*`; build a non-owning view over the strings
    // owned by `args.metrics` (which outlives the decode thread).
    char_metrics.clear();
    char_metrics.reserve(args.metrics.size());
    for (auto const& metric : args.metrics) {
      char_metrics.push_back(metric.c_str());
    }

    size_t num_passes = 0;
    CUPTI_API_CALL(pmSamplingHost.CreateConfigImage(char_metrics, configImage, num_passes));
    if (num_passes > 1) {
      throw std::invalid_argument("The configured profiling metrics require " +
                                  std::to_string(num_passes) +
                                  " passes. Reduce or change your metrics to use only one pass.");
    }

    cuptiPmSamplingTarget.SetUp(args.deviceIndex);
    target_setup_ = true;

    // 1. Enable PM sampling and set config for the PM sampling data collection.
    CUPTI_API_CALL(cuptiPmSamplingTarget.EnablePmSampling(args.deviceIndex));
    sampling_enabled_ = true;
    CUPTI_API_CALL(
      cuptiPmSamplingTarget.SetConfig(configImage, args.hardwareBufferSize, args.samplingInterval));

    // 2. Create counter data image
    CUPTI_API_CALL(cuptiPmSamplingTarget.CreateCounterDataImage(
      args.maxSamples, char_metrics, counterDataImage));

    // 3. Launch the decode thread
    decode_thread = std::thread(decode_counter_data,
                                std::ref(counterDataImage),
                                std::ref(char_metrics),
                                std::ref(startDecodeThread),
                                std::ref(stopDecodeThread),
                                std::ref(is_decode_thread_running),
                                std::ref(cuptiPmSamplingTarget),
                                std::ref(pmSamplingHost),
                                std::ref(decode_thread_result),
                                std::ref(decode_exception),
                                std::cref(sampleCallback));
  }

  void setSampleCallback(pm_sampler::sample_callback cb) { sampleCallback = std::move(cb); }

  void start()
  {
    // 4. Start the PM sampling and launch the CUDA workload
    CUPTI_API_CALL(cuptiPmSamplingTarget.StartPmSampling());
    startDecodeThread = true;

    while (!is_decode_thread_running.load()) {
      std::this_thread::yield();  // Wait for decode thread to start logging.
    }
  }

  std::vector<pm_sampler::sample> stop()
  {
    // 5. Stop the PM sampling and join the decode thread
    CUPTI_API_CALL(cuptiPmSamplingTarget.StopPmSampling());

    stopDecodeThread = true;
    joinDecodeThread();

    std::vector<pm_sampler::sample> samples;
    samples.reserve(pmSamplingHost.GetSamplerRange().size());

    for (auto& old_sample : pmSamplingHost.GetSamplerRange()) {
      pm_sampler::sample sample;
      sample.range_index     = old_sample.rangeIndex;
      sample.start_timestamp = old_sample.startTimestamp;
      sample.end_timestamp   = old_sample.endTimestamp;
      sample.metric_values   = old_sample.metricValues;

      samples.push_back(std::move(sample));
    }

    return samples;
  }

  void teardown() { release_resources(/*may_throw=*/true); }

  // Release the CUPTI resources acquired in setup(), in reverse order, each
  // guarded by its acquisition flag so a partially-completed setup() (or a
  // second call) is safe. On the destructor path (may_throw == false) any
  // CUPTI failure is logged and swallowed rather than propagated.
  void release_resources(bool may_throw)
  {
    auto guarded = [&](auto&& fn) {
      try {
        fn();
      } catch (...) {
        if (may_throw) { throw; }
        std::cerr << "pm_sampler: CUPTI teardown error (suppressed in dtor)\n";
      }
    };

    // 7. Clean up
    if (sampling_enabled_) {
      sampling_enabled_ = false;
      guarded([&] { CUPTI_API_CALL(cuptiPmSamplingTarget.DisablePmSampling()); });
    }
    if (target_setup_) {
      target_setup_ = false;
      guarded([&] { cuptiPmSamplingTarget.TearDown(); });
    }
    if (host_setup_) {
      host_setup_ = false;
      guarded([&] { pmSamplingHost.TearDown(); });
    }
  }

  int query_metrics()
  {
    // RAII guard so every return path (including the two early returns
    // below) tears down the CUPTI host. SetUp leaks CUPTI-internal state
    // otherwise — harmless for this short-lived CLI entry point, but the
    // RAII pattern keeps reviewers happy and future-proofs the function.
    CuptiProfilerHost pmSamplingHost;
    pmSamplingHost.SetUp(chipName, counterAvailibilityImage);
    auto teardown_guard = std::unique_ptr<CuptiProfilerHost, void (*)(CuptiProfilerHost*)>(
      &pmSamplingHost, [](CuptiProfilerHost* h) { h->TearDown(); });

    if (args.queryBaseMetrics) {
      std::vector<std::string> baseMetrics;
      CUPTI_API_CALL(pmSamplingHost.GetSupportedBaseMetrics(baseMetrics));
      printf("Base Metrics:\n");
      for (const auto& metric : baseMetrics) {
        printf("  %s\n", metric.c_str());
      }
      return 0;
    }

    if (args.queryMetricProperties) {
      for (const auto& metricName : args.metrics) {
        std::vector<std::string> subMetrics;
        CUPTI_API_CALL(pmSamplingHost.GetSubMetrics(metricName, subMetrics));
        printf("Sub Metrics for %s:\n", metricName.c_str());
        for (const auto& metric : subMetrics) {
          printf("  %s\n", metric.c_str());
        }

        std::string metricDescription;
        CUpti_MetricType metricType;
        CUPTI_API_CALL(
          pmSamplingHost.GetMetricProperties(metricName, metricType, metricDescription));

        printf("Metric Description: %s\n", metricDescription.c_str());
        printf("Metric Type: %s\n",
               metricType == CUPTI_METRIC_TYPE_COUNTER   ? "Counter"
               : (metricType == CUPTI_METRIC_TYPE_RATIO) ? "Ratio"
                                                         : "Throughput");
        printf("\n");
      }
      return 0;
    }

    return 0;
  }

 private:
  pm_sampler::configuration args;
  // Non-owning `const char*` view over `args.metrics`, rebuilt in setup().
  // Kept as a member so the decode thread can hold a stable reference to it.
  std::vector<const char*> char_metrics;
  std::string chipName;
  std::vector<uint8_t> counterAvailibilityImage;
  std::vector<uint8_t> configImage;
  std::vector<uint8_t> counterDataImage;
  std::thread decode_thread;
  std::atomic<bool> is_decode_thread_running;
  std::atomic<bool> startDecodeThread;
  std::atomic<bool> stopDecodeThread;
  std::atomic<CUptiResult> decode_thread_result;
  // Set by the decode thread if the user callback threw; rethrown from stop().
  std::exception_ptr decode_exception;
  CuptiProfilerHost pmSamplingHost;
  CuptiPmSampling cuptiPmSamplingTarget;
  pm_sampler::sample_callback sampleCallback;
  // Track which CUPTI resources have been acquired so the destructor can
  // release them on any path (incl. a setup() that throws partway).
  bool host_setup_       = false;
  bool target_setup_     = false;
  bool sampling_enabled_ = false;
};

}  // namespace impl

pm_sampler::pm_sampler(configuration const& args)
  : _impl(std::make_unique<impl::pm_sampler_impl>(args))
{
}

pm_sampler::~pm_sampler() = default;

void pm_sampler::setup() { _impl->setup(); }

void pm_sampler::start() { _impl->start(); }

std::vector<pm_sampler::sample> pm_sampler::stop() { return _impl->stop(); }

void pm_sampler::teardown() { _impl->teardown(); }

int pm_sampler::query_metrics() { return _impl->query_metrics(); }

void pm_sampler::set_sample_callback(sample_callback cb)
{
  _impl->setSampleCallback(std::move(cb));
}

}  // namespace utility
}  // namespace gqe
