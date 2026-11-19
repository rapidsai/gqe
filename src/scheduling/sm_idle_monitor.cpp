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

#include <gqe/scheduling/sm_idle_monitor.hpp>

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/utility/cupti_pm_sampler.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace gqe {
namespace scheduling {

namespace {
// Translate a metric selector (see optimization_parameters::sm_monitor_metric) into the CUPTI
// PM metric name. Choices trade off what "SM busy" means for the offload dispatcher:
//
//   warps_active_elapsed (default): "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
//     Average resident warps as a fraction of the SM's max warp slots across the sampling
//     window. Distinguishes low-occupancy kernels from saturated SMs and stays proportional
//     across the whole occupancy range — the right default for "how busy is the SM?".
//
//   cycles_active: "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"
//     Reports ~100 whenever the SM has at least one warp resident — does NOT distinguish
//     "1 warp" from "saturated". Use only for the coarse "is any kernel running at all?"
//     question (e.g. gating background work between queries).
//
//   warps_active: "sm__warps_active.avg.pct_of_peak_sustained_active"
//     Achieved occupancy of the running kernel measured over active cycles only (idle gaps
//     factored out). Use when profiling a kernel's warp-slot efficiency rather than overall
//     SM busyness.
std::string_view resolve_metric_name(std::string_view selector)
{
  if (selector == "warps_active_elapsed") {
    return "sm__warps_active.avg.pct_of_peak_sustained_elapsed";
  }
  if (selector == "warps_active") { return "sm__warps_active.avg.pct_of_peak_sustained_active"; }
  if (selector == "cycles_active") { return "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"; }
  if (selector == "dram") { return "dram__throughput.avg.pct_of_peak_sustained_elapsed"; }
  GQE_LOG_WARN(
    "Unrecognized SM monitor metric selector '{}'; using "
    "'warps_active_elapsed'",
    selector);
  return "sm__warps_active.avg.pct_of_peak_sustained_elapsed";
}
}  // namespace

SmIdleMonitor::Config SmIdleMonitor::config_from_environment()
{
  return config_from_environment(Config{});
}

SmIdleMonitor::Config SmIdleMonitor::config_from_environment(Config config)
{
  // Environment parsing is centralized in optimization_parameters; pull the sm_monitor_* knobs
  // from there rather than reading getenv directly. device_id is not env-driven, so preserve the
  // caller's value.
  auto const params         = gqe::make_optimization_parameters();
  config.metric             = params.sm_monitor_metric;
  config.sample_interval_ns = params.sm_monitor_sample_interval_ns;
  config.window_samples     = params.sm_monitor_window_samples;
  config.debug              = params.sm_monitor_debug;
  config.trace_json_file    = params.sm_monitor_trace_json_file;
  config.trace_max_entries  = params.sm_monitor_trace_max_entries;
  return config;
}

SmIdleMonitor::SmIdleMonitor(Config config) : _config(std::move(config))
{
  // Smoothing window (1 = no smoothing).
  _window_size = std::max<size_t>(_config.window_samples, 1);

  // Respect the CUDA device already selected by the task manager unless the caller supplied an
  // explicit device ordinal.
  if (_config.device_id == -1) { GQE_CUDA_TRY(cudaGetDevice(&_config.device_id)); }

  // Optional per-sample JSON trace, for offline analysis.
  if (!_config.trace_json_file.empty()) {
    _trace_file.open(_config.trace_json_file);
    if (_trace_file.is_open()) {
      _ctor_time = std::chrono::steady_clock::now();
    } else {
      GQE_LOG_ERROR("Failed to open SM monitor JSON trace file '{}'", _config.trace_json_file);
    }
  }

  // CUPTI Profiler init (called from pm_sampler ctor below) requires a CUDA
  // context to exist. Force runtime init by touching the device. cudaFree(0)
  // is the canonical idiom for this.
  GQE_CUDA_TRY(cudaSetDevice(_config.device_id));
  GQE_CUDA_TRY(cudaFree(nullptr));

  std::string const metric_key{resolve_metric_name(_config.metric)};
  gqe::utility::pm_sampler::configuration cfg{};
  cfg.deviceIndex      = _config.device_id;
  cfg.samplingInterval = _config.sample_interval_ns;
  cfg.metrics          = {metric_key};

  _sampler = std::make_unique<gqe::utility::pm_sampler>(cfg);

  // Register the per-sample callback BEFORE setup() so the decode thread
  // sees it from its first iteration.
  bool const debug_samples = _config.debug;
  _sampler->set_sample_callback(
    [this, metric_key, debug_samples](gqe::utility::pm_sampler::sample const& s) {
      // Optional first-samples metric dump for debugging CUPTI metric selection.
      static std::atomic<int> debug_logged{0};
      if (debug_samples) {
        if (debug_logged.fetch_add(1, std::memory_order_relaxed) < 3) {
          std::string dump;
          for (auto const& [k, v] : s.metric_values) {
            dump += k + "=" + std::to_string(v) + "  ";
          }
          GQE_LOG_DEBUG("SmIdleMonitor sample: {}", dump);
        }
      }
      auto it = s.metric_values.find(metric_key);
      if (it == s.metric_values.end()) {
        // The configured metric was missing from a decoded sample. This should
        // never happen (we ask CUPTI for exactly this metric), so warn — but
        // throttle to a handful of lines so a persistent misconfiguration does
        // not flood the log from the decode thread.
        static std::atomic<int> missing_logged{0};
        if (missing_logged.fetch_add(1, std::memory_order_relaxed) < 5) {
          GQE_LOG_WARN("SmIdleMonitor: configured metric '{}' absent from CUPTI sample; skipping",
                       metric_key);
        }
        return;
      }
      // Metric is a percentage 0-100; rescale to [0, 1].
      this->on_sample(it->second / 100.0);
    });

  // setup() and start() go through CUPTI and may throw. pm_sampler's destructor
  // joins the decode thread and releases held CUPTI resources, but that only
  // runs once _sampler is fully owned and this constructor completes — if
  // start() throws here, tear the sampler down explicitly first so we never
  // leave PM sampling enabled on the device. release_resources() is safe to
  // call more than once, so the subsequent member destruction is harmless.
  try {
    _sampler->setup();
    _sampler->start();
  } catch (...) {
    try {
      _sampler->teardown();
    } catch (...) {
      // Best-effort cleanup on the failure path; surface the original error.
    }
    throw;
  }
}

SmIdleMonitor::~SmIdleMonitor()
{
  // stop() returns the accumulated samples; we don't need them — the
  // per-sample callback already updated our atomics. Discard the
  // nodiscard return value explicitly.
  //
  // stop() and teardown() each go through CUPTI_API_CALL and can throw.
  // We must NOT skip teardown() just because stop() threw — that would
  // leave PM sampling enabled on the device and leak the host profiler
  // state. Wrap each call in its own try/catch so a stop() failure does
  // not bypass teardown().
  if (_sampler) {
    try {
      std::ignore = _sampler->stop();
    } catch (...) {
      // dtor: swallow.
    }
    try {
      _sampler->teardown();
    } catch (...) {
      // dtor: swallow.
    }
  }
}

void SmIdleMonitor::on_sample(double util)
{
  // Clamp to [0, 1] as a defensive guard against metric weirdness.
  util = std::clamp(util, 0.0, 1.0);

  _latest_sm_util.store(util, std::memory_order_relaxed);
  if (_window_size > 1) {
    std::lock_guard<std::mutex> const lock(_samples_mutex);
    _recent_samples.push_back(util);
    if (_recent_samples.size() > _window_size) { _recent_samples.pop_front(); }
  }
  _samples_received.fetch_add(1, std::memory_order_relaxed);

  // Optional JSON-Lines trace (debug). Single-writer (bg decode thread), so no lock is needed.
  if (_trace_file) {
    if (_trace_entries_written >= _config.trace_max_entries) {
      GQE_LOG_WARN("SM monitor JSON trace reached its {}-entry limit; stopping trace output",
                   _config.trace_max_entries);
      _trace_file.close();
    } else {
      auto const rel_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - _ctor_time)
                            .count();
      _trace_file << "{\"t_us\": " << rel_us << ", \"sm_util\": " << util << ", \"metric\": \""
                  << _config.metric << "\"}\n";
      ++_trace_entries_written;
    }
  }
}

double SmIdleMonitor::get_smoothed_sm_util() const
{
  if (_window_size <= 1) { return get_current_sm_util(); }

  std::lock_guard<std::mutex> const lock(_samples_mutex);
  if (_recent_samples.empty()) { return get_current_sm_util(); }

  double sum = 0.0;
  for (double const sample : _recent_samples) {
    sum += sample;
  }
  return sum / static_cast<double>(_recent_samples.size());
}

}  // namespace scheduling
}  // namespace gqe
