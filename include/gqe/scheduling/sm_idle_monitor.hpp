/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

namespace gqe {

namespace utility {
class pm_sampler;
}  // namespace utility

namespace scheduling {

/**
 * @brief Live monitor of GPU SM utilization for the dynamic decompression
 *        offload mechanism.
 *
 * Wraps a CUPTI PM sampler and exposes the most recent SM-utilization reading
 * through a hot-path-friendly accessor: a reader on the decompression dispatch
 * path obtains the latest value with a single atomic load and never calls into
 * the CUPTI driver itself. An optional moving average smooths the reading over
 * a configurable window.
 *
 * Lifecycle: construction sets up and starts sampling; destruction stops and
 * tears it down. CUPTI initialization failure is fatal (no soft fallback).
 *
 * Single-GPU only; multi-GPU would need one instance per device.
 */
class SmIdleMonitor {
 public:
  struct Config {
    int device_id               = -1;  ///< CUDA device index (-1 uses the current device).
    std::string metric          = "warps_active_elapsed";  ///< SM-utilization metric selector.
    uint64_t sample_interval_ns = 1'000'000;               ///< CUPTI sample period (1 ms default).
    size_t window_samples       = 1;          ///< Moving-average window (1 = no smoothing).
    bool debug                  = false;      ///< Log the first few decoded samples.
    std::string trace_json_file = "";         ///< If non-empty, JSON trace output file path.
    size_t trace_max_entries    = 1'000'000;  ///< Maximum JSON trace records.
  };

  /** @brief Construct with explicit configuration. */
  explicit SmIdleMonitor(Config config);

  /** @brief Build default monitor config and apply supported environment overrides. */
  [[nodiscard]] static Config config_from_environment();

  /** @brief Apply supported environment overrides to an existing config. */
  [[nodiscard]] static Config config_from_environment(Config config);

  /** @brief Convenience overload with default configuration. */
  SmIdleMonitor() : SmIdleMonitor(config_from_environment()) {}

  /** @brief Stop sampling, join bg thread, tear down CUPTI. */
  ~SmIdleMonitor();

  SmIdleMonitor(SmIdleMonitor const&)            = delete;
  SmIdleMonitor& operator=(SmIdleMonitor const&) = delete;

  /** @brief Latest sampled SM utilization (0.0 to 1.0). */
  [[nodiscard]] double get_current_sm_util() const
  {
    return _latest_sm_util.load(std::memory_order_relaxed);
  }

  /**
   * @brief Smoothed SM utilization — moving average of the last W samples
   *        (W = get_window_size()). Falls back to the latest sample when no
   *        samples have been received yet.
   *
   * The window size W is fixed at construction. A window size of 1 returns the
   * latest sample without smoothing. Access to the sample history is
   * synchronized with the sampler thread.
   */
  [[nodiscard]] double get_smoothed_sm_util() const;

  /** @brief Configured window size (number of samples averaged). */
  [[nodiscard]] size_t get_window_size() const { return _window_size; }

  /** @brief Aggregate counters for evaluation. */
  void record_decision(bool offloaded_to_sm)
  {
    _total_decisions.fetch_add(1, std::memory_order_relaxed);
    if (offloaded_to_sm) { _offload_decisions.fetch_add(1, std::memory_order_relaxed); }
  }

  [[nodiscard]] uint64_t get_total_decisions() const
  {
    return _total_decisions.load(std::memory_order_relaxed);
  }

  [[nodiscard]] uint64_t get_offload_decisions() const
  {
    return _offload_decisions.load(std::memory_order_relaxed);
  }

  /** @brief Number of samples received from CUPTI since startup. */
  [[nodiscard]] uint64_t get_samples_received() const
  {
    return _samples_received.load(std::memory_order_relaxed);
  }

 private:
  void on_sample(double sm_util);

  Config _config;
  std::atomic<double> _latest_sm_util{0.0};
  std::atomic<uint64_t> _samples_received{0};
  std::atomic<uint64_t> _offload_decisions{0};
  std::atomic<uint64_t> _total_decisions{0};

  // Recent samples for moving-average smoothing. The deque is bounded by the
  // configured window size and protected because the sampler thread writes it
  // while dispatch threads may read it.
  mutable std::mutex _samples_mutex;
  std::deque<double> _recent_samples;
  size_t _window_size{1};

  std::unique_ptr<gqe::utility::pm_sampler> _sampler;

  // Optional JSON trace (debug): when a trace path is configured the monitor
  // appends one JSON record per CUPTI sample ({"t_us", "sm_util", "metric"}).
  std::ofstream _trace_file;
  size_t _trace_entries_written{0};
  std::chrono::steady_clock::time_point _ctor_time;
};

}  // namespace scheduling
}  // namespace gqe
