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

#include <gqe/utility/cupti_activity.hpp>

#include <gqe/utility/cupti_common.hpp>
#include <gqe/utility/error.hpp>

#include <cuda.h>

#include <cupti_result.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace gqe {
namespace utility {
namespace detail {
namespace {
// The data is used to allocate the CUPTI buffer.
constexpr size_t buffer_size  = 8 * 1024 * 1024;  // 8 MB per buffer
constexpr size_t buffer_align = 8;                // CUPTI requires 8-byte-aligned buffers
}  // namespace

using cupti_marker_id = uint32_t;

/**
 * @brief CUPTI Activity profiler implementation.
 *
 * See `activity_profiler` for API documentation.
 */
class activity_profiler_impl {
 public:
  activity_profiler_impl(std::vector<CUpti_ActivityKind> kinds)
  {
    // We only support the following activity kinds capturing for now.
    // If other kinds are requested, an exception would be thrown.
    static const auto supported_kinds = std::vector<CUpti_ActivityKind>{
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
      CUPTI_ACTIVITY_KIND_MEMCPY,
      CUPTI_ACTIVITY_KIND_MARKER,
      CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS,
    };
    for (auto k : kinds) {
      if (std::find(supported_kinds.begin(), supported_kinds.end(), k) == supported_kinds.end()) {
        throw std::invalid_argument(
          "Unsupported activity kind: " + std::to_string(static_cast<int>(k)) +
          ". Supported kinds are: CONCURRENT_KERNEL, MEMCPY, MARKER, MEM_DECOMPRESS.");
      }
    }

    if (g_active_profiler != nullptr) {
      throw std::logic_error(
        "Only one activity_profiler may exist at a time. The CUPTI Activity API uses "
        "process-global buffer callbacks.");
    }
    g_active_profiler = this;

    GQE_CUPTI_TRY(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));

    for (auto k : kinds) {
      GQE_CUPTI_TRY(cuptiActivityEnable(k));
      _kinds.push_back(k);
    }
  }

  ~activity_profiler_impl() noexcept(false)
  {
    if (_is_running) { stop(); }  // stop() is void — safe to call without using return value

    teardown();
  }

  void start()
  {
    if (_is_running) {
      throw std::logic_error("Start called on activity_profiler that is already running.");
    }

    // Ensure all activity records are generated and flushed.
    GQE_CUPTI_TRY(cuptiActivityFlushAll(1));
    _pending = {};
    _open_markers.clear();
    _is_running = true;
  }

  void stop()
  {
    if (!_is_running) {
      throw std::logic_error("Stop called on activity_profiler that is not running.");
    }
    _is_running = false;

    // Deliver all records that are still buffered inside CUPTI.
    GQE_CUPTI_TRY(cuptiActivityFlushAll(1));
  }

  [[nodiscard]] activity_records decode_result()
  {
    if (_is_running) {
      throw std::logic_error(
        "Decode result called on activity_profiler that is still running. "
        "Call stop() first.");
    }

    // Discard any MARKER_START records that never received a matching END.
    _open_markers.clear();

    return std::move(_pending);
  }

  void teardown()
  {
    if (_is_running) {
      throw std::logic_error(
        "Teardown called on profiler that is still running. Profiler should be stopped first.");
    }
    GQE_CUPTI_TRY(cuptiActivityFlushAll(1));
    for (auto k : _kinds) {
      GQE_CUPTI_TRY(cuptiActivityDisable(k));
    }
    g_active_profiler = nullptr;
  }

 private:
  /**
   * @brief Walk through the CUPTI buffer and append event records to `_pending`.
   */
  void process_buffer(uint8_t* buffer, size_t valid_size)
  {
    CUpti_Activity* record = nullptr;

    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
      switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          auto* k = reinterpret_cast<CUpti_ActivityKernel9*>(record);
          kernel_event ev;
          ev.name                = k->name ? k->name : "";
          ev.interval.start_time = k->start;
          ev.interval.end_time   = k->end;
          _pending.kernels.push_back(std::move(ev));
          break;
        }

        case CUPTI_ACTIVITY_KIND_MEMCPY: {
          auto* m = reinterpret_cast<CUpti_ActivityMemcpy4*>(record);
          memcpy_event ev;
          ev.kind                = m->copyKind;
          ev.interval.start_time = m->start;
          ev.interval.end_time   = m->end;
          _pending.memcopies.push_back(ev);
          break;
        }

        // NVTX ranges arrive as paired START / END records sharing the same id.
        // Store the START record in `_open_markers` and then look up the START record when the END
        // record is encountered.
        case CUPTI_ACTIVITY_KIND_MARKER: {
          auto* m = reinterpret_cast<CUpti_ActivityMarker2*>(record);
          if (m->flags & CUPTI_ACTIVITY_FLAG_MARKER_START) {
            nvtx_event partial{};
            partial.name                = m->name ? m->name : "";
            partial.interval.start_time = m->timestamp;
            _open_markers[m->id]        = std::move(partial);
          } else if (m->flags & CUPTI_ACTIVITY_FLAG_MARKER_END) {
            auto it = _open_markers.find(m->id);
            if (it != _open_markers.end()) {
              nvtx_event ev        = std::move(it->second);
              ev.interval.end_time = m->timestamp;
              _pending.markers.push_back(std::move(ev));
              _open_markers.erase(it);
            }
          }
          break;
        }

        case CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS: {
          auto* d = reinterpret_cast<CUpti_ActivityMemDecompress*>(record);
          mem_decompress_event ev;
          ev.interval.start_time = d->start;
          ev.interval.end_time   = d->end;
          _pending.mem_decompress.push_back(ev);
          break;
        }

        default: break;
      }
    }
  }

  // Global CUPTI buffer callback.
  // CUPTI requires free functions or static member functions for its callbacks.
  static void buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records)
  {
    *size            = buffer_size;
    *buffer          = static_cast<uint8_t*>(aligned_alloc(buffer_align, buffer_size));
    *max_num_records = 0;  // 0 = let CUPTI fill the whole buffer
  }

  // Global CUPTI buffer callback.
  // Extract the activity records from the buffer and append to `_pending`.
  static void buffer_completed(
    CUcontext ctx, uint32_t stream_id, uint8_t* buffer, size_t size, size_t valid_size)
  {
    if (valid_size > 0) { g_active_profiler->process_buffer(buffer, valid_size); }
    free(buffer);
  }

  std::vector<CUpti_ActivityKind> _kinds;
  bool _is_running{false};

  activity_records _pending;

  // Stack of MARKER_START records waiting for a matching MARKER_END.
  // Use a map to quickly look up the START record by id.
  std::unordered_map<cupti_marker_id, nvtx_event> _open_markers;

  // Process-global pointer to the single live profiler. Set in the constructor,
  // cleared in the destructor. The buffer callbacks read this to find where to
  // deposit parsed records.
  static activity_profiler_impl* g_active_profiler;
};

activity_profiler_impl* activity_profiler_impl::g_active_profiler = nullptr;

}  // namespace detail

// ── Public API ────────────────────────────────────────────────────────────────

activity_profiler::activity_profiler(std::vector<CUpti_ActivityKind> kinds)
  : _impl(std::make_unique<detail::activity_profiler_impl>(std::move(kinds)))
{
}

activity_profiler::~activity_profiler() noexcept(false) = default;

void activity_profiler::start() { _impl->start(); }

activity_records activity_profiler::stop()
{
  _impl->stop();
  return _impl->decode_result();
}

namespace detail {
// Merge the intervals of the activity events if there is overlap.
// E.g. [[1, 3], [2, 4]] -> [[1, 4]].
std::vector<event_interval> merge_intervals(std::vector<event_interval>& intervals)
{
  std::vector<event_interval> merged;
  // Sort the intervals by start_time.
  std::sort(
    intervals.begin(), intervals.end(), [](const event_interval& a, const event_interval& b) {
      return a.start_time < b.start_time;
    });
  // Merge the intervals.
  for (auto& interval : intervals) {
    if (merged.empty() || interval.start_time > merged.back().end_time) {
      merged.push_back(interval);
    } else {
      merged.back().end_time = std::max(merged.back().end_time, interval.end_time);
    }
  }
  return merged;
}

// Sum the duration of the intervals in seconds.
double sum_duration(const std::vector<event_interval>& intervals)
{
  double duration = 0;
  for (auto& interval : intervals) {
    duration += interval.end_time - interval.start_time;
  }
  return duration / 1e9;
}

// Merge multiple vectors of the same type into a single vector.
// The input vectors are no longer valid after this operation.
std::vector<event_interval> merge(std::vector<std::vector<event_interval>>&& vecs)
{
  size_t total = 0;
  for (const auto& v : vecs)
    total += v.size();

  std::vector<event_interval> out;
  out.reserve(total);

  for (auto& v : vecs) {
    out.insert(out.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
  }

  return out;
}

template <class T>
std::vector<event_interval> to_intervals(std::vector<T>& events)
{
  std::vector<event_interval> intervals;
  intervals.reserve(events.size());
  for (auto& event : events) {
    intervals.push_back(event.interval);
  }
  return intervals;
}

}  // namespace detail

time_breakdown activity_profiler::get_time_breakdown(
  activity_records& records, const std::vector<std::string>& io_kernel_filter_list)
{
  time_breakdown tb;
  auto io_kernel_intervals      = std::vector<event_interval>();
  auto compute_kernel_intervals = std::vector<event_interval>();
  for (auto& kernel : records.kernels) {
    if (std::any_of(io_kernel_filter_list.begin(),
                    io_kernel_filter_list.end(),
                    [&](const std::string& filter) {
                      return kernel.name.find(filter) != std::string::npos;
                    })) {
      io_kernel_intervals.push_back(kernel.interval);
    } else {
      compute_kernel_intervals.push_back(kernel.interval);
    }
  }

  // Only keep the `in_memory_read_task` marker activity.
  auto read_task_intervals = std::vector<event_interval>();
  for (auto& marker : records.markers) {
    if (marker.name.find("in_memory_read_task") != std::string::npos) {
      read_task_intervals.push_back(marker.interval);
    }
  }

  // Only keep the H2D memcpy activity.
  auto h2d_memcpy_intervals = std::vector<event_interval>();
  for (auto& memcpy : records.memcopies) {
    if (memcpy.kind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD) {
      h2d_memcpy_intervals.push_back(memcpy.interval);
    }
  }

  auto merged_read_task_intervals      = detail::merge_intervals(read_task_intervals);
  auto merged_io_kernel_intervals      = detail::merge_intervals(io_kernel_intervals);
  auto merged_compute_kernel_intervals = detail::merge_intervals(compute_kernel_intervals);
  auto merged_h2d_memcpy_intervals     = detail::merge_intervals(h2d_memcpy_intervals);
  auto mem_decompress_intervals        = detail::to_intervals(records.mem_decompress);
  auto merged_mem_decompress_intervals = detail::merge_intervals(mem_decompress_intervals);

  tb.in_memory_read_task_s = detail::sum_duration(merged_read_task_intervals);
  tb.io_kernel_s           = detail::sum_duration(merged_io_kernel_intervals);
  tb.compute_kernel_s      = detail::sum_duration(merged_compute_kernel_intervals);
  tb.memcpy_s              = detail::sum_duration(merged_h2d_memcpy_intervals);
  tb.mem_decompress_s      = detail::sum_duration(merged_mem_decompress_intervals);

  // Merge all io activities into a single interval.
  auto io_intervals        = detail::merge({std::move(merged_io_kernel_intervals),
                                            std::move(merged_h2d_memcpy_intervals),
                                            std::move(merged_mem_decompress_intervals)});
  auto merged_io_intervals = detail::merge_intervals(io_intervals);
  tb.merged_io_activity_s  = detail::sum_duration(merged_io_intervals);

  return tb;
}

}  // namespace utility
}  // namespace gqe
