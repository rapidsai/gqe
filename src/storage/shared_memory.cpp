/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/storage/shared_memory.hpp>

#include <gqe/utility/logger.hpp>

#include <atomic>
#include <cstddef>
#include <format>
#include <mutex>
#include <stdexcept>

namespace gqe {

namespace {

std::atomic<bool> poisoned{false};

// The flag shared through the attached segment (see `shared_segment_manager::poisoned_flag`).
using segment_flag_type = std::atomic<std::uint32_t>;
std::atomic<segment_flag_type*> segment_flag{nullptr};

void set_segment_flag() noexcept
{
  if (auto* flag = segment_flag.load(std::memory_order_acquire)) {
    flag->store(1, std::memory_order_release);
  }
}

// Guards `poison_cause`. Only the first poisoning stores a cause, so the lock is taken at most once
// per failure report and never on the fast path of `is_shared_memory_poisoned`.
std::mutex poison_cause_mutex;
std::string poison_cause;

std::string describe_lock_exception(std::string_view operation,
                                    boost::interprocess::lock_exception const& error)
{
  auto const* const reason = error.get_error_code() == boost::interprocess::not_recoverable
                               ? "a process died while holding a lock in the shared-memory segment"
                               : "a lock in the shared-memory segment failed";
  return std::format("{}: {} ({}, error code {})",
                     operation,
                     reason,
                     error.what(),
                     static_cast<int>(error.get_error_code()));
}

}  // namespace

bool is_shared_memory_poisoned() noexcept
{
  if (poisoned.load(std::memory_order_acquire)) { return true; }
  auto const* flag = segment_flag.load(std::memory_order_acquire);
  return flag != nullptr && flag->load(std::memory_order_acquire) != 0;
}

std::string shared_memory_poison_cause()
{
  {
    std::scoped_lock guard(poison_cause_mutex);
    if (!poison_cause.empty()) { return poison_cause; }
  }
  if (is_shared_memory_poisoned()) {
    return "another process recorded an unrecoverable shared-memory error in the segment";
  }
  return {};
}

void attach_shared_memory_poison_flag(shared_segment& segment) noexcept
{
  segment_flag.store(&segment.get_segment_manager()->poisoned_flag(), std::memory_order_release);
  if (poisoned.load(std::memory_order_acquire)) { set_segment_flag(); }
}

void detach_shared_memory_poison_flag(shared_segment const& segment) noexcept
{
  auto* flag = &segment.get_segment_manager()->poisoned_flag();
  segment_flag.compare_exchange_strong(flag, nullptr, std::memory_order_acq_rel);
}

void mark_shared_memory_poisoned(std::string_view cause) noexcept
{
  try {
    {
      std::scoped_lock guard(poison_cause_mutex);
      if (poisoned.load(std::memory_order_relaxed)) {
        GQE_LOG_ERROR("Shared memory is unrecoverable (already reported): {}", cause);
        return;
      }
      poison_cause.assign(cause);
      poisoned.store(true, std::memory_order_release);
      set_segment_flag();
    }
    GQE_LOG_CRITICAL("Shared memory is unrecoverable: {}", cause);
  } catch (...) {
    // Recording the cause or logging failed. The flag is what matters, so set it regardless.
    poisoned.store(true, std::memory_order_release);
    set_segment_flag();
  }
}

bool record_if_shared_memory_unrecoverable(std::exception_ptr error,
                                           std::string_view operation) noexcept
{
  if (error == nullptr) { return false; }
  try {
    std::rethrow_exception(error);
  } catch (boost::interprocess::lock_exception const& lock_error) {
    try {
      mark_shared_memory_poisoned(describe_lock_exception(operation, lock_error));
    } catch (...) {
      mark_shared_memory_poisoned(operation);
    }
    return true;
  } catch (shared_memory_unrecoverable_error const& unrecoverable) {
    // `throw_shared_memory_unrecoverable` has already set the flag. Mark only if the error was
    // raised some other way, so the cause is not logged twice.
    if (!poisoned.load(std::memory_order_acquire)) {
      mark_shared_memory_poisoned(unrecoverable.what());
    }
    return true;
  } catch (...) {
    return false;
  }
}

void throw_shared_memory_unrecoverable(std::string_view operation,
                                       boost::interprocess::lock_exception const& error)
{
  auto message = describe_lock_exception(operation, error);
  mark_shared_memory_poisoned(message);
  throw shared_memory_unrecoverable_error(
    std::format("Shared memory is unrecoverable: {}", message));
}

#ifdef GQE_ENABLE_FAULT_INJECTION
namespace detail {

void reset_shared_memory_poisoned_for_testing() noexcept
{
  std::scoped_lock guard(poison_cause_mutex);
  poison_cause.clear();
  poisoned.store(false, std::memory_order_release);
  if (auto* flag = segment_flag.load(std::memory_order_acquire)) {
    flag->store(0, std::memory_order_release);
  }
}

}  // namespace detail
#endif

}  // namespace gqe
