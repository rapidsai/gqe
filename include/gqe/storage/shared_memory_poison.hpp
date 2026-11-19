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

#pragma once

#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>

// GQE relies on robust Boost.Interprocess mutexes. When a process dies while holding a lock in the
// segment, the next lock must fail with `lock_exception` instead of blocking forever.
// Boost enables robust mutexes only when the POSIX feature macros advertise them, so a translation
// unit compiled without them would build non-robust mutexes into the segment or ignore
// `EOWNERDEAD`. Every header that uses Boost.Interprocess includes this header, so the check runs
// in every such translation unit, under both the host compiler and nvcc.
#ifndef BOOST_INTERPROCESS_POSIX_ROBUST_MUTEXES
#error "GQE requires robust Boost.Interprocess mutexes (BOOST_INTERPROCESS_POSIX_ROBUST_MUTEXES)"
#endif

namespace gqe {

// ----- Unrecoverable shared memory -----

/**
 * @brief Error raised when the shared-memory segment can no longer be trusted.
 *
 * # Design
 *
 * The segment holds several kinds of interprocess lock. Each arena of @ref shared_segment has an
 * allocator lock, taken by every allocation and deallocation in that arena, the segment's name map
 * has its own lock, and each shared table storage has its own mutex. All are robust POSIX mutexes.
 * If a process dies while holding one, Boost marks the mutex permanently unrecoverable instead of
 * handing it to the next owner, and every later lock throws `boost::interprocess::lock_exception`.
 *
 * The data the dead process was writing may be partially updated, so the segment's contents can no
 * longer be guaranteed. Code that observes a `lock_exception` therefore records the segment as
 * *poisoned* (see @ref mark_shared_memory_poisoned) and raises this error instead. Every call into
 * @ref shared_segment does this, and @ref lock_shared_memory_mutex does it for the table mutexes.
 * The node then tears down and exits, and the operator or supervising harness restarts it and
 * reloads the data.
 *
 * ## Poisoned flag
 *
 * Some code cannot propagate an exception, such as `noexcept` deallocation, destructors of
 * containers in the segment, and cleanup paths that log and continue. Those paths only set the
 * poisoned flag. The flag is sticky. It is kept per process, and also in the segment once the
 * process has attached it (see @ref attach_shared_memory_poison_flag), so that every process that
 * maps the segment sees it. A task-manager restart therefore cannot lose it.
 *
 * The fault boundary (see gqe/node_manager/fault_boundary.hpp) checks the flag before and after
 * every task-manager operation and every node-manager query or statement, and the node manager
 * checks it before restarting task managers. Poisoning is reported even when no exception reaches
 * any of these boundaries.
 */
class shared_memory_unrecoverable_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Return whether the shared memory is known to be unrecoverable.
 *
 * True if this process has observed an unrecoverable shared-memory error, or if any process has
 * recorded one in the segment this process attached. Once set, the flag stays set.
 */
[[nodiscard]] bool is_shared_memory_poisoned() noexcept;

/**
 * @brief Return the cause recorded by this process's first call to @ref
 * mark_shared_memory_poisoned.
 *
 * If only another process recorded the poisoning, returns a generic description. Returns an empty
 * string if the segment is not poisoned.
 */
[[nodiscard]] std::string shared_memory_poison_cause();

/**
 * @brief Record that the shared-memory segment is unrecoverable.
 *
 * The first call logs @p cause as a critical error and stores it. Later calls only log their cause
 * at error level.
 *
 * @param cause Description of the failed operation and the error that was observed.
 */
void mark_shared_memory_poisoned(std::string_view cause) noexcept;

/**
 * @brief Set the poisoned flag if @p error is an unrecoverable shared-memory error.
 *
 * An unrecoverable shared-memory error is either a `boost::interprocess::lock_exception` or a
 * @ref shared_memory_unrecoverable_error. Code that catches exceptions broadly calls this so that
 * swallowing or rethrowing the exception cannot lose the poisoning.
 *
 * @param error The caught exception. A null pointer is ignored.
 * @param operation Name of the operation that caught it, used in the recorded cause.
 * @return Whether @p error is an unrecoverable shared-memory error.
 */
bool record_if_shared_memory_unrecoverable(std::exception_ptr error,
                                           std::string_view operation) noexcept;

/**
 * @brief Set the poisoned flag and throw @ref shared_memory_unrecoverable_error.
 *
 * @param operation Name of the operation that observed @p error.
 * @param error The lock failure raised by Boost.Interprocess.
 * @throw shared_memory_unrecoverable_error Always.
 */
[[noreturn]] void throw_shared_memory_unrecoverable(
  std::string_view operation, boost::interprocess::lock_exception const& error);

/**
 * @brief Lock an interprocess mutex that lives in the shared-memory segment.
 *
 * Translates a lock failure into @ref shared_memory_unrecoverable_error and sets the poisoned flag.
 *
 * @tparam Mutex A Boost.Interprocess mutex type, such as `boost::interprocess::interprocess_mutex`.
 * @param mutex The mutex to lock.
 * @param operation Name of the operation taking the lock, used in the error message.
 * @return A guard that owns the lock.
 * @throw shared_memory_unrecoverable_error If a process died while holding @p mutex.
 */
template <typename Mutex>
[[nodiscard]] boost::interprocess::scoped_lock<Mutex> lock_shared_memory_mutex(
  Mutex& mutex, std::string_view operation)
{
  try {
    return boost::interprocess::scoped_lock<Mutex>(mutex);
  } catch (boost::interprocess::lock_exception const& error) {
    throw_shared_memory_unrecoverable(operation, error);
  }
}

#ifdef GQE_ENABLE_FAULT_INJECTION
namespace detail {

/**
 * @brief Clear the poisoned flag and its recorded cause.
 *
 * Only for tests that poison a segment deliberately. Production code never clears the flag. Also
 * clears the flag in the attached segment, if any.
 *
 * Declared only in a build with the CMake option `GQE_ENABLE_FAULT_INJECTION`, which is off by
 * default, so that a regular build cannot un-poison a process.
 */
void reset_shared_memory_poisoned_for_testing() noexcept;

}  // namespace detail
#endif

}  // namespace gqe
