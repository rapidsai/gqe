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

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

namespace gqe::task_manager {

/**
 * @brief The one thread that initializes a task manager and coordinates its operations.
 *
 * # Design
 *
 * A task manager's CUDA context, NVSHMEM state, and execution context all have per-thread effects,
 * such as the current device, the current memory resource, and the CPU affinity. Running all of
 * their setup and use on one thread keeps these effects consistent. gRPC instead runs each handler
 * on any thread of its pool, so handlers only queue jobs here and wait for them.
 *
 * Jobs run one at a time, in submission order. The owner thread may start helper threads within a
 * job, such as executor workers, but joins them before the job ends.
 *
 * ## Thread Safety
 *
 * `submit` and `stop` may be called concurrently from any thread except the owner thread itself.
 * Calling `stop` from a job would deadlock, because `stop` joins the owner thread.
 */
class owner_thread {
 public:
  /** @brief Start the owner thread with an empty queue. */
  owner_thread();

  /** @brief Stop the owner thread, as `stop` does. */
  ~owner_thread();

  owner_thread(owner_thread const&)            = delete;
  owner_thread& operator=(owner_thread const&) = delete;
  owner_thread(owner_thread&&)                 = delete;
  owner_thread& operator=(owner_thread&&)      = delete;

  /**
   * @brief Queue `job` to run on the owner thread.
   *
   * @param job The job to run. Exceptions it throws are stored in the returned future.
   * @return A future that becomes ready when `job` has run. If the owner thread stops before
   *         running `job`, the future holds a `std::future_error` with
   *         `std::future_errc::broken_promise`.
   * @throw std::runtime_error if `stop` has already been called.
   */
  std::future<void> submit(std::function<void()> job);

  /**
   * @brief Wait for the running job, if any, then discard queued jobs and join the owner thread.
   *
   * Discarded jobs never run, and their futures report `std::future_errc::broken_promise`. Later
   * calls have no effect.
   */
  void stop();

 private:
  void run();

  std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<std::packaged_task<void()>> _jobs;  ///< Protected by `_mutex`.
  bool _is_stopping = false;                     ///< Protected by `_mutex`.
  std::mutex _join_mutex;                        ///< Serializes joins of `_thread`.
  std::thread _thread;                           ///< Declared last so it starts after the queue.
};

}  // namespace gqe::task_manager
