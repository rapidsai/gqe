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

#include <gqe/task_manager/owner_thread.hpp>

#include <stdexcept>
#include <utility>

namespace gqe::task_manager {

owner_thread::owner_thread() : _thread([this] { run(); }) {}

owner_thread::~owner_thread() { stop(); }

std::future<void> owner_thread::submit(std::function<void()> job)
{
  std::packaged_task<void()> task(std::move(job));
  auto done = task.get_future();
  {
    std::scoped_lock guard(_mutex);
    if (_is_stopping) { throw std::runtime_error("The task manager's owner thread has stopped"); }
    _jobs.push_back(std::move(task));
  }
  _cv.notify_one();
  return done;
}

void owner_thread::stop()
{
  std::deque<std::packaged_task<void()>> discarded;
  {
    std::scoped_lock guard(_mutex);
    _is_stopping = true;
    discarded.swap(_jobs);
  }
  _cv.notify_one();
  std::scoped_lock guard(_join_mutex);
  if (_thread.joinable()) { _thread.join(); }
  // Destroying a packaged task that never ran makes its future report `broken_promise`.
}

void owner_thread::run()
{
  while (true) {
    std::packaged_task<void()> job;
    {
      std::unique_lock guard(_mutex);
      _cv.wait(guard, [this] { return _is_stopping || !_jobs.empty(); });
      if (_is_stopping) { return; }
      job = std::move(_jobs.front());
      _jobs.pop_front();
    }
    job();
  }
}

}  // namespace gqe::task_manager
