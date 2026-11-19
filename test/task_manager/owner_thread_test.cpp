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

#include <gtest/gtest.h>

#include <future>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

using gqe::task_manager::owner_thread;

/**
 * @brief Test that jobs run in submission order, all on one thread other than the caller's.
 */
TEST(OwnerThreadTest, RunsJobsInOrderOnOneThread)
{
  owner_thread owner;
  constexpr int num_jobs = 16;
  std::vector<int> order;
  std::vector<std::thread::id> thread_ids;
  std::vector<std::future<void>> done;
  for (int job = 0; job < num_jobs; ++job) {
    done.push_back(owner.submit([&order, &thread_ids, job] {
      order.push_back(job);
      thread_ids.push_back(std::this_thread::get_id());
    }));
  }
  for (auto& job_done : done) {
    job_done.get();
  }

  ASSERT_EQ(order.size(), num_jobs);
  for (int job = 0; job < num_jobs; ++job) {
    EXPECT_EQ(order[job], job);
    EXPECT_EQ(thread_ids[job], thread_ids.front());
  }
  EXPECT_NE(thread_ids.front(), std::this_thread::get_id());
}

/**
 * @brief Test that a job's exception reaches its future and does not stop later jobs.
 */
TEST(OwnerThreadTest, PropagatesExceptions)
{
  owner_thread owner;
  auto failed = owner.submit([] { throw std::runtime_error("job failed"); });
  EXPECT_THROW(failed.get(), std::runtime_error);

  bool is_later_job_run = false;
  owner.submit([&] { is_later_job_run = true; }).get();
  EXPECT_TRUE(is_later_job_run);
}

/**
 * @brief Test that `stop` waits for the running job and discards the queued ones.
 */
TEST(OwnerThreadTest, StopWaitsForRunningJobAndDiscardsQueuedJobs)
{
  owner_thread owner;

  std::promise<void> started;
  std::promise<void> release;
  auto running = owner.submit([&started, released = release.get_future().share()] {
    started.set_value();
    released.wait();
  });
  started.get_future().wait();

  bool is_queued_job_run = false;
  auto queued            = owner.submit([&] { is_queued_job_run = true; });

  std::thread stopper([&] { owner.stop(); });
  // `submit` throws once `stop` has discarded the queue. Jobs submitted before then are discarded
  // too.
  while (true) {
    try {
      std::ignore = owner.submit([] {});
      std::this_thread::yield();
    } catch (std::runtime_error const&) {
      break;
    }
  }
  release.set_value();
  stopper.join();

  EXPECT_NO_THROW(running.get());
  try {
    queued.get();
    ADD_FAILURE() << "The queued job's future did not report an error";
  } catch (std::future_error const& e) {
    EXPECT_EQ(e.code(), std::future_errc::broken_promise);
  }
  EXPECT_FALSE(is_queued_job_run);
}

/**
 * @brief Test that `submit` throws after `stop`, and that `stop` can be called again.
 */
TEST(OwnerThreadTest, SubmitAfterStopThrows)
{
  owner_thread owner;
  owner.stop();
  EXPECT_THROW(std::ignore = owner.submit([] {}), std::runtime_error);
  EXPECT_NO_THROW(owner.stop());
}
