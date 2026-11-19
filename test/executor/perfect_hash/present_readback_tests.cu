/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/executor/perfect_hash/unique_indices.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <gtest/gtest.h>

#include <cuda/stream>
#include <cudf/column/column_view.hpp>
#include <rmm/cuda_stream.hpp>

#include <atomic>
#include <cstdint>
#include <random>
#include <set>
#include <thread>
#include <vector>

namespace {

struct present_readback_test : public cudf::test::BaseFixture {};

// The regression test: a key space this narrow is sparse, so every call runs build_rank.
TEST_F(present_readback_test, the_distinct_count_matches_an_independent_oracle_under_concurrency)
{
  constexpr int THREADS       = 8;
  constexpr int ITERATIONS    = 1000;
  constexpr int ROWS          = 1 << 20;
  constexpr int64_t KEY_SPACE = 1 << 18;  // Under the identity hasher's 20 significant bits.

  // Built on this thread: the cudf test wrappers are not meant to be constructed concurrently.
  std::vector<cudf::test::fixed_width_column_wrapper<int64_t>> keys;
  std::vector<std::size_t> expected;
  keys.reserve(THREADS);
  expected.reserve(THREADS);
  for (int t = 0; t < THREADS; t++) {
    std::mt19937_64 rng(static_cast<uint64_t>(t));
    std::vector<int64_t> host(ROWS);
    for (auto& value : host) {
      value = static_cast<int64_t>(rng() % KEY_SPACE);
    }
    expected.push_back(std::set<int64_t>(host.begin(), host.end()).size());
    keys.emplace_back(host.begin(), host.end());
  }

  std::atomic<int> mismatches{0};
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(THREADS);
  for (int t = 0; t < THREADS; t++) {
    workers.emplace_back([&, t] {
      rmm::cuda_stream owned;
      cuda::stream_ref const stream = owned;
      std::vector<cudf::column_view> const key_columns{keys[static_cast<std::size_t>(t)]};
      cudf::column_view const no_mask;
      for (int i = 0; i < ITERATIONS; i++) {
        try {
          auto const [representatives, group_indices] =
            gqe::perfect_hash::unique_indices(key_columns, no_mask, stream);
          if (representatives.size() != expected[static_cast<std::size_t>(t)]) { mismatches++; }
        } catch (std::exception const&) {
          // The capacity guard throwing is the failure this test exists to catch.
          failures++;
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }

  EXPECT_EQ(mismatches.load(), 0) << "out of " << THREADS * ITERATIONS << " builds";
  EXPECT_EQ(failures.load(), 0) << "out of " << THREADS * ITERATIONS << " builds";
}

}  // namespace
