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

#include <gtest/gtest.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <cstddef>
#include <numeric>
#include <vector>

namespace {

TEST(TbbIntegrationTest, ParallelForRunsWork)
{
  constexpr std::size_t num_values = 1024;
  std::vector<int> values(num_values, 0);

  tbb::parallel_for(tbb::blocked_range<std::size_t>{0, values.size()},
                    [&values](tbb::blocked_range<std::size_t> const& range) {
                      for (auto i = range.begin(); i != range.end(); ++i) {
                        values[i] = static_cast<int>(i + 1);
                      }
                    });

  auto const sum = std::accumulate(values.begin(), values.end(), 0);
  EXPECT_EQ(sum, static_cast<int>(num_values * (num_values + 1) / 2));
}

}  // namespace
