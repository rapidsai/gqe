/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/executor/task.hpp>
#include <gqe/scheduler.hpp>
#include <unordered_set>

namespace gqe {

std::unordered_set<int32_t> all_to_all_scheduler::get_execution_ranks(task* t)
{
  std::unordered_set<int32_t> execution_ranks;
  for (int32_t i = 0; i < _num_ranks; ++i) {
    execution_ranks.insert(i);
  }
  return execution_ranks;
}

std::unordered_set<int32_t> round_robin_scheduler::get_execution_ranks(task* t)
{
  auto pipeline_ids = t->pipeline_ids();
  assert(pipeline_ids.size() != 0);
  std::unordered_set<int32_t> execution_ranks;
  for (auto id : pipeline_ids) {
    execution_ranks.insert(id % _num_ranks);
  }
  return execution_ranks;
}

void explicit_scheduler::set_execution_ranks(task* t, std::unordered_set<int32_t> execution_ranks)
{
  _execution_ranks[t] = execution_ranks;
}

std::unordered_set<int32_t> explicit_scheduler::get_execution_ranks(task* t)
{
  if (_execution_ranks.find(t) == _execution_ranks.end()) { return std::unordered_set<int32_t>(); }
  return _execution_ranks[t];
}

}  // namespace gqe
