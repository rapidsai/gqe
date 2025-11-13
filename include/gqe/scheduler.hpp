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

#pragma once
#include <gqe/executor/task.hpp>
#include <unordered_set>

namespace gqe {

enum SCHEDULER_TYPE { ALL_TO_ALL, ROUND_ROBIN };

class scheduler {
 public:
  virtual std::unordered_set<int32_t> get_execution_ranks(task* t) = 0;
};

/**
 * @brief Scheduler that assigns all tasks to all ranks.
 */
class all_to_all_scheduler : public scheduler {
 public:
  all_to_all_scheduler(int32_t num_ranks) : _num_ranks(num_ranks) {}
  std::unordered_set<int32_t> get_execution_ranks(task* t) override;

 private:
  int32_t _num_ranks;
};

class round_robin_scheduler : public scheduler {
 public:
  round_robin_scheduler(int32_t num_ranks) : _num_ranks(num_ranks) {}
  std::unordered_set<int32_t> get_execution_ranks(task* t) override;

 private:
  int32_t _num_ranks;
};

class explicit_scheduler : public scheduler {
 public:
  std::unordered_set<int32_t> get_execution_ranks(task* t) override;
  void set_execution_ranks(task* t, std::unordered_set<int32_t> execution_ranks);

 private:
  std::unordered_map<task*, std::unordered_set<int32_t>> _execution_ranks;
};

}  // namespace gqe