/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once
#include <gqe/executor/task.hpp>
#include <unordered_set>

namespace gqe {

class scheduler {
 public:
  virtual std::unordered_set<int32_t> get_execution_ranks(task* t) = 0;
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