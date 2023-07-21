/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <memory>

namespace gqe {

class write_task_base : public task {
 public:
  write_task_base(int32_t task_id, int32_t stage_id, std::shared_ptr<task> input);

  write_task_base(const write_task_base&) = delete;
  write_task_base& operator=(const write_task_base&) = delete;
};

}  // namespace gqe
