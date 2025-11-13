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

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>

#include <cstdint>
#include <memory>

namespace gqe {

class write_task_base : public task {
 public:
  write_task_base(context_reference ctx_ref,
                  int32_t task_id,
                  int32_t stage_id,
                  std::shared_ptr<task> input);

  write_task_base(const write_task_base&)            = delete;
  write_task_base& operator=(const write_task_base&) = delete;
};

}  // namespace gqe
