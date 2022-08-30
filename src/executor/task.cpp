/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/task.hpp>

#include <stdexcept>

namespace gqe {

task::task(std::vector<std::shared_ptr<task>> dependencies, int32_t task_id, int32_t stage_id)
  : _result_status(result_status_type::not_available),
    _dependencies(std::move(dependencies)),
    _task_id(task_id),
    _stage_id(stage_id)
{
}

void task::migrate() { throw std::logic_error("tash::migrate() has not been implemented"); }

void task::update_result_cache(std::unique_ptr<cudf::table> new_result)
{
  _result_cache = std::move(new_result);
  _result       = _result_cache->view();
}

}  // namespace gqe
