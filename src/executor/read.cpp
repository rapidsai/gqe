/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/read.hpp>

namespace gqe {

read_task_base::read_task_base(int32_t task_id,
                               int32_t stage_id,
                               std::vector<std::shared_ptr<task>> subquery_tasks)
  : task(task_id, stage_id, {}, std::move(subquery_tasks))
{
}

}  // namespace gqe
