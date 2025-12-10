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

#include "utilities.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/project.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

class TaskGraphGenerationTest : public ::testing::Test {
 protected:
  TaskGraphGenerationTest()
    : params(true),
      task_manager_ctx(params),
      query_ctx(params),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }

  gqe::optimization_parameters params;
  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
};

TEST_F(TaskGraphGenerationTest, AssignPipelineIds)
{
  // Dummy task graph
  // (task1, stage 0) <- (task2, stage 1) <- (task3, stage 1)
  //                     (task2, stage 1) <- (task4, stage1)

  auto task1 =
    std::make_shared<gqe::test::executed_task>(ctx_ref, 1, 0, std::make_unique<cudf::table>());

  auto task2 = std::make_shared<gqe::test::executed_task>(
    ctx_ref, 2, 1, std::make_unique<cudf::table>(), std::vector<std::shared_ptr<gqe::task>>{task1});

  auto task3 = std::make_shared<gqe::test::executed_task>(
    ctx_ref, 3, 1, std::make_unique<cudf::table>(), std::vector<std::shared_ptr<gqe::task>>{task2});

  auto task4 = std::make_shared<gqe::test::executed_task>(
    ctx_ref, 4, 1, std::make_unique<cudf::table>(), std::vector<std::shared_ptr<gqe::task>>{task2});

  // pipelines are assinged starting at root tasks for a stage
  task1->assign_pipeline(0);

  // During actual execution, pipeline ids are assinged starting from 0 for a given stage. Here we
  // have them unique across stages for testing purposes
  task3->assign_pipeline(1);
  task4->assign_pipeline(2);

  // Check pipeline assignments are propagated correctly, i.e dont cross stage boundaries
  auto task1_pipelines = task1->pipeline_ids();
  ASSERT_EQ(task1_pipelines.size(), 1);
  ASSERT_TRUE(task1_pipelines.find(0) != task1_pipelines.end());

  // task2 belongs to both pipelines 1 and 2
  auto task2_pipelines = task2->pipeline_ids();
  ASSERT_EQ(task2_pipelines.size(), 2);
  ASSERT_TRUE(task2_pipelines.find(1) != task2_pipelines.end());
  ASSERT_TRUE(task2_pipelines.find(2) != task2_pipelines.end());

  // task3 belongs to pipeline 1
  auto task3_pipelines = task3->pipeline_ids();
  ASSERT_EQ(task3_pipelines.size(), 1);
  ASSERT_TRUE(task3_pipelines.find(1) != task3_pipelines.end());

  // task4 belongs to pipeline 2
  auto task4_pipelines = task4->pipeline_ids();
  ASSERT_EQ(task4_pipelines.size(), 1);
  ASSERT_TRUE(task4_pipelines.find(2) != task4_pipelines.end());
}
