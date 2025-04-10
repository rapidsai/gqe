/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

class TaskGraphGenerationTest : public ::testing::Test {
 protected:
  class DummyTask : public gqe::task {
   public:
    DummyTask(int32_t task_id,
              int32_t stage_id,
              std::vector<std::shared_ptr<gqe::task>> dependencies,
              std::vector<std::shared_ptr<gqe::task>> subquery_tasks)
      : gqe::task(gqe::context_reference{}, task_id, stage_id, dependencies, subquery_tasks)
    {
    }
    void execute() override { throw std::runtime_error("Not implemented"); }
  };
};

TEST_F(TaskGraphGenerationTest, AssignPipelineIds)
{
  // Dummy task graph
  // (task1, stage 0) <- (task2, stage 1) <- (task3, stage 1)
  //                     (task2, stage 1) <- (task4, stage1)

  auto task1 = std::make_shared<DummyTask>(
    1, 0, std::vector<std::shared_ptr<gqe::task>>{}, std::vector<std::shared_ptr<gqe::task>>{});

  auto task2 = std::make_shared<DummyTask>(2,
                                           1,
                                           std::vector<std::shared_ptr<gqe::task>>{},
                                           std::vector<std::shared_ptr<gqe::task>>{task1});

  auto task3 = std::make_shared<DummyTask>(3,
                                           1,
                                           std::vector<std::shared_ptr<gqe::task>>{task2},
                                           std::vector<std::shared_ptr<gqe::task>>{});

  auto task4 = std::make_shared<DummyTask>(4,
                                           1,
                                           std::vector<std::shared_ptr<gqe::task>>{task2},
                                           std::vector<std::shared_ptr<gqe::task>>{});

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
