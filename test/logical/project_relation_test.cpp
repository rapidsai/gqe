/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>

#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace {

/**
 * @brief Count schema requests at a leaf without reading table data.
 */
class counted_read_relation : public gqe::logical::read_relation {
 public:
  counted_read_relation()
    : read_relation({},
                    {"a", "b", "c", "d"},
                    {cudf::data_type{cudf::type_id::INT32},
                     cudf::data_type{cudf::type_id::FLOAT64},
                     cudf::data_type{cudf::type_id::STRING},
                     cudf::data_type{cudf::type_id::BOOL8}},
                    "t",
                    nullptr)
  {
  }

  [[nodiscard]] std::vector<cudf::data_type> data_types() const override
  {
    ++_schema_calls;
    return read_relation::data_types();
  }

  [[nodiscard]] std::size_t schema_calls() const noexcept { return _schema_calls; }

 private:
  mutable std::size_t _schema_calls{};
};

}  // namespace

/**
 * @brief Nested Projects query the leaf once per root schema request, without a persistent cache.
 */
TEST(ProjectRelationTest, NestedProjectsQueryLeafOnce)
{
  auto input                                   = std::make_shared<counted_read_relation>();
  std::shared_ptr<gqe::logical::relation> root = input;
  for (int depth = 0; depth < 6; ++depth) {
    std::vector<std::unique_ptr<gqe::expression>> outputs;
    for (cudf::size_type i = 0; i < 4; ++i) {
      outputs.push_back(std::make_unique<gqe::column_reference_expression>(i));
    }
    root = std::make_shared<gqe::logical::project_relation>(
      root, std::vector<std::shared_ptr<gqe::logical::relation>>{}, std::move(outputs));
  }
  std::vector<cudf::data_type> const expected{cudf::data_type{cudf::type_id::INT32},
                                              cudf::data_type{cudf::type_id::FLOAT64},
                                              cudf::data_type{cudf::type_id::STRING},
                                              cudf::data_type{cudf::type_id::BOOL8}};

  EXPECT_EQ(root->data_types(), expected);
  EXPECT_EQ(input->schema_calls(), 1u);
  EXPECT_EQ(root->data_types(), expected);
  EXPECT_EQ(input->schema_calls(), 2u);
}
