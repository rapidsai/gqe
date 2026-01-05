/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/relation.hpp>

#include <cxx_gqe/api.hpp>
#include <cxx_gqe/logical.hpp>

namespace cxx_gqe {

/*
 * @brief Directly exposes physical relation as an opaque type to Rust.
 */
using physical_relation = gqe::physical::relation;

/*
 * @brief Physical plan builder wrapper.
 *
 * This class is exported to Rust as an opaque type. It exposes an FFI-compatible API of the GQE
 * physical plan builder.
 */
class physical_plan_builder {
 public:
  explicit physical_plan_builder(gqe::physical_plan_builder&& builder)
    : _builder(std::move(builder))
  {
  }

  physical_plan_builder()                                        = delete;
  physical_plan_builder(physical_plan_builder const&)            = delete;
  physical_plan_builder& operator=(physical_plan_builder const&) = delete;

  std::shared_ptr<physical_relation> build(logical_relation const& logical_relation);

 private:
  gqe::physical_plan_builder _builder;
};

/*
 * @brief Returns a new physical plan builder wrapper.
 */
std::unique_ptr<physical_plan_builder> new_physical_plan_builder(catalog& catalog);

}  // namespace cxx_gqe
