/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cxx_gqe/api.hpp>
#include <cxx_gqe/logical.hpp>

#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/relation.hpp>

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

  physical_plan_builder()                             = delete;
  physical_plan_builder(physical_plan_builder const&) = delete;
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
