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

#include <cxx_gqe/physical.hpp>

#include <memory>

namespace cxx_gqe {

std::shared_ptr<physical_relation> physical_plan_builder::build(
  logical_relation const& logical_relation)
{
  return _builder.build(&logical_relation);
}

std::unique_ptr<physical_plan_builder> new_physical_plan_builder(catalog& catalog)
{
  return std::make_unique<physical_plan_builder>(
    std::move(gqe::physical_plan_builder(&catalog.get())));
}

}  // namespace cxx_gqe
