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

#pragma once

#include <gqe/catalog.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/optimizer/estimator.hpp>
#include <gqe/physical/relation.hpp>

#include <memory>
#include <unordered_map>

namespace gqe {

/**
 * @brief A builder for generating the physical plan from the logical plan.
 */
class physical_plan_builder {
 public:
  /**
   * @brief Construct a physical plan builder.
   *
   * @param[in] cat Catalog containing table metadata.
   */
  physical_plan_builder(catalog const* cat) : _estimator(cat) {}

  /**
   * @brief Generate a physical plan from a logical plan.
   *
   * @param[in] logical_plan Root relation of the logical plan.
   *
   * @return Generated physical plan.
   */
  std::shared_ptr<physical::relation> build(logical::relation const* logical_plan);

 private:
  // A cache for keeping track of the previously-generated physical relations. Without the cache,
  // the builder can only generate a tree instead of a DAG.
  std::unordered_map<logical::relation const*, std::weak_ptr<physical::relation>> _cache;
  estimator _estimator;
};

}  // namespace gqe
