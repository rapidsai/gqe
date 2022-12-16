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
#include <gqe/logical/relation.hpp>

#include <cstdint>
#include <unordered_map>

namespace gqe {

/**
 * @brief An estimator used to approximate table statistics for a logical relation.
 *
 * Note that an estimator caches the table statistics internally. If statistics need to be
 * recomputed, a new estimator needs to be constructed.
 */
class estimator {
 public:
  /**
   * @brief Construct a new estimator.
   *
   * @param[in] cat Catalog containing table metadata.
   */
  estimator(catalog const* cat) : _catalog(cat) {}

  /**
   * @brief Return the estimated statistics of the output of a logical relation.
   */
  table_statistics operator()(logical::relation const* input_relation) const;

 private:
  mutable std::unordered_map<logical::relation const*, table_statistics> _cache;
  catalog const* _catalog;
};

}  // namespace gqe
