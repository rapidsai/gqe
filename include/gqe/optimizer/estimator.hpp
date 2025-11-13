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
