/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <array>

namespace gqe_test {

/**
 * @brief TPC-H query numbers supported by the SQL integration tests.
 *
 * Q15 is excluded from this set and from `test/end_to_end/` because its results are flaky.
 */
inline constexpr auto kTpchSqlQueries =
  std::to_array({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22});

/**
 * @brief TPC-H query numbers supported by the physical-plan integration tests.
 *
 * Q8 and Q14 are excluded because no plan-builder fixture is available. Q15 is excluded because
 * its results are flaky, matching @ref kTpchSqlQueries. Q11 is included because its `.pb` fixture
 * has the scale-factor-substituted threshold baked in.
 */
inline constexpr auto kTpchPhysicalPlanQueries =
  std::to_array({1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22});

}  // namespace gqe_test
