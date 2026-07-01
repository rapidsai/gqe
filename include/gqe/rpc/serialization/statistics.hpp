/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/types.hpp>

#include <proto/statistics.pb.h>

namespace gqe::rpc {

/**
 * @brief Serialize a C++ table_statistics to a proto TableStatistics.
 */
[[nodiscard]] proto::TableStatistics serialize_table_statistics(table_statistics const& stats);

/**
 * @brief Deserialize a proto TableStatistics to a C++ table_statistics.
 */
[[nodiscard]] table_statistics deserialize_table_statistics(proto::TableStatistics const& proto);

}  // namespace gqe::rpc
