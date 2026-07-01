/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/physical/relation.hpp>

#include <proto/physical_plan.pb.h>

#include <memory>

namespace gqe::rpc {

/** @brief Serialize a physical relation tree to protobuf. */
[[nodiscard]] proto::PhysicalRelation serialize_physical_plan(physical::relation const* plan);

/**
 * @brief Deserialize a protobuf to a physical relation tree.
 *
 * Storage descriptors in read/write relations are deserialized to
 * storage::descriptor values that can be passed to a storage::table_provider.
 */
[[nodiscard]] std::shared_ptr<physical::relation> deserialize_physical_plan(
  proto::PhysicalRelation const& proto);

}  // namespace gqe::rpc
