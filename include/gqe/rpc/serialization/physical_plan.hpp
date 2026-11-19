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

#include <gqe/physical/plan.hpp>

#include <proto/physical_plan.pb.h>

namespace gqe::rpc {

/**
 * @brief Serialize an identified physical relation tree to protobuf, preserving its node IDs.
 *
 * @throws std::logic_error If a reachable relation does not have a node ID.
 */
[[nodiscard]] proto::PhysicalRelation serialize_physical_plan(
  physical::identified_plan const& plan);

/**
 * @brief Deserialize a protobuf to a physical plan and retain its relation node IDs.
 *
 * Storage descriptors in read/write relations are deserialized to
 * storage::descriptor values that can be passed to a storage::table_provider.
 *
 * @throws std::logic_error If a reachable relation has a non-positive or duplicate node ID, or if
 * a `ReferenceRel` refers to an unknown node ID.
 */
[[nodiscard]] physical::identified_plan deserialize_physical_plan(
  proto::PhysicalRelation const& proto);

}  // namespace gqe::rpc
