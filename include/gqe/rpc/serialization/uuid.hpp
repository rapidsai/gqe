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

#include <gqe/utility/uuid.hpp>

#include <proto/uuid.pb.h>

namespace gqe::rpc {

/**
 * @brief Serialize a UUID to a protobuf Uuid message.
 */
[[nodiscard]] proto::Uuid serialize_uuid(utility::uuid const& id);

/**
 * @brief Deserialize a UUID from a protobuf Uuid message.
 * @throws std::invalid_argument if the value is not exactly 16 bytes.
 */
[[nodiscard]] utility::uuid deserialize_uuid(proto::Uuid const& proto);

}  // namespace gqe::rpc
