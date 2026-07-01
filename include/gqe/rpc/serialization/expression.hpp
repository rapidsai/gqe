/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/expression/expression.hpp>

#include <proto/expression.pb.h>

#include <memory>

namespace gqe::rpc {

/**
 * @brief Serialize an expression tree to its protobuf representation.
 */
[[nodiscard]] proto::Expression serialize_expression(expression const* expr);

/**
 * @brief Deserialize a protobuf expression to a unique_ptr expression tree.
 */
[[nodiscard]] std::unique_ptr<expression> deserialize_expression(proto::Expression const& proto);

}  // namespace gqe::rpc
