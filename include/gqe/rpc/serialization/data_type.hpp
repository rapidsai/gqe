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

#include <cudf/types.hpp>
#include <proto/data_type.pb.h>

namespace gqe::rpc {

/**
 * @brief Serialize a cudf data type to its protobuf representation.
 */
[[nodiscard]] proto::DataType serialize_data_type(cudf::data_type dt);

/**
 * @brief Deserialize a protobuf data type to a cudf data type.
 */
[[nodiscard]] cudf::data_type deserialize_data_type(proto::DataType const& dt);

}  // namespace gqe::rpc
