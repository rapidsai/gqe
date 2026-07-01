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

#include <gqe/rpc/serialization/data_type.hpp>

namespace gqe::rpc {

proto::DataType serialize_data_type(cudf::data_type dt)
{
  proto::DataType out;
  out.set_type_id(static_cast<::proto::DataType>(static_cast<int>(dt.id())));
  if (dt.id() == cudf::type_id::DECIMAL32 || dt.id() == cudf::type_id::DECIMAL64 ||
      dt.id() == cudf::type_id::DECIMAL128) {
    out.set_scale(dt.scale());
  }
  return out;
}

cudf::data_type deserialize_data_type(proto::DataType const& dt)
{
  auto type_id = static_cast<cudf::type_id>(static_cast<int>(dt.type_id()));
  if (dt.has_scale()) { return cudf::data_type{type_id, dt.scale()}; }
  return cudf::data_type{type_id};
}

}  // namespace gqe::rpc
