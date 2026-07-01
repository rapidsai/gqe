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

#include <gqe/rpc/serialization/uuid.hpp>

#include <cstddef>
#include <format>
#include <span>
#include <stdexcept>

namespace gqe::rpc {

proto::Uuid serialize_uuid(utility::uuid const& id)
{
  auto bytes = id.bytes();
  proto::Uuid proto;
  proto.set_value(bytes.data(), bytes.size());
  return proto;
}

utility::uuid deserialize_uuid(proto::Uuid const& proto)
{
  auto const& bytes = proto.value();
  if (bytes.size() != 16) {
    throw std::invalid_argument(std::format("Expected 16-byte UUID, got {} bytes", bytes.size()));
  }
  return utility::uuid::from_bytes(
    std::span<std::byte const, 16>{reinterpret_cast<std::byte const*>(bytes.data()), 16});
}

}  // namespace gqe::rpc
