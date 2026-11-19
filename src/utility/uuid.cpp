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

#include <gqe/utility/uuid.hpp>

#include <boost/container_hash/hash.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_hash.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <algorithm>
#include <cstring>

namespace gqe::utility {

uuid uuid::generate()
{
  static thread_local boost::uuids::random_generator gen;
  auto boost_uuid = gen();

  uuid result;
  static_assert(sizeof(result._data) == sizeof(boost_uuid));
  std::memcpy(result._data.data(), &boost_uuid, sizeof(boost_uuid));
  return result;
}

uuid uuid::from_bytes(std::span<std::byte const, 16> bytes)
{
  uuid result;
  std::copy(bytes.begin(), bytes.end(), result._data.begin());
  return result;
}

uuid uuid::from_string(std::string_view str)
{
  try {
    boost::uuids::string_generator gen;
    auto boost_uuid = gen(str.begin(), str.end());

    uuid result;
    std::memcpy(result._data.data(), &boost_uuid, sizeof(boost_uuid));
    return result;
  } catch (std::runtime_error const& e) {
    throw std::invalid_argument(e.what());
  }
}

std::span<std::byte const, 16> uuid::bytes() const { return _data; }

std::string uuid::to_string() const
{
  boost::uuids::uuid boost_uuid;
  static_assert(sizeof(_data) == sizeof(boost_uuid));
  std::memcpy(&boost_uuid, _data.data(), sizeof(boost_uuid));
  return boost::uuids::to_string(boost_uuid);
}

}  // namespace gqe::utility

auto fmt::formatter<gqe::utility::uuid>::format(gqe::utility::uuid const& u,
                                                fmt::format_context& ctx) const
  -> fmt::format_context::iterator
{
  return fmt::formatter<std::string>::format(u.to_string(), ctx);
}

std::size_t std::hash<gqe::utility::uuid>::operator()(gqe::utility::uuid const& u) const noexcept
{
  boost::uuids::uuid boost_uuid;
  std::memcpy(&boost_uuid, u.bytes().data(), sizeof(boost_uuid));
  return boost::hash<boost::uuids::uuid>{}(boost_uuid);
}
