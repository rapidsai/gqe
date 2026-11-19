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

#include <gqe/storage/descriptor.hpp>

#include <proto/storage.pb.h>

namespace gqe::rpc {

/** @brief Convert a C++ storage::descriptor to a proto StorageDescriptor. */
[[nodiscard]] proto::StorageDescriptor serialize_storage_descriptor(
  storage::descriptor const& desc);

/** @brief Convert a proto StorageDescriptor to a C++ storage::descriptor. */
[[nodiscard]] storage::descriptor deserialize_storage_descriptor(
  proto::StorageDescriptor const& proto);

}  // namespace gqe::rpc
