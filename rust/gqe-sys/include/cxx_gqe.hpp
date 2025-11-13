/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * This is the FFI header imported by the Rust cxxbridge. It must include all
 * APIs for which bindings are generated.
 *
 * The design of the `cxx_gqe` bindings is described in the Rust documentation
 * of this `gqe-sys` crate.
 */

#include <cxx_gqe/api.hpp>
#include <cxx_gqe/executor.hpp>
#include <cxx_gqe/logical.hpp>
#include <cxx_gqe/physical.hpp>
#include <cxx_gqe/query_context.hpp>
#include <cxx_gqe/storage.hpp>
#include <cxx_gqe/task_manager_context.hpp>
#include <cxx_gqe/types.hpp>
