/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/compiler/Conversion/RelAlgToSCF/RelAlgToSCF.hpp>

namespace gqe {
namespace compiler {

// clang-format off
/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include <gqe/compiler/Conversion/Passes.h.inc>
// clang-format on

}  // namespace compiler
}  // namespace gqe
