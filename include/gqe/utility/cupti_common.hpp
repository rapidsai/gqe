/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cupti_result.h>

namespace gqe {
namespace utility {

#define GQE_CUPTI_TRY(api_function_call)                                                          \
  do {                                                                                            \
    CUptiResult status = api_function_call;                                                       \
    if (status != CUPTI_SUCCESS) {                                                                \
      const char* error_string;                                                                   \
      cuptiGetResultString(status, &error_string);                                                \
                                                                                                  \
      auto error_message = std::string("CUPTI error: ") + __FILE__ + ":" +                        \
                           std::to_string(__LINE__) + ": Function " + #api_function_call +        \
                           " failed with error(" + std::to_string(status) + "): " + error_string; \
                                                                                                  \
      throw gqe::cuda_error{error_message};                                                       \
    }                                                                                             \
  } while (0)

}  // namespace utility
}  // namespace gqe
