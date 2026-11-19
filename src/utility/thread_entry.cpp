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

#include <gqe/utility/thread_entry.hpp>

#include <gqe/device_properties.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/linux.hpp>

#include <cuda_runtime_api.h>

namespace gqe::utility {

void enter_device_thread(rmm::cuda_device_id device)
{
  GQE_CUDA_TRY(cudaSetDevice(device.value()));
  set_thread_affinity(
    device_properties::instance().get<device_properties::property::cpuAffinity>(device));
}

}  // namespace gqe::utility
