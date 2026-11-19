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

#pragma once

#include <rmm/cuda_device.hpp>

/**
 * @file
 * @brief Per-thread setup for threads that do GPU work.
 *
 * # Design
 *
 * The CUDA current device and the CPU affinity are both per-thread state. A thread that does GPU
 * work must select its device and pin itself to the CPUs local to that device before its first
 * CUDA call. Every such thread calls `enter_device_thread` on entry, so this setup lives in one
 * place instead of being repeated inline at each thread's start.
 *
 * The caller passes the device explicitly. A process decides its device once and hands the same
 * value to every thread it starts. Deriving the device again on each thread, for example from a
 * rank, lets threads of one process disagree on their device.
 *
 * A process calls `capture_process_affinity` at startup, before any thread is pinned. Pinned
 * threads widen their affinity back to that mask for CPU-heavy work, such as CPU compression in
 * `scoped_cpu_affinity`.
 *
 * ## Thread Safety
 *
 * `enter_device_thread` is safe to call concurrently. Apart from the calling thread's device and
 * affinity, its first call in a process also initializes the CUDA primary context and the cached
 * `device_properties`.
 */

namespace gqe::utility {

/**
 * @brief Set up the calling thread to do GPU work on `device`.
 *
 * Sets `device` as the thread's current CUDA device, then pins the thread to the CPUs local to
 * `device`.
 *
 * @param device The device the calling thread does GPU work on.
 *
 * @throw cuda_error if setting the device fails.
 * @throw std::runtime_error if the device's CPU affinity cannot be queried from NVML.
 */
void enter_device_thread(rmm::cuda_device_id device);

}  // namespace gqe::utility
