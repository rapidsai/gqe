/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
#include <cxx_gqe/storage.hpp>
#include <cxx_gqe/types.hpp>
