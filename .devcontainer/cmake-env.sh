#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sourced by VS Code CMake Tools via the `cmake.environmentSetupScript` setting
# before every configure/build. It activates the conda `gqe` environment so the
# GUI build uses the exact same toolchain as a terminal build: conda's gcc 14.3.0
# and nvcc with a matching `-ccbin` (NVCC_PREPEND_FLAGS), plus CXXFLAGS,
# CUDAToolkit paths, and the ar/ranlib/linker wrappers.
#
# Without this, CMake Tools configures outside the env and nvcc falls back to a
# gcc-13 host frontend while including conda's 14.3.0 libstdc++ headers, which
# fails with errors like "__type_pack_element is not a template".
source /conda/bin/activate gqe
