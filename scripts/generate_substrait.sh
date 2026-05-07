#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Substrait C++ code generation
substrait_in_dir=$1
substrait_out_dir=$2
if [ -d ${substrait_out_dir}/substrait ]
then
    echo "Directory ${substrait_out_dir}/substrait exists"
    echo "Substrait C++ code will NOT be re-generated"
else
    echo "Directory ${substrait_out_dir}/substrait does not exist"
    echo "Generating Substrait C++ code ..."
    echo "Assuming substrait submodule is already initialized"
    cd ${substrait_in_dir}/proto
    protoc substrait/*.proto substrait/extensions/*.proto --cpp_out=${substrait_out_dir}
    echo "Done generating Substrait C++ code"
fi
