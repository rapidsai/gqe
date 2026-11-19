#!/bin/bash

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

set -euo pipefail

# Record the host IDs before the image build. Remote-SSH does not export UID/GID
# shell variables, so the Dockerfile reads these values from the generated file.
dev_uid="$(id -u)"
dev_gid="$(id -g)"
output_path="${BASH_SOURCE[0]%/*}/devcontainer-user-ids"

printf 'DEV_UID=%s\nDEV_GID=%s\n' "$dev_uid" "$dev_gid" > "$output_path"

echo "Configured dev container user IDs as $dev_uid:$dev_gid"
