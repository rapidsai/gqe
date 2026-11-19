#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# The repository is bind-mounted into a container owned by the host runner UID.
# Mark that mount safe before pre-commit asks Git to inspect every tracked file.
git config --global --add safe.directory "${PWD}" || true

set +u
source /conda/bin/activate gqe
set -u

pre-commit run --all-files --show-diff-on-failure
