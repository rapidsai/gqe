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

add_mlir_dialect(RelAlgOps relalg)
add_mlir_doc(RelAlgOps RelAlgOps Dialects/ -gen-op-doc -dialect=relalg)

set(LLVM_TARGET_DEFINITIONS RelAlgOps.td)
mlir_tablegen(RelAlgOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(RelAlgOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(GQECompilerRelAlgOpsEnumsGen)

set(LLVM_TARGET_DEFINITIONS RelAlgOps.td)
mlir_tablegen(RelAlgOpsAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=relalg)
mlir_tablegen(RelAlgOpsAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=relalg)
add_public_tablegen_target(GQECompilerRelAlgOpsAttrsIncGen)
