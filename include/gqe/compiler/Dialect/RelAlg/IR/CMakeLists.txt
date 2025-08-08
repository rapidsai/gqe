# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
