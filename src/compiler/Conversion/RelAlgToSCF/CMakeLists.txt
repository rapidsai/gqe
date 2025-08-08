# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

add_mlir_conversion_library(GQECompilerRelAlgToSCF
  DeclarativeConversion.cpp
  NamedIUTuple.cpp
  RelAlgToSCF.cpp

  CudfTableScan.cpp
  Filter.cpp
  GetIU.cpp
  Map.cpp
  ScalarAggregate.cpp
  Yield.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/gqe/compiler/Conversion/RelAlgToSCF

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
)
