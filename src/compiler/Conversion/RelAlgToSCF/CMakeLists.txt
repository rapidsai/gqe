# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

add_mlir_conversion_library(
  GQECompilerRelAlgToSCF
  DeclarativeConversion.cpp
  NamedIUTuple.cpp
  RelAlgToSCF.cpp
  CudfTableMaterialize.cpp
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
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRSCFDialect)
