//===- RelAlgTraits.td - Relational Algebra dialect traits -*- tablegen -*-===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
//===----------------------------------------------------------------------===//

#ifndef RELALG_TRAITS
#define RELALG_TRAITS

/// This class provides verification for ops that consume tuple streams. At
/// least one operand must be of type `TupleStreamType`.
def TupleStreamConsumer : RelAlg_Trait<"TupleStreamConsumer">;

#endif // RELALG_TRAITS
