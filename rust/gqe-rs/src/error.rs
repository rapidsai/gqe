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

//! The GQE error and result types.

use thiserror::Error;

/// The GQE result type.
pub type Result<T> = std::result::Result<T, Error>;

/// The GQE error type.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("GQE has thrown an internal exception: {0}")]
    Internal(#[from] cxx::Exception),
    #[error("Could not convert string to UTF8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("Could not parse string to integer: {0}")]
    ParseInt(#[from] std::num::ParseIntError),
    #[error("Could not parse string to boolean: {0}")]
    ParseBool(#[from] std::str::ParseBoolError),
}
