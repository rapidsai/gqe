/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
