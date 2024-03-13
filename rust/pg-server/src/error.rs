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

use pgwire::error::{ErrorInfo, PgWireError};

pub(crate) type Result<T> = std::result::Result<T, PgWireError>;

// List of codes: https://www.postgresql.org/docs/current/errcodes-appendix.html
#[derive(Copy, Clone, Debug)]
#[allow(unused)]
pub(crate) enum PgErrorCode {
    FeatureNotSupported,
    InvalidParameterValue,
    SyntaxError,
    InvalidName,
    InvalidColumnDefinition,
    DataTypeMismatch,
    UndefinedTable,
    UndefinedParameter,
    InvalidPreparedStatementDefinition,
    SystemError,
    IoError,
    UndefinedFile,
    DuplicateFile,
    InternalError,
}

impl PgErrorCode {
    pub(crate) fn as_code(&self) -> &str {
        match self {
            PgErrorCode::FeatureNotSupported => "0A000",
            PgErrorCode::InvalidParameterValue => "22023",
            PgErrorCode::SyntaxError => "42601",
            PgErrorCode::InvalidName => "42602",
            PgErrorCode::InvalidColumnDefinition => "42611",
            PgErrorCode::DataTypeMismatch => "42804",
            PgErrorCode::UndefinedTable => "42P01",
            PgErrorCode::UndefinedParameter => "42P02",
            PgErrorCode::InvalidPreparedStatementDefinition => "42P14",
            PgErrorCode::SystemError => "58000",
            PgErrorCode::IoError => "58030",
            PgErrorCode::UndefinedFile => "58P01",
            PgErrorCode::DuplicateFile => "58P02",
            PgErrorCode::InternalError => "XX000",
        }
    }
}

// List of severity levels: https://www.postgresql.org/docs/current/protocol-error-fields.html
#[derive(Copy, Clone, Debug)]
#[allow(unused)]
pub(crate) enum PgErrorSeverity {
    Error,
    Fatal,
    Panic,
    Warning,
    Notice,
    Debug,
    Info,
    Log,
}

impl ToString for PgErrorSeverity {
    fn to_string(&self) -> String {
        match self {
            PgErrorSeverity::Error => "ERROR",
            PgErrorSeverity::Fatal => "FATAL",
            PgErrorSeverity::Panic => "PANIC",
            PgErrorSeverity::Warning => "WARNING",
            PgErrorSeverity::Notice => "NOTICE",
            PgErrorSeverity::Debug => "DEBUG",
            PgErrorSeverity::Info => "INFO",
            PgErrorSeverity::Log => "LOG",
        }
        .to_owned()
    }
}

pub(crate) fn pg_wire_usererror(
    severity: PgErrorSeverity,
    code: PgErrorCode,
    message: &str,
) -> PgWireError {
    PgWireError::UserError(Box::new(ErrorInfo::new(
        severity.to_string(),
        code.as_code().to_string(),
        message.to_string(),
    )))
}
