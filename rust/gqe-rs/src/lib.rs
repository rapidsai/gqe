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

//! Idiomatic Rust bindings for GQE.
//!
//! `gqe-rs` wraps the raw C++ API exposed by [`gqe_sys`] into an easy-to-use
//! Rust-style API. For example:
//!
//! - C++ wrapper types such as [`cxx::UniquePtr`] and [`cxx::Vector`] are
//! hidden behind opaque Rust types so that users don't need to import the
//! [`cxx`] crate.
//!
//! - The [`gqe_sys`] must convert C++ `std::variant` types to `enum` + `union`
//! + `struct` combinations in its FFI. These are converted to Rust `enum`
//! types.
//!
//! The overall purpose is to reduce code complexity and duplication for tools
//! building on top of the Rust GQE bindings.

pub mod api;
pub mod error;
pub mod executor;
pub mod logical;
pub mod physical;
pub mod task_manager_context;
pub mod query_context;
pub mod storage;
pub mod utility;
