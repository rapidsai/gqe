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
