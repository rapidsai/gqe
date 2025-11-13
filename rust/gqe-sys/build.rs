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

use std::env;

static GQE_PATH: &str = "../..";

fn main() {
    // Get Cargo's environment variables.
    let pkg_name = env::var("CARGO_PKG_NAME").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();

    // Generate C++ helper files that wrap some Rust std types.
    let _ = cxx_build::bridge("src/lib.rs");

    // Build the C++ part of the bindings with cmake. This also builds GQE.
    let mut cmake = cmake::Config::new(GQE_PATH);
    cmake.build_target("gqe-cxx");
    cmake.define("CXX_GQE_BINDINGS_DIR", format!("{out_dir}/cxxbridge")); // Set a cmake variable with the path to the C++ helper files.
    let destination = cmake.build();

    // Watch these files and rerun build.rs if they change.
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=src/lib.rs");

    // Link the C++ part of the bindings.
    println!(
        "cargo:rustc-link-search=native={}/build/rust/{}",
        destination.display(),
        pkg_name
    );
    println!("cargo:rustc-link-lib=static=gqe-cxx");

    // Link GQE and Substrait.
    println!(
        "cargo:rustc-link-search=native={}/build/src",
        destination.display()
    );
    println!("cargo:rustc-link-lib=static=gqe");
    println!("cargo:rustc-link-lib=static=substrait");

    // Set library search paths.
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        println!("cargo:rustc-link-search=native={}/lib", conda_prefix);
    }
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    // Link CUDA RT.
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Link system libraries.
    //
    // These could potentially be linked with cmake via the Corrosion tool in
    // future. However, Corrosion's `corrosion_link_libraries` doesn't link
    // transitive dependencies into the final Rust binary. This is problematic,
    // because the `gqe.a` static library depends on, e.g., `cudf.so`.
    //
    // See: https://corrosion-rs.github.io/corrosion/
    // See: https://github.com/micahsnyder/cmake-rust-demo
    println!("cargo:rustc-link-lib=dylib=cudf");
    println!("cargo:rustc-link-lib=dylib=protobuf");
    println!("cargo:rustc-link-lib=dylib=spdlog");
    println!("cargo:rustc-link-lib=dylib=numa");

    // Link NVcomp.
    //
    // Needs to be linked after spdlog, as otherwise GQE pulls in the spdlog symbols from nvcomp.
    println!("cargo:rustc-link-lib=dylib=nvcomp");

    // Link Conda-specific system libraries
    if env::var("CONDA_PREFIX").is_ok() {
        println!("cargo:rustc-link-lib=dylib=absl_log_internal_check_op");
        println!("cargo:rustc-link-lib=dylib=absl_log_internal_message");
    }
}
