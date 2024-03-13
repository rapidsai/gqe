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

    // Link Conda-specific system libraries
    if env::var("CONDA_PREFIX").is_ok() {
        println!("cargo:rustc-link-lib=dylib=absl_log_internal_check_op");
        println!("cargo:rustc-link-lib=dylib=absl_log_internal_message");
    }
}
