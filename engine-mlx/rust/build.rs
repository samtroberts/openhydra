// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Links the Swift engine that `engine-mlx/build.sh` built: packs its object files into one static
//! library, points the linker at the system Swift runtime, and pins the Metal kernel library's SHA-256
//! into the binary (`OHMLX_METALLIB_SHA256`), refusing a build whose metallib differs from the committed
//! `engine-mlx/METALLIB_SHA256` unless told otherwise. Putting `mlx.metallib` beside an executable is a separate,
//! explicit step (`engine-mlx/stage.sh`); this script writes nothing outside `OUT_DIR`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_os != "macos" || target_arch != "aarch64" {
        panic!("openhydra-mlx-engine builds only for Apple Silicon macOS (target {target_arch}-{target_os})");
    }
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let products = std::env::var_os("OHMLX_PRODUCTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../.xcbuild/Build/Products/Release"));
    let metallib = products.join("mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib");
    if !products.join("OHMLX.o").exists() || !metallib.exists() {
        panic!(
            "openhydra-mlx-engine: no engine build in {} — run engine-mlx/build.sh first (or set OHMLX_PRODUCTS)",
            products.display()
        );
    }
    let expected_file = manifest_dir.join("../METALLIB_SHA256");
    println!("cargo:rerun-if-env-changed=OHMLX_PRODUCTS");
    println!("cargo:rerun-if-changed={}", products.join("OHMLX.o").display());
    println!("cargo:rerun-if-changed={}", metallib.display());
    println!("cargo:rerun-if-changed={}", expected_file.display());

    // One archive from every module object xcodebuild produced (the engine and its Swift dependencies).
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let mut objects: Vec<PathBuf> = std::fs::read_dir(&products)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "o"))
        .collect();
    objects.sort();
    let archive = out.join("libohmlx.a");
    let _ = std::fs::remove_file(&archive);
    run(Command::new("libtool").arg("-static").arg("-o").arg(&archive).args(&objects));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=ohmlx");

    // The Swift runtime ships with macOS (/usr/lib/swift); the toolchain's directory supplies the
    // back-deployment shims that the objects' autolink entries name. Executables also need
    // `-rpath /usr/lib/swift`, which only the final package can add (see agent/build.rs).
    let toolchain_swift = capture(Command::new("xcrun").args(["--toolchain", "default", "--find", "swift"]));
    let toolchain_lib = Path::new(toolchain_swift.trim()).parent().unwrap().join("../lib/swift/macosx");
    let sdk = capture(Command::new("xcrun").args(["--sdk", "macosx", "--show-sdk-path"]));
    println!("cargo:rustc-link-search=native={}", toolchain_lib.display());
    println!("cargo:rustc-link-search=native={}/usr/lib/swift", sdk.trim());
    println!("cargo:rustc-link-search=native=/usr/lib/swift");
    // This package's own executables (the smoke example, its tests).
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    for framework in ["Foundation", "Metal", "MetalPerformanceShaders", "Accelerate", "CoreImage", "AVFoundation", "Security"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    println!("cargo:rustc-link-lib=dylib=c++");
    // Clang's runtime, which clang links for every C++ program and rustc doesn't: MLX's `@available`
    // checks call `__isPlatformVersionAtLeast`. Rust's std has a weak copy that LTO drops, so a release
    // build fails to link without this.
    let clang_resources = capture(Command::new("xcrun").args(["clang", "-print-resource-dir"]));
    println!("cargo:rustc-link-search=native={}/lib/darwin", clang_resources.trim());
    println!("cargo:rustc-link-lib=static=clang_rt.osx");

    // Pin the AOT kernel library (the JIT kernels ship as sources in the binary): the agent refuses an
    // engine whose metallib hashes differently.
    let digest = capture(Command::new("shasum").args(["-a", "256"]).arg(&metallib));
    let digest = digest.split_whitespace().next().unwrap_or_default().to_string();
    assert_eq!(digest.len(), 64, "openhydra-mlx-engine: could not hash {}", metallib.display());
    // The committed digest is what the pinned mlx-swift builds to with the recorded Xcode. A different one
    // means another toolchain (or an altered build tree): allowed for development, never for a release.
    let expected = std::fs::read_to_string(&expected_file).unwrap_or_default();
    let expected = expected.split_whitespace().next().unwrap_or_default();
    println!("cargo:rerun-if-env-changed=OHMLX_ALLOW_UNPINNED_METALLIB");
    if expected != digest {
        if std::env::var_os("OHMLX_ALLOW_UNPINNED_METALLIB").is_some_and(|v| v == "1") {
            println!(
                "cargo:warning=openhydra-mlx-engine: default.metallib is {digest}, not the committed {expected} \
                 (allowed by OHMLX_ALLOW_UNPINNED_METALLIB=1: development only, never a release)"
            );
        } else {
            panic!(
                "openhydra-mlx-engine: default.metallib is {digest}, not the committed {expected} \
                 (engine-mlx/METALLIB_SHA256) — a different Xcode or an altered build tree. For development \
                 only, set OHMLX_ALLOW_UNPINNED_METALLIB=1"
            );
        }
    }
    println!("cargo:rustc-env=OHMLX_METALLIB_SHA256={digest}");
    println!("cargo:rustc-env=OHMLX_METALLIB_PATH={}", metallib.display());
}

fn run(cmd: &mut Command) {
    let status = cmd.status().unwrap_or_else(|e| panic!("openhydra-mlx-engine: {cmd:?}: {e}"));
    assert!(status.success(), "openhydra-mlx-engine: {cmd:?} failed");
}

fn capture(cmd: &mut Command) -> String {
    let out = cmd.output().unwrap_or_else(|e| panic!("openhydra-mlx-engine: {cmd:?}: {e}"));
    assert!(out.status.success(), "openhydra-mlx-engine: {cmd:?} failed");
    String::from_utf8_lossy(&out.stdout).into_owned()
}
