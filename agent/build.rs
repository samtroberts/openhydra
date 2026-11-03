// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! With the built-in Mac engine (`--features mlx-swift`), the agent's executables need the system
//! Swift runtime on their run path: Swift's libraries are found through `@rpath`, and a dependency's
//! build script can't add link arguments to the final binaries. Nothing happens in other builds.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let mlx = std::env::var_os("CARGO_FEATURE_MLX_SWIFT").is_some();
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    if mlx && macos {
        // Every linked target: the binaries and the unit-test harness.
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
