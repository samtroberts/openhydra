// swift-tools-version: 5.9
// OpenHydra's built-in Mac engine: Apple's MLX (mlx-swift + mlx-swift-lm) behind a small C ABI that the
// Rust agent links in-process (see `include/ohmlx.h` and `build.sh`). Built with xcodebuild, because the
// SwiftPM command line can't compile MLX's Metal shaders.
import PackageDescription

let package = Package(
    name: "OHMLX",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "OHMLX", type: .static, targets: ["OHMLX"]),
    ],
    dependencies: [
        // Pinned exactly (mlx-swift and mlx-swift-lm at the versions measured in ATTESTED_ENGINE_TIER_PLAN_v1 §4.3).
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.4"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "3.31.3"),
        // Tokenizers for local model folders (Hugging Face's own package, Apache-2.0).
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.5"),
    ],
    targets: [
        .target(
            name: "OHMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
    ]
)
