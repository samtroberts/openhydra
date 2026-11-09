# Third-party code in the built-in Mac engine

A `--features mlx-swift` build of `openhydra-agent` links the code below (statically, from
`engine-mlx/.xcbuild`). Builds without the feature contain none of it. Versions are pinned in
`Package.swift` and `Package.resolved`. A release of a feature build must ship each project's licence text
with the binary; the release notices tooling (`scorer/notices.py`, `desktop/stage-notices.sh`) doesn't
collect these yet.

| Component | Licence | Source |
| --- | --- | --- |
| mlx-swift 0.31.4 | MIT | https://github.com/ml-explore/mlx-swift |
| — MLX (C++ core, vendored in Cmlx) | MIT | https://github.com/ml-explore/mlx |
| — mlx-c (vendored in Cmlx) | MIT | https://github.com/ml-explore/mlx-c |
| — metal-cpp (vendored in Cmlx) | Apache-2.0 | https://developer.apple.com/metal/cpp/ |
| — fmt (vendored in Cmlx) | MIT | https://github.com/fmtlib/fmt |
| — nlohmann/json (vendored in Cmlx) | MIT | https://github.com/nlohmann/json |
| mlx-swift-lm 3.31.3 | MIT | https://github.com/ml-explore/mlx-swift-lm |
| swift-transformers 1.3.5 (Tokenizers, Hub) | Apache-2.0 | https://github.com/huggingface/swift-transformers |
| swift-huggingface (linked through Hub; never called) | Apache-2.0 | https://github.com/huggingface/swift-huggingface |
| swift-jinja | Apache-2.0 | https://github.com/huggingface/swift-jinja |
| swift-collections | Apache-2.0 | https://github.com/apple/swift-collections |
| swift-crypto, swift-asn1 | Apache-2.0 | https://github.com/apple/swift-crypto |
| swift-numerics | Apache-2.0 | https://github.com/apple/swift-numerics |
| yyjson | MIT | https://github.com/ibireme/yyjson |
| EventSource | MIT | https://github.com/mattt/EventSource |

swift-syntax (Apache-2.0) is resolved for mlx-swift-lm's macros but isn't compiled into the engine.
