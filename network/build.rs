fn main() {
    // Generate Rust types from peer.proto for zero-copy activation handling.
    // The generated code goes to OUT_DIR and is included via include!(concat!(...)).
    prost_build::compile_protos(&["proto/peer.proto"], &["proto/"])
        .expect("Failed to compile peer.proto");
}
