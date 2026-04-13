fn main() {
    // Generate Rust types from peer.proto for zero-copy activation handling.
    // The generated code goes to OUT_DIR and is included via include!(concat!(...)).
    prost_build::compile_protos(&["../peer/peer.proto"], &["../peer/"])
        .expect("Failed to compile peer.proto");
}
