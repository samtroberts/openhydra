# Build MLX with JIT-compiled Metal kernels: a ~1.4 MB mlx.metallib instead of ~174 MB, so the
# scorer can ship inside the desktop app. Applied to mlx-sys's CMake build via .cargo/config.toml.
set(MLX_METAL_JIT ON CACHE BOOL "MLX: JIT-compile Metal kernels" FORCE)
