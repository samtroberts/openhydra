# engine-mlx: the built-in Mac engine

Apple's MLX, through [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm), linked into `openhydra-agent`
so a Mac provider can serve an MLX model with no separate engine running. This is Phase 1a of the attested
engine tier (`docs/ATTESTED_ENGINE_TIER_PLAN_v1.md`, local): the engine runs in-process, which is what a
later hardware attestation of the running binary (Phase 2) needs. On its own it proves nothing more than
any other engine: the model digest it reports is the agent's own word, shown like every engine's.

## Build

Apple Silicon and Xcode (with its Metal toolchain) are required; the SwiftPM command line can't compile
MLX's Metal shaders, so the package is built with `xcodebuild`.

```bash
engine-mlx/build.sh
cargo build --release -p openhydra-agent --features mlx-swift
engine-mlx/stage.sh target/release
```

The engine's Swift objects are built for macOS 14, but the agent binary declares 11.0. Release builds with
the engine should set `MACOSX_DEPLOYMENT_TARGET=14.0`, so older systems refuse the binary instead of failing
somewhere inside it (not yet tried).

`build.sh` writes the Swift objects and `default.metallib` to `engine-mlx/.xcbuild/`. The
`openhydra-mlx-engine` crate (`engine-mlx/rust/`) packs the objects into one static library, links the
system Swift runtime and pins the metallib's SHA-256 into the binary. `stage.sh` copies the metallib beside
an executable as `mlx.metallib`, where MLX looks first; the agent refuses to load a model unless that file
(beside both the executable as invoked and its real path) matches the pinned digest. `METALLIB_SHA256` is
the digest the pinned mlx-swift builds to with the recorded Xcode; a build tree whose metallib differs
fails the crate's build and `stage.sh` (`OHMLX_ALLOW_UNPINNED_METALLIB=1` lets it through, for
development only, never for a release).

The metallib is MLX's ahead-of-time kernel library only (argument reductions, convolutions, gemv, norms,
RoPE, attention and the like). mlx-swift builds MLX's JIT variant: most kernels, including the quantized
matmul every 4-bit model runs and the models' own `metalKernel`s, are compiled at run time by macOS's Metal
compiler from sources inside the binary. The pin covers the AOT library; the JIT sources are covered only
by the binary itself.

Without `--features mlx-swift` nothing here is compiled: other platforms and the normal Mac build never
touch Swift or Metal. The crate is kept out of the root Cargo workspace for the same reason.

## Run

```bash
openhydra-agent provide --engine-kind mlx-swift --model-dir <MLX model folder> [--model-name org/name] [--max-context N] [--prompt-scoring auto]
```

The folder is an MLX model as Hugging Face or LM Studio store it: `config.json`, the weights
(`*.safetensors`, with `model.safetensors.index.json` when sharded), `tokenizer.json`,
`tokenizer_config.json`. A Hugging Face cache snapshot or an LM Studio models folder is named `org/name`
automatically. With `--prompt-scoring auto` the engine answers log-prob challenges itself.

It serves one request at a time, and a serve that arrives while it is busy is refused at once
(`Overloaded`) so its consumer fails over, rather than queued until the consumer times out
(`--max-concurrency` doesn't apply). When every provider of the model is busy, the consumer's gateway
waits and tries them again (0.5 s, doubling to 4 s, jittered) for up to 60 s before answering
`503 upstream_overloaded`, so requests that arrive together at a single provider are served one after
another. A streaming client gets its headers at once and keep-alives while it waits. Sheds cost a
provider no reputation. Log-prob challenges, model-info and attestation requests have their
own workers; a score that finds a generation holding the engine for 20 s answers busy. Leechers are not
throttled here: with one serve worker, a throttle sleep would block everyone else.

The context a request may use is the model's, lowered to what fits in memory: the KV cache per token is
worked out from `config.json` (full-attention layers × K and V × KV heads × head size × 2 bytes) and fitted
into Metal's recommended working set beside the weights. `--max-context` lowers it further. A Mac where
less than 2,048 tokens fit refuses to start. Requests larger than 8 bytes per allowed token are refused
before the chat template runs, and prompts that don't fit after it. Sampling the request leaves unset
follows the model's `generation_config.json`; the length defaults to the rest of the context, at most
8,192 tokens.

## What it reports

- `weight_digest`: the same digest an LM Studio provider of the folder reports (`agent/src/weights.rs`:
  `config.json` plus the weights files), computed from the engine's own hashes of the files it loaded.
  The engine loads exactly those files plus the config, tokenizer and template files, through a private
  folder of links to them, so any other `*.safetensors` in the folder (a leftover LoRA, an old shard, a
  hidden file) is not loaded; it is listed as ignored at startup. The engine hashes the files it loads
  before and after loading and refuses to start if they differ. That catches persistent changes only: a
  file rewritten and restored between the two hashes is not detected (hashing the bytes as MLX reads them
  is Phase 2 work).
- `config` (startup log): a SHA-256 over the files that decide how the model is read and prompted
  (config, generation config, tokenizer files, chat template), the engine and MLX versions, and the AOT
  kernel library's digest.

Both are reported data, the agent's own word, shown like every other engine's digest: nothing here is an
attestation or raises a trust tier.

## Safety limits

- Token ids, from scoring challenges and from the tokenizer alike, are checked against the vocabulary
  (the smaller of config's and the embedding table's rows), and prompts against the context, before they
  reach MLX (which doesn't bounds-check). An MLX error raised while a call builds or evaluates its arrays
  fails that call. Some still end the process (one inside a Metal completion handler, or one a later
  `item()` trips over); the input and memory bounds are what keep those rare.
- A stop is noticed between generated pieces, and between 512-token prefill steps, so a long prompt can be
  cancelled before its first token. A streaming consumer that hangs up is noticed by the provider within
  about 250 ms even while no text flows; a stop is then passed to the engine within about 20 ms. The next
  request starts only after the previous generation has finished.
- A panic while delivering text cancels the generation, then propagates once the engine has stopped.

## What's linked in

See [NOTICES.md](NOTICES.md). The tokenizer is loaded from the model folder only; nothing contacts a hub.

## Limits

- One request at a time per model (MLX has no batching; a scheduler is future work).
- Text models only, MLX format (safetensors) only.
- Tool calls are parsed by mlx-swift-lm's per-model tool-call format; a model whose format it doesn't
  know returns its calls as plain text. A malformed `tools` list fails the request. Tool schemas render
  with their keys sorted (swift-jinja), so prompts with tools differ slightly from a Python template's.
- Kernels compiled at run time go through the OS's Metal compiler and its per-user shader cache, which an
  attestation of the binary alone does not cover (see the plan's Phase 2).
