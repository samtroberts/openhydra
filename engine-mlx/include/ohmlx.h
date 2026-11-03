// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// The C ABI of OpenHydra's built-in Mac engine (Apple MLX via mlx-swift-lm), implemented in
// Sources/OHMLX/OHMLX.swift and linked into openhydra-agent by the `openhydra-mlx-engine` crate (engine-mlx/rust).
//
// Conventions:
// - Every call is synchronous and may block (model load, generation); call from a plain thread, never
//   from an async runtime's worker.
// - Strings in are UTF-8, NUL-terminated. Strings out are allocated by the engine and must be released
//   with ohmlx_free_string. An `err` out-pointer receives a message on failure (status != 0).
// - A handle runs one call at a time; the engine serialises calls on the same handle. ohmlx_cancel is the
//   exception: it may be called from another thread while ohmlx_generate runs.
// - Status codes: 0 ok, 1 failed (*err says why), 2 busy (only ohmlx_score: another call held the engine
//   too long; try again shortly).
// - MLX errors raised while a call builds or evaluates its arrays stop and fail that call. Some still end
//   the process (one inside a Metal completion handler, or one a later item() trips over), so inputs from
//   remote peers (request size, token ids, prompt and generated length) are bounded first.

#ifndef OHMLX_H
#define OHMLX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OhmlxEngine OhmlxEngine;

// Called with each piece of generated text (not NUL-terminated). Return 0 to continue, 1 to stop.
typedef int32_t (*ohmlx_on_text)(void *user, const char *text, size_t len);

// {"engine":"mlx-swift-lm","mlx_swift":"…","mlx_swift_lm":"…"} — the versions compiled in.
char *ohmlx_version(void);

// Loads the model from exactly the files named in `files_json` (a JSON array of plain names in
// `model_dir`: the weights, config, tokenizer and template files), hashing them before and after (they
// must not change while loading). MLX reads them through a private folder of links to those files only,
// so other files in `model_dir` are never loaded. NULL on failure, with *err set.
OhmlxEngine *ohmlx_load(const char *model_dir, const char *files_json, char **err);

// The load-time manifest, JSON: {"files":[{"name","size","sha256"}],"metallib_sha256","model_type",
// "vocab_size","context_length","gpu_working_set","mlx_swift","mlx_swift_lm"}. Files are the loaded ones;
// digests are hex SHA-256; metallib_sha256 is of `mlx.metallib` beside the executable's real path ("" if
// absent); vocab_size is the smaller of config's and the embedding table's; gpu_working_set is Metal's
// recommended maximum in bytes.
char *ohmlx_manifest(OhmlxEngine *engine);

// Chat completion. `messages_json`: a JSON array of {"role","content",…} objects (passed to the chat
// template as-is). `params_json`: {"max_context","max_tokens","temperature","top_p","seed","tools":[…],
// "enable_thinking"} (all optional). max_context lowers the model's context (the caller's memory budget);
// the request is refused before templating if its JSON exceeds 8 bytes per allowed token, and after if the
// prompt doesn't fit. Unset sampling falls back to the model's generation_config.json; max_tokens to the
// rest of the context, at most 8192. Streams text through `on_text`; on success *result receives
// {"prompt_tokens","completion_tokens","stop_reason":"stop|length|cancelled","tool_calls":[…],
// "prompt_ms","generate_ms"}.
int32_t ohmlx_generate(OhmlxEngine *engine, const char *messages_json, const char *params_json,
                       ohmlx_on_text on_text, void *user, char **result, char **err);

// Stop the generation running on `engine`, if any, including during prefill (noticed between 512-token
// steps); ohmlx_generate then returns with stop_reason "cancelled" and the exact token counts.
void ohmlx_cancel(OhmlxEngine *engine);

// Teacher-forced prompt scoring: writes the log-probability of ids[i + 1] given ids[0..=i] into
// out[i], for i in 0..n-1 (so `out` holds n - 1 floats). Fails for an id outside the vocabulary or more
// ids than the context holds; returns 2 (busy) if a generation keeps the engine for 20 s.
int32_t ohmlx_score(OhmlxEngine *engine, const uint32_t *ids, size_t n, float *out, char **err);

// The model tokenizer's ids for `text` (special tokens added as the tokenizer does), as a JSON array.
char *ohmlx_tokenize(OhmlxEngine *engine, const char *text, char **err);

void ohmlx_free(OhmlxEngine *engine);
void ohmlx_free_string(char *s);

#ifdef __cplusplus
}
#endif

#endif // OHMLX_H
