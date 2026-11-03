// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// OpenHydra's built-in Mac engine: an MLX model loaded with mlx-swift-lm, exposed through the C ABI in
// include/ohmlx.h so the Rust agent can run it in-process. Plumbing only: load, chat, prompt scoring,
// tokenizing, and a manifest of the files it loaded. What the hashes are worth is decided on the Rust side.
//
// Everything that reaches MLX runs inside `withError`, and generation and scoring check its error box as
// they go, so an MLX error stops and fails that one call instead of reaching mlx-swift's default handler
// (which ends the process). Some still end it: one raised inside a Metal completion handler, or one at
// graph-build time that a later `item()` trips over. Inputs a remote peer controls (request size, token
// ids, prompt length, generated length) are bounded first, which is what keeps those rare.

import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - Versions compiled in (kept in step with Package.swift's exact pins)

let mlxSwiftVersion = "0.31.4"
let mlxSwiftLMVersion = "3.31.3"

/// The most ids one scoring call accepts (log-prob challenges send at most 512).
let maxScoreIds = 8192

/// How long a scoring call waits for the engine before answering "busy" (a generation holds it).
let scoreWaitSeconds = 20.0

/// MLX keeps freed GPU buffers for reuse; bound that cache so memory comes back after a long prompt.
let gpuCacheLimitBytes = 512 << 20

/// `max_tokens` when a request sets none: the rest of the context, but at most this many.
let defaultMaxTokens = 8192

/// Requests are refused before the chat template runs when their messages and tools JSON exceeds this
/// many bytes per token of the context the caller allows (a token is about 4 bytes of text).
let maxBytesPerContextToken = 8

/// Prompt tokens evaluated per step during prefill; a cancel is noticed between steps.
let prefillStep = 512

// MARK: - Tokenizer bridge (swift-transformers -> mlx-swift-lm)

/// Adapts a swift-transformers tokenizer to mlx-swift-lm's protocol. Equivalent to what the
/// `MLXHuggingFace` macros generate, written out so the build needs no macro plugin.
struct TokenizerBridge: MLXLMCommon.Tokenizer {
    let upstream: any Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try upstream.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}

/// Loads the tokenizer from the model folder itself; never contacts a hub.
struct LocalTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        TokenizerBridge(upstream: try await AutoTokenizer.from(modelFolder: directory))
    }
}

// MARK: - Errors and small helpers

struct EngineError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

final class ResultBox<T>: @unchecked Sendable {
    var result: Result<T, Error>?
}

/// Runs an async operation to completion on the calling (non-async) thread.
func runBlocking<T>(_ operation: @escaping @Sendable () async throws -> T) -> Result<T, Error> {
    let box = ResultBox<T>()
    let done = DispatchSemaphore(value: 0)
    Task.detached {
        do { box.result = .success(try await operation()) } catch { box.result = .failure(error) }
        done.signal()
    }
    done.wait()
    return box.result!
}

func describe(_ error: Error) -> String {
    (error as? EngineError)?.description ?? String(describing: error)
}

func cString(_ s: String) -> UnsafeMutablePointer<CChar> { strdup(s)! }

func setError(_ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?, _ message: String) {
    err?.pointee = cString(message)
}

func jsonString(_ value: Any) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
        let s = String(data: data, encoding: .utf8)
    else { return "{}" }
    return s
}

/// JSON values from JSONSerialization, made `Sendable` for mlx-swift-lm's message and tool types. A
/// JSON null becomes an empty optional, which the template engine reads as null.
func sendable(_ value: Any) -> any Sendable {
    switch value {
    case is NSNull: return String?.none
    case let s as String: return s
    case let n as NSNumber:
        if CFGetTypeID(n) == CFBooleanGetTypeID() { return n.boolValue }
        if CFNumberIsFloatType(n) { return n.doubleValue }
        return n.intValue
    case let a as [Any]: return a.map(sendable)
    case let d as [String: Any]: return d.mapValues(sendable)
    default: return String(describing: value)
    }
}

func parseObject(_ json: UnsafePointer<CChar>?) throws -> [String: Any] {
    guard let json else { return [:] }
    let data = Data(String(cString: json).utf8)
    if data.isEmpty { return [:] }
    guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw EngineError("expected a JSON object")
    }
    return obj
}

func parseArray(_ json: UnsafePointer<CChar>?) throws -> [[String: Any]] {
    guard let json else { throw EngineError("missing messages") }
    guard let arr = try JSONSerialization.jsonObject(with: Data(String(cString: json).utf8)) as? [[String: Any]] else {
        throw EngineError("messages must be a JSON array of objects")
    }
    return arr
}

func number(_ value: Any?) -> Double? {
    (value as? NSNumber).flatMap { CFGetTypeID($0) == CFBooleanGetTypeID() ? nil : $0.doubleValue }
}

// MARK: - File manifest

struct FileEntry: Equatable {
    let name: String
    let size: UInt64
    let sha256: String
}

/// SHA-256 of a file, read in chunks (symlinks, as in the Hugging Face cache, are followed). Each chunk is
/// released as it goes: callers run on threads without an autorelease pool.
func sha256(of url: URL) throws -> (String, UInt64) {
    let handle = try FileHandle(forReadingFrom: url.resolvingSymlinksInPath())
    defer { try? handle.close() }
    var hasher = SHA256()
    var size: UInt64 = 0
    while true {
        let more = try autoreleasepool { () throws -> Bool in
            let chunk = try handle.read(upToCount: 8 << 20) ?? Data()
            if chunk.isEmpty { return false }
            size += UInt64(chunk.count)
            hasher.update(data: chunk)
            return true
        }
        if !more { break }
    }
    return (hex(hasher.finalize()), size)
}

func hex<D: Sequence>(_ digest: D) -> String where D.Element == UInt8 {
    digest.map { String(format: "%02x", $0) }.joined()
}

/// The named files in `dir` (plain names in that folder only), each a regular file, hashed, in order.
func hashFiles(_ dir: URL, _ names: [String]) throws -> [FileEntry] {
    try names.sorted().map { name in
        let url = dir.appendingPathComponent(name)
        guard (try? url.resolvingSymlinksInPath().resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
        else { throw EngineError("\(name) is missing or not a regular file") }
        let (digest, size) = try sha256(of: url)
        return FileEntry(name: name, size: size, sha256: digest)
    }
}

/// A private folder holding symlinks to exactly `names` in `dir`, so MLX's loader (which takes every
/// `*.safetensors` it finds under the folder it's given) and the tokenizer see those files and nothing else.
func stage(_ dir: URL, _ names: [String]) throws -> URL {
    let staging = FileManager.default.temporaryDirectory.appendingPathComponent("ohmlx-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false,
                                            attributes: [.posixPermissions: 0o700])
    for name in names {
        try FileManager.default.createSymbolicLink(
            at: staging.appendingPathComponent(name),
            withDestinationURL: dir.appendingPathComponent(name).resolvingSymlinksInPath())
    }
    return staging
}

/// The Metal kernel library MLX loads first: `mlx.metallib` beside the executable (its real path).
func metallibDigest() -> String {
    guard let exe = Bundle.main.executableURL?.resolvingSymlinksInPath() else { return "" }
    let path = exe.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
    return (try? sha256(of: path).0) ?? ""
}

/// `key` from a model config, or from its `text_config` (multimodal configs nest the text model's).
func configInt(_ config: [String: Any], _ key: String) -> Int? {
    number(config[key]).map(Int.init) ?? (config["text_config"] as? [String: Any]).flatMap { number($0[key]).map(Int.init) }
}

// MARK: - The engine

final class Engine: @unchecked Sendable {
    let container: ModelContainer
    let manifest: String
    /// Ids below this have an embedding row (the smaller of config `vocab_size` and the loaded table).
    let vocabSize: Int
    let contextLength: Int
    /// Sampling defaults from the model's `generation_config.json`, used when a request sets none.
    let defaults: [String: Double]
    /// One request at a time: MLX has no batching, and concurrent evaluation on one model isn't safe.
    let lock = NSLock()
    /// Guards `current` and `cancelled`, which `ohmlx_cancel` touches while a generation holds `lock`.
    let taskLock = NSLock()
    var current: Task<Void, Never>?
    var cancelled = false

    init(container: ModelContainer, manifest: String, vocabSize: Int, contextLength: Int, defaults: [String: Double]) {
        self.container = container
        self.manifest = manifest
        self.vocabSize = vocabSize
        self.contextLength = contextLength
        self.defaults = defaults
    }

    func cancel() {
        taskLock.lock()
        cancelled = true
        let task = current
        taskLock.unlock()
        task?.cancel()
    }
}

func load(modelDir: String, files: [String]) throws -> Engine {
    let dir = URL(fileURLWithPath: modelDir)
    for name in files where name.isEmpty || name.contains("/") || name == "." || name == ".." {
        throw EngineError("unexpected file name \(name)")
    }
    if !files.contains(where: { $0.hasSuffix(".safetensors") }) {
        throw EngineError("no .safetensors weights listed for \(modelDir)")
    }
    // Without these, swift-transformers falls back to configs in a resource bundle that a plain
    // executable doesn't carry (and crashes looking for it), so require the model's own.
    for required in ["config.json", "tokenizer.json", "tokenizer_config.json"] where !files.contains(required) {
        throw EngineError("\(modelDir) has no \(required)")
    }
    let before = try hashFiles(dir, files)
    guard let configData = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
        let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any]
    else { throw EngineError("\(modelDir)/config.json is not a JSON object") }
    guard let configVocab = configInt(config, "vocab_size"), configVocab > 0 else {
        throw EngineError("\(modelDir)/config.json has no vocab_size")
    }
    let contextLength = configInt(config, "max_position_embeddings") ?? 4096
    var defaults: [String: Double] = [:]
    if files.contains("generation_config.json"),
        let data = try? Data(contentsOf: dir.appendingPathComponent("generation_config.json")),
        let gen = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    {
        for key in ["temperature", "top_p", "top_k", "repetition_penalty"] {
            if let v = number(gen[key]) { defaults[key] = v }
        }
    }

    MLX.Memory.cacheLimit = min(MLX.Memory.cacheLimit, gpuCacheLimitBytes)
    let staging = try stage(dir, files)
    defer { try? FileManager.default.removeItem(at: staging) }
    // The LLM factory directly: the registry finds factories by class name at runtime, which a
    // statically linked binary doesn't keep, and text models are all this engine serves.
    let container = try runBlocking {
        try await withError {
            try await LLMModelFactory.shared.loadContainer(from: staging, using: LocalTokenizerLoader())
        }
    }.get()
    if try hashFiles(dir, files) != before {
        throw EngineError("model files changed while loading")
    }
    // The embedding table can be smaller than config's vocab_size claims; ids past it would be read
    // out of bounds (MLX doesn't check gathers).
    let embedRows = try runBlocking {
        await container.perform { context in
            context.model.parameters().flattened()
                .first { $0.0.hasSuffix("embed_tokens.weight") }
                .map { $0.1.dim(0) }
        }
    }.get()
    let vocabSize = min(configVocab, embedRows ?? configVocab)
    let manifest: [String: Any] = [
        "files": before.map { ["name": $0.name, "size": $0.size, "sha256": $0.sha256] },
        "metallib_sha256": metallibDigest(),
        "model_type": config["model_type"] as? String ?? "",
        "vocab_size": vocabSize,
        "context_length": contextLength,
        "gpu_working_set": GPU.deviceInfo().maxRecommendedWorkingSetSize,
        "mlx_swift": mlxSwiftVersion,
        "mlx_swift_lm": mlxSwiftLMVersion,
    ]
    return Engine(container: container, manifest: jsonString(manifest), vocabSize: vocabSize,
                  contextLength: contextLength, defaults: defaults)
}

struct GenerateOutcome: Sendable {
    var promptTokens = 0
    var completionTokens = 0
    var stopReason = "stop"
    var toolCalls: [String] = []  // JSON objects, OpenAI shape
    var promptMs = 0.0
    var generateMs = 0.0
}

/// Calls back into Rust with each text piece; a non-zero return asks to stop.
struct TextSink: @unchecked Sendable {
    let callback: @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?, Int) -> Int32
    let user: UnsafeMutableRawPointer?

    func send(_ text: String) -> Bool {
        if text.isEmpty { return true }
        var bytes = Array(text.utf8)
        return bytes.withUnsafeMutableBufferPointer { buf in
            buf.baseAddress!.withMemoryRebound(to: CChar.self, capacity: buf.count) {
                callback(user, $0, buf.count) == 0
            }
        }
    }
}

func toolCallJSON(_ call: ToolCall) -> String {
    let args = (try? JSONEncoder().encode(call.function.arguments)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    let id = "call_" + hex((0..<8).map { _ in UInt8.random(in: 0...255) })
    return jsonString([
        "id": id,
        "type": "function",
        "function": ["name": call.function.name, "arguments": args],
    ] as [String: Any])
}

/// The context a request may use: the model's, lowered by the caller's `max_context` (its memory budget).
func allowedContext(_ engine: Engine, _ params: [String: Any]) -> Int {
    min(engine.contextLength, number(params["max_context"]).map { max(1, Int($0)) } ?? engine.contextLength)
}

func generate(engine: Engine, messages: [[String: Any]], params: [String: Any], sink: TextSink) throws -> GenerateOutcome {
    let defaults = engine.defaults
    var parameters = GenerateParameters()
    if let t = number(params["temperature"]) ?? defaults["temperature"] { parameters.temperature = Float(t) }
    if let p = number(params["top_p"]) ?? defaults["top_p"] { parameters.topP = Float(p) }
    if let k = defaults["top_k"] { parameters.topK = Int(k) }
    if let r = defaults["repetition_penalty"], r != 1 { parameters.repetitionPenalty = Float(r) }
    let requestedMax = number(params["max_tokens"]).map { max(1, Int($0)) }
    // Each request gets its own random state: a fixed one when seeded, else a fresh one, never what
    // an earlier request left behind.
    let seed = (params["seed"] as? NSNumber).map { UInt64(bitPattern: $0.int64Value) } ?? UInt64.random(in: 0...UInt64.max)
    var tools: [[String: any Sendable]]? = nil
    if let raw = params["tools"] {
        guard let list = raw as? [Any], list.allSatisfy({ $0 is [String: Any] }) else {
            throw EngineError("tools must be a JSON array of objects")
        }
        tools = list.map { ($0 as! [String: Any]).mapValues(sendable) }
    }
    var context: [String: any Sendable]? = nil
    if let think = params["enable_thinking"] as? Bool { context = ["enable_thinking": think] }
    let chat = messages.map { $0.mapValues(sendable) }
    let input = UserInput(messages: chat, tools: tools, additionalContext: context)
    let contextLength = allowedContext(engine, params)
    let vocabSize = engine.vocabSize

    engine.taskLock.lock()
    engine.cancelled = false
    engine.taskLock.unlock()
    let isCancelled = { () -> Bool in
        engine.taskLock.lock()
        defer { engine.taskLock.unlock() }
        return engine.cancelled
    }

    return try runBlocking {
        try await withError { errors in
            try await engine.container.perform { context in
                let prepared = try await context.processor.prepare(input: input)
                try errors.check()
                let promptCount = prepared.text.tokens.size
                if promptCount >= contextLength {
                    throw EngineError("the prompt has \(promptCount) tokens; this engine allows \(contextLength)")
                }
                if promptCount > 0, prepared.text.tokens.max().item(Int.self) >= vocabSize {
                    throw EngineError("the tokenizer produced an id outside this model's embedding table")
                }
                var params = parameters
                params.maxTokens = min(requestedMax ?? defaultMaxTokens, contextLength - promptCount)

                // Prefill in steps, so a cancel (or an MLX error) is noticed between them rather than only
                // after the whole prompt; the iterator then gets the short tail and the filled cache.
                var outcome = GenerateOutcome()
                outcome.promptTokens = promptCount
                let cache = context.model.newCache(parameters: params)
                var tail = prepared.text
                let prefillStart = Date.timeIntervalSinceReferenceDate
                while tail.tokens.size > prefillStep {
                    if isCancelled() {
                        outcome.stopReason = "cancelled"
                        return outcome
                    }
                    _ = context.model(tail[.newAxis, ..<prefillStep], cache: cache, state: nil)
                    eval(cache)
                    try errors.check()
                    tail = tail[prefillStep...]
                }
                let prefillMs = (Date.timeIntervalSinceReferenceDate - prefillStart) * 1000

                let (stream, task) = try withRandomState(MLXRandom.RandomState(seed: seed)) {
                    let iterator = try TokenIterator(input: LMInput(text: tail), model: context.model, cache: cache, parameters: params)
                    return generateTask(
                        promptTokenCount: promptCount, modelConfiguration: context.configuration,
                        tokenizer: context.tokenizer, iterator: iterator)
                }
                engine.taskLock.lock()
                engine.current = task
                let cancelledAlready = engine.cancelled
                engine.taskLock.unlock()
                if cancelledAlready { task.cancel() }

                var stopped = false
                var failed = false
                // Read the stream to its end even after a stop: cancelling the task ends it, and its last
                // item carries the exact token counts.
                for await item in stream {
                    if !failed && errors.firstError != nil {
                        // An MLX error leaves the loop sampling garbage: stop it and fail the call.
                        failed = true
                        task.cancel()
                    }
                    switch item {
                    case .chunk(let text):
                        if !stopped && !failed && !sink.send(text) {
                            stopped = true
                            task.cancel()
                        }
                    case .info(let info):
                        outcome.completionTokens = info.generationTokenCount
                        outcome.promptMs = prefillMs + info.promptTime * 1000
                        outcome.generateMs = info.generateTime * 1000
                        switch info.stopReason {
                        case .stop: outcome.stopReason = "stop"
                        case .length: outcome.stopReason = "length"
                        case .cancelled: outcome.stopReason = "cancelled"
                        }
                    case .toolCall(let call):
                        if !stopped && !failed { outcome.toolCalls.append(toolCallJSON(call)) }
                    }
                }
                // The model and its caches are free only once the task has finished.
                await task.value
                engine.taskLock.lock()
                engine.current = nil
                let cancelledByCaller = engine.cancelled
                engine.taskLock.unlock()
                try errors.check()
                if stopped || cancelledByCaller {
                    outcome.stopReason = "cancelled"
                } else if outcome.stopReason == "cancelled", let cap = params.maxTokens, outcome.completionTokens >= cap {
                    // mlx-swift-lm 3.31.3 checks the length on a copy of its iterator (its count stays 0)
                    // and reports a generation that hit maxTokens as cancelled. Nobody cancelled this one.
                    outcome.stopReason = "length"
                }
                return outcome
            }
        }
    }.get()
}

/// log p(ids[i+1] | ids[0...i]) for every i, from one forward pass, normalised in float32 chunks so
/// the full-vocabulary log-softmax never has to exist at once.
func score(engine: Engine, ids: [UInt32]) throws -> [Float] {
    if ids.count < 2 { return [] }
    let allowed = min(maxScoreIds, engine.contextLength)
    if ids.count > allowed {
        throw EngineError("\(ids.count) ids is more than this engine scores at once (\(allowed))")
    }
    if let bad = ids.first(where: { Int($0) >= engine.vocabSize }) {
        throw EngineError("token id \(bad) is outside this model's vocabulary (\(engine.vocabSize))")
    }
    return try runBlocking {
        try await withError { errors in
            try await engine.container.perform { context in
                let tokens = MLXArray(ids.map { Int32($0) })[.newAxis, 0...]
                let logits = context.model(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits[0]
                try errors.check()
                let targets = MLXArray(ids.dropFirst().map { Int32($0) })
                let rows = ids.count - 1
                var out: [Float] = []
                out.reserveCapacity(rows)
                var start = 0
                while start < rows {
                    let end = min(start + 128, rows)
                    let chunk = logits[start ..< end].asType(.float32)
                    let lse = logSumExp(chunk, axis: -1)
                    let picked = takeAlong(chunk, targets[start ..< end].expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
                    let lp = picked - lse
                    eval(lp)
                    try errors.check()
                    out.append(contentsOf: lp.asArray(Float.self))
                    start = end
                }
                return out
            }
        }
    }.get()
}

func tokenize(engine: Engine, text: String) throws -> [Int] {
    try runBlocking {
        await engine.container.perform { context in context.tokenizer.encode(text: text) }
    }.get()
}

// MARK: - C ABI

/// Status codes: 0 ok, 1 failed, 2 busy (try again shortly).
let statusOK: Int32 = 0
let statusFailed: Int32 = 1
let statusBusy: Int32 = 2

@_cdecl("ohmlx_version")
public func ohmlx_version() -> UnsafeMutablePointer<CChar> {
    cString(jsonString(["engine": "mlx-swift-lm", "mlx_swift": mlxSwiftVersion, "mlx_swift_lm": mlxSwiftLMVersion]))
}

@_cdecl("ohmlx_load")
public func ohmlx_load(
    _ modelDir: UnsafePointer<CChar>?, _ filesJSON: UnsafePointer<CChar>?,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
    guard let modelDir, let filesJSON,
        let files = try? JSONSerialization.jsonObject(with: Data(String(cString: filesJSON).utf8)) as? [String]
    else {
        setError(err, "missing model directory or file list")
        return nil
    }
    do {
        let engine = try load(modelDir: String(cString: modelDir), files: files)
        return Unmanaged.passRetained(engine).toOpaque()
    } catch {
        setError(err, describe(error))
        return nil
    }
}

func engineFrom(_ ptr: UnsafeMutableRawPointer?) -> Engine? {
    ptr.map { Unmanaged<Engine>.fromOpaque($0).takeUnretainedValue() }
}

@_cdecl("ohmlx_manifest")
public func ohmlx_manifest(_ ptr: UnsafeMutableRawPointer?) -> UnsafeMutablePointer<CChar>? {
    engineFrom(ptr).map { cString($0.manifest) }
}

@_cdecl("ohmlx_generate")
public func ohmlx_generate(
    _ ptr: UnsafeMutableRawPointer?,
    _ messagesJSON: UnsafePointer<CChar>?,
    _ paramsJSON: UnsafePointer<CChar>?,
    _ onText: (@convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?, Int) -> Int32)?,
    _ user: UnsafeMutableRawPointer?,
    _ result: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let engine = engineFrom(ptr), let onText else {
        setError(err, "missing engine or callback")
        return statusFailed
    }
    engine.lock.lock()
    defer { engine.lock.unlock() }
    do {
        let params = try parseObject(paramsJSON)
        // Bound the request before the chat template and tokenizer see it (they'd spend minutes and GBs on
        // a huge one before the token bound applies).
        let bytes = (messagesJSON.map { strlen($0) } ?? 0) + (paramsJSON.map { strlen($0) } ?? 0)
        let allowedBytes = allowedContext(engine, params) * maxBytesPerContextToken
        if bytes > allowedBytes {
            throw EngineError("the request is \(bytes) bytes; this engine takes at most \(allowedBytes)")
        }
        let outcome = try generate(
            engine: engine, messages: try parseArray(messagesJSON), params: params,
            sink: TextSink(callback: onText, user: user))
        let toolCalls = outcome.toolCalls.compactMap {
            try? JSONSerialization.jsonObject(with: Data($0.utf8))
        }
        result?.pointee = cString(jsonString([
            "prompt_tokens": outcome.promptTokens,
            "completion_tokens": outcome.completionTokens,
            "stop_reason": outcome.stopReason,
            "tool_calls": toolCalls,
            "prompt_ms": outcome.promptMs,
            "generate_ms": outcome.generateMs,
        ] as [String: Any]))
        return statusOK
    } catch {
        setError(err, describe(error))
        return statusFailed
    }
}

/// Stop the generation running on `engine`, if any (callable from another thread while it runs).
@_cdecl("ohmlx_cancel")
public func ohmlx_cancel(_ ptr: UnsafeMutableRawPointer?) {
    engineFrom(ptr)?.cancel()
}

@_cdecl("ohmlx_score")
public func ohmlx_score(
    _ ptr: UnsafeMutableRawPointer?,
    _ ids: UnsafePointer<UInt32>?,
    _ n: Int,
    _ out: UnsafeMutablePointer<Float>?,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let engine = engineFrom(ptr), let ids, let out, n >= 2 else {
        setError(err, "missing engine, ids or output, or fewer than 2 ids")
        return statusFailed
    }
    guard engine.lock.lock(before: Date().addingTimeInterval(scoreWaitSeconds)) else {
        setError(err, "the engine is busy generating")
        return statusBusy
    }
    defer { engine.lock.unlock() }
    do {
        let scores = try score(engine: engine, ids: Array(UnsafeBufferPointer(start: ids, count: n)))
        guard scores.count == n - 1 else { throw EngineError("scored \(scores.count) positions, expected \(n - 1)") }
        out.update(from: scores, count: scores.count)
        return statusOK
    } catch {
        setError(err, describe(error))
        return statusFailed
    }
}

@_cdecl("ohmlx_tokenize")
public func ohmlx_tokenize(
    _ ptr: UnsafeMutableRawPointer?, _ text: UnsafePointer<CChar>?,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutablePointer<CChar>? {
    guard let engine = engineFrom(ptr), let text else {
        setError(err, "missing engine or text")
        return nil
    }
    engine.lock.lock()
    defer { engine.lock.unlock() }
    do {
        return cString(jsonString(try tokenize(engine: engine, text: String(cString: text))))
    } catch {
        setError(err, describe(error))
        return nil
    }
}

@_cdecl("ohmlx_free")
public func ohmlx_free(_ ptr: UnsafeMutableRawPointer?) {
    if let ptr { Unmanaged<Engine>.fromOpaque(ptr).release() }
}

@_cdecl("ohmlx_free_string")
public func ohmlx_free_string(_ s: UnsafeMutablePointer<CChar>?) {
    free(s)
}
