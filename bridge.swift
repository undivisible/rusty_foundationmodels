import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

// ─── Availability reason codes (must stay in sync with lib.rs) ───────────────
private let FM_AVAILABLE: Int32 = 0
private let FM_DEVICE_NOT_ELIGIBLE: Int32 = 1
private let FM_NOT_ENABLED: Int32 = 2
private let FM_MODEL_NOT_READY: Int32 = 3
private let FM_UNKNOWN: Int32 = 4

// ─── Callback type aliases ────────────────────────────────────────────────────

/// Called once with the result or an error string. Exactly one of `result` / `error` is non-nil.
typealias ResultCallback = @convention(c) (
    UnsafeMutableRawPointer?,  // Rust context pointer (passed back to caller)
    UnsafePointer<CChar>?,     // result (nil on error)
    UnsafePointer<CChar>?      // error  (nil on success)
) -> Void

/// Called for each streaming token/snapshot.
typealias TokenCallback = @convention(c) (
    UnsafeMutableRawPointer?,  // Rust context pointer
    UnsafePointer<CChar>       // token text (never nil)
) -> Void

/// Called once when streaming finishes. `error` is nil on success.
typealias DoneCallback = @convention(c) (
    UnsafeMutableRawPointer?,  // Rust context pointer
    UnsafePointer<CChar>?      // error message (nil on success)
) -> Void

/// Called by Swift to dispatch a tool invocation to Rust. Rust must call `resultCb` exactly once
/// (synchronously) to deliver the tool result before returning.
typealias ToolDispatchCallback = @convention(c) (
    UnsafeMutableRawPointer?,  // tool_ctx (Rust ToolsContext raw ptr)
    UnsafePointer<CChar>,      // tool name
    UnsafePointer<CChar>,      // arguments as JSON string
    UnsafeMutableRawPointer?,  // result_ctx passed back to resultCb
    ResultCallback             // write result / error here
) -> Void

// ─── Private JSON schema description types ────────────────────────────────────
// These decode the schema JSON that Rust passes for structured generation.

private struct SchemaPropertyDesc: Decodable {
    let name: String
    let description: String?
    let type: String
    let optional: Bool
}

private struct SchemaDesc: Decodable {
    let name: String
    let description: String?
    let properties: [SchemaPropertyDesc]
}

// ─── Session holder ───────────────────────────────────────────────────────────

/// ARC-managed wrapper around LanguageModelSession.
/// Only instantiated on macOS 26+; the guard in every @_cdecl function ensures this.
@available(macOS 26.0, *)
private final class SessionHolder {
    let session: LanguageModelSession

    init(_ session: LanguageModelSession) {
        self.session = session
    }
}

// ─── Tool support ─────────────────────────────────────────────────────────────

/// Holds the result of a synchronous tool call dispatched to Rust.
private final class ToolCallResult {
    var result: String?
    var error: String?
}

/// Static C-compatible callback written to by Rust's `tool_dispatch` after executing a tool.
/// `ctx` is an unretained `ToolCallResult` pointer allocated by `DynamicTool.call`.
private let writeToolCallResult: ResultCallback = { ctx, result, error in
    guard let ctx = ctx else { return }
    let holder = Unmanaged<ToolCallResult>.fromOpaque(ctx).takeUnretainedValue()
    if let result = result {
        holder.result = String(cString: result)
    } else if let error = error {
        holder.error = String(cString: error)
    }
}

/// A `Tool` implementation that forwards calls to a Rust `ToolsContext` via a C callback.
@available(macOS 26.0, *)
private final class DynamicTool: Tool {
    typealias Arguments = GeneratedContent
    typealias Output = String

    let name: String
    let description: String
    let parameters: GenerationSchema
    private let toolCtx: UnsafeMutableRawPointer?
    private let dispatch: ToolDispatchCallback

    init(
        name: String,
        description: String,
        schema: GenerationSchema,
        ctx: UnsafeMutableRawPointer?,
        dispatch: ToolDispatchCallback
    ) {
        self.name = name
        self.description = description
        self.parameters = schema
        self.toolCtx = ctx
        self.dispatch = dispatch
    }

    func call(arguments: GeneratedContent) async throws -> String {
        let argsJson = contentToJson(arguments)
        let resultHolder = ToolCallResult()
        // Use unretained — resultHolder is alive on the stack for the duration of the call.
        let resultHolderPtr = Unmanaged.passUnretained(resultHolder).toOpaque()
        name.withCString { namePtr in
            argsJson.withCString { argsPtr in
                dispatch(toolCtx, namePtr, argsPtr, resultHolderPtr, writeToolCallResult)
            }
        }
        if let error = resultHolder.error {
            throw ToolDispatchError(message: error)
        }
        return resultHolder.result ?? ""
    }
}

private struct ToolDispatchError: Error {
    let message: String
}

// ─── Helper: GeneratedContent → JSON string ───────────────────────────────────

/// Serialises a `GeneratedContent` value to a JSON string.
/// `Kind` cases: null, bool, number (f64), string, array, structure.
@available(macOS 26.0, *)
private func contentToJson(_ content: GeneratedContent) -> String {
    switch content.kind {
    case .null:
        return "null"
    case .bool(let b):
        return b ? "true" : "false"
    case .number(let n):
        // Emit as integer when the value is a whole number to keep JSON tidy.
        if !n.isInfinite && !n.isNaN && n.truncatingRemainder(dividingBy: 1) == 0 {
            return String(Int64(n))
        }
        return String(n)
    case .string(let s):
        return jsonQuote(s)
    case .array(let elements):
        let items = elements.map { contentToJson($0) }.joined(separator: ",")
        return "[\(items)]"
    case .structure(let props, let orderedKeys):
        let pairs = orderedKeys.compactMap { key -> String? in
            guard let value = props[key] else { return nil }
            return "\(jsonQuote(key)):\(contentToJson(value))"
        }.joined(separator: ",")
        return "{\(pairs)}"
    @unknown default:
        return "null"
    }
}

/// Returns a JSON-encoded string literal (double-quoted, with necessary escapes).
private func jsonQuote(_ s: String) -> String {
    var out = "\""
    for scalar in s.unicodeScalars {
        switch scalar.value {
        case 0x22: out += "\\\""
        case 0x5C: out += "\\\\"
        case 0x0A: out += "\\n"
        case 0x0D: out += "\\r"
        case 0x09: out += "\\t"
        case 0x00...0x1F: out += String(format: "\\u%04X", scalar.value)
        default:   out += String(scalar)
        }
    }
    out += "\""
    return out
}

// ─── Schema building helper ───────────────────────────────────────────────────

/// Builds a `DynamicGenerationSchema.Property` from a decoded property descriptor.
/// Returns `nil` for unknown type strings.
@available(macOS 26.0, *)
private func buildProperty(_ prop: SchemaPropertyDesc) -> DynamicGenerationSchema.Property? {
    let propSchema: DynamicGenerationSchema
    switch prop.type {
    case "string":  propSchema = DynamicGenerationSchema(type: String.self)
    case "integer": propSchema = DynamicGenerationSchema(type: Int.self)
    case "double":  propSchema = DynamicGenerationSchema(type: Double.self)
    case "bool":    propSchema = DynamicGenerationSchema(type: Bool.self)
    default:        return nil
    }
    return DynamicGenerationSchema.Property(
        name: prop.name,
        description: prop.description,
        schema: propSchema,
        isOptional: prop.optional
    )
}

/// Builds a `GenerationSchema` from a decoded `SchemaDesc`. Returns `nil` on failure.
@available(macOS 26.0, *)
private func buildGenerationSchema(_ desc: SchemaDesc) -> GenerationSchema? {
    let props = desc.properties.compactMap { buildProperty($0) }
    let dynSchema = DynamicGenerationSchema(
        name: desc.name,
        description: desc.description,
        properties: props
    )
    return try? GenerationSchema(root: dynSchema, dependencies: [])
}

// ─── Availability ─────────────────────────────────────────────────────────────

/// Returns FM_AVAILABLE if Apple Intelligence is ready, otherwise an FM_* reason code.
@_cdecl("fm_availability_reason")
func availabilityReason() -> Int32 {
    #if canImport(FoundationModels)
    guard #available(macOS 26.0, *) else { return FM_DEVICE_NOT_ELIGIBLE }
    switch SystemLanguageModel.default.availability {
    case .available:
        return FM_AVAILABLE
    case .unavailable(let reason):
        switch reason {
        case .deviceNotEligible:           return FM_DEVICE_NOT_ELIGIBLE
        case .appleIntelligenceNotEnabled: return FM_NOT_ENABLED
        case .modelNotReady:               return FM_MODEL_NOT_READY
        @unknown default:                  return FM_UNKNOWN
        }
    }
    #else
    return FM_DEVICE_NOT_ELIGIBLE
    #endif
}

// ─── Session lifecycle ────────────────────────────────────────────────────────

/// Creates a new LanguageModelSession with the given system instructions.
/// Returns an opaque pointer to an ARC-retained SessionHolder, or NULL on failure.
@_cdecl("fm_session_create")
func sessionCreate(instructionsPtr: UnsafePointer<CChar>) -> UnsafeMutableRawPointer? {
    #if canImport(FoundationModels)
    guard #available(macOS 26.0, *) else { return nil }
    let session = LanguageModelSession(instructions: String(cString: instructionsPtr))
    return Unmanaged.passRetained(SessionHolder(session)).toOpaque()
    #else
    return nil
    #endif
}

/// Creates a session pre-loaded with tools defined by `toolsJsonPtr` (a JSON array of
/// `{"name","description","properties":[{"name","type","description","optional"}]}` objects).
/// `toolCtx` and `toolDispatch` are forwarded to each `DynamicTool` so it can call back into Rust.
@_cdecl("fm_session_create_with_tools")
func sessionCreateWithTools(
    instructionsPtr: UnsafePointer<CChar>,
    toolsJsonPtr: UnsafePointer<CChar>,
    toolCtx: UnsafeMutableRawPointer?,
    toolDispatch: ToolDispatchCallback
) -> UnsafeMutableRawPointer? {
    #if canImport(FoundationModels)
    guard #available(macOS 26.0, *) else { return nil }

    let instructions = String(cString: instructionsPtr)
    let toolsJson = String(cString: toolsJsonPtr)

    guard
        let data = toolsJson.data(using: .utf8),
        let toolDescs = try? JSONDecoder().decode([SchemaDesc].self, from: data)
    else { return nil }

    let tools: [any Tool] = toolDescs.compactMap { desc -> DynamicTool? in
        guard let schema = buildGenerationSchema(desc) else { return nil }
        return DynamicTool(
            name: desc.name,
            description: desc.description ?? "",
            schema: schema,
            ctx: toolCtx,
            dispatch: toolDispatch
        )
    }

    let session = LanguageModelSession(tools: tools, instructions: instructions)
    return Unmanaged.passRetained(SessionHolder(session)).toOpaque()
    #else
    return nil
    #endif
}

/// Releases the ARC-retained SessionHolder created by fm_session_create / fm_session_create_with_tools.
/// Must be called exactly once per handle.
@_cdecl("fm_session_destroy")
func sessionDestroy(handlePtr: UnsafeMutableRawPointer) {
    #if canImport(FoundationModels)
    guard #available(macOS 26.0, *) else { return }
    Unmanaged<SessionHolder>.fromOpaque(handlePtr).release()
    #endif
}

// ─── Single-shot response ─────────────────────────────────────────────────────

/// Sends a prompt to the model and calls `callback` exactly once when done.
///
/// - `temperature`: generation temperature in [0.0, 2.0]. Pass -1.0 to use the model default.
/// - `maxTokens`:   maximum response tokens. Pass -1 to use the model default.
@_cdecl("fm_session_respond")
func sessionRespond(
    handlePtr: UnsafeMutableRawPointer,
    promptPtr: UnsafePointer<CChar>,
    temperature: Double,
    maxTokens: Int64,
    callbackCtx: UnsafeMutableRawPointer?,
    callback: ResultCallback
) {
    #if canImport(FoundationModels)
    guard #available(macOS 26.0, *) else {
        "Apple Intelligence requires macOS 26 or later".withCString { callback(callbackCtx, nil, $0) }
        return
    }

    let holder = Unmanaged<SessionHolder>.fromOpaque(handlePtr).takeUnretainedValue()
    let prompt = String(cString: promptPtr)
    var options = GenerationOptions()
    if temperature >= 0.0 { options.temperature = temperature }
    if maxTokens >= 0     { options.maximumResponseTokens = Int(maxTokens) }

    Task {
        do {
            let response = try await holder.session.respond(to: prompt, options: options)
            response.content.withCString { callback(callbackCtx, $0, nil) }
        } catch {
            error.localizedDescription.withCString { callback(callbackCtx, nil, $0) }
        }
    }
    #else
    "FoundationModels framework not available in this build".withCString { callback(callbackCtx, nil, $0) }
    #endif
}

// ─── Structured generation ────────────────────────────────────────────────────

/// Like `fm_session_respond` but constrains the output to the JSON schema described by
/// `schemaJsonPtr`. The callback receives the model output serialised as a JSON string.
///
/// `schemaJsonPtr` must be a UTF-8 JSON object:
/// `{"name":"T","description":"...","properties":[{"name":"x","type":"string","description":"...","optional":false}]}`
/// Supported types: `"string"`, `"integer"`, `"double"`, `"bool"`.
@_cdecl("fm_session_respond_structured")
func sessionRespondStructured(
    handlePtr: UnsafeMutableRawPointer,
    promptPtr: UnsafePointer<CChar>,
    schemaJsonPtr: UnsafePointer<CChar>,
    temperature: Double,
    maxTokens: Int64,
    callbackCtx: UnsafeMutableRawPointer?,
    callback: ResultCallback
) {
    #if canImport(FoundationModels)
    guard #available(macOS 26.0, *) else {
        "Apple Intelligence requires macOS 26 or later".withCString { callback(callbackCtx, nil, $0) }
        return
    }

    let holder = Unmanaged<SessionHolder>.fromOpaque(handlePtr).takeUnretainedValue()
    let prompt = String(cString: promptPtr)
    let schemaJson = String(cString: schemaJsonPtr)

    guard
        let schemaData = schemaJson.data(using: .utf8),
        let schemaDesc = try? JSONDecoder().decode(SchemaDesc.self, from: schemaData),
        let genSchema = buildGenerationSchema(schemaDesc)
    else {
        "Invalid or unsupported schema JSON".withCString { callback(callbackCtx, nil, $0) }
        return
    }

    var options = GenerationOptions()
    if temperature >= 0.0 { options.temperature = temperature }
    if maxTokens >= 0     { options.maximumResponseTokens = Int(maxTokens) }

    Task {
        do {
            let response = try await holder.session.respond(to: prompt, schema: genSchema, options: options)
            let json = contentToJson(response.content)
            json.withCString { callback(callbackCtx, $0, nil) }
        } catch {
            error.localizedDescription.withCString { callback(callbackCtx, nil, $0) }
        }
    }
    #else
    "FoundationModels framework not available in this build".withCString { callback(callbackCtx, nil, $0) }
    #endif
}

// ─── Streaming response ───────────────────────────────────────────────────────

/// Streams the model response, calling `onToken` for each text chunk and `onDone` when finished.
///
/// Each `onToken` call delivers an incremental snapshot of the response. `onDone` is called
/// exactly once, with a non-nil error string on failure and nil on success.
/// After `onDone` returns, `callbackCtx` must not be used.
///
/// - `temperature`: generation temperature in [0.0, 2.0]. Pass -1.0 to use the model default.
/// - `maxTokens`:   maximum response tokens. Pass -1 to use the model default.
@_cdecl("fm_session_stream")
func sessionStream(
    handlePtr: UnsafeMutableRawPointer,
    promptPtr: UnsafePointer<CChar>,
    temperature: Double,
    maxTokens: Int64,
    callbackCtx: UnsafeMutableRawPointer?,
    onToken: TokenCallback,
    onDone: DoneCallback
) {
    #if canImport(FoundationModels)
    guard #available(macOS 26.0, *) else {
        "Apple Intelligence requires macOS 26 or later".withCString { onDone(callbackCtx, $0) }
        return
    }

    let holder = Unmanaged<SessionHolder>.fromOpaque(handlePtr).takeUnretainedValue()
    let prompt = String(cString: promptPtr)
    var options = GenerationOptions()
    if temperature >= 0.0 { options.temperature = temperature }
    if maxTokens >= 0     { options.maximumResponseTokens = Int(maxTokens) }

    Task {
        do {
            for try await chunk in holder.session.streamResponse(to: prompt, options: options) {
                chunk.content.withCString { onToken(callbackCtx, $0) }
            }
            onDone(callbackCtx, nil)
        } catch {
            error.localizedDescription.withCString { onDone(callbackCtx, $0) }
        }
    }
    #else
    "FoundationModels framework not available in this build".withCString { onDone(callbackCtx, $0) }
    #endif
}
