import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

// ─── Availability reason codes (must stay in sync with AvailabilityCode in lib.rs) ───
private let FM_AVAILABLE: Int32 = 0
private let FM_DEVICE_NOT_ELIGIBLE: Int32 = 1
private let FM_NOT_ENABLED: Int32 = 2
private let FM_MODEL_NOT_READY: Int32 = 3
private let FM_UNKNOWN: Int32 = 4

// ─── Callback type aliases ────────────────────────────────────────────────────────────

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

// ─── Session holder ───────────────────────────────────────────────────────────────────

/// ARC-managed wrapper around LanguageModelSession.
/// Only instantiated on macOS 26+; the guard in every @_cdecl function ensures this.
@available(macOS 26.0, *)
private final class SessionHolder {
    let session: LanguageModelSession

    init(_ session: LanguageModelSession) {
        self.session = session
    }
}

// ─── Availability ─────────────────────────────────────────────────────────────────────

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
        case .deviceNotEligible:      return FM_DEVICE_NOT_ELIGIBLE
        case .appleIntelligenceNotEnabled: return FM_NOT_ENABLED
        case .modelNotReady:          return FM_MODEL_NOT_READY
        @unknown default:             return FM_UNKNOWN
        }
    }
    #else
    return FM_DEVICE_NOT_ELIGIBLE
    #endif
}

// ─── Session lifecycle ────────────────────────────────────────────────────────────────

/// Creates a new LanguageModelSession with the given system instructions.
/// Returns an opaque pointer to an ARC-retained SessionHolder, or NULL on failure.
@_cdecl("fm_session_create")
func sessionCreate(instructionsPtr: UnsafePointer<CChar>) -> UnsafeMutableRawPointer? {
    #if canImport(FoundationModels)
    guard #available(macOS 26.0, *) else { return nil }
    let instructions = String(cString: instructionsPtr)
    let session = LanguageModelSession(instructions: instructions)
    let holder = SessionHolder(session)
    return Unmanaged.passRetained(holder).toOpaque()
    #else
    return nil
    #endif
}

/// Releases the ARC-retained SessionHolder created by fm_session_create.
/// Must be called exactly once per handle, and never after that.
@_cdecl("fm_session_destroy")
func sessionDestroy(handlePtr: UnsafeMutableRawPointer) {
    #if canImport(FoundationModels)
    guard #available(macOS 26.0, *) else { return }
    Unmanaged<SessionHolder>.fromOpaque(handlePtr).release()
    #endif
}

// ─── Single-shot response ─────────────────────────────────────────────────────────────

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
        "Apple Intelligence requires macOS 26 or later".withCString {
            callback(callbackCtx, nil, $0)
        }
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
    "FoundationModels framework not available in this build".withCString {
        callback(callbackCtx, nil, $0)
    }
    #endif
}

// ─── Streaming response ───────────────────────────────────────────────────────────────

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
