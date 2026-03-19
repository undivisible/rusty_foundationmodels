//! Safe Rust bindings for Apple's [FoundationModels] on-device AI framework
//! (Apple Intelligence).
//!
//! # Overview
//!
//! FoundationModels gives access to an on-device ≈3B-parameter language model that runs
//! entirely locally — no network requests, no API keys, no data leaving the device.
//!
//! This crate exposes the full session lifecycle:
//! - **Availability checking** — [`is_available`] / [`availability`]
//! - **Single-shot generation** — top-level [`respond`] / [`respond_with_options`]
//! - **Session-based multi-turn generation** — [`Session::respond`]
//! - **Streaming** — [`Session::stream`] returns a [`ResponseStream`] that implements
//!   [`futures_core::Stream`]
//!
//! # Requirements
//!
//! | Requirement | Value |
//! |---|---|
//! | macOS | 26 (Tahoe) or later |
//! | Hardware | Apple Silicon (M1 or later) |
//! | Setting | Apple Intelligence enabled in System Settings |
//! | Build tool | Xcode with the macOS 26 SDK |
//!
//! On unsupported hardware or older macOS the crate still compiles and links; all APIs
//! return [`Error::Unavailable`] at runtime.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn example() -> Result<(), rusty_foundationmodels::Error> {
//! use rusty_foundationmodels::{is_available, respond};
//!
//! if !is_available() {
//!     eprintln!("Apple Intelligence not available on this device");
//!     return Ok(());
//! }
//!
//! let answer = respond("What is the capital of France?").await?;
//! println!("{answer}");
//! # Ok(()) }
//! ```
//!
//! # Multi-turn conversation
//!
//! ```no_run
//! # async fn example() -> Result<(), rusty_foundationmodels::Error> {
//! use rusty_foundationmodels::Session;
//!
//! let mut session = Session::with_instructions("You are a concise Rust expert.")?;
//! let r1 = session.respond("What is ownership?").await?;
//! let r2 = session.respond("Give me a one-line example.").await?;
//! println!("{r1}\n{r2}");
//! # Ok(()) }
//! ```
//!
//! # Streaming
//!
//! ```no_run
//! # async fn example() -> Result<(), rusty_foundationmodels::Error> {
//! use futures_core::Stream;
//! use rusty_foundationmodels::Session;
//!
//! let session = Session::new()?;
//! let mut stream = session.stream("Tell me a short story.")?;
//!
//! use std::pin::Pin;
//! use std::task::{Context, Poll, Waker};
//! // Use your preferred async executor to drive the stream...
//! # Ok(()) }
//! ```
//!
//! [FoundationModels]: https://developer.apple.com/documentation/foundationmodels

use std::pin::Pin;
use std::task::{Context as StdContext, Poll};

use futures_core::Stream;

#[cfg(foundation_models_bridge)]
use std::ffi::{CStr, CString, c_char, c_void};

#[cfg(foundation_models_bridge)]
use futures_channel::{mpsc, oneshot};

// ─── FFI declarations ──────────────────────────────────────────────────────────

#[cfg(foundation_models_bridge)]
unsafe extern "C" {
    fn fm_availability_reason() -> i32;
    fn fm_session_create(instructions: *const c_char) -> *mut c_void;
    fn fm_session_destroy(handle: *mut c_void);
    fn fm_session_respond(
        handle: *mut c_void,
        prompt: *const c_char,
        temperature: f64,
        max_tokens: i64,
        ctx: *mut c_void,
        callback: extern "C" fn(*mut c_void, *const c_char, *const c_char),
    );
    fn fm_session_stream(
        handle: *mut c_void,
        prompt: *const c_char,
        temperature: f64,
        max_tokens: i64,
        ctx: *mut c_void,
        on_token: extern "C" fn(*mut c_void, *const c_char),
        on_done: extern "C" fn(*mut c_void, *const c_char),
    );
}

// ─── Error ─────────────────────────────────────────────────────────────────────

/// Reasons why Apple Intelligence is not available on the current device.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnavailabilityReason {
    /// The device does not have compatible hardware (requires Apple Silicon M1 or later).
    #[error("device is not eligible (requires Apple Silicon M1 or later)")]
    DeviceNotEligible,
    /// Apple Intelligence is supported but has not been enabled in System Settings.
    #[error("Apple Intelligence is not enabled in System Settings")]
    NotEnabled,
    /// The on-device model is still downloading or is otherwise not ready.
    #[error("the on-device model is not ready yet")]
    ModelNotReady,
    /// An unrecognized availability state was returned by the framework.
    #[error("unknown availability state")]
    Unknown,
}

/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Apple Intelligence is not available on this device.
    #[error("Apple Intelligence unavailable: {0}")]
    Unavailable(#[source] UnavailabilityReason),

    /// The model produced an error during text generation.
    #[error("generation error: {0}")]
    Generation(String),

    /// An argument contained a null byte and could not be converted to a C string.
    #[error("argument contains a null byte: {0}")]
    NullByte(#[from] std::ffi::NulError),

    /// A `temperature` value outside the valid range [0.0, 2.0] was supplied.
    #[error("temperature {0} is out of range; expected 0.0 – 2.0")]
    InvalidTemperature(f64),
}

// ─── GenerationOptions ─────────────────────────────────────────────────────────

/// Tuning parameters for a single generation request.
///
/// All fields are optional; `None` uses the model's built-in default.
#[derive(Debug, Default, Clone)]
pub struct GenerationOptions {
    /// Controls output randomness.
    ///
    /// Range: `0.0` (fully deterministic) to `2.0` (very creative). Default: model-chosen.
    pub temperature: Option<f64>,
    /// Maximum number of tokens to generate.
    ///
    /// The model's session has a combined context window of 4 096 tokens (instructions +
    /// all prompts + all responses). Leaving this `None` lets the model decide.
    pub max_tokens: Option<usize>,
}

impl GenerationOptions {
    /// Returns an error if any field contains an out-of-range value.
    pub fn validate(&self) -> Result<(), Error> {
        if let Some(t) = self.temperature {
            if !(0.0..=2.0).contains(&t) {
                return Err(Error::InvalidTemperature(t));
            }
        }
        Ok(())
    }

    /// Raw temperature value for FFI: the temperature if set, otherwise -1.0 (= use default).
    fn ffi_temperature(&self) -> f64 {
        self.temperature.unwrap_or(-1.0)
    }

    /// Raw max-tokens value for FFI: the limit if set, otherwise -1 (= use default).
    fn ffi_max_tokens(&self) -> i64 {
        self.max_tokens.map(|n| n as i64).unwrap_or(-1)
    }
}

// ─── Availability ──────────────────────────────────────────────────────────────

const FM_AVAILABLE: i32 = 0;
const FM_DEVICE_NOT_ELIGIBLE: i32 = 1;
const FM_NOT_ENABLED: i32 = 2;
const FM_MODEL_NOT_READY: i32 = 3;

/// Returns `true` if Apple Intelligence is available and ready on this device.
///
/// This is a cheap synchronous check. See [`availability`] for the specific reason
/// when this returns `false`.
pub fn is_available() -> bool {
    availability().is_ok()
}

/// Returns `Ok(())` if Apple Intelligence is ready, or the specific [`UnavailabilityReason`].
pub fn availability() -> Result<(), UnavailabilityReason> {
    #[cfg(foundation_models_bridge)]
    {
        let code = unsafe { fm_availability_reason() };
        match code {
            FM_AVAILABLE => Ok(()),
            FM_DEVICE_NOT_ELIGIBLE => Err(UnavailabilityReason::DeviceNotEligible),
            FM_NOT_ENABLED => Err(UnavailabilityReason::NotEnabled),
            FM_MODEL_NOT_READY => Err(UnavailabilityReason::ModelNotReady),
            _ => Err(UnavailabilityReason::Unknown),
        }
    }
    #[cfg(not(foundation_models_bridge))]
    Err(UnavailabilityReason::DeviceNotEligible)
}

// ─── Convenience top-level functions ──────────────────────────────────────────

/// Sends a single prompt to the model and returns the response text.
///
/// Each call creates a fresh session with no prior context. For multi-turn
/// conversations use [`Session`] directly.
///
/// # Errors
///
/// Returns [`Error::Unavailable`] if Apple Intelligence is not available.
pub async fn respond(prompt: &str) -> Result<String, Error> {
    respond_with_options(prompt, &GenerationOptions::default()).await
}

/// Like [`respond`] but allows tuning generation via [`GenerationOptions`].
pub async fn respond_with_options(
    prompt: &str,
    options: &GenerationOptions,
) -> Result<String, Error> {
    let session = Session::new()?;
    session.respond_with_options(prompt, options).await
}

// ─── Session ───────────────────────────────────────────────────────────────────

/// A stateful conversation session backed by a `LanguageModelSession`.
///
/// The session automatically maintains a conversation transcript, so each
/// successive call to [`respond`][Session::respond] has access to the full
/// prior context (subject to the 4 096-token context window limit).
///
/// # Thread safety
///
/// `Session` is `Send + Sync`. Concurrent calls are forwarded to the underlying
/// Swift session, which handles them via its internal async actor. Note however
/// that concurrent calls will interleave entries in the transcript in an
/// unspecified order; for predictable multi-turn behaviour call sequentially.
///
/// # Drop behaviour
///
/// Dropping a `Session` releases the underlying Swift object. Any in-flight
/// `respond` futures or active `ResponseStream`s that still hold a reference to
/// the raw handle will have that handle invalidated. Always ensure futures and
/// streams complete or are dropped before dropping the parent `Session`.
pub struct Session {
    #[cfg(foundation_models_bridge)]
    handle: *mut c_void,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").finish_non_exhaustive()
    }
}

// Safety: The underlying Swift LanguageModelSession is designed for concurrent
// use via Swift's async/await actor system; we never dereference the raw pointer
// from Rust — we only pass it to C-exported Swift functions.
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    /// Creates a new session with no system instructions.
    pub fn new() -> Result<Self, Error> {
        Self::with_instructions("")
    }

    /// Creates a new session with the given system instructions.
    ///
    /// Instructions act as a persistent system prompt that guides all subsequent
    /// responses in this session. They must come from developer code, never from
    /// user input, to prevent prompt-injection attacks.
    pub fn with_instructions(instructions: &str) -> Result<Self, Error> {
        availability().map_err(Error::Unavailable)?;

        #[cfg(foundation_models_bridge)]
        {
            let c_instructions = CString::new(instructions)?;
            let handle = unsafe { fm_session_create(c_instructions.as_ptr()) };
            if handle.is_null() {
                return Err(Error::Unavailable(UnavailabilityReason::Unknown));
            }
            Ok(Self { handle })
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = instructions;
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }

    /// Sends a prompt and returns the full response text.
    ///
    /// The response is appended to this session's transcript, so subsequent
    /// calls have access to prior context.
    pub async fn respond(&self, prompt: &str) -> Result<String, Error> {
        self.respond_with_options(prompt, &GenerationOptions::default())
            .await
    }

    /// Like [`respond`][Session::respond] but allows tuning generation.
    pub async fn respond_with_options(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<String, Error> {
        options.validate()?;

        #[cfg(foundation_models_bridge)]
        {
            let (tx, rx) = oneshot::channel::<Result<String, String>>();
            let ctx = Box::into_raw(Box::new(tx)) as *mut c_void;
            let c_prompt = CString::new(prompt)?;

            unsafe {
                fm_session_respond(
                    self.handle,
                    c_prompt.as_ptr(),
                    options.ffi_temperature(),
                    options.ffi_max_tokens(),
                    ctx,
                    respond_callback,
                );
            }

            rx.await
                .map_err(|_| Error::Generation("session was dropped before responding".into()))?
                .map_err(Error::Generation)
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = (prompt, options);
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }

    /// Returns a [`ResponseStream`] that yields text chunks as the model generates them.
    ///
    /// Each yielded chunk is an incremental snapshot of the response text. Drive the
    /// stream with your preferred async executor.
    ///
    /// ```no_run
    /// # async fn example() -> Result<(), rusty_foundationmodels::Error> {
    /// use rusty_foundationmodels::Session;
    ///
    /// let session = Session::new()?;
    /// let stream = session.stream("Count to ten.")?;
    /// # Ok(()) }
    /// ```
    pub fn stream(&self, prompt: &str) -> Result<ResponseStream, Error> {
        self.stream_with_options(prompt, &GenerationOptions::default())
    }

    /// Like [`stream`][Session::stream] but allows tuning generation.
    pub fn stream_with_options(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<ResponseStream, Error> {
        options.validate()?;

        #[cfg(foundation_models_bridge)]
        {
            let (tx, rx) = mpsc::unbounded::<Result<String, String>>();
            let ctx = Box::into_raw(Box::new(StreamContext { tx })) as *mut c_void;
            let c_prompt = CString::new(prompt)?;

            unsafe {
                fm_session_stream(
                    self.handle,
                    c_prompt.as_ptr(),
                    options.ffi_temperature(),
                    options.ffi_max_tokens(),
                    ctx,
                    stream_token_callback,
                    stream_done_callback,
                );
            }

            Ok(ResponseStream { rx })
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = (prompt, options);
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        #[cfg(foundation_models_bridge)]
        unsafe {
            fm_session_destroy(self.handle);
        }
    }
}

// ─── ResponseStream ────────────────────────────────────────────────────────────

/// An async stream of text chunks produced by [`Session::stream`].
///
/// Each item is `Ok(String)` for a new chunk, or `Err(Error)` if generation failed.
/// The stream ends when the model finishes generating.
///
/// Implements [`futures_core::Stream`]; use with `.next()` from `StreamExt` or
/// any executor that can drive `Stream`.
pub struct ResponseStream {
    rx: mpsc::UnboundedReceiver<Result<String, String>>,
}

impl Stream for ResponseStream {
    type Item = Result<String, Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut StdContext<'_>,
    ) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx)
            .poll_next(cx)
            .map(|opt| opt.map(|r| r.map_err(Error::Generation)))
    }
}

// ─── FFI callbacks ─────────────────────────────────────────────────────────────

/// Callback for single-shot respond. Called exactly once by Swift.
#[cfg(foundation_models_bridge)]
extern "C" fn respond_callback(
    ctx: *mut c_void,
    result: *const c_char,
    error: *const c_char,
) {
    // Safety: ctx is always a Box<oneshot::Sender<...>> allocated in respond_with_options.
    let tx = unsafe { Box::from_raw(ctx as *mut oneshot::Sender<Result<String, String>>) };

    if !error.is_null() {
        let msg = unsafe { CStr::from_ptr(error).to_string_lossy().into_owned() };
        tx.send(Err(msg)).ok();
    } else if !result.is_null() {
        let text = unsafe { CStr::from_ptr(result).to_string_lossy().into_owned() };
        tx.send(Ok(text)).ok();
    }
}

/// Internal state for a streaming request; owned by the Swift Task via a raw pointer.
#[cfg(foundation_models_bridge)]
struct StreamContext {
    tx: mpsc::UnboundedSender<Result<String, String>>,
}

/// Token callback for streaming. May be called many times before stream_done_callback.
#[cfg(foundation_models_bridge)]
extern "C" fn stream_token_callback(ctx: *mut c_void, token: *const c_char) {
    // Safety: ctx is a Box<StreamContext> allocated in stream_with_options; it remains
    // valid until stream_done_callback drops it.
    let stream_ctx = unsafe { &*(ctx as *const StreamContext) };
    let text = unsafe { CStr::from_ptr(token).to_string_lossy().into_owned() };
    // Failure here means the Rust ResponseStream was dropped; ignore silently.
    stream_ctx.tx.unbounded_send(Ok(text)).ok();
}

/// Done callback for streaming. Called exactly once; takes ownership of StreamContext.
#[cfg(foundation_models_bridge)]
extern "C" fn stream_done_callback(ctx: *mut c_void, error: *const c_char) {
    // Safety: takes ownership of the Box<StreamContext> that was created in stream_with_options.
    let stream_ctx = unsafe { Box::from_raw(ctx as *mut StreamContext) };
    if !error.is_null() {
        let msg = unsafe { CStr::from_ptr(error).to_string_lossy().into_owned() };
        stream_ctx.tx.unbounded_send(Err(msg)).ok();
    }
    // stream_ctx drops here, closing the channel and ending the ResponseStream.
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Always-runnable unit tests ────────────────────────────────────────────

    #[test]
    fn test_is_available_returns_without_panic() {
        let _ = is_available();
    }

    #[test]
    fn test_availability_result_is_consistent() {
        let avail = availability();
        assert_eq!(is_available(), avail.is_ok());
    }

    #[test]
    fn test_options_default_is_valid() {
        let opts = GenerationOptions::default();
        assert!(opts.validate().is_ok());
        assert_eq!(opts.ffi_temperature(), -1.0);
        assert_eq!(opts.ffi_max_tokens(), -1);
    }

    #[test]
    fn test_options_valid_temperature() {
        for temp in [0.0_f64, 1.0, 2.0] {
            let opts = GenerationOptions { temperature: Some(temp), ..Default::default() };
            assert!(opts.validate().is_ok(), "temperature {temp} should be valid");
        }
    }

    #[test]
    fn test_options_invalid_temperature() {
        for temp in [-0.1_f64, 2.001, f64::INFINITY, f64::NAN] {
            let opts = GenerationOptions { temperature: Some(temp), ..Default::default() };
            assert!(
                opts.validate().is_err(),
                "temperature {temp} should be invalid"
            );
        }
    }

    #[test]
    fn test_session_creation_fails_gracefully_when_unavailable() {
        if is_available() {
            return; // skip — integration tests cover the available path
        }
        let err = Session::new().unwrap_err();
        assert!(matches!(err, Error::Unavailable(_)));
    }

    #[test]
    fn test_null_byte_in_prompt_returns_error() {
        if !is_available() {
            return;
        }
        let result = futures_executor::block_on(respond("hello\0world"));
        assert!(matches!(result, Err(Error::NullByte(_))));
    }

    // ── Integration tests (require Apple Intelligence) ─────────────────────────
    //
    // Run with:  cargo test -- --include-ignored

    #[test]
    #[ignore = "requires Apple Intelligence (macOS 26+, Apple Silicon, AI enabled)"]
    fn test_simple_respond() {
        let response = futures_executor::block_on(respond(
            "Reply with only the number: what is 2 + 2?",
        ))
        .expect("respond failed");
        assert!(response.contains('4'), "expected '4' in: {response:?}");
    }

    #[test]
    #[ignore = "requires Apple Intelligence"]
    fn test_respond_with_low_temperature() {
        let opts = GenerationOptions {
            temperature: Some(0.0),
            ..Default::default()
        };
        let r = futures_executor::block_on(respond_with_options(
            "Reply with only the word: capital of France?",
            &opts,
        ))
        .expect("respond_with_options failed");
        assert!(
            r.to_lowercase().contains("paris"),
            "expected Paris in: {r:?}"
        );
    }

    #[test]
    #[ignore = "requires Apple Intelligence"]
    fn test_multi_turn_session() {
        let session =
            Session::with_instructions("Reply to every message with exactly one word.")
                .expect("Session::with_instructions failed");
        let r1 = futures_executor::block_on(session.respond("Say hello."))
            .expect("first respond failed");
        let r2 = futures_executor::block_on(session.respond("Say goodbye."))
            .expect("second respond failed");
        assert!(!r1.is_empty(), "first response was empty");
        assert!(!r2.is_empty(), "second response was empty");
    }

    #[test]
    #[ignore = "requires Apple Intelligence"]
    fn test_streaming_yields_chunks() {
        let session = Session::new().expect("Session::new failed");
        let stream = session.stream("Count: one two three").expect("stream failed");

        // Collect all chunks — block_on_stream drives the stream synchronously;
        // it cannot be called from inside another block_on.
        let chunks: Vec<String> = futures_executor::block_on_stream(stream)
            .map(|r| r.expect("stream item was error"))
            .collect();

        assert!(!chunks.is_empty(), "stream produced no chunks");
        let full = chunks.join("");
        assert!(!full.is_empty(), "concatenated response was empty");
    }
}
