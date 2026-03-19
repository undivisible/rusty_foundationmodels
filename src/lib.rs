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
//! - **Structured generation** — [`Session::respond_as`] returns any `serde::Deserialize` type
//! - **Tool calling** — [`Session::with_tools`] registers Rust closures the model can invoke
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
//! let session = Session::with_instructions("You are a concise Rust expert.")?;
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
//! use rusty_foundationmodels::Session;
//!
//! let session = Session::new()?;
//! let stream = session.stream("Tell me a short story.")?;
//! # Ok(()) }
//! ```
//!
//! # Structured generation
//!
//! ```no_run
//! # async fn example() -> Result<(), rusty_foundationmodels::Error> {
//! use serde::Deserialize;
//! use rusty_foundationmodels::{Session, Schema, SchemaProperty, SchemaPropertyType};
//!
//! #[derive(Deserialize)]
//! struct CityInfo { name: String, population: f64, country: String }
//!
//! let session = Session::new()?;
//! let schema = Schema::new("CityInfo")
//!     .property(SchemaProperty::new("name", SchemaPropertyType::String))
//!     .property(SchemaProperty::new("population", SchemaPropertyType::Double))
//!     .property(SchemaProperty::new("country", SchemaPropertyType::String));
//!
//! let info: CityInfo = session.respond_as("Describe Paris.", &schema).await?;
//! println!("{} has {} people", info.name, info.population);
//! # Ok(()) }
//! ```
//!
//! # Tool calling
//!
//! ```no_run
//! # async fn example() -> Result<(), rusty_foundationmodels::Error> {
//! use rusty_foundationmodels::{Session, ToolDefinition, Schema, SchemaProperty, SchemaPropertyType};
//!
//! let tool = ToolDefinition::new(
//!     "get_weather",
//!     "Get current weather for a city",
//!     Schema::new("GetWeatherArgs")
//!         .property(SchemaProperty::new("city", SchemaPropertyType::String)
//!             .description("City name")),
//!     |args| {
//!         let city = args["city"].as_str().unwrap_or("unknown");
//!         Ok(format!("Weather in {city}: sunny, 72°F"))
//!     },
//! );
//!
//! let session = Session::with_tools("You are a weather assistant.", vec![tool])?;
//! let response = session.respond("What's the weather in Tokyo?").await?;
//! println!("{response}");
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
use std::ptr::null;

#[cfg(foundation_models_bridge)]
use std::sync::Arc;

#[cfg(foundation_models_bridge)]
use futures_channel::{mpsc, oneshot};

// ─── FFI declarations ──────────────────────────────────────────────────────────

#[cfg(foundation_models_bridge)]
unsafe extern "C" {
    fn fm_availability_reason() -> i32;
    fn fm_session_create(instructions: *const c_char) -> *mut c_void;
    fn fm_session_create_with_tools(
        instructions: *const c_char,
        tools_json: *const c_char,
        tool_ctx: *mut c_void,
        tool_dispatch: extern "C" fn(
            *mut c_void,
            *const c_char,
            *const c_char,
            *mut c_void,
            extern "C" fn(*mut c_void, *const c_char, *const c_char),
        ),
    ) -> *mut c_void;
    fn fm_session_destroy(handle: *mut c_void);
    fn fm_session_respond(
        handle: *mut c_void,
        prompt: *const c_char,
        temperature: f64,
        max_tokens: i64,
        ctx: *mut c_void,
        callback: extern "C" fn(*mut c_void, *const c_char, *const c_char),
    );
    fn fm_session_respond_structured(
        handle: *mut c_void,
        prompt: *const c_char,
        schema_json: *const c_char,
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

    /// JSON serialisation or deserialisation failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A tool invoked by the model returned an error.
    #[error("tool '{name}' failed: {message}")]
    ToolError { name: String, message: String },
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

    fn ffi_temperature(&self) -> f64 {
        self.temperature.unwrap_or(-1.0)
    }

    fn ffi_max_tokens(&self) -> i64 {
        self.max_tokens.map(|n| n as i64).unwrap_or(-1)
    }
}

// ─── Schema types for structured generation ────────────────────────────────────

/// The type of a single property in a [`Schema`].
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SchemaPropertyType {
    /// UTF-8 text.
    String,
    /// Whole number (serialised as JSON integer).
    Integer,
    /// Floating-point number.
    Double,
    /// Boolean true/false.
    Bool,
}

/// A single property within a [`Schema`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct SchemaProperty {
    /// Property name (matches the JSON key in the model output).
    pub name: String,
    /// Optional human-readable hint that guides the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The expected type of this property.
    #[serde(rename = "type")]
    pub property_type: SchemaPropertyType,
    /// Whether the model may omit this property.
    #[serde(default)]
    pub optional: bool,
}

impl SchemaProperty {
    /// Creates a required property with the given name and type.
    pub fn new(name: impl Into<String>, property_type: SchemaPropertyType) -> Self {
        Self {
            name: name.into(),
            description: None,
            property_type,
            optional: false,
        }
    }

    /// Attaches a human-readable description that guides the model.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Marks this property as optional (the model may omit it).
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }
}

/// Describes the JSON object shape that the model must produce for structured generation.
///
/// Build one using the builder methods, then pass it to [`Session::respond_as`].
///
/// ```
/// use rusty_foundationmodels::{Schema, SchemaProperty, SchemaPropertyType};
///
/// let schema = Schema::new("Point")
///     .property(SchemaProperty::new("x", SchemaPropertyType::Double))
///     .property(SchemaProperty::new("y", SchemaPropertyType::Double));
/// ```
#[derive(Debug, Clone, serde::Serialize)]
pub struct Schema {
    /// Internal type name used by the model's structured generation system.
    pub name: String,
    /// Optional description of what this type represents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The properties the model must populate.
    pub properties: Vec<SchemaProperty>,
}

impl Schema {
    /// Creates a new empty schema with the given type name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            properties: Vec::new(),
        }
    }

    /// Attaches a description of this type.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Adds a property to this schema.
    pub fn property(mut self, property: SchemaProperty) -> Self {
        self.properties.push(property);
        self
    }
}

// ─── Tool calling ──────────────────────────────────────────────────────────────

/// A function that the model can invoke when responding to a prompt.
///
/// The `handler` receives the model's arguments as a [`serde_json::Value`] and must return
/// either a string result (delivered back to the model) or an error string.
///
/// Build one with [`ToolDefinition::new`].
pub struct ToolDefinition {
    /// Name the model uses to reference this tool. Must be unique within a session.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// Schema describing the arguments the model must supply when calling this tool.
    pub parameters: Schema,
    pub(crate) handler: Box<dyn Fn(serde_json::Value) -> Result<String, String> + Send + Sync>,
}

impl ToolDefinition {
    /// Creates a new tool definition.
    ///
    /// - `name`: identifier used by the model.
    /// - `description`: explains what the tool does.
    /// - `parameters`: schema for the tool's input arguments.
    /// - `handler`: closure called when the model invokes the tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Schema,
        handler: impl Fn(serde_json::Value) -> Result<String, String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            handler: Box::new(handler),
        }
    }
}

impl std::fmt::Debug for ToolDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolDefinition")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// Internal context that holds tool handlers. A raw pointer to this is passed to Swift
/// as `tool_ctx` and lives for the full `Session` lifetime via `Arc`.
#[cfg(foundation_models_bridge)]
struct ToolsContext {
    tools: Vec<(String, Box<dyn Fn(serde_json::Value) -> Result<String, String> + Send + Sync>)>,
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
    /// Keeps the tool handlers alive for the full session lifetime.
    /// A raw pointer to the Arc payload is passed to Swift as `tool_ctx`.
    #[cfg(foundation_models_bridge)]
    _tools: Option<Arc<ToolsContext>>,
}

// Safety: The underlying Swift LanguageModelSession is designed for concurrent
// use via Swift's async/await actor system; we never dereference the raw pointer
// from Rust — we only pass it to C-exported Swift functions.
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").finish_non_exhaustive()
    }
}

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
            Ok(Self { handle, _tools: None })
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = instructions;
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }

    /// Creates a session pre-loaded with the given tools.
    ///
    /// The model will use these tools automatically when appropriate during `respond` calls.
    /// Tool names must be unique within the session.
    pub fn with_tools(instructions: &str, tools: Vec<ToolDefinition>) -> Result<Self, Error> {
        availability().map_err(Error::Unavailable)?;

        #[cfg(foundation_models_bridge)]
        {
            // Serialize tool definitions for Swift.
            let tool_descs: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "properties": t.parameters.properties,
                    })
                })
                .collect();
            let tools_json = serde_json::to_string(&tool_descs)?;

            let tools_ctx = Arc::new(ToolsContext {
                tools: tools
                    .into_iter()
                    .map(|t| (t.name, t.handler))
                    .collect(),
            });
            // Pass the raw Arc payload pointer to Swift. The Arc keeps it alive.
            let tool_ctx_ptr = Arc::as_ptr(&tools_ctx) as *mut c_void;

            let c_instructions = CString::new(instructions)?;
            let c_tools_json = CString::new(tools_json)?;

            let handle = unsafe {
                fm_session_create_with_tools(
                    c_instructions.as_ptr(),
                    c_tools_json.as_ptr(),
                    tool_ctx_ptr,
                    tool_dispatch,
                )
            };
            if handle.is_null() {
                return Err(Error::Unavailable(UnavailabilityReason::Unknown));
            }
            Ok(Self { handle, _tools: Some(tools_ctx) })
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = (instructions, tools);
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

    /// Sends a prompt and deserialises the response into `T` using the provided schema.
    ///
    /// The model generates output conforming to `schema` and this method deserialises it.
    /// Derive [`serde::Deserialize`] on `T` and ensure the field names match the schema
    /// property names exactly.
    pub async fn respond_as<T: serde::de::DeserializeOwned>(
        &self,
        prompt: &str,
        schema: &Schema,
    ) -> Result<T, Error> {
        self.respond_as_with_options(prompt, schema, &GenerationOptions::default())
            .await
    }

    /// Like [`respond_as`][Session::respond_as] but allows tuning generation.
    pub async fn respond_as_with_options<T: serde::de::DeserializeOwned>(
        &self,
        prompt: &str,
        schema: &Schema,
        options: &GenerationOptions,
    ) -> Result<T, Error> {
        options.validate()?;

        #[cfg(foundation_models_bridge)]
        {
            let (tx, rx) = oneshot::channel::<Result<String, String>>();
            let ctx = Box::into_raw(Box::new(tx)) as *mut c_void;
            let c_prompt = CString::new(prompt)?;
            let schema_json = serde_json::to_string(schema)?;
            let c_schema_json = CString::new(schema_json)?;

            unsafe {
                fm_session_respond_structured(
                    self.handle,
                    c_prompt.as_ptr(),
                    c_schema_json.as_ptr(),
                    options.ffi_temperature(),
                    options.ffi_max_tokens(),
                    ctx,
                    respond_callback,
                );
            }

            let json = rx
                .await
                .map_err(|_| Error::Generation("session was dropped before responding".into()))?
                .map_err(Error::Generation)?;
            Ok(serde_json::from_str(&json)?)
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = (prompt, schema, options);
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

/// Callback for single-shot respond and structured respond. Called exactly once by Swift.
#[cfg(foundation_models_bridge)]
extern "C" fn respond_callback(
    ctx: *mut c_void,
    result: *const c_char,
    error: *const c_char,
) {
    // Safety: ctx is always a Box<oneshot::Sender<...>> allocated in respond_with_options
    // or respond_as_with_options.
    let tx = unsafe { Box::from_raw(ctx as *mut oneshot::Sender<Result<String, String>>) };

    if !error.is_null() {
        let msg = unsafe { CStr::from_ptr(error).to_string_lossy().into_owned() };
        tx.send(Err(msg)).ok();
    } else if !result.is_null() {
        let text = unsafe { CStr::from_ptr(result).to_string_lossy().into_owned() };
        tx.send(Ok(text)).ok();
    }
}

/// Internal state for a streaming request; owned by Swift via raw pointer until stream_done_callback.
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

/// Dispatches a tool call from Swift to the appropriate Rust handler in `ToolsContext`.
/// Calls `result_cb(result_ctx, result, null)` or `result_cb(result_ctx, null, error)`.
#[cfg(foundation_models_bridge)]
extern "C" fn tool_dispatch(
    ctx: *mut c_void,
    name_ptr: *const c_char,
    args_ptr: *const c_char,
    result_ctx: *mut c_void,
    result_cb: extern "C" fn(*mut c_void, *const c_char, *const c_char),
) {
    // Safety: ctx is Arc::as_ptr(&tools_ctx) cast to *mut c_void; the Arc outlives this call.
    let tools = unsafe { &*(ctx as *const ToolsContext) };
    let name = unsafe { CStr::from_ptr(name_ptr).to_string_lossy() };
    let args_str = unsafe { CStr::from_ptr(args_ptr).to_string_lossy() };

    let args: serde_json::Value = match serde_json::from_str(&args_str) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("invalid tool args JSON: {e}");
            if let Ok(c) = CString::new(msg) {
                result_cb(result_ctx, null(), c.as_ptr());
            }
            return;
        }
    };

    match tools.tools.iter().find(|(n, _)| n == name.as_ref()) {
        Some((_, handler)) => match handler(args) {
            Ok(result) => {
                if let Ok(c) = CString::new(result) {
                    result_cb(result_ctx, c.as_ptr(), null());
                }
            }
            Err(err) => {
                if let Ok(c) = CString::new(err) {
                    result_cb(result_ctx, null(), c.as_ptr());
                }
            }
        },
        None => {
            let msg = format!("unknown tool: {name}");
            if let Ok(c) = CString::new(msg) {
                result_cb(result_ctx, null(), c.as_ptr());
            }
        }
    }
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
            assert!(opts.validate().is_err(), "temperature {temp} should be invalid");
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

    #[test]
    fn test_schema_builder() {
        let schema = Schema::new("Point")
            .description("A 2D point")
            .property(SchemaProperty::new("x", SchemaPropertyType::Double).description("X axis"))
            .property(SchemaProperty::new("y", SchemaPropertyType::Double));
        assert_eq!(schema.name, "Point");
        assert_eq!(schema.properties.len(), 2);
        let json = serde_json::to_string(&schema).unwrap();
        assert!(json.contains("\"x\""));
        assert!(json.contains("\"double\""));
    }

    #[test]
    fn test_tool_definition_builder() {
        let tool = ToolDefinition::new(
            "add",
            "Add two numbers",
            Schema::new("AddArgs")
                .property(SchemaProperty::new("a", SchemaPropertyType::Double))
                .property(SchemaProperty::new("b", SchemaPropertyType::Double)),
            |args| {
                let a = args["a"].as_f64().unwrap_or(0.0);
                let b = args["b"].as_f64().unwrap_or(0.0);
                Ok(format!("{}", a + b))
            },
        );
        assert_eq!(tool.name, "add");
        let result = (tool.handler)(serde_json::json!({"a": 3.0, "b": 4.0}));
        assert_eq!(result.unwrap(), "7");
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
        assert!(r.to_lowercase().contains("paris"), "expected Paris in: {r:?}");
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

        let chunks: Vec<String> = futures_executor::block_on_stream(stream)
            .map(|r| r.expect("stream item was error"))
            .collect();

        assert!(!chunks.is_empty(), "stream produced no chunks");
        let full = chunks.join("");
        assert!(!full.is_empty(), "concatenated response was empty");
    }

    #[test]
    #[ignore = "requires Apple Intelligence"]
    fn test_structured_generation() {
        use serde::Deserialize;

        #[derive(Debug, Deserialize)]
        struct MathAnswer {
            value: f64,
            explanation: String,
        }

        let session = Session::new().expect("Session::new failed");
        let schema = Schema::new("MathAnswer")
            .description("A numeric answer with a brief explanation")
            .property(
                SchemaProperty::new("value", SchemaPropertyType::Double)
                    .description("The numeric result"),
            )
            .property(
                SchemaProperty::new("explanation", SchemaPropertyType::String)
                    .description("One-sentence explanation"),
            );

        let answer: MathAnswer =
            futures_executor::block_on(session.respond_as("What is 6 × 7?", &schema))
                .expect("respond_as failed");

        assert!(
            (answer.value - 42.0).abs() < 0.5,
            "expected 42, got {}",
            answer.value
        );
        assert!(!answer.explanation.is_empty(), "explanation was empty");
    }

    #[test]
    #[ignore = "requires Apple Intelligence"]
    fn test_tool_calling() {
        let tool = ToolDefinition::new(
            "add_numbers",
            "Add two numbers together and return the sum",
            Schema::new("AddArgs")
                .property(
                    SchemaProperty::new("a", SchemaPropertyType::Double)
                        .description("First number"),
                )
                .property(
                    SchemaProperty::new("b", SchemaPropertyType::Double)
                        .description("Second number"),
                ),
            |args| {
                let a = args["a"].as_f64().unwrap_or(0.0);
                let b = args["b"].as_f64().unwrap_or(0.0);
                Ok(format!("{}", a + b))
            },
        );

        let session = Session::with_tools(
            "You are a calculator. Use the add_numbers tool when asked to add.",
            vec![tool],
        )
        .expect("Session::with_tools failed");

        let response = futures_executor::block_on(session.respond("What is 15 + 27?"))
            .expect("respond failed");

        assert!(
            response.contains("42"),
            "expected 42 in response: {response:?}"
        );
    }
}
