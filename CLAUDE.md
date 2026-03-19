# rusty_foundationmodels — developer notes

## Architecture

This crate is a thin safe Rust wrapper around Apple's FoundationModels framework. All platform-specific code lives in two files:

- `bridge.swift` — Swift source compiled at build time into a static library. Exports C-ABI functions via `@_cdecl`.
- `build.rs` — Invokes `xcrun swiftc` to compile `bridge.swift`. Sets the `foundation_models_bridge` Cargo `cfg` on success; emits a warning and skips silently on failure.
- `src/lib.rs` — All public Rust API. FFI declarations, safe wrappers, async bridging via `futures_channel`.

## Build behaviour

The `foundation_models_bridge` cfg is **only set when the Swift bridge compiles successfully**. All code paths that touch FFI are gated behind `#[cfg(foundation_models_bridge)]`. When the cfg is absent, every public API returns `Err(Error::Unavailable)` — no panics, no linker errors.

## Async bridging pattern

- **Single-shot** (`respond`): A `Box<oneshot::Sender<Result<String, String>>>` is heap-allocated and passed as a raw `*mut c_void` context pointer to Swift. The Swift `Task` calls `respond_callback` exactly once, which reconstitutes the box via `Box::from_raw`, sends the result, and drops the box. The Rust `async fn` awaits the `oneshot::Receiver`.

- **Streaming** (`stream`): A `Box<StreamContext>` containing an `mpsc::UnboundedSender` is heap-allocated and passed to Swift. The `stream_token_callback` borrows the pointer (does NOT take ownership) to send each chunk. The `stream_done_callback` takes ownership via `Box::from_raw`, optionally sends an error, then drops the box — which closes the channel and terminates the `ResponseStream`.

## Session handle lifecycle

`fm_session_create` returns an ARC-retained Swift `SessionHolder` as an opaque `*mut c_void`. Rust stores this in `Session.handle`. `Drop` for `Session` calls `fm_session_destroy`, which calls `Unmanaged.fromOpaque(ptr).release()`. Every live `respond` future or `ResponseStream` must complete or be dropped before the parent `Session` is dropped, since they hold a raw copy of the handle.

## Swift API surface used

From the macOS 26 SDK (`FoundationModels.framework`):

- `SystemLanguageModel.default.availability` — sync availability check
- `LanguageModelSession(instructions:)` — session creation
- `session.respond(to:options:)` — async single-shot, returns `Response<String>`; access text via `.content`
- `session.streamResponse(to:options:)` — returns `ResponseStream<String>`; iterate with `for try await chunk in ...`; access text via `chunk.content`
- `GenerationOptions` — struct with `temperature: Double?` and `maximumResponseTokens: Int?`

## Running tests

Unit tests (always runnable):
```sh
cargo test
```

Integration tests (require macOS 26, Apple Silicon, Apple Intelligence enabled):
```sh
cargo test -- --include-ignored
```

## Linting

```sh
cargo clippy -- -D warnings
```

## Publishing checklist

- [ ] Bump version in `Cargo.toml`
- [ ] `cargo test -- --include-ignored` passes (requires Apple Intelligence hardware)
- [ ] `cargo doc --open` renders correctly
- [ ] `cargo publish --dry-run` succeeds
