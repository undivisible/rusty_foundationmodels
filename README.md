# rusty_foundationmodels

Safe Rust bindings for Apple's [FoundationModels] on-device AI framework (Apple Intelligence).

Run a local ≈3B-parameter language model entirely on-device — no network requests, no API keys, no data leaving the device.

## Requirements

| Requirement | Value |
|---|---|
| macOS | 26 (Tahoe) or later |
| Hardware | Apple Silicon (M1 or later) |
| Setting | Apple Intelligence enabled in System Settings |
| Build tool | Xcode with the macOS 26 SDK |

The crate still **compiles and links** on unsupported hardware or older macOS; all APIs return `Err(Error::Unavailable)` at runtime so you can handle the fallback gracefully.

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
rusty_foundationmodels = "0.1"
```

### Single-shot response

```rust
use rusty_foundationmodels::{is_available, respond};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !is_available() {
        eprintln!("Apple Intelligence not available on this device");
        return Ok(());
    }

    let answer = respond("What is the capital of France?").await?;
    println!("{answer}");
    Ok(())
}
```

### Multi-turn conversation

```rust
use rusty_foundationmodels::Session;

let mut session = Session::with_instructions("You are a concise Rust expert.")?;
let r1 = session.respond("What is ownership?").await?;
let r2 = session.respond("Give me a one-line example.").await?;
println!("{r1}\n{r2}");
```

### Streaming

```rust
use rusty_foundationmodels::Session;
use futures_util::StreamExt as _;

let session = Session::new()?;
let mut stream = session.stream("Tell me a short story.")?;

while let Some(chunk) = stream.next().await {
    print!("{}", chunk?);
}
println!();
```

### Structured generation

Describe the shape of the output with a `Schema` and the model returns a value that deserialises into your type:

```rust
use rusty_foundationmodels::{Session, Schema, SchemaProperty, SchemaPropertyType};
use serde::Deserialize;

#[derive(Deserialize)]
struct Capital {
    city: String,
    country: String,
    population: i64,
}

let session = Session::new()?;
let schema = Schema::new("Capital", "A country capital")
    .property(SchemaProperty::new("city", SchemaPropertyType::String).description("City name"))
    .property(SchemaProperty::new("country", SchemaPropertyType::String).description("Country name"))
    .property(SchemaProperty::new("population", SchemaPropertyType::Integer).description("Approximate population"));

let capital: Capital = session.respond_as("What is the capital of France?", &schema).await?;
println!("{} is in {} with ~{} people", capital.city, capital.country, capital.population);
```

### Tool calling

Define Rust functions that the model can invoke during generation:

```rust
use rusty_foundationmodels::{Session, Schema, SchemaProperty, SchemaPropertyType, ToolDefinition};
use serde_json::Value;

let tools = vec![
    ToolDefinition::new(
        "get_weather",
        "Return current weather for a city",
        Schema::new("WeatherArgs", "Arguments for get_weather")
            .property(SchemaProperty::new("city", SchemaPropertyType::String).description("City name")),
    )
    .handler(|args: Value| {
        let city = args["city"].as_str().unwrap_or("unknown");
        Ok(format!("{city}: 22°C, sunny"))
    }),
];

let mut session = Session::with_tools(tools, None)?;
let response = session.respond("What is the weather in Tokyo?").await?;
println!("{response}");
```

### Generation options

```rust
use rusty_foundationmodels::{respond_with_options, GenerationOptions};

let opts = GenerationOptions {
    temperature: Some(0.2),
    max_tokens: Some(256),
};
let response = respond_with_options("Summarize Rust in one sentence.", &opts).await?;
```

### Availability checking

```rust
use rusty_foundationmodels::{availability, UnavailabilityReason};

match availability() {
    Ok(()) => println!("Apple Intelligence is ready"),
    Err(UnavailabilityReason::DeviceNotEligible) => {
        eprintln!("Requires Apple Silicon M1 or later")
    }
    Err(UnavailabilityReason::NotEnabled) => {
        eprintln!("Enable Apple Intelligence in System Settings → Apple Intelligence & Siri")
    }
    Err(UnavailabilityReason::ModelNotReady) => {
        eprintln!("Model is still downloading, try again in a few minutes")
    }
    Err(UnavailabilityReason::Unknown) => eprintln!("Unknown availability state"),
}
```

## Running the integration tests

The integration tests require Apple Intelligence to be active. Once you have macOS 26 on Apple Silicon with Apple Intelligence enabled:

```sh
cargo test -- --include-ignored
```

## How it works

The crate ships a Swift source file (`bridge.swift`) that is compiled by `build.rs` at build time using `xcrun swiftc`. The resulting static library exposes a small set of C-ABI functions that Rust calls through an `unsafe extern "C"` block. Swift's `async/await` tasks are bridged to Rust futures using `futures_channel` oneshot and mpsc channels.

If `xcrun` is not found or compilation fails (e.g. on non-macOS hosts or without the macOS 26 SDK), the build still succeeds — the Swift bridge is simply skipped and all public APIs return `Err(Error::Unavailable)` at runtime.

## License

Licensed under the [Mozilla Public License 2.0](LICENSE).

[FoundationModels]: https://developer.apple.com/documentation/foundationmodels
