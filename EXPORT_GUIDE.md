# Codon-Infer Project Export Guide

## Overview
This guide summarizes the essential information needed to export, build, and use the Codon-Infer project.

## Project Structure
```
codon-infer/
├── src/
│   ├── lib.rs          # Core library: EsmConfig, EsmWeights, RunState, primitives, forward
│   ├── bin/
│   │   ├── cli.rs      # codon-cli entry point
│   │   └── mcp.rs      # codon-mcp entry point
│   └── mcp/            # MCP protocol implementation
├── tests/
│   ├── parity_test.rs  # Numerical parity vs PyTorch
│   └── mcp_test.rs     # MCP integration tests
├── benches/
│   └── inference_bench.rs
├── tools/
│   ├── export_weights.py
│   └── export_parity_reference.py
├── scripts/
│   └── run_autonomous.sh
├── Cargo.toml
├── README.md
├── CHANGELOG.md
├── CHECKPOINTS.md
├── AGENTS.md
└─ EXPORT_GUIDE.md   (this file)
```

## Dependencies
- **Rust**: stable channel (tested with 1.70+)
- **libm** = "0.2" (for erf function in exact GELU)
- **serde** = { version = "1.0", features = ["derive"] }
- **serde_json** = "1.0"
- **Development only**:
  - criterion = "0.5" (for benchmarks)

All dependencies are specified in `Cargo.toml`.

## Build Instructions
### Standard Build (dynamic linking)
```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

### Static Build (musl target)
Requires `x86_64-unknown-linux-musl` target and musl toolchain installed.
```bash
# Add target if not present
rustup target add x86_64-unknown-linux-musl

# Build static release binaries
cargo build --release --target x86_64-unknown-linux-musl
```
Output binaries:
- `target/x86_64-unknown-linux-musl/release/codon-cli`
- `target/x86_64-unknown-linux-musl/release/codon-mcp`

These are fully static (static-pie linked) and can run on any compatible Linux distribution without additional Rust runtime.

## Usage

### codon-cli (Protein Sequence Generation)
```bash
# Basic usage
./target/release/codon-cli --sequence "MKT"

# Custom model weights
./target/release/codon-cli --model /path/to/esm2_8m_weights.bin --sequence "MKT"

# Help (no explicit flag; defaults to sequence "M" and prints report)
./target/release/codon-cli
```
The CLI loads the model, processes the prompt sequence, then generates proteins autoregressively with stochastic sampling (temperature 0.7 by default). It prints a performance report upon completion.

### codon-mcp (MCP Server)
```bash
# Start server (JSON-RPC over stdio)
./target/release/codon-mcp
```
The server waits for JSON-RPC requests on stdin and writes responses to stdout. It exposes a single tool: `generate_protein`.

#### Example Interaction
```bash
# Initialize
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' |
./target/release/codon-mcp

# List tools
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' |
./target/release/codon-mcp

# Call generate_protein
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"generate_protein","arguments":{"prompt":"MK","max_tokens":5,"temperature":0.0}}}' |
./target/release/codon-mcp
```
Each request should return a valid JSON-RPC response.

## Testing
```bash
# Unit tests (library)
cargo test --lib

# Numerical parity vs PyTorch
cargo test --test parity_test

# MCP integration tests
cargo test --test mcp_test

# All tests
cargo test
```
All tests should pass with no warnings when `cargo clippy --all-targets -- -D warnings` is run.

## Benchmarks
```bash
cargo bench
```
Runs criterion benchmarks measuring:
- Prefill latency
- Decode throughput
- KV cache memory usage
- Full inference latency

Results are printed in the terminal; detailed reports appear in `target/criterion`.

## Zero‑Allocation Hot Path Guarantee
The functions `forward()`, `layer_norm`, `matmul`, `gelu`, `softmax`, `apply_rope` in `src/lib.rs` perform **no allocations**:
- No `vec![]`, `Box::new()`, `.to_vec()`, `.clone()`
- All mutations are in‑place on `&mut [f32]` slices supplied by `RunState`

## Cross‑Platform Notes
- Static binaries built for `x86_64-unknown-linux-musl` run on most modern Linux distributions.
- For other targets (e.g., Windows, macOS) use the standard `cargo build --release` which produces dynamically linked executables; ensure the Rust runtime and dependencies are present.

## License
MIT – see `LICENSE` file (if present) or the repository header.

## Version
- Current: 0.2.0 (see `Cargo.toml` and `CHANGELOG.md`)

--- 
*This guide was generated from the Codon-Infer repository as of 2026-10-05.*