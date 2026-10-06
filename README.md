# Codon-Infer

Production-ready Rust library for ESM-2 protein language model inference with zero-allocation hot paths.

## Architecture

```
codon-infer/
├── src/
│   ├── lib.rs          # Core library: EsmConfig, EsmWeights, RunState, primitives, forward
│   ├── bin/
│   │   ├── cli.rs      # codon-cli: High-performance terminal interface
│   │   └── mcp.rs      # codon-mcp: Local MCP server (JSON-RPC over stdio)
│   └── mcp/            # MCP protocol implementation
├── tests/
│   ├── parity_test.rs  # Numerical parity vs PyTorch
│   └── mcp_test.rs     # MCP integration tests
├── benches/
│   └── inference_bench.rs  # Criterion benchmarks
├── tools/
│   ├── export_weights.py
│   └── export_parity_reference.py
└── scripts/
    └── run_autonomous.sh
```

## Installation

```bash
cargo build --release
```

## Usage

### CLI (`codon-cli`)

```bash
# Generate protein sequence from prompt
cargo run --release --bin codon-cli -- --sequence "MCMPCFTTDHQMARKCDD"

# Use custom model weights
cargo run --release --bin codon-cli -- --model esm2_8m_weights.bin --sequence "MKT..."
```

### MCP Server (`codon-mcp`)

```bash
# Start MCP server (JSON-RPC over stdio)
cargo run --release --bin codon-mcp
```

The server exposes the `generate_protein` tool for AI clients.

## Configuration

ESM-2 8M (esm2_t6_8M_UR50D) architecture:
- `dim`: 320
- `hidden_dim`: 1280
- `n_layers`: 6
- `n_heads`: 20
- `vocab_size`: 33
- `max_seq_len`: 1024

## Zero-Allocation Guarantees

All hot-path operations (`forward`, primitives) use in-place mutations on `&mut [f32]` slices provided by `RunState`. No `vec![]`, `Box::new()`, `.to_vec()`, or `.clone()` in the hot path.

## Benchmarks

Run with Criterion:

```bash
cargo bench
```

| Metric | Value |
|--------|-------|
| Prefill Latency (128 tokens) | ~ms |
| Decode Throughput | ~tokens/sec |
| KV Cache Memory (1024 ctx) | ~MB |

## Testing

- Unit tests: `cargo test --lib`
- Integration tests (parity): `cargo test --test parity_test` (Note: slow in debug mode due to O(n^2) attention complexity; recommend running in release mode for performance)
- Integration tests (MCP): `cargo test --test mcp_test`
- All tests: `cargo test`

## Numerical Parity

Target: MAE < 1e-3 vs PyTorch `esm2_t6_8M_UR50D` (FP32).

Current status: **In progress** - architecture alignment needed.

## Model Weights

Export from fair-esm:

```bash
python tools/export_weights.py
```

Generates `esm2_8m_weights.bin` with PyTorch-native weight layout.

## Development

### Autonomous Nightly Sessions

```bash
chmod +x scripts/run_autonomous.sh
tmux new -s nightly-run
./scripts/run_autonomous.sh
# Ctrl+B, D to detach
```

### Self-Verification Cycle

Before marking tasks complete:
```bash
cargo clippy --all-targets -- -D warnings
cargo test
```

## License

MIT