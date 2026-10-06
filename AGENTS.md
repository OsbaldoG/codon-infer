# AGENTS.md - Operational Rules for codon-infer

## Project Overview
Production-ready Rust library (`codon-core`) with two executables:
- `codon-cli`: High-performance terminal interface for local inference and batch benchmarking
- `codon-mcp`: Local MCP server (JSON-RPC over stdio) exposing `generate_protein` tool

## Core Constraints (MANDATORY)

### Zero-Allocation in Hot Path
- **FORBIDDEN** in `forward()` and primitives in `src/lib.rs`:
  - `vec![]`, `Box::new()`, `.to_vec()`, `.clone()`
  - Any allocation on the hot path
- All mutations must be **in-place** on mutable slices `&mut [f32]` provided by `RunState`

### Self-Verification Cycle (Run Before Marking Task Complete)
```bash
cargo clippy --all-targets -- -D warnings
cargo test
```
If either fails: diagnose, fix, re-run. Only then update documentation.

### Atomic Documentation Updates
Upon completing a milestone:
1. Update `README.md` with new commands/modules
2. Record changes in `CHECKPOINTS.md` (files modified, test status)

## Current Phase Status

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1: Modularization | `src/lib.rs`, `src/bin/cli.rs` | **DONE** |
| 2: MCP Protocol | `src/bin/mcp.rs`, `src/mcp/` | **DONE** |
| 3: PyTorch Parity | `tools/export_parity.py`, `tests/parity.rs` | **DONE** |
| 4: Benchmarks | `benches/inference_bench.rs` | **DONE** |
| 5: Hardening | Release flags, static binary | **COMPLETED** (LTO enabled, static binary built, CHANGELOG v0.2.0 created) |

## Git Workflow
- Each autonomous task creates a commit on branch `feat/autonomous-nightly-YYYYMMDD`
- Commit message format: `auto(<branch>): <task summary>`
- Failed tasks: revert `src/` and `Cargo.toml`, log failure, stop

## Testing Requirements
- Unit tests in `src/lib.rs` (primitives: layer_norm, matmul, gelu, softmax, rope)
- Integration tests in `tests/` (parity, MCP protocol)
- Benchmarks in `benches/` (criterion: prefill latency, decode throughput)

## File Structure Target
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
└── scripts/
    └── run_autonomous.sh
```