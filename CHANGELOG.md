# Changelog

## [v0.2.0] - 2026-10-05

### Added
- Static binary compilation support via musl target
- Release profile optimizations (LTO, codegen-units=1)
- Exact GELU implementation using erf (matches PyTorch)
- Precise RoPE INV_FREQ values from PyTorch float32

### Changed
- Updated embed_scale from sqrt(dim) to 1.0 (ESM-2 eval mode)
- Corrected lm_head order: dense → gelu → layer_norm → out
- Updated version to 0.2.0

### Fixed
- Numerical parity with PyTorch ESM-2 8M (MAE < 1e-3)
- All clippy warnings
- Fixed parity test to collect logits for all positions and handle dtype correctly
- Fixed dependency on `npyz` to include `npz` feature and correct version

### Dependencies
- Added libm = "0.2" for erf function

### binaries
- Static codon-cli and codon-mcp available for x86_64-unknown-linux-musl