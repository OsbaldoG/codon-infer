# codon-infer 🧬⚡

> **High-Performance, Zero-Allocation Multi-Scale ESM-2 Protein Language Model Inference Engine & Bioinformatic Pipeline in Pure Rust (v1.4.0).**

[🇪🇸 Read in Spanish (README_ES.md)](README_ES.md)

[![Rust](https://img.shields.io/badge/rust-2024_edition-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-1.4.0-blue.svg)]()
[![Dependencies](https://img.shields.io/badge/dependencies-0%20(pure%20std)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/model-ESM--2%20(8M%20%7C%2035M%20%7C%20150M)-blueviolet.svg)](https://github.com/facebookresearch/esm)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%2BFMA%20%7C%20AVX1-red.svg)]()
[![Throughput](https://img.shields.io/badge/CPU%20throughput->200%20tokens%2Fsec-success.svg)]()

`codon-infer` is an ultra-lightweight, dependency-free inference engine and bioinformatic analysis suite for the Evolutionary Scale Modeling (**ESM-2**) protein language model family (`8M`, `35M`, and `150M`), engineered from first principles in **100% safe, idiomatic Rust (`std` only)**.

By eliminating Python, PyTorch, C/CUDA runtimes, and external third-party crates, `codon-infer` executes ESM-2 autoregressive generation, bidirectional active-site inpainting, in silico deep mutational scanning (DMS), and structural contact prediction with dynamic SIMD vectorization at **over 200 tokens/second on standard x86_64 CPUs** with a static memory footprint under 40 MB for 8M models.

---

## ⚡ Highlights (Version 1.4)

* **Multi-Scale Architecture & Parametric Autodetection (`detect_from_file_len`):**
  * Instant, zero-config model instantiation for **ESM-2 8M**, **35M**, and **150M** variants directly from binary file byte length, completely eliminating the need for detached JSON configs or manifest files.
* **Heterogeneous Dynamic SIMD Attention Kernels ($d_k \in \{16, 24, 32\}$):**
  * Hand-crafted 256-bit AVX vector kernels mapped to 2, 3, or 4 `__m256` registers per head, delivering maximum vector compute density with bit-exact scalar parity ($\Delta < 10^{-5}$) across both causal autoregressive and bidirectional forward passes.
* **Production CLI Subcommands (`generate`, `inpaint`, `dms`, `contacts`, `benchmark`):**
  * Expressive, unified terminal interface supporting subcommands, global flags (`--model`, `--color`, `--no-color`), automatic TTY detection, and clear error diagnostics.
* **Zero-Dependency TrueColor UX (`ui.rs`):**
  * **Physicochemical Palette:** Amino acid residues styled using TrueColor ANSI RGB according to biochemical properties (Aliphatic, Aromatic, Basic, Acidic, Polar, Cysteine, Conformationally flexible).
  * **Continuous Thermal Gradient Heatmap:** Dynamic RGB interpolation (`lerp_rgb`) from Deep Crimson (`#7F1D1D`) for lethal substitutions ($\Delta \log P \le -4.0$) through Amber, Neutral Slate, to Electric Blue (`#1E3A8A`) for beneficial variants, preserving ASCII density glyphs (`█`, `▓`, `▒`, `░`, `·`) for non-color terminals.
  * **Stderr Dynamic Progress Bar:** Clean ANSI `\r` updates reporting throughput (`muts/s`) and estimated completion time (`ETA`) without polluting redirected `stdout`.
* **Multi-FASTA & Stream Pipeline (`fasta.rs`):**
  * High-throughput zero-copy streaming parser for multi-record FASTA files (`--fasta <FILE>`), standard input (`--fasta -`), or direct string sequences (`--sequence <STR>`).
* **Publication-Grade Export Formats (`export.rs`):**
  * Structured rectangular CSV matrices and comprehensive JSON reports with residue mutational tolerances, Shannon entropy, and top contact pairs.
* **Sliding Window Evaluator for Full-Length Proteins (`pipeline.rs`):**
  * Processes arbitrary-length proteins using adaptive sliding windows ($W=128, S=64$) centered on evaluated residues to prevent RoPE boundary distortion and guarantee cache locality.
* **Dual SIMD Hardware Acceleration & Pure Rust Core:**
  * **AVX2 + FMA:** 256-bit Fused Multiply-Add vector pipeline for modern AMD Ryzen and Intel Haswell+.
  * **AVX1:** 256-bit SIMD kernel for legacy Intel Sandy Bridge / Ivy Bridge processors.
  * **Dynamic Runtime CPU Dispatch:** Evaluates CPU capabilities at startup via `is_x86_feature_detected!`.
* **Exact Mathematical & Architectural Parity with ESM-2:**
  * Complete `EsmLMHead` with numerical `erf` GELU approximation and split-half RoPE positional embeddings.
* **Strict Zero-Allocation Runtime (`RunState`):** All transformer buffers, scratchpads, attention matrices, and KV caches pre-allocated. Zero memory churn during forward passes.

---

## 📊 Performance & Benchmarks

Benchmarked on Linux x86_64 (CPU single-thread execution, native vector instructions AVX/AVX2+FMA):

### 1. Contrast vs. Python / PyTorch Baseline
The architectural migration to pure Rust with deterministic zero-allocation guarantees dramatically outperforms the standard Python/PyTorch ecosystem across all operational metrics:

| Metric | PyTorch / Python Baseline (ESM-2 8M) | `codon-infer` v1.4 (Pure Rust + SIMD) | Engineering Advantage |
| :--- | :--- | :--- | :--- |
| **Dependencies** | Python, PyTorch, LibTorch, CUDA (~4–7 GB) | **Pure `std` (0 crates, 0 dynamic dependencies)** | **~99.9% smaller footprint** |
| **Executable Size** | Multi-gigabyte runtime environment | **~1.5 MB release binary** | **Ultra-portable single binary** |
| **Model Load Latency** | ~2,500 – 4,000 ms (PyTorch initialization) | **34.8 ms (8M) / 117.4 ms (35M) / 466.8 ms (150M)** | **~60x – 100x faster startup** |
| **Memory Footprint (RAM)** | ~1.5 – 2.5 GB RSS baseline | **~35 MB static RSS (8M FP32) / ~12 MB (Q8_0)** | **~50x leaner memory consumption** |
| **Inference Throughput** | ~15 – 35 tokens/sec (CPU single-thread) | **~333.3 tokens/sec (8M) / 84.7 tok/s (35M)** | **~10x – 22x faster execution** |
| **Memory Allocation** | Continuous GC pauses & tensor churn | **0 heap allocations in inference hot-path** | **Deterministic real-time guarantees** |

### 2. Multi-Scale Real Performance Matrix (ESM-2 8M, 35M, and 150M)
Measured with `codon-infer benchmark --suite full` under CPU single-thread execution with native SIMD dispatch:

| Model Scale | Layer Depth ($L$) | Hidden Dim ($d$) | Attention Heads ($H \times d_k$) | Weight Load Latency | Autoregressive Throughput | Bidirectional ($L=32$) | Working State Buffer ($L=128$) | Cache Hierarchy & Memory Regime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ESM-2 8M** | 6 | 320 | $20 \times 16$ (2 AVX regs) | **34.8 ms** | **~333.3 tokens/s** (3.0 ms/tok) | **~418.8 tokens/s** (76.4 ms) | **911.0 KB** | **100% L3 Resident** ($\le 32$ MB) · Compute-Bound |
| **ESM-2 35M** | 12 | 480 | $20 \times 24$ (3 AVX regs) | **117.4 ms** | **~84.7 tokens/s** (11.8 ms/tok) | **~89.9 tokens/s** (356.1 ms) | **1,321.2 KB** | **Partial L3 / DDR Streaming** · Balanced Cache/DRAM |
| **ESM-2 150M** | 30 | 640 | $20 \times 32$ (4 AVX regs) | **466.8 ms** | **~18.9 tokens/s** (53.0 ms/tok) | **~20.2 tokens/s** (1,583.7 ms) | **1,731.5 KB** | **RAM Streaming Bound** ($>500$ MB) · Memory-Bus Limited |

---

## 🧬 Biological Exploration & Validation

During validation, `codon-infer` was seeded with a cysteine-rich motif prompt (`MCMPCFTTDHQMARKCDD`, 18 aa). The engine generated candidate sequences under canonical amino acid constraints:

```text
Prompt:     MCMPCFTTDHQMARKCDD
Generation: MCMPCFTTDHQMARKCDDDPCCIIVQCCRQEIIQRCCLRIDRNCCIIIVYCCNCY
```

Submitting generated candidates to the **NCBI BLASTP** database against non-redundant protein sequences confirmed alignment against genuine chlorotoxin crystal structures (PDB: [`7X41_A`](https://www.rcsb.org/structure/7X41), E-value: $0.001$), demonstrating structural motif conservation followed by coherent *de novo* peptide space exploration.

---

## 🧬 Multi-Scale ESM-2 Architecture & Dynamic SIMD Engine

Starting with version **v1.4.0**, `codon-infer` provides native, first-class support for the primary three scale tiers of the ESM-2 protein language model family:

```text
                                  ┌────────────────────────┐
                                  │ Binary Model Weights   │
                                  │ (*.bin / FP32 / Q8_0)  │
                                  └───────────┬────────────┘
                                              │ File length heuristic
                                              ▼
                             EsmConfig::detect_from_file_len()
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
      ESM-2 8M (t6_8M)               ESM-2 35M (t12_35M)             ESM-2 150M (t30_150M)
   L=6, d=320, H=20, dk=16        L=12, d=480, H=20, dk=24        L=30, d=640, H=20, dk=32
   2 AVX Regs (__m256 x 2)        3 AVX Regs (__m256 x 3)         4 AVX Regs (__m256 x 4)
              │                               │                               │
              └───────────────────────────────┼───────────────────────────────┘
                                              ▼
                             Unified Zero-Allocation Runtime
                            (RunState / BidirectionalRunState)
```

### 1. Parametric Autodetection without Metadata (`detect_from_file_len`)
Traditional Python deep learning frameworks require external configuration files (`config.json`, tokenizer configurations, model manifests, etc.) alongside weight tensors. `codon-infer` eliminates this friction through a deterministic byte-length heuristic:

```rust
pub fn detect_from_file_len(file_len: u64) -> io::Result<Self> {
    let floats = file_len / 4;
    if floats >= 140_000_000 {
        Ok(Self::esm2_150m()) // 150M (~591 MB FP32 / ~178 MB Q8_0)
    } else if floats >= 30_000_000 {
        Ok(Self::esm2_35m())  // 35M (~133 MB FP32 / ~48 MB Q8_0)
    } else if floats >= 7_000_000 {
        Ok(Self::esm2_8m())   // 8M (~30 MB FP32 / ~12 MB Q8_0)
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, "Unrecognized ESM-2 weights binary size"))
    }
}
```
Any custom weights binary supplied via `--model <PATH>` is identified and validated at boot time in $< 0.1\text{ ms}$, automatically resizing all internal inference buffers, KV-caches, and embedding projections with zero runtime allocation overhead.

### 2. SIMD Optimization and Heterogeneous Attention Dynamic Dispatch
In multi-head attention, every head has dimension $d_k = d_{\text{model}} / H$. Because all ESM-2 variants standardize on $H=20$ attention heads, the head dimensions differ across model scales:
* **8M:** $d_k = 320 / 20 = 16$ floats
* **35M:** $d_k = 480 / 20 = 24$ floats
* **150M:** $d_k = 640 / 20 = 32$ floats

Because each 256-bit AVX register (`__m256`) holds exactly 8 single-precision (`f32`) floats, every $d_k$ is an exact integer multiple of 8:
$$\frac{16}{8} = 2 \text{ registers}, \quad \frac{24}{8} = 3 \text{ registers}, \quad \frac{32}{8} = 4 \text{ registers}$$

`codon-infer` implements dedicated, unrolled assembly-grade vector kernels for dot products and value accumulations:
* **Scaled Dot-Product ($Q \cdot K^\top$):** `dot_product_head_avx` evaluates 16, 24, or 32 dimensions in parallel without remaining scalar loops or loop bounds checks, performing pairwise multiplication (`_mm256_mul_ps`), vector addition (`_mm256_add_ps`), and an ultra-fast horizontal sum (`hsum256_ps_avx`).
* **Value Vector Accumulation ($V \times \text{prob}$):** `accumulate_value_head_avx` broadcasts the scalar probability across an entire vector (`_mm256_set1_ps`) and executes direct vector multiply-accumulate operations (`_mm256_add_ps` + `_mm256_mul_ps`) straight into the output head accumulator.
* **Scalar Parity Guarantee:** Unit tests (`test_attention_simd_parity_dk_16_24_32`) enforce mathematical equivalence with scalar reference loops under a strict tolerance threshold ($\Delta < 10^{-5}$) across all 3 head dimensions.

### 3. Architectural and Performance Technical Comparison Matrix

| Parameter / Metric | ESM-2 8M (`esm2_t6_8M_UR50D`) | ESM-2 35M (`esm2_t12_35M_UR50D`) | ESM-2 150M (`esm2_t30_150M_UR50D`) |
| :--- | :--- | :--- | :--- |
| **Transformer Layers ($L$)** | 6 | 12 | 30 |
| **Hidden Dimension ($d_{\text{model}}$)** | 320 | 480 | 640 |
| **Attention Heads ($H$)** | 20 | 20 | 20 |
| **Head Dimension ($d_k$)** | 16 | 24 | 32 |
| **AVX Registers (`__m256`) per Head** | **2 registers** (16 floats) | **3 registers** (24 floats) | **4 registers** (32 floats) |
| **FFN Dimension ($d_{\text{ffn}}$)** | 1,280 | 1,920 | 2,560 |
| **Vocabulary** | 33 tokens | 33 tokens | 33 tokens |
| **Total Parameters** | 7,511,809 (~7.5M) | 33,524,449 (~33.5M) | 148,011,297 (~148M) |
| **Binary Size (FP32)** | ~30.0 MB | ~134.1 MB | ~592.5 MB |
| **Quantized Binary Size (`Q8_0`)** | ~11.5 MB | ~48.2 MB | ~178.6 MB |
| **Weight Load Latency** | **34 – 38 ms** | **117 – 122 ms** | **466 – 487 ms** |
| **Autoregressive Throughput (Generate)** | **>315 tokens/s** (~3.1 ms/tok) | **~80 – 85 tokens/s** (~11.8 ms/tok) | **~18 – 20 tokens/s** (~53.0 ms/tok) |
| **Bidirectional Throughput (Inpaint)** | **>400 tokens/s** (~76 ms / $L=32$) | **~85 – 90 tokens/s** (~356 ms / $L=32$) | **~18 – 21 tokens/s** (~1,580 ms / $L=32$) |
| **Working State Memory ($L=128$)** | **~911 KB** | **~1,321 KB** | **~1,732 KB** |
| **Cache Hierarchy Residency** | **100% in L3 Cache** ($\le 32$ MB) | Partial Residency in L3 / DDR | Streaming from RAM ($> 500$ MB) |
| **Memory Bandwidth Demand** | **Compute-Bound (AVX/SIMD)** | **Balanced Cache/DRAM** | **Memory-Bus Limited** |
| **Primary Use Case** | Fast in silico DMS, real-time inpainting | High-resolution DMS, peptide screening | 3D contact prediction, de novo design |

---

## 🏗️ Transformer Architecture Pipeline

The runtime pipeline faithfully executes the ESM-2 architecture (RoBERTa-based):

```text
                    ┌────────────────────────┐
                    │  Input Prompt (String) │
                    └───────────┬────────────┘
                                │ Tokenizer::encode()
                                ▼
                    ┌────────────────────────┐
                    │ Token IDs ([0, ..., 2])│
                    └───────────┬────────────┘
                                │
 ┌──────────────────────────────▼───────────────────────────────┐
 │ Token Embedding Table (33 tokens × dim)                      │
 └──────────────────────────────┬───────────────────────────────┘
                                │  x [dim]
 ┌──────────────────────────────▼───────────────────────────────┐ ◄──┐
 │ Transformer Layer (×L Layers: 6, 12, or 30)                  │    │
 │  ├── LayerNorm 1 (dim, eps 1e-5)                             │    │
 │  ├── Multi-Head Attention (20 Heads × dk Dim: 16, 24, or 32) │    │
 │  │    ├── Projections: Q, K, V (SIMD Matmul AVX2+FMA / AVX1)  │    │
 │  │    ├── RoPE Split-Half: x[:half], x[half:dk] rotated      │    │
 │  │    ├── Static Ring KV-Cache Update (pos × dim)            │    │
 │  │    ├── Scaled Dot-Product Attention (AVX dk-unrolled)     │    │
 │  │    └── Output Projection (Wo: dim × dim + Bias)           │    │
 │  ├── Residual Connection 1: x = x + att_out                  │    │
 │  ├── LayerNorm 2 (dim, eps 1e-5)                             │    │
 │  ├── Feed-Forward Network (FFN)                              │    │
 │  │    ├── FC1 Dense: dim -> hidden_dim + Bias                │    │
 │  │    ├── GELU Activation (Numerical erf approximation)      │    │
 │  │    └── FC2 Dense: hidden_dim -> dim + Bias                │    │
 │  └── Residual Connection 2: x = x + ffn_out                  │    │
 └──────────────────────────────┬───────────────────────────────┘ ───┘
                                │  x [dim] (after L layers)
 ┌──────────────────────────────▼───────────────────────────────┐
 │ Final LayerNorm (ln_f, dim, eps 1e-5)                        │
 └──────────────────────────────┬───────────────────────────────┘
                                │
 ┌──────────────────────────────▼───────────────────────────────┐
 │ EsmLMHead (Classification Head)                              │
 │  ├── Dense Layer: dim × dim + Bias                           │
 │  ├── GELU Activation (erf-based)                             │
 │  └── Tied Embedding Projection (dim × 33) + Decoder Bias     │
 └──────────────────────────────┬───────────────────────────────┘
                                │  logits [33]
 ┌──────────────────────────────▼───────────────────────────────┐
 │ Biological Logit Masking (-inf on tokens <4 or >23)          │
 │ Stochastic Temperature Sampling (SimpleRng / XORShift64)     │
 └──────────────────────────────┬───────────────────────────────┘
                                ▼
                    Next Canonical Amino Acid Token
```

---

## 🚀 Quickstart

### Prerequisites
* Rust toolchain (2024 edition or 1.85+):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 1. Build from Source
Clone the repository and build optimized binaries:
```bash
git clone https://github.com/OsbaldoG/codon-infer.git
cd codon-infer

# Build release binaries with native CPU SIMD optimizations
cargo build --release
```

### 2. Pretrained Weights & GitHub Releases

`codon-infer` standardizes model binary files on clean, explicit nomenclature:
* **ESM-2 8M (`esm2_8m_weights.bin`):** Pre-packaged in the repository root (~30 MB FP32) for instant out-of-the-box inference, local development, and rapid unit testing without external downloads.
* **ESM-2 35M (`esm2_35m_weights.bin`):** Higher capacity model (~133 MB FP32), hosted in [GitHub Releases](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.4.0) to prevent Git repository bloat.
* **ESM-2 150M (`esm2_150m_weights.bin`):** Publication-grade structural contact and high-fidelity model (~590 MB FP32), hosted in [GitHub Releases](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.4.0).

#### Download Weights via `curl`
Directly download the desired scaled binary model into your repository root:
```bash
# Download ESM-2 35M weights (~133 MB)
curl -L -o esm2_35m_weights.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.4.0/esm2_35m_weights.bin

# Download ESM-2 150M weights (~590 MB)
curl -L -o esm2_150m_weights.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.4.0/esm2_150m_weights.bin

# Or use the automated helper script:
./scripts/download_weights.sh 35m
./scripts/download_weights.sh 150m
```

> [!TIP]
> You can also export custom weights directly from HuggingFace Transformers or Meta `fair-esm` using the provided Python utility:
> ```bash
> python3 scripts/export_weights.py --model 35M
> ```

### 3. Run Test Suite
Verify numerical primitives, SIMD vectorization parity, FASTA parser, and export formats:
```bash
cargo test
```
Expected output:
```text
running 22 tests
test export::tests::test_json_string_serializers ... ok
test fasta::tests::test_parse_multi_fasta_with_masks ... ok
test fasta::tests::test_parse_simple_fasta ... ok
test fasta::tests::test_parse_raw_sequence_fallback ... ok
test mcp::tests::test_mcp_json_parser_and_formatter ... ok
test mcp::tests::test_mcp_resolve_record_fallback ... ok
test export::tests::test_dms_row_and_csv_generation ... ok
test mcp::tests::test_mcp_tools_list_schema ... ok
test tests::test_attention_simd_parity_dk_16_24_32 ... ok
test tests::test_esm_config_autodetection ... ok
test tests::test_forward_pass_with_weights_if_available ... ok
test tests::test_layer_norm_stability ... ok
test tests::test_log_softmax_canonical ... ok
test tests::test_mask_token_encoding_and_decoding ... ok
test tests::test_matmul_avx1_and_dispatch_parity ... ok
test tests::test_q8_0_quantization_roundtrip ... ok
test tests::test_simd_benchmark_parity_run ... ok
test tests::test_softmax_sum_to_one ... ok
test tests::test_tokenizer_boundaries ... ok
test tests::test_bidirectional_logits_entropy ... ok
test tests::test_forward_bidirectional_if_available ... ok
test mcp::tests::test_mcp_dispatch_with_weights_if_available ... ok

test result: ok. 22 passed; 0 failed; 0 ignored; 0 measured; finished in 7.45s
```

### 4. Subcommands & Bioinformatic Pipelines

All subcommands support the global `--model <PATH>` flag. `codon-infer` automatically detects whether the binary corresponds to **ESM-2 8M**, **35M**, or **150M** (FP32 or quantized `Q8_0`) and configures the hardware kernels accordingly.

#### A. Autoregressive Sequence Generation (`generate`)
Execute de novo sequence generation or motif extension:
```bash
# Default (ESM-2 8M)
codon-infer generate --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7 --max-tokens 40

# Multi-scale: Run with ESM-2 35M or 150M
codon-infer generate --model esm2_35m_weights.bin --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7
codon-infer generate --model esm2_150m_weights.bin --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7
```

#### B. Bidirectional Inpainting & Motif Infilling (`inpaint`)
Reconstruct missing, ambiguous, or masked active-site motifs (marked with `*` or `<mask_1>`):
```bash
# Default (ESM-2 8M)
codon-infer inpaint --sequence "MCMPC*TTDHQMARKCDD"

# With higher capacity ESM-2 35M weights:
codon-infer inpaint --model esm2_35m_weights.bin --sequence "MCMPC*TTDHQMARKCDD"
```

#### C. In Silico Deep Mutational Scanning (`dms`)
Perform comprehensive single-point saturation mutagenesis with TrueColor thermal heatmap rendering and tabular export:
```bash
# High-throughput screening with ESM-2 8M:
codon-infer dms --fasta proteins.fasta --window 128 --export-csv dms_matrix.csv --export-json dms_report.json

# High-fidelity scanning with ESM-2 150M:
codon-infer dms --model esm2_150m_weights.bin --fasta target.fasta --window 128 --export-csv dms_150m.csv
```

#### D. Zero-Shot Structural Contact Prediction (`contacts`)
Compute epistatic residue couplings using multi-head attention contact maps (APC-corrected) and identify putative disulfide bonds:
```bash
# Predict 3D contact pairs with ESM-2 150M for publication-grade precision:
codon-infer contacts --model esm2_150m_weights.bin --fasta proteins.fasta --export-csv contacts.csv --export-json contacts.json
```

#### E. Hardware & Biological Benchmark Suite (`benchmark`)
Execute the full hardware stress and biological validation suite across any model size:
```bash
# Execute comparative cross-model benchmark across all discovered weights (8M, 35M, 150M):
codon-infer benchmark --suite full

# Benchmark a specific model variant:
codon-infer benchmark --model esm2_35m_weights.bin --suite full

# Export structured performance report for CI/CD pipelines:
codon-infer benchmark --suite full --export-json benchmark_report.json
```

Available suites:
* `full`: Multi-model comparative evaluation (weight loading latency, tokens/s throughput, working memory, cache profile) plus algorithmic tests.
* `simd`: Multi-scale AVX2+FMA vs AVX1 numerical parity and matrix-vector microbenchmarks across projection dimensions ($320 \to 1280$, $480 \to 1920$, $640 \to 2560$) and head dimensions $d_k \in \{16, 24, 32\}$.
* `mlm`: Evolutionary pseudo-perplexity and wild-type vs disruptive mutation penalties.
* `dms`: Deep mutational scanning calibration on bacterial catalytic peptidase motifs.
* `contacts`: Structural contact prediction and Cys-Cys spatial proximity scoring.
* `inpainting`: Multi-position bidirectional masked sequence infilling.
* `stress`: Dynamic context continuous inference cache-thrashing stability test with slice Q8 quantization.
* `scaling`: Context length scaling ($L = 32 \to 256$) memory and latency benchmark adapting to model depth.

### 5. Model Context Protocol (MCP) Server
`codon-infer-mcp` exposes the entire bioinformatic engine as native tools for LLM agents (Claude Desktop, Cursor, Gemini, etc.) over stdio JSON-RPC:
```bash
# Launch with default ESM-2 8M weights
cargo run --release --bin codon-infer-mcp

# Launch with higher-capacity ESM-2 35M or 150M weights
cargo run --release --bin codon-infer-mcp -- --model esm2_35m_weights.bin
```

#### Tools Exposed over MCP:
* `generate_protein`: Autoregressive sequence generation given `prompt`, `temperature`, `max_tokens`.
* `codon_infer_inpaint`: Reconstructs masked residues (`*`) returning top-5 amino acid candidates with probabilities and physicochemical classifications.
* `codon_infer_dms`: Runs in silico saturation mutagenesis from raw sequence or FASTA path, returning critical catalytic sites ($\Delta \log P \le -3.8$) and mutational tolerances.
* `codon_infer_contacts`: Computes epistatic contact pairs (APC) and identifies putative disulfide bridges (`Cys-Cys`).

---

## 🛠️ Unified CLI Options Reference

### Global Flags
| Option | Description | Default |
| :--- | :--- | :--- |
| `-h`, `--help` | Display help banner with subcommand guide | - |
| `-V`, `--version` | Display version and active SIMD vector engine | - |
| `--model <PATH>` | Custom ESM-2 raw FP32 weights binary | `esm2_8m_weights.bin` |
| `--color <auto|always|never>` | Control ANSI TrueColor terminal styling | `auto` |
| `--no-color` | Strictly disable ANSI terminal escapes (NO_COLOR standard) | `false` |

### Pipeline Options
| Option | Applicable Subcommands | Description | Default |
| :--- | :--- | :--- | :--- |
| `--sequence <STR>` | `generate`, `inpaint`, `dms`, `contacts` | Direct input amino acid sequence | `"M"` |
| `--fasta <PATH>` | `inpaint`, `dms`, `contacts` | Path to FASTA file (or `-` for stdin) | - |
| `--export-csv <PATH>`| `dms`, `contacts` | Destination path for rectangular matrix / table | - |
| `--export-json <PATH>`| `dms`, `contacts`, `benchmark` | Destination path for structured JSON analysis / report | - |
| `--window <USIZE>` | `dms` | Sliding window context size for long proteins | `128` |
| `--stride <USIZE>` | `dms` | Sliding window stride step size | `64` |
| `--temperature <F32>`| `generate` | Sampling temperature (0.0 = greedy argmax) | `0.7` |
| `--max-tokens <USIZE>`| `generate` | Maximum new residues to generate | `1024` |
| `--suite <STR>` | `benchmark` | Benchmark suite (full, simd, mlm, dms, contacts, inpainting, stress, scaling) | `"full"` |

---

## 📂 Project Structure

```text
codon-infer/
├── .cargo/
│   └── config.toml         # Native CPU optimization flags (-C target-cpu=native)
├── .gitignore              # Clean repository filter (build artifacts & backups ignored)
├── Cargo.toml              # Zero-dependency specification (v1.4.0, edition 2024)
├── esm2_8m_weights.bin     # ESM-2 8M FP32 weights (7.51M parameters, ~30 MB)
├── scripts/
│   ├── download_weights.sh # Fast GitHub Releases weights downloader
│   └── export_weights.py   # HuggingFace & fair-esm weight exporter
├── src/
│   ├── lib.rs              # Core engine: dual SIMD matmul, RoPE split-half, Q8_0, EsmLMHead
│   ├── main.rs             # CLI dispatcher & subcommand argument parser
│   ├── ui.rs               # TrueColor ANSI palettes, thermal gradients & progress bar
│   ├── fasta.rs            # Zero-copy multi-FASTA & stream sequence reader
│   ├── export.rs           # Publication-grade CSV and JSON data serializers
│   ├── pipeline.rs         # Sliding-window DMS, contacts, and inpainting drivers
│   ├── benchmark.rs        # Automated bioinformatic & hardware stress suite
│   ├── mcp.rs              # Zero-dependency stdio JSON-RPC MCP server
│   └── mcp_main.rs         # Standalone MCP server executable entrypoint
├── AUDIT_REPORT.md         # Comprehensive architectural & mathematical audit report
├── LICENSE                 # MIT License
├── README.md               # Main project documentation (English)
└── README_ES.md            # Complete project documentation (Spanish)
```

---

## 📜 Roadmap

- [x] Pure Rust migration from C prototype (**ADR-01**).
- [x] Strict zero-allocation heap runtime (`RunState`).
- [x] Dual SIMD vectorization: `AVX2+FMA` and `AVX1 Sandy Bridge` kernels with runtime dispatch.
- [x] RoPE `split-half` formulation and numerical `erf` GELU approximation.
- [x] Full `EsmLMHead` classifier integration (102,849 weights).
- [x] Block `Q8_0` Quantization for L3 cache optimization.
- [x] Standard Model Context Protocol (MCP) JSON-RPC daemon (`codon-infer-mcp`).
- [x] Modular separation: `lib.rs` (core library), `main.rs` (CLI), `mcp.rs` (MCP daemon).
- [ ] ARM NEON SIMD kernels for Apple Silicon (M1/M2/M3/M4) and Raspberry Pi.
- [ ] Sub-byte INT4 quantization (`Q4_0`) for ultra-low power embedded bio-wearables.

---

## 🔬 Citation & Acknowledgments

If you use `codon-infer` in your bioinformatic workflows, structural biology, or synthetic biology research, please cite:
* **ESM-2 Original Architecture:** Lin, Z., et al. *"Evolutionary-scale prediction of atomic-level protein structure with a language model."* Science 379.6637 (2023): eade2574.
* **Author & Lead Architect:** Manuel Osbaldo Gastélum Islas ([@OsbaldoG](https://github.com/OsbaldoG))
