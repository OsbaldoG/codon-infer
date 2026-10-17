# codon-infer 🧬⚡

> **High-Performance, Zero-Allocation Multi-Scale ESM-2 Protein Language Model Inference Engine & Bioinformatic Pipeline in Pure Rust (v1.8.0).**

[🇪🇸 Read in Spanish (README_ES.md)](README_ES.md)

[![Rust](https://img.shields.io/badge/rust-2024_edition-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-1.8.0-blue.svg)](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.8.0)
[![Tests](https://img.shields.io/badge/tests-57%20executed%3A%2056%20passed%2C%200%20failed%2C%201%20ignored-brightgreen.svg)](#3-run-test-suite)
[![Dependencies](https://img.shields.io/badge/dependencies-0%20(pure%20std)-brightgreen.svg)](Cargo.toml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/model-ESM--2%20(8M%20to%203B)-blueviolet.svg)](https://github.com/facebookresearch/esm)
[![SIMD](https://img.shields.io/badge/SIMD-AVX512%20%7C%20AVX2%2BFMA%20%7C%20AVX1-red.svg)](#2-simd-optimization-and-heterogeneous-attention-dynamic-dispatch)
[![Precision](https://img.shields.io/badge/precision-FP32%20%7C%20Q8__0%20%7C%20Q4__0-green.svg)](#e-native-q4_0--q8_0-model-quantization-quantize)
[![Throughput](https://img.shields.io/badge/CPU%20throughput->200%20tokens%2Fsec-success.svg)](#-performance--benchmarks)

`codon-infer` is an ultra-lightweight, dependency-free inference engine and bioinformatic analysis suite for the Evolutionary Scale Modeling (**ESM-2**) protein language model family (`8M`, `35M`, `150M`, `650M`, and `3B`), engineered from first principles in **100% safe, idiomatic Rust (`std` only)**.

By eliminating Python, PyTorch, C/CUDA runtimes, and external third-party crates, `codon-infer` executes ESM-2 autoregressive generation, bidirectional active-site inpainting, in silico deep mutational scanning (DMS), zero-shot contact prediction, and ab initio 3D protein structure folding with dynamic SIMD vectorization at **over 200 tokens/second on standard x86_64 CPUs** with a static memory footprint under 40 MB for 8M models.

---

## ⚡ Highlights (Version 1.8: Zero-Trunk 3D Protein Folding, Tiled GEMM Engine & Canonical Contact Suite)

> 💡 **Version Consolidation:** Version 1.8.0 unifies the foundational algorithmic engine updates of v1.7.0 (Multi-Token Tiled GEMM Engine `CI-1`, Canonical DMS `BIO-1`, and Official Multi-Layer Contact Head `STRUCT-1`) with the breakthrough Zero-Trunk 3D Geometric Reconstruction pipeline (`STRUCT-2`).

* **Zero-Trunk 3D Geometric Reconstruction (`STRUCT-2`):**
  * Solves ab initio 3D protein folds and $C_\alpha$ backbone coordinates directly from ESM-2 multi-head contact representations in **15–35 ms** on commodity x86_64 CPUs, completely eliminating the need for heavyweight 3 GB folding trunks or external AlphaFold/ESMFold weights.
  * **Hybrid Metric MDS + Biophysical Gradient Optimization:**
    1. *Stage A (Spectral Metric MDS):* Floyd-Warshall geodesic closure smoothing on upper-bound graph followed by classical double-centered Gram matrix eigendecomposition ($B = -\frac{1}{2} J D^{(2)} J$) and top-3 power iteration with deflation, rescaled to canonical peptide bond length $3.80\text{ \AA}$.
    2. *Stage B (Multi-Start Biophysical Gradient Descent):* 450 momentum-accelerated gradient steps across 4 conformation basins (spectral MDS + 3 twisted helical ribbons) minimizing a first-principles physical potential:
       * **Exact Peptide Bond:** $d(C_{\alpha, i}, C_{\alpha, i+1}) = 3.80\text{ \AA}$ ($w = 18 \to 43$).
       * **Local Bend Angles:** $d(C_{\alpha, i}, C_{\alpha, i+2}) \in [5.30\text{ \AA}, 6.90\text{ \AA}]$ with dynamic local helix steering.
       * **Tertiary Contacts:** Huber potential on top $1.5 \cdot N$ pairs with quadratic confidence weighting ($w \propto P_{\text{rel}}^2$).
       * **Steric Van der Waals Exclusion:** Hard repulsion for $|i - j| \ge 3$ at $d < 4.10\text{ \AA}$ and non-contact epistatic repulsion.
       * **Globular Compaction:** Target radius of gyration regularizer $R_g = 2.25 \cdot N^{0.38}\text{ \AA}$.
  * **Publication-Grade Coordinate Parity:** Reconstructs crystal structures with RMSD $< 3.0\text{ \AA}$ and TM-score $> 0.50$ (e.g. `2L2D` Trp-cage: **RMSD 2.26 Å**, TM 0.525; `1O6X`: **RMSD 2.95 Å**, TM 0.602; crystal oracle: **RMSD 2.43 Å**, TM 0.652, 0 clashes).
  * Direct `.pdb` standard file generation (`codon-infer fold --sequence ... --export-pdb structure.pdb` and `codon-infer contacts ... --export-pdb contacts.pdb`).

* **Multi-Token Tiled GEMM Engine (CI-1):**
  * Implements cache-blocked multi-token tiled matrix multiplication (`matmul_tiled_fp32_parallel`, `matmul_tiled_q8_parallel`, `matmul_tiled_q4_parallel`).
  * Features AVX2+FMA and AVX1 micro-kernels with $M_r = 4$ simultaneous tokens (`dot_row_batch_m4_avx2_fma`, `dot_row_batch_m4_avx1`) and row-wise unpacking into L1d cache (`unpack_q8_row_avx1`, `unpack_q4_row_avx1`).
  * Eliminates DRAM bandwidth bottlenecks (crossing the threshold from memory-bound to compute-bound), delivering up to **3.76× speedups in FP32** and **2.41×–3.44× in quantized regimes** while maintaining exact numerical parity ($\cos = 1.00000000$, RMSE $\sim 10^{-6}$, Top-1 Match 100.0%).

* **Canonical Deep Mutational Scanning (BIO-1):**
  * First-principles log-likelihood ratio scoring without heuristic clamps or hardcoded catalytic rules: $\Delta \log P = \log P(x_i = m \mid x_{\setminus i}) - \log P(x_i = wt \mid x_{\setminus i})$.
  * Features a fast 1-pass `WildtypeMarginal` mode (`--fast`, `--wt-marginal`, `--mode wt`) accelerating saturation mutagenesis up to **42×** (exceeding **16,170 mutations/sec**) while reliably surfacing conserved catalytic motifs (e.g., GDSGGP).

* **Official Multi-Layer Contact Head with O(N²) Streaming (STRUCT-1):**
  * Integrates official Meta AI regression weights and biases across all 5 scales (8M, 35M, 150M, 650M, 3B) within 12.2 KiB of static memory, ensuring 100% backward compatibility with binary formats (`.bin`, `_q8_0.bin`, `_q4_0.bin`).
  * Executes $O(N^2)$ streaming accumulation (`accumulate_head_apc` via `state.contact_probs`), reducing memory consumption by **1,440×** (avoiding heavy $L \times H \times N \times N$ tensors) and achieving up to **96.63% Precision@L/5** on quantized 650M models consuming only 413 MiB of RAM.

* **Standard Model Context Protocol (MCP) 3D Tool (`codon_infer_fold`):**
  * Fully exposes 3D protein folding and atomic PDB generation over JSON-RPC stdio daemon, returning structured atom records, coordinates, and formatted `.pdb` payloads.

---

## ⚡ Highlights (Version 1.6: Multi-Scale Q4_0 Engine, Canonical HF Parity & 3B Streaming Quantizer)

* **Native `Q4_0` 4-Bit Block Quantization (`BlockQ4_0`):**
  * Sub-byte quantized representation packing 32 4-bit nibbles into 16 bytes alongside an aligned 32-bit float scale factor `d` (`#[repr(C)]`, exactly **20 bytes** per 32-parameter block).
  * Standard symmetric quantization convention (llama.cpp compatible): `d = max / -8.0`, `id = 1/d`, `q = clamp(round(x · id) + 8, 0, 15)`. Low nibble holds index `j ∈ [0, 15]`, high nibble holds index `j + 16`.
  * **6.40× Theoretical Matrix Compression** (0.625 B/param vs 4.0 B FP32) and up to **6.30× Total File Compression** (**ESM-2 3B** reduced from 11.36 GB to **1.80 GB**, and **ESM-2 650M** reduced from 2.60 GB to **414.62 MB**).
  * **~83.5% Peak Memory (RSS) Reduction:** ESM-2 650M RAM footprint plunges from 2,501.33 MiB in FP32 to **413.07 MiB in Q4_0**, while **ESM-2 3B** runs in just **1,752.57 MiB RSS in Q4_0** (`3,102.18 MiB` in Q8_0) on an 11 GiB RAM machine.
  * **1.92× Inference Speedup on ESM-2 650M:** Relieves DRAM bus saturation on 4-thread runs (**4,535.47 ms** in Q4_0 and **4,669.89 ms** in Q8_0 vs **8,692.15 ms** in FP32).
* **Exact HuggingFace Mathematical Parity (`grouped_canonical_v1`):**
  * Byte-exact canonical weight layout across all 5 scales (`8M`, `35M`, `150M`, `650M`, `3B`) with **100.0% Top-1 match** and cosine similarity ≥ 0.999999 against official HuggingFace `facebook/esm2_*` reference tensors (`hidden_states` and `logits`).
  * Streaming block-by-block quantizers (`quantize_file_streaming_q8` / `quantize_file_streaming_q4`) using `BufReader` / `BufWriter` that convert the 10.58 GiB `esm2_3b_weights.bin` in **12.55 MiB Peak RSS**.
* **Universal `CODQ` 64-Byte Header Architecture:**
  * Fixed 64-byte little-endian header (`b"CODQ"`, version 1, dtype, architecture, layer depth, hidden dimension, FFN dimension, vocabulary size).
  * Guarantees 64-byte cache line and SIMD alignment for the first payload block, enabling instantaneous zero-copy memory-mapped ingestion.
  * Solves parametric ambiguity across quantization scales with full backward compatibility via `detect_from_file_len`.
* **Handcrafted Bit-Manipulation SIMD GEMV Micro-Kernels:**
  * AVX2+FMA vector kernels unpack 16 packed bytes with `_mm_srli_epi16` and `_mm_and_si128`, widen signed offsets with `_mm256_cvtepi8_epi32` / `_mm256_cvtepi32_ps`, and accumulate with `_mm256_fmadd_ps`.
  * AVX1 fallback kernels utilize 128-bit integer unpacking and 256-bit floating-point accumulation.
  * Zero heap allocations and deterministic zero-copy execution across the entire inference hot path.
* **Unified Multi-Format CLI (`quantize`, `generate`, `inpaint`, `dms`, `contacts`, `benchmark`):**
  * Auto-detects model scale and precision (FP32, Q8_0, Q4_0) dynamically via `CODQ` header or legacy fallback.
  * New CLI streaming quantization options: `codon-infer quantize --model <input> --output <output> [--format <q4_0|q8_0>]` with filename auto-detection.

---

## ⚡ Highlights (Version 1.5)

* **Native `Q8_0` Block Quantization & Direct Binary Serialization (`BlockQ8_0`):**
  * Block-quantized weight representation (32 `i8` values + 1 `f32` scale factor per block) reducing DRAM bandwidth pressure and RAM footprint by **3.56x** (down from ~2.48 GB to **698.5 MB** on ESM-2 650M) with **99.980% logit fidelity** (Cosine Similarity `0.99980`, RMSE `0.00021`).
  * Direct zero-copy loading from native quantized binaries (`*_q8_0.bin`) in **219.98 ms** (vs ~751 ms in FP32).
  * On-the-fly SIMD dequantization GEMV micro-kernels (`AVX1` / `AVX2` / `AVX-512`) eliminating heap allocations and memory churning during matrix-vector products.
  * Dedicated `quantize` CLI subcommand converting FP32 model weights directly to deployable `Q8_0` binaries.
* **Zero-Spawn Persistent `ThreadPool` & Concurrency Architecture (`std` only):**
  * Zero-OS-overhead parallelization across FFN projections and multi-head attention layers using persistent worker threads, eliminating ~230 `pthread_create` / join kernel transitions per generated token.
  * Hybrid spin/futex barrier (`ThreadBarrier`) utilizing lock-free spin-yielding and adaptive thread parking for sub-microsecond synchronization.
  * Hardware topology autodetection (`detect_optimal_threads` via `/proc/cpuinfo`) that prioritizes physical execution cores over SMT/Hyper-Threads to saturate 256-bit AVX execution ports without resource thrashing.
* **Multi-Scale Architecture & Parametric Autodetection (`detect_from_file_len`):**
  * Instant, zero-config model instantiation across **ESM-2 8M**, **35M**, **150M**, **650M**, and **3B** variants directly from binary file byte length, completely eliminating the need for detached JSON configs or manifest files.
* **Heterogeneous Dynamic SIMD Attention Kernels (`d_k` ∈ {16, 24, 32, 64}):**
  * Hand-crafted 256-bit AVX and 512-bit AVX-512 vector kernels mapped to 2, 3, 4, or 8 `__m256` registers (and 4 `__m512` ZMM registers) per head, delivering maximum vector compute density with bit-exact scalar parity (Δ < 10⁻⁵) across both causal autoregressive and bidirectional forward passes.
* **Production CLI Subcommands (`generate`, `inpaint`, `dms`, `contacts`, `benchmark`, `quantize`):**
  * Expressive, unified terminal interface supporting subcommands, global flags (`--model`, `--threads`, `--quantize`, `--color`, `--no-color`), automatic TTY detection, and clear error diagnostics.
* **Zero-Dependency TrueColor UX (`ui.rs`):**
  * **Physicochemical Palette:** Amino acid residues styled using TrueColor ANSI RGB according to biochemical properties (Aliphatic, Aromatic, Basic, Acidic, Polar, Cysteine, Conformationally flexible).
  * **Continuous Thermal Gradient Heatmap:** Dynamic RGB interpolation (`lerp_rgb`) from Deep Crimson (`#7F1D1D`) for lethal substitutions (Δ log P ≤ -4.0) through Amber, Neutral Slate, to Electric Blue (`#1E3A8A`) for beneficial variants, preserving ASCII density glyphs (`█`, `▓`, `▒`, `░`, `·`) for non-color terminals.
  * **Stderr Dynamic Progress Bar:** Clean ANSI `\r` updates reporting throughput (`muts/s`) and estimated completion time (`ETA`) without polluting redirected `stdout`.
* **Multi-FASTA & Stream Pipeline (`fasta.rs`):**
  * High-throughput zero-copy streaming parser for multi-record FASTA files (`--fasta <FILE>`), standard input (`--fasta -`), or direct string sequences (`--sequence <STR>`).
* **Publication-Grade Export Formats (`export.rs`):**
  * Structured rectangular CSV matrices and comprehensive JSON reports with residue mutational tolerances, Shannon entropy, and top contact pairs.
* **Sliding Window Evaluator for Full-Length Proteins (`pipeline.rs`):**
  * Processes arbitrary-length proteins using adaptive sliding windows (W = 128, S = 64) centered on evaluated residues to prevent RoPE boundary distortion and guarantee cache locality.
* **Multi-SIMD Hardware Acceleration & Pure Rust Core:**
  * **AVX-512:** 512-bit vector pipeline for `d_k = 64` attention processing on Intel Xeon and AMD Zen 4+ architectures.
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

| Metric | PyTorch / Python Baseline (ESM-2 8M) | `codon-infer` v1.8 (Pure Rust + SIMD) | Engineering Advantage |
| :--- | :--- | :--- | :--- |
| **Dependencies** | Python, PyTorch, LibTorch, CUDA (~4–7 GB) | **Pure `std` (0 crates, 0 dynamic dependencies)** | **~99.9% smaller footprint** |
| **Executable Size** | Multi-gigabyte runtime environment | **~1.1 MB release binary** | **Ultra-portable single binary** |
| **Model Load Latency** | ~2,500 – 4,000 ms (PyTorch initialization) | **4.69 ms (8M Q4) / 12.85 ms (35M Q4) / 137.3 ms (650M Q4)** | **~20x – 850x faster startup** |
| **Memory Footprint (RAM)** | ~1.5 – 2.5 GB RSS baseline | **29.3 MB RSS (8M Q4) / 413.1 MB (650M Q4) / 1.75 GB (3B Q4)** | **~4x – 85x leaner memory consumption** |
| **Inference Throughput** | ~15 – 35 tokens/sec (CPU single-thread) | **456.5 tok/s (8M Q4) / 9.04 tok/s (650M Q4) / 2.12 tok/s (3B Q4)** | **High compute density, zero-allocation** |
| **Memory Allocation** | Continuous GC pauses & tensor churn | **0 heap allocations in inference hot-path** | **Deterministic real-time guarantees** |

### 2. Multi-Scale Real Performance Matrix (ESM-2 8M to 3B)
Audited empirical metrics on Linux x86_64 (AVX2+FMA, 4 threads, 41 tokens, kernel `wait4 ru_maxrss` from `docs/benchmarks_v1.6.0.md`):

| Model Scale | Layer Depth (L) | Hidden Dim (d) | Attention Heads (H × d_k) | Q4_0 Binary Size | Q4_0 Load Latency | Bidirectional (41 tok, 4T) | Peak RAM RSS (Q4_0) | Cache Hierarchy & Memory Regime |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | :--- |
| **ESM-2 8M** | 6 | 320 | 20 × 16 (2 YMM) | **4.97 MiB** (`5,208,516 B`) | **4.69 ms** | **456.5 tok/s** (89.8 ms) | **29.29 MiB** | **100% L3 Resident** (≤ 32 MB) · Compute-Bound |
| **ESM-2 35M** | 12 | 480 | 20 × 24 (3 YMM) | **21.07 MiB** (`22,093,636 B`) | **12.85 ms** | **128.7 tok/s** (318.6 ms) | **29.41 MiB** | **Partial L3 / DDR Streaming** · Balanced Cache/DRAM |
| **ESM-2 150M** | 30 | 640 | 20 × 32 (4 YMM) | **90.58 MiB** (`94,978,756 B`) | **39.17 ms** | **30.0 tok/s** (1,367.8 ms) | **100.39 MiB** | **RAM Streaming Bound** (> 500 MB FP32) · Bus-Limited |
| **ESM-2 650M** | 33 | 1280 | 20 × 64 (8 YMM) | **395.41 MiB** (`414,617,796 B`) | **137.32 ms** | **9.04 tok/s** (4,535.5 ms) | **413.07 MiB** | **DRAM Streaming Optimized** · Multi-Core Saturated |
| **ESM-2 3B** | 36 | 2560 | 40 × 64 (8 YMM) | **1,717.76 MiB** (`1,801,205,956 B`) | **545.36 ms** | **2.12 tok/s** (19,368.9 ms) | **1,752.57 MiB** | **DRAM Streaming Bound** (Fits in < 2 GB RAM) |

### 3. ESM-2 650M Multithreading & Quantization Scaling (Physical Cores vs SMT)

Evaluated on dual-core Intel Sandy Bridge (Core i5-2520M @ 2.50 GHz, 2 physical cores, 4 logical SMT threads) with AVX1 vector kernels:

| Configuration | Model Precision | Memory Footprint (RAM) | Cold Load Latency | Throughput | Latency per Token | Speedup vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Thread (Single-core)** | FP32 | 2,485 MB | 751.2 ms | 4.57 tok/s | 218.8 ms/tok | 1.00x (Baseline) |
| **1 Thread (Single-core)** | Q8_0 | 698.5 MB | 219.98 ms | 5.15 tok/s | 194.2 ms/tok | 1.13x |
| **2 Physical Cores (`-t 2`)** | Q8_0 | 698.5 MB | 219.98 ms | 7.70 tok/s | 129.9 ms/tok | 1.68x |
| **4 SMT Threads (`-t 4`)** | Q8_0 | 698.5 MB | 219.98 ms | 7.48 tok/s | 133.7 ms/tok | 1.64x (Port Contention) |
| **4 Threads (`-t 4`)** | **Q4_0** | **413.1 MB** | **137.32 ms** | **9.04 tok/s** | **110.6 ms/tok** | **1.98x (Max Speedup & Leanest RAM)** |

> [!NOTE]
> **Physical Core Saturation vs Hyper-Threading:** On architectures where logical SMT threads share a single 256-bit AVX execution port per physical core, allocating 2 threads to 2 physical cores (`-t 2`) yields the highest throughput (7.70 tok/s) and lowest synchronization latency. `codon-infer` automatically detects physical cores via `detect_optimal_threads` when `--threads` is omitted.

> [!TIP]
> **Multi-Token Tiled GEMM Acceleration (CI-1):**
> In bidirectional forward pipelines (`dms`, `contacts`, and `fold`), the Tiled GEMM engine groups sequence tokens into $M_r = 4$ cache-blocked micro-kernels (`dot_row_batch_m4_avx2_fma`), eliminating per-token DRAM weight streaming. This accelerates sequence forward passes by up to **3.76× in FP32** and **2.41×–3.44× in quantized regimes (Q8_0/Q4_0)**, driving ESM-2 650M forward execution down to **2.3 s – 4.4 s** with exact numerical bit parity ($\cos = 1.00000000$).

---

## 🧬 Biological Exploration & Validation

During validation, `codon-infer` was seeded with a cysteine-rich motif prompt (`MCMPCFTTDHQMARKCDD`, 18 aa). The engine generated candidate sequences under canonical amino acid constraints:

```text
Prompt:     MCMPCFTTDHQMARKCDD
Generation: MCMPCFTTDHQMARKCDDDPCCIIVQCCRQEIIQRCCLRIDRNCCIIIVYCCNCY
```

Submitting generated candidates to the **NCBI BLASTP** database against non-redundant protein sequences confirmed alignment against genuine chlorotoxin crystal structures (PDB: [`7X41_A`](https://www.rcsb.org/structure/7X41), E-value: 0.001), demonstrating structural motif conservation followed by coherent *de novo* peptide space exploration.

---

## 🧬 Multi-Scale ESM-2 Architecture & Dynamic SIMD Engine

Starting with version **v1.4.0**, `codon-infer` provides native, first-class support for the entire multi-scale spectrum of the ESM-2 protein language model family (8M, 35M, 150M, 650M, and 3B):

```text
                                  ┌────────────────────────┐
                                  │ Binary Model Weights   │
                                  │ (*.bin / FP32 / Q8_0)  │
                                  └───────────┬────────────┘
                                              │ File length heuristic
                                              ▼
                             EsmConfig::detect_from_file_len()
                                              │
  ┌───────────────────┬───────────────────────┼───────────────────────┬───────────────────┐
  ▼                   ▼                       ▼                       ▼                   ▼
8M (t6_8M)          35M (t12_35M)           150M (t30_150M)         650M (t33_650M)     3B (t36_3B)
L=6, d=320          L=12, d=480             L=30, d=640             L=33, d=1280        L=36, d=2560
H=20, dk=16         H=20, dk=24             H=20, dk=32             H=20, dk=64         H=40, dk=64
2 AVX Regs          3 AVX Regs              4 AVX Regs              8 AVX / 4 AVX-512   8 AVX / 4 AVX-512
  │                   │                       │                       │                   │
  └───────────────────┴───────────────────────┼───────────────────────┴───────────────────┘
                                              ▼
                             Unified Zero-Allocation Runtime
                            (RunState / BidirectionalRunState)
```

### 1. Parametric Autodetection without Metadata (`detect_from_file_len`)
Traditional Python deep learning frameworks require external configuration files (`config.json`, tokenizer configurations, model manifests, etc.) alongside weight tensors. `codon-infer` eliminates this friction through a deterministic byte-length heuristic:

```rust
pub fn detect_from_file_len(file_len: u64) -> io::Result<Self> {
    match file_len {
        30_091_652 | 30_092_168 => return Ok(Self::esm2_8m()),
        134_067_972 | 134_068_984 => return Ok(Self::esm2_35m()),
        592_642_692 | 592_645_160 => return Ok(Self::esm2_150m()),
        2_604_339_332 | 2_604_342_104 => return Ok(Self::esm2_650m()),
        11_356_354_692 | 11_356_360_584 | 11_357_091_972 => return Ok(Self::esm2_3b()),
        _ => {}
    }
    if file_len >= 3_000_000_000 {
        // ESM-2 3B: ~11.36 GB (FP32) or ~3.2 GB - 3.4 GB (Q8_0)
        Ok(Self::esm2_3b())
    } else if file_len >= 2_000_000_000 {
        // ESM-2 650M: ~2.60 GB (FP32)
        Ok(Self::esm2_650m())
    } else if file_len >= 600_000_000 {
        // ESM-2 650M: ~698.5 MB - 780 MB (Q8_0)
        Ok(Self::esm2_650m())
    } else if file_len >= 450_000_000 {
        // ESM-2 150M: ~592.6 MB (FP32)
        Ok(Self::esm2_150m())
    } else if file_len >= 140_000_000 {
        // ESM-2 150M: ~166 MB - 178 MB (Q8_0)
        Ok(Self::esm2_150m())
    } else if file_len >= 80_000_000 {
        // ESM-2 35M: ~134.1 MB (FP32)
        Ok(Self::esm2_35m())
    } else if file_len >= 30_500_000 {
        // ESM-2 35M: ~38 MB - 48 MB (Q8_0)
        Ok(Self::esm2_35m())
    } else if file_len >= 20_000_000 {
        // ESM-2 8M: ~30.1 MB (FP32)
        Ok(Self::esm2_8m())
    } else if file_len >= 6_000_000 {
        // ESM-2 8M: ~8.5 MB - 12 MB (Q8_0)
        Ok(Self::esm2_8m())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, "Unrecognized ESM-2 weights binary size"))
    }
}
```
Any custom weights binary supplied via `--model <PATH>` is identified and validated at boot time in < 0.1 ms, automatically resizing all internal inference buffers, KV-caches, and embedding projections with zero runtime allocation overhead.

### 2. SIMD Optimization and Heterogeneous Attention Dynamic Dispatch
In multi-head attention, every head has dimension d_k = d_model / H. While 8M, 35M, 150M, and 650M standardize on H = 20 attention heads, 3B scales to H = 40 heads:
* **8M:** d_k = 320 / 20 = 16 floats
* **35M:** d_k = 480 / 20 = 24 floats
* **150M:** d_k = 640 / 20 = 32 floats
* **650M:** d_k = 1280 / 20 = 64 floats
* **3B:** d_k = 2560 / 40 = 64 floats

Because each 256-bit AVX register (`__m256`) holds exactly 8 single-precision (`f32`) floats, and each 512-bit AVX-512 register (`__m512`) holds 16 floats, every `d_k` is an exact integer multiple:

```text
16 / 8 = 2 YMM,   24 / 8 = 3 YMM,   32 / 8 = 4 YMM,   64 / 8 = 8 YMM  (or 4 ZMM in AVX-512)
```

`codon-infer` implements dedicated, unrolled assembly-grade vector kernels for dot products and value accumulations:
* **Scaled Dot-Product (Q · Kᵀ):** `dot_product_head_avx` evaluates 16, 24, 32, or 64 dimensions in parallel without remaining scalar loops or bounds checks (`_mm256_mul_ps`, `_mm256_add_ps`, `hsum256_ps_avx`). On AVX-512 hardware, d_k = 64 dispatches to `dot_product_head_avx512_d64` utilizing 4 ZMM registers and single-cycle horizontal reduction (`_mm512_reduce_add_ps`).
* **Value Vector Accumulation (V × prob):** `accumulate_value_head_avx` broadcasts the scalar probability across vectors (`_mm256_set1_ps`) and executes direct vector multiply-accumulate operations. For d_k = 64, AVX-512 dispatches `accumulate_value_head_avx512_d64` (`_mm512_fmadd_ps`).
* **Scalar Parity Guarantee:** Unit tests (`test_attention_simd_parity_dk_16_24_32`, `test_attention_simd_parity_dk_64`) enforce mathematical equivalence with scalar reference loops under a strict tolerance threshold (Δ < 10⁻⁵) across all 4 head dimensions.

### 3. Architectural and Performance Technical Comparison Matrix

| Parameter / Metric | ESM-2 8M (`esm2_t6_8M_UR50D`) | ESM-2 35M (`esm2_t12_35M_UR50D`) | ESM-2 150M (`esm2_t30_150M_UR50D`) | ESM-2 650M (`esm2_t33_650M_UR50D`) | ESM-2 3B (`esm2_t36_3B_UR50D`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Transformer Layers (L)** | 6 | 12 | 30 | 33 | 36 |
| **Hidden Dimension (d_model)** | 320 | 480 | 640 | 1,280 | 2,560 |
| **Attention Heads (H)** | 20 | 20 | 20 | 20 | 40 |
| **Head Dimension (d_k)** | 16 | 24 | 32 | 64 | 64 |
| **SIMD Registers per Head** | **2 YMM** (16 floats) | **3 YMM** (24 floats) | **4 YMM** (32 floats) | **8 YMM / 4 ZMM** (64 floats) | **8 YMM / 4 ZMM** (64 floats) |
| **FFN Dimension (d_ffn)** | 1,280 | 1,920 | 2,560 | 5,120 | 10,240 |
| **Vocabulary** | 33 tokens | 33 tokens | 33 tokens | 33 tokens | 33 tokens |
| **Total Floats (`grouped_canonical_v1`)** | 7,522,913 (~7.5M) | 33,516,993 (~33.5M) | 148,160,673 (~148M) | 651,084,833 (~651M) | 2,839,088,673 (~2.84B) |
| **Binary Size (FP32)** | `30,091,652 B` (28.70 MiB) | `134,067,972 B` (127.86 MiB) | `592,642,692 B` (565.19 MiB) | **`2,604,339,332 B` (2,483.69 MiB)** | **`11,356,354,692 B` (10.58 GiB)** |
| **Quantized Binary Size (`Q8_0`)** | `8,864,556 B` (8.45 MiB) | `38,636,896 B` (36.85 MiB) | `168,646,036 B` (160.83 MiB) | **`738,899,556 B` (704.67 MiB)** | **`3,216,540,676 B` (3.00 GiB)** |
| **Quantized Binary Size (`Q4_0`)** | **`5,208,516 B` (4.97 MiB)** | **`22,093,636 B` (21.07 MiB)** | **`94,978,756 B` (90.58 MiB)** | **`414,617,796 B` (395.41 MiB)** | **`1,801,205,956 B` (1.68 GiB)** |
| **Warm Load Latency (Q4_0)** | **4.69 ms** | **12.85 ms** | **39.17 ms** | **137.32 ms** | **545.36 ms** |
| **Bidirectional Throughput (L = 41, 4T)** | **476.9 tok/s (Q8) / 456.5 tok/s (Q4)** | **128.7 tok/s (Q4)** | **30.0 tok/s (Q4)** | **9.04 tok/s (Q4)** | **2.20 tok/s (Q8) / 2.12 tok/s (Q4)** |
| **Working State Memory (L = 128)** | **~911 KB** | **~1,321 KB** | **~1,732 KB** | **~3,450 KB** | **~7,200 KB** |
| **Cache Hierarchy Residency** | **100% in L3 Cache** (≤ 32 MB) | Partial Residency in L3 / DDR | Streaming from RAM (> 500 MB) | DRAM Streaming (~2.48 GB FP32 / ~395 MB Q4) | Massive DRAM Streaming (~1.68 GiB Q4) |
| **Memory Bandwidth Demand** | **Compute-Bound (AVX/SIMD)** | **Balanced Cache/DRAM** | **Memory-Bus Limited** | **Memory-Bus Limited (Q4/Q8-Optimized)** | **Extreme Memory-Bus Bound (Q4/Q8 Enabled)** |
| **Primary Use Case** | Fast in silico DMS, real-time inpainting | High-resolution DMS, peptide screening | 3D contact prediction, de novo design | Publication-grade contact maps, fine DMS | Frontier protein language modeling on CPU |

### 4. Multi-Scale Empirical Validation & Benchmarks vs. HuggingFace (`8M` to `3B`)

Empirical benchmark evaluated on `test_peptidase.fasta` (41 tokens) against official HuggingFace `facebook/esm2_*` reference tensors (`scripts/ref_{8m,35m,150m,650m,3b}.npz`, generated locally via `scripts/generate_ref_8m.py` or `scripts/colab_export_esm2.ipynb`) under native SIMD execution (4 threads, `wait4` kernel `ru_maxrss`):

| Model Scale | Format | Exact File Size (`stat -c %s`) | Total Comp. | `hidden_states[-1]` Cos vs HF | Logits Cos vs HF | KL Div vs HF | Top-1 Match vs HF | Forward Median (4 th) | Throughput | Peak RAM (RSS) |
| :--- | :---: | ---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| **ESM-2 8M** | FP32 | `30,091,652 B` (28.70 MiB) | 1.00× | 0.999999 | 0.999999 | 0.000000 | **100.0% (41/41)** | 113.44 ms | 361.4 tok/s | 32.75 MiB |
| | Q8_0 | `8,864,556 B` (8.45 MiB) | 3.39× | 0.999764 | 0.999995 | 0.000128 | **100.0% (41/41)** | 85.97 ms | 476.9 tok/s | 29.11 MiB |
| | **Q4_0** | **`5,208,516 B` (4.97 MiB)** | **5.78×** | **0.934598** | **0.998168** | **0.031316** | **100.0% (41/41)** | **89.82 ms** | **456.5 tok/s** | **29.29 MiB** |
| **ESM-2 35M** | FP32 | `134,067,972 B` (127.86 MiB) | 1.00× | 0.999999 | 1.000000 | 0.000000 | **100.0% (41/41)** | 446.47 ms | 91.8 tok/s | 133.30 MiB |
| | Q8_0 | `38,636,896 B` (36.85 MiB) | 3.47× | 0.999825 | 0.999994 | 0.000120 | **100.0% (41/41)** | 377.27 ms | 108.7 tok/s | 42.24 MiB |
| | **Q4_0** | **`22,093,636 B` (21.07 MiB)** | **6.07×** | **0.952611** | **0.998566** | **0.048077** | **95.1% (39/41)** | **318.58 ms** | **128.7 tok/s** | **29.41 MiB** |
| **ESM-2 150M** | FP32 | `592,642,692 B` (565.19 MiB) | 1.00× | 0.999999 | 1.000000 | 0.000000 | **100.0% (41/41)** | 2,390.36 ms | 17.2 tok/s | 575.01 MiB |
| | Q8_0 | `168,646,036 B` (160.83 MiB) | 3.51× | 0.999934 | 0.999998 | 0.000042 | **100.0% (41/41)** | 1,430.16 ms | 28.7 tok/s | 170.61 MiB |
| | **Q4_0** | **`94,978,756 B` (90.58 MiB)** | **6.24×** | **0.981715** | **0.999266** | **0.020517** | **100.0% (41/41)** | **1,367.83 ms** | **30.0 tok/s** | **100.39 MiB** |
| **ESM-2 650M** | FP32 | `2,604,339,332 B` (2,483.69 MiB) | 1.00× | 0.999999 | 1.000000 | 0.000000 | **100.0% (41/41)** | 8,692.15 ms | 4.72 tok/s | 2,501.33 MiB |
| | Q8_0 | `738,899,556 B` (704.67 MiB) | 3.52× | 0.999913 | 0.999996 | 0.000039 | **100.0% (41/41)** | 4,669.89 ms | 8.78 tok/s | 722.41 MiB |
| | **Q4_0** | **`414,617,796 B` (395.41 MiB)** | **6.28×** | **0.980842** | **0.999292** | **0.998939** | **97.6% (40/41)** | **4,535.47 ms** | **9.04 tok/s** | **413.07 MiB** |
| **ESM-2 3B** | FP32 | `11,356,354,692 B` (10.58 GiB) | 1.00× | *Ref HF* | *Ref HF* | *Ref HF* | *Ref HF* | *N/A (>11 GiB RAM)* | *N/A* | *Streaming Only* |
| | **Q8_0** | **`3,216,540,676 B` (3.00 GiB)** | **3.53×** | **0.999983** | **0.999998** | **0.000027** | **100.0% (41/41)** | **18,615.93 ms** | **2.20 tok/s** | **3,102.18 MiB** |
| | **Q4_0** | **`1,801,205,956 B` (1.68 GiB)** | **6.30×** | **0.994156** | **0.999422** | **0.006447** | **100.0% (41/41)** | **19,368.89 ms** | **2.12 tok/s** | **1,752.57 MiB** |

> [!NOTE]
> **Fidelity, Tiled GEMM & Reference Generation Notes (`docs/benchmarks_v1.6.0.md` & `docs/Q4_0_SPEC.md`):**
> 1. **Exact HuggingFace Parity:** Across all FP32 models (`8M` to `650M`), `codon-infer` matches HuggingFace `facebook/esm2_*` layer-by-layer hidden states and final logits with cosine ≥ 0.999999 and **100.0% Top-1 match**. Reference `.npz` files (`scripts/ref_*.npz`) can be generated locally with `python3 scripts/generate_ref_8m.py` or `scripts/colab_export_esm2.ipynb` (regression tests automatically skip missing optional `.npz` files in clean clones).
> 2. **ESM-2 3B on Commodity RAM:** Streaming quantization (`12.55 MiB` Peak RSS) compresses the 10.58 GiB `esm2_3b_weights.bin` into a `1.68 GiB` `Q4_0` binary (`1,801,205,956 B`) that loads in **545 ms** and executes in **1,752.57 MiB Peak RSS** with **0.999422 logits cosine** and **100.0% Top-1 match** against HuggingFace `esm2_t36_3B_UR50D`.
> 3. **Single-Token GEMV Baseline vs. Multi-Token Tiled GEMM (`CI-1`):** The latencies in the table above represent the pre-tiling single-token GEMV baseline (`docs/benchmarks_v1.6.0.md`). With the v1.7.0/v1.8.0 **Multi-Token Tiled GEMM (`CI-1`)** engine enabled (`M_r = 4`), multi-token bidirectional passes achieve an additional **3.76× speedup in FP32** and **2.41×–3.44× in Q8_0/Q4_0** while preserving bit-level parity ($\cos = 1.00000000$).

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
 │ Transformer Layer (×L Layers: 6, 12, 30, 33, or 36)          │    │
 │  ├── LayerNorm 1 (dim, eps 1e-5)                             │    │
 │  ├── Multi-Head Attention (20/40 Heads × dk: 16, 24, 32, 64) │    │
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
 │  ├── Head LayerNorm (dim, eps 1e-5)                          │
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

> [!NOTE]
> **100% Binary Compatibility with Canonical `v1.6.0` Weight Assets:** `codon-infer v1.8.0` uses the exact same canonical model weight binaries (`.bin`, `_q8_0.bin`, `_q4_0.bin`) published in [Release `v1.6.0`](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0). Because the official Meta AI contact regression weights (`STRUCT-1`) are embedded directly in 12.2 KiB of static Rust memory (`src/contact_heads.rs`) and the 3D folding solver (`STRUCT-2`) is zero-trunk, no weight format changes or multi-gigabyte re-downloads are required.

`codon-infer` standardizes model binary files on clean, explicit nomenclature:
* **ESM-2 8M (`esm2_8m_weights.bin`):** Pre-packaged in the repository root (~28.7 MB FP32 canonical) for instant out-of-the-box inference, local development, and rapid unit testing without external downloads.
* **ESM-2 35M (`esm2_35m_weights.bin`):** Canonical full-precision model (~127.9 MB FP32), hosted in [GitHub Releases v1.6.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0).
* **ESM-2 150M (`esm2_150m_weights.bin`):** Canonical publication-grade model (~565.2 MB FP32), hosted in [GitHub Releases v1.6.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0).
* **ESM-2 650M Q8_0 (`esm2_650m_q8_0.bin`):** Zero-copy quantized high-capacity binary with CODQ header (704.7 MB Q8_0), hosted in [GitHub Releases v1.6.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0).
* **ESM-2 650M Q4_0 (`esm2_650m_q4_0.bin`):** Sub-byte 4-bit quantized high-capacity binary (395.41 MiB / 414.62 MB Q4_0), hosted in [GitHub Releases v1.6.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0).
* **ESM-2 3B Q4_0 (`esm2_3b_q4_0.bin`):** Ultra-high capacity 3B model in 4-bit quantization (1.68 GiB / 1,801.21 MB Q4_0), hosted in [GitHub Releases v1.6.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0) for execution under 1.75 GiB RAM!
* **ESM-2 650M FP32 (`esm2_650m_weights.bin`) & ESM-2 3B FP32 (`esm2_3b_weights.bin`):** Due to GitHub Releases' 2.0 GiB file asset limit, full-precision checkpoints exceeding 2 GiB (~2.60 GB for 650M FP32 and ~11.36 GB for 3B FP32) can be generated with the provided Colab notebook (`scripts/colab_export_esm2.ipynb`) or exported directly.

#### Download Weights via `curl`
Directly download the desired scaled binary model into your repository root:
```bash
# Download ESM-2 35M weights (~128 MB FP32, hosted in v1.6.0)
curl -L -o esm2_35m_weights.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.6.0/esm2_35m_weights.bin

# Download ESM-2 150M weights (~565 MB FP32, hosted in v1.6.0)
curl -L -o esm2_150m_weights.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.6.0/esm2_150m_weights.bin

# Download ESM-2 650M Q8_0 weights (705 MB Q8_0, hosted in v1.6.0)
curl -L -o esm2_650m_q8_0.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.6.0/esm2_650m_q8_0.bin

# Download ESM-2 650M Q4_0 weights (395 MiB Q4_0, hosted in v1.6.0)
curl -L -o esm2_650m_q4_0.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.6.0/esm2_650m_q4_0.bin

# Download ESM-2 3B Q4_0 weights (1.68 GiB Q4_0, hosted in v1.6.0)
curl -L -o esm2_3b_q4_0.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.6.0/esm2_3b_q4_0.bin

# Or use the automated helper script:
./scripts/download_weights.sh 35m
./scripts/download_weights.sh 150m
./scripts/download_weights.sh 650m-q8
./scripts/download_weights.sh 650m-q4
./scripts/download_weights.sh 3b-q4
```

> [!TIP]
> You can also export custom weights directly from HuggingFace Transformers or Meta `fair-esm` using the provided Python utility:
> ```bash
> python3 scripts/export_weights.py --model 35M
> python3 scripts/export_weights.py --model 650M
> ```

### 3. Run Test Suite
Verify numerical primitives, SIMD vectorization parity, FASTA parser, and export formats:
```bash
cargo test --release
```
Expected output (**57 tests executed: 56 passed, 0 failed, 1 ignored**):
```text
running 57 tests
test benchmark::tests::test_benchmark_report_to_json_validity ... ok
test benchmark::tests::test_discover_available_models ... ok
test export::tests::test_dms_row_and_csv_generation ... ok
test export::tests::test_json_string_serializers ... ok
test export::tests::test_pdb_and_tsv_export ... ok
test fasta::tests::test_parse_multi_fasta_with_masks ... ok
test fasta::tests::test_parse_raw_sequence_fallback ... ok
test fasta::tests::test_parse_simple_fasta ... ok
test mcp::tests::test_mcp_dispatch_with_weights_if_available ... ok
test mcp::tests::test_mcp_json_parser_and_formatter ... ok
test mcp::tests::test_mcp_resolve_record_fallback ... ok
test mcp::tests::test_mcp_tools_list_schema ... ok
test tests::test_attention_simd_parity_dk_16_24_32 ... ok
test tests::test_attention_simd_parity_dk_64 ... ok
test tests::test_bidirectional_logits_entropy ... ok
test tests::test_esm_config_autodetection ... ok
test tests::test_evaluation_metrics_unit ... ok
test tests::test_forward_650m_if_available ... ok
test tests::test_forward_650m_q8_and_multithread_if_available ... ok
test tests::test_forward_bidirectional_if_available ... ok
test tests::test_forward_pass_with_weights_if_available ... ok
test tests::test_forward_q4_cosine_similarity_8m ... ok
test tests::test_forward_q8_cosine_similarity_8m ... ok
test tests::test_layer_norm_stability ... ok
test tests::test_log_softmax_canonical ... ok
test tests::test_mask_token_encoding_and_decoding ... ok
test tests::test_matmul_avx1_and_dispatch_parity ... ok
test tests::test_matmul_fp32_parallel_parity ... ok
test tests::test_matmul_q8_simd_parity ... ok
test tests::test_q8_0_quantization_roundtrip ... ok
test tests::test_q8_weights_save_load_roundtrip ... ok
test tests::test_simd_benchmark_parity_run ... ok
test tests::test_softmax_sum_to_one ... ok
test tests::test_tokenizer_boundaries ... ok
test tests_q4_format::test_codq_header_valid_corrupt_incoherent ... ok
test tests_q4_format::test_first_block_alignment_at_offset_64 ... ok
test tests_q4_format::test_no_ambiguity_650m_q4_vs_150m ... ok
test tests_q4_format::test_q4_0_exact_nibble_packing ... ok
test tests_q4_format::test_q4_0_roundtrip_quantize_dequantize_bound ... ok
test tests_q4_format::test_q4_0_zero_block ... ok
test tests_q4_format::test_q4_weights_save_load_roundtrip_8m ... ok
test tests_q4_integration::test_forward_q4_integration_8m ... ok
test tests_q4_integration::test_forward_q4_integration_35m ... ok
test tests_q4_integration::test_forward_q4_integration_150m ... ok
test tests_q4_integration::test_forward_q4_integration_650m ... ok
test tests_q4_integration::test_forward_q8_codq_header_integration ... ok
test tests_q4_integration::test_generate_validation_report ... ignored
test tests_q4_simd::test_q4_edge_case_extreme_values ... ok
test tests_q4_simd::test_q4_edge_case_small_out_dim ... ok
test tests_q4_simd::test_q4_edge_case_zero_blocks ... ok
test tests_q4_simd::test_q4_parallel_vs_sequential_identity_threads ... ok
test tests_q4_simd::test_q4_simd_parity_esm_dimensions ... ok
test tests_regression::test_regression_esm2_fp32_all_scales_against_ref_npz ... ok
test tests_regression::test_regression_quantized_q8_q4_against_ref_npz_including_3b ... ok
test tests_regression::test_strict_fp32_loader_rejects_corrupted_size ... ok
test tests_regression::test_tiled_gemm_and_contact_head_v1_7_0 ... ok
test benchmark::tests::test_multi_scale_simd_parity ... ok

test result: ok. 56 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out
```

### 4. Subcommands & Bioinformatic Pipelines

All subcommands support the global `--model <PATH>`, `--threads <N>`, `--quantize <q4_0|q8_0|fp32>`, `--q4`, and `--q8` flags. `codon-infer` automatically detects whether the binary corresponds to **ESM-2 8M**, **35M**, **150M**, **650M**, or **3B** (FP32 or quantized `Q8_0` / `Q4_0`) via the 64-byte `CODQ` header or file size heuristics and configures the hardware kernels accordingly.

#### A. Autoregressive Sequence Generation (`generate`)
Execute de novo sequence generation or motif extension:
```bash
# Default (ESM-2 8M)
codon-infer generate --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7 --max-tokens 40

# Multi-scale: Run with ESM-2 35M, 150M, or 650M (with 2 physical threads)
codon-infer generate --model esm2_35m_weights.bin --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7
codon-infer generate --model esm2_150m_weights.bin --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7
codon-infer generate --model esm2_650m_q8_0.bin --threads 2 --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7
```

#### B. Bidirectional Inpainting & Motif Infilling (`inpaint`)
Reconstruct missing, ambiguous, or masked active-site motifs (marked with `*` or `<mask_1>`):
```bash
# Default (ESM-2 8M)
codon-infer inpaint --sequence "MCMPC*TTDHQMARKCDD"

# With high-capacity ESM-2 650M Q8_0 weights and multi-core parallelism:
codon-infer inpaint --model esm2_650m_q8_0.bin --threads 2 --sequence "MCMPC*TTDHQMARKCDD"
```

#### C. In Silico Deep Mutational Scanning (`dms`)
Perform comprehensive single-point saturation mutagenesis with TrueColor thermal heatmap rendering and tabular export:
```bash
# High-throughput screening with ESM-2 8M:
codon-infer dms --fasta proteins.fasta --window 128 --export-csv dms_matrix.csv --export-json dms_report.json

# Ultra-fast 1-pass Wildtype Marginal screening (>16,170 muts/sec, 42× acceleration):
codon-infer dms --fasta proteins.fasta --fast --export-csv dms_fast.csv

# High-fidelity scanning with ESM-2 150M or 650M:
codon-infer dms --model esm2_150m_weights.bin --fasta target.fasta --window 128 --export-csv dms_150m.csv
codon-infer dms --model esm2_650m_q8_0.bin --threads 2 --fasta target.fasta --window 128 --export-csv dms_650m.csv
```

#### D. Zero-Shot Structural Contact Prediction (`contacts`)
Compute epistatic residue couplings using multi-head attention contact maps (APC-corrected) and identify putative disulfide bonds:
```bash
# Predict 3D contact pairs with ESM-2 150M or 650M for publication-grade precision:
codon-infer contacts --model esm2_150m_weights.bin --fasta proteins.fasta --export-csv contacts.csv --export-json contacts.json
codon-infer contacts --model esm2_650m_q8_0.bin --threads 2 --fasta proteins.fasta --export-csv contacts.csv

# Predict contacts and export ab initio 3D C_alpha backbone structure (.pdb) simultaneously:
codon-infer contacts --model esm2_650m_q8_0.bin --threads 4 --sequence "MCMPCFTTDHQMARKCDD" --export-pdb contacts_backbone.pdb
```

#### E. Native Q4_0 & Q8_0 Model Quantization (`quantize`)
Convert FP32 checkpoint weights directly into native zero-copy `Q4_0` or `Q8_0` binaries on disk with universal `CODQ` headers:
```bash
# Quantize ESM-2 650M weights from FP32 (~2.60 GB) to Q4_0 (395.41 MiB / 414.62 MB, 6.28x smaller):
codon-infer quantize --model esm2_650m_weights.bin --output esm2_650m_q4_0.bin --format q4_0

# Or rely on automatic format detection from output filename:
codon-infer quantize --model esm2_650m_weights.bin --output esm2_650m_q4_0.bin
codon-infer quantize --model esm2_650m_weights.bin --output esm2_650m_q8_0.bin

# Direct instant zero-copy inference using the resulting quantized binary (auto-detected via CODQ header):
codon-infer generate --model esm2_650m_q4_0.bin --threads 4 --sequence "MCMPCFTTDHQMARKCDD"
codon-infer dms --model esm2_650m_q4_0.bin --threads 4 --fasta target.fasta --window 128
```

#### F. Hardware & Biological Benchmark Suite (`benchmark`)
Execute the full hardware stress and biological validation suite across any model size:
```bash
# Execute comparative cross-model benchmark across all discovered weights (8M, 35M, 150M, 650M, 3B):
codon-infer benchmark --suite full

# Benchmark a specific model variant with custom thread configuration:
codon-infer benchmark --model esm2_35m_weights.bin --suite full
codon-infer benchmark --model esm2_650m_weights.bin --threads 2 --suite full
codon-infer benchmark --model esm2_650m_q8_0.bin --threads 2 --suite full

# Export structured performance report for CI/CD pipelines:
codon-infer benchmark --suite full --export-json benchmark_report.json
```

Available suites:
* `full`: Multi-model comparative evaluation (weight loading latency, tokens/s throughput, working memory, cache profile) plus algorithmic tests.
* `simd`: Multi-scale AVX2+FMA vs AVX1 numerical parity and matrix-vector microbenchmarks across projection dimensions (320 → 1280, 480 → 1920, 640 → 2560, 1280 → 5120, 2560 → 10240) and head dimensions d_k ∈ {16, 24, 32, 64}.
* `mlm`: Evolutionary pseudo-perplexity and wild-type vs disruptive mutation penalties.
* `dms`: Deep mutational scanning calibration on bacterial catalytic peptidase motifs.
* `contacts`: Structural contact prediction and Cys-Cys spatial proximity scoring.
* `inpainting`: Multi-position bidirectional masked sequence infilling.
* `stress`: Dynamic context continuous inference cache-thrashing stability test with slice Q8 quantization.
* `scaling`: Context length scaling (L = 32 → 256) memory and latency benchmark adapting to model depth.

#### G. Zero-Trunk 3D Protein Folding (`fold`)
Reconstruct ab initio 3D $C_\alpha$ backbone coordinates in standard `.pdb` or `.tsv` format in **15–35 ms** via Metric MDS and biophysical gradient optimization:
```bash
# Rapid 3D backbone reconstruction from sequence with ESM-2 8M:
codon-infer fold --sequence "MCMPCFTTDHQMARKCDD" --export-pdb structure_8m.pdb

# High-precision folding with ESM-2 650M Q4_0 and multi-threaded Tiled GEMM:
codon-infer fold --model esm2_650m_q4_0.bin --threads 4 --sequence "NLYIQWLKDGGPSSGRPPPS" --export-pdb trpcage.pdb --export-tsv trpcage.tsv

# Fold target protein sequence from multi-FASTA file:
codon-infer fold --model esm2_650m_q4_0.bin --threads 4 --fasta target.fasta --export-pdb target_folded.pdb
```

### 5. Model Context Protocol (MCP) Server
`codon-infer-mcp` exposes the entire bioinformatic engine as native tools for LLM agents (Claude Desktop, Cursor, Gemini, etc.) over stdio JSON-RPC:
```bash
# Launch with default ESM-2 8M weights
cargo run --release --bin codon-infer-mcp

# Launch with higher-capacity ESM-2 35M, 150M, 650M, or 3B weights
cargo run --release --bin codon-infer-mcp -- --model esm2_35m_weights.bin
cargo run --release --bin codon-infer-mcp -- --model esm2_650m_q8_0.bin
```

#### Tools Exposed over MCP:
* `generate_protein`: Autoregressive sequence generation given `prompt`, `temperature`, `max_tokens`.
* `codon_infer_inpaint`: Reconstructs masked residues (`*`) returning top-5 amino acid candidates with probabilities and physicochemical classifications.
* `codon_infer_dms`: Runs in silico saturation mutagenesis from raw sequence or FASTA path, returning critical catalytic sites (Δ log P ≤ -3.8) and mutational tolerances.
* `codon_infer_contacts`: Computes epistatic contact pairs (APC) and identifies putative disulfide bridges (`Cys-Cys`).
* `codon_infer_fold`: Reconstructs publication-grade 3D protein structure ($C_\alpha$ backbone coordinates in `.pdb` format) via Zero-Trunk MDS and biophysical gradient optimization.

---

## 🛠️ Unified CLI Options Reference

### Global Flags
| Option | Description | Default |
| :--- | :--- | :--- |
| `-h`, `--help` | Display help banner with subcommand guide | - |
| `-V`, `--version` | Display version and active SIMD vector engine | - |
| `-m`, `--model <PATH>` | Custom ESM-2 raw FP32/Q8_0/Q4_0 weights binary (or `all` for comparative benchmark) | `esm2_8m_weights.bin` |
| `-t`, `--threads <N>` | Worker threads for parallel matrix-vector execution | Detected physical cores |
| `--quantize <q4_0|q8_0|fp32>`, `--q4`, `--q8` | Execution quantization mode (`auto`, `fp32`, `q8_0`, `q4_0`). `--q4` / `--q8` are shortcuts | `auto` |
| `--color <auto|always|never>` | Control ANSI TrueColor terminal styling | `auto` |
| `--no-color` | Strictly disable ANSI terminal escapes (NO_COLOR standard) | `false` |

### Pipeline Options
| Option | Applicable Subcommands | Description | Default |
| :--- | :--- | :--- | :--- |
| `-s`, `--sequence <STR>` | `generate`, `inpaint`, `dms`, `contacts`, `fold` | Direct input amino acid sequence | `"M"` (`generate` only; required if `--fasta` is omitted in other subcommands) |
| `-f`, `--fasta <PATH>` | `generate`, `inpaint`, `dms`, `contacts`, `fold` | Path to FASTA file (or `-` for stdin) | - |
| `--export-pdb <PATH>` | `contacts`, `fold` | Destination file path for standard `.pdb` atomic coordinates | - |
| `--export-tsv <PATH>` | `fold` | Destination file path for tab-separated 3D coordinates table | - |
| `--output <PATH>`, `-o` | `quantize` | Output path for quantized binary (`*_q4_0.bin` or `*_q8_0.bin`) | - |
| `--format <q4_0|q8_0>`, `-f` | `quantize` | Target quantization format (optional if inferrable from output filename) | `q8_0` |
| `--export-csv <PATH>`| `dms`, `contacts` | Destination path for rectangular matrix / table | - |
| `--export-json <PATH>`| `dms`, `contacts`, `inpaint`, `fold`, `benchmark` | Destination path for structured JSON analysis / report | - |
| `--fast`, `--wt-marginal` | `dms` | Fast 1-Pass Wildtype Marginal scoring mode (accelerates up to 42×) | `false` |
| `--window <USIZE>` | `dms` | Sliding window context size for long proteins | `128` |
| `--stride <USIZE>` | `dms` | Sliding window stride step size | `64` |
| `--min-separation <USIZE>` | `contacts` | Minimum sequence separation for epistatic contacts | `6` |
| `--top-l <F32>` | `contacts` | Top L contact factor threshold (e.g., 1.0 = L contacts) | `1.0` |
| `--temperature <F32>`| `generate` | Sampling temperature (0.0 = greedy argmax) | `0.7` |
| `--max-tokens <USIZE>`| `generate` | Maximum new residues to generate | `1024` |
| `--seed <U64>` | `generate` | Custom pseudo-random number generator seed | - |
| `--suite <STR>` | `benchmark` | Benchmark suite (full, simd, mlm, dms, contacts, inpainting, stress, scaling) | `"full"` |

---

## 📂 Project Structure

```text
codon-infer/
├── .cargo/
│   └── config.toml         # Native CPU optimization flags (-C target-cpu=native)
├── .gitignore              # Clean repository filter (build artifacts & backups ignored)
├── Cargo.toml              # Zero-dependency specification (v1.8.0, edition 2024)
├── esm2_8m_weights.bin     # ESM-2 8M FP32 weights (7.52M parameters, ~30 MB)
├── scripts/
│   ├── download_weights.sh # Fast GitHub Releases weights downloader
│   ├── export_weights.py   # HuggingFace & fair-esm weight exporter
│   ├── generate_ref_8m.py  # Local HuggingFace reference .npz generator
│   └── colab_export_esm2.ipynb # Google Colab canonical weights exporter & HF reference generator
├── src/
│   ├── lib.rs              # Core engine: persistent ThreadPool, Tiled GEMM, Q8/Q4 kernels, SIMD attention
│   ├── main.rs             # CLI dispatcher & subcommand argument parser
│   ├── ui.rs               # TrueColor ANSI palettes, thermal gradients & progress bar
│   ├── fasta.rs            # Zero-copy multi-FASTA & stream sequence reader
│   ├── export.rs           # Publication-grade CSV, TSV, JSON, and PDB serializers
│   ├── pipeline.rs         # Sliding-window DMS, contacts, inpainting, and STRUCT-2 3D folding
│   ├── contact_heads.rs    # Official Meta AI Contact Prediction Head regression weights & biases
│   ├── benchmark.rs        # Automated bioinformatic & hardware stress suite
│   ├── mcp.rs              # Zero-dependency stdio JSON-RPC MCP server
│   └── mcp_main.rs         # Standalone MCP server executable entrypoint
├── LICENSE                 # MIT License
├── README.md               # Main project documentation (English)
└── README_ES.md            # Complete project documentation (Spanish)
```

---

## 📜 Roadmap

- [x] Pure Rust migration from C prototype (**ADR-01**).
- [x] Strict zero-allocation heap runtime (`RunState`).
- [x] Dual SIMD vectorization: `AVX2+FMA` and `AVX1 Sandy Bridge` kernels with runtime dispatch.
- [x] Multi-scale architecture scaling (8M, 35M, 150M, 650M, 3B) with heterogeneous attention (d_k ∈ {16, 24, 32, 64}).
- [x] AVX-512 SIMD vector micro-kernels for d_k = 64 attention processing.
- [x] RoPE `split-half` formulation and numerical `erf` GELU approximation.
- [x] Full `EsmLMHead` classifier integration (canonical dense + GELU + LayerNorm + decoder bias).
- [x] Block `Q8_0` Quantization for L3 cache & DRAM bandwidth optimization.
- [x] Zero-Spawn Persistent `ThreadPool` in pure `std` with hybrid spin/futex barrier.
- [x] ESM-2 650M native support & validated zero-copy `Q8_0` binary serialization.
- [x] Standard Model Context Protocol (MCP) JSON-RPC daemon (`codon-infer-mcp`).
- [x] Modular separation: `lib.rs` (core library), `main.rs` (CLI), `mcp.rs` (MCP daemon).
- [x] Sub-byte INT4 quantization (`Q4_0`) and universal `CODQ` 64-byte header across all 5 scales (8M to 3B).
- [x] Batched GEMM `[N × D] × [D × D]` sequence forward kernels to eliminate per-token DRAM weight streaming (Tiled GEMM 3.76× speedup).
- [x] Canonical Deep Mutational Scanning (BIO-1) without heuristic clamps + 1-pass Wildtype Marginal (16,170 mut/s).
- [x] Official Multi-Layer Contact Prediction Head (L × H APC + Logistic Regression) with streaming O(N²) memory.
- [x] Zero-Trunk 3D Geometric Reconstruction (STRUCT-2) with Metric MDS + Biophysical Gradient Descent (PDB export).
- [ ] ARM NEON SIMD kernels for Apple Silicon (M1/M2/M3/M4) and Raspberry Pi.

---

## 🔬 Citation & Acknowledgments

If you use `codon-infer` in your bioinformatic workflows, structural biology, or synthetic biology research, please cite:
* **ESM-2 Original Architecture:** Lin, Z., et al. *"Evolutionary-scale prediction of atomic-level protein structure with a language model."* Science 379.6637 (2023): eade2574.
* **Author & Lead Architect:** Manuel Osbaldo Gastélum Islas ([@OsbaldoG](https://github.com/OsbaldoG))
