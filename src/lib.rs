#![allow(
    unused_unsafe,
    unsafe_op_in_unsafe_fn,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::missing_safety_doc,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::excessive_precision,
    clippy::collapsible_if,
    clippy::manual_strip,
    clippy::let_and_return,
    clippy::chunks_exact_to_as_chunks
)]
use std::fs::File;
use std::io::{self, Read, Seek, Write};
use std::sync::{Arc, Mutex, Condvar, OnceLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

pub mod mcp;
pub mod benchmark;
pub mod ui;
pub mod fasta;
pub mod export;
pub mod pipeline;

pub use ui::TerminalTheme;
pub use fasta::{FastaRecord, parse_fasta_file, parse_raw_sequence};
pub use export::{
    DmsPositionRow, export_dms_csv, export_dms_json, export_contacts_csv, export_contacts_json,
    dms_to_json_string, contacts_to_json_string, inpaint_to_json_string,
};
pub use pipeline::{
    run_pipeline_dms, run_pipeline_dms_parallel,
    run_pipeline_inpaint, run_pipeline_inpaint_parallel,
    run_pipeline_contacts, run_pipeline_contacts_parallel,
    render_dms_heatmap, compute_pipeline_inpaint, compute_pipeline_inpaint_parallel,
    compute_pipeline_contacts, compute_pipeline_contacts_parallel,
    InpaintPipelineResult, InpaintSite, InpaintCandidate, ContactsPipelineResult,
};
pub use benchmark::{
    ModelBenchmarkMetrics, MultiModelBenchmarkReport, MultiScaleSimdReport,
    benchmark_model_performance, discover_available_models, export_benchmark_json,
    render_comparative_table, run_benchmark_cli, run_multi_model_benchmark_suite,
    run_simd_benchmark, run_multi_scale_simd_benchmark,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EsmModelArch {
    Esm2_8M,
    Esm2_35M,
    Esm2_150M,
    Esm2_650M,
    Esm2_3B,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EsmConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub arch: EsmModelArch,
}

impl EsmConfig {
    pub fn esm2_8m() -> Self {
        Self {
            dim: 320,
            hidden_dim: 1280,
            n_layers: 6,
            n_heads: 20,
            vocab_size: 33,
            max_seq_len: 1024,
            arch: EsmModelArch::Esm2_8M,
        }
    }

    pub fn esm2_35m() -> Self {
        Self {
            dim: 480,
            hidden_dim: 1920,
            n_layers: 12,
            n_heads: 20,
            vocab_size: 33,
            max_seq_len: 1024,
            arch: EsmModelArch::Esm2_35M,
        }
    }

    pub fn esm2_150m() -> Self {
        Self {
            dim: 640,
            hidden_dim: 2560,
            n_layers: 30,
            n_heads: 20,
            vocab_size: 33,
            max_seq_len: 1024,
            arch: EsmModelArch::Esm2_150M,
        }
    }

    pub fn esm2_650m() -> Self {
        Self {
            dim: 1280,
            hidden_dim: 5120,
            n_layers: 33,
            n_heads: 20,
            vocab_size: 33,
            max_seq_len: 1024,
            arch: EsmModelArch::Esm2_650M,
        }
    }

    pub fn esm2_3b() -> Self {
        Self {
            dim: 2560,
            hidden_dim: 10240,
            n_layers: 36,
            n_heads: 40,
            vocab_size: 33,
            max_seq_len: 1024,
            arch: EsmModelArch::Esm2_3B,
        }
    }

    #[inline]
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn model_name(&self) -> &'static str {
        match self.arch {
            EsmModelArch::Esm2_8M => "ESM-2 8M (esm2_t6_8M_UR50D)",
            EsmModelArch::Esm2_35M => "ESM-2 35M (esm2_t12_35M_UR50D)",
            EsmModelArch::Esm2_150M => "ESM-2 150M (esm2_t30_150M_UR50D)",
            EsmModelArch::Esm2_650M => "ESM-2 650M (esm2_t33_650M_UR50D)",
            EsmModelArch::Esm2_3B => "ESM-2 3B (esm2_t36_3B_UR50D)",
        }
    }

    #[inline]
    pub fn lm_head_floats(&self) -> usize {
        3 * self.dim + self.dim * self.dim + self.vocab_size * self.dim + self.vocab_size
    }

    #[inline]
    pub fn grouped_canonical_floats(&self) -> usize {
        self.vocab_size * self.dim
            + self.n_layers
                * (4 * self.dim * self.dim
                    + 9 * self.dim
                    + 2 * self.dim * self.hidden_dim
                    + self.hidden_dim)
            + 2 * self.dim
            + self.lm_head_floats()
    }

    #[inline]
    pub fn grouped_canonical_bytes(&self) -> u64 {
        (self.grouped_canonical_floats() as u64) * 4
    }

    #[inline]
    pub fn sequential_hf_floats(&self) -> usize {
        let head_dim = self.dim / self.n_heads;
        let inv_freq_len = head_dim / 2;
        let contact_head_len = self.n_layers * self.n_heads + 1;
        self.grouped_canonical_floats() + inv_freq_len + contact_head_len
    }

    #[inline]
    pub fn sequential_hf_bytes(&self) -> u64 {
        (self.sequential_hf_floats() as u64) * 4
    }

    /// Autodetects ESM-2 model architecture from weights file length in bytes.
    /// Deterministically supports exact `grouped_canonical_v1` FP32 sizes (`30,091,652`,
    /// `134,067,972`, `592,642,692`, `2,604,339,332`, `11,356,354,692`), `sequential_hf` FP32 sizes,
    /// and Q8_0 (~1.125 bytes/param) representations across all 5 scales (8M, 35M, 150M, 650M, 3B).
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
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Unrecognized ESM-2 weights size: {} bytes. Expected 30,091,652 B (8M), 134,067,972 B (35M), 592,642,692 B (150M), 2,604,339,332 B (650M), or 11,356,354,692 B (3B).",
                    file_len
                ),
            ))
        }
    }

    /// Opens file on disk and autodetects ESM-2 configuration.
    /// Inspects CODQ header if present; otherwise falls back to file-length autodetection.
    pub fn detect_from_file(path: &str) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_len = metadata.len();
        if file_len >= 64 {
            let mut magic = [0u8; 4];
            if file.read_exact(&mut magic).is_ok() && magic == CODQ_MAGIC {
                file.seek(io::SeekFrom::Start(0))?;
                let header = CodqHeader::read_from(&mut file)?;
                if header.version != CODQ_VERSION {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Unsupported CODQ version: {}", header.version),
                    ));
                }
                return header.to_config();
            }
        }
        Self::detect_from_file_len(file_len)
    }
}

pub struct EsmWeights {
    pub token_embedding_table: Vec<f32>, pub ln_1_weight: Vec<f32>, pub ln_1_bias: Vec<f32>,
    pub wq: Vec<f32>, pub bq: Vec<f32>, pub wk: Vec<f32>, pub bk: Vec<f32>,
    pub wv: Vec<f32>, pub bv: Vec<f32>, pub wo: Vec<f32>, pub bo: Vec<f32>,
    pub ln_2_weight: Vec<f32>, pub ln_2_bias: Vec<f32>,
    pub fc1_weight: Vec<f32>, pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<f32>, pub fc2_bias: Vec<f32>,
    pub ln_f_weight: Vec<f32>, pub ln_f_bias: Vec<f32>, pub classifier_weights: Vec<f32>,
}

impl EsmWeights {
    pub fn from_file(model_path: &str, config: &EsmConfig) -> io::Result<Self> {
        let mut file = File::open(model_path)?;
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_ok() && magic == CODQ_MAGIC {
            file.rewind()?;
            let header = CodqHeader::read_from(&mut file)?;
            header.validate(config)?;
            if header.dtype != DTYPE_FP32 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Expected FP32 model in EsmWeights::from_file, found dtype={}", header.dtype),
                ));
            }
            file.seek(io::SeekFrom::Start(64))?;
        } else {
            file.rewind()?;
        }
        let head_dim = config.dim / config.n_heads;
        let inv_freq_len = head_dim / 2;
        let contact_head_len = config.n_layers * config.n_heads + 1;
        let lm_head_len = config.lm_head_floats();
        let grouped_canonical_floats = config.grouped_canonical_floats();
        let sequential_hf_floats = config.sequential_hf_floats();

        let start_pos = file.stream_position()?;
        let file_len = file.metadata()?.len();
        if start_pos > file_len {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Stream position exceeds file length"));
        }
        let remaining_bytes_u64 = file_len - start_pos;
        if remaining_bytes_u64 % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unaligned trailing bytes: {} is not a multiple of 4", remaining_bytes_u64),
            ));
        }
        let remaining_floats = (remaining_bytes_u64 / 4) as usize;

        let (
            token_embedding_table,
            ln_1_weight,
            ln_1_bias,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wo,
            bo,
            ln_2_weight,
            ln_2_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            ln_f_weight,
            ln_f_bias,
            classifier_weights,
        ) = if remaining_floats == grouped_canonical_floats {
            // Strict grouped_canonical_v1 layout
            let token_emb = read_floats(&mut file, config.vocab_size * config.dim)?;
            let ln_1_w = read_floats(&mut file, config.n_layers * config.dim)?;
            let ln_1_b = read_floats(&mut file, config.n_layers * config.dim)?;
            let w_q = read_floats(&mut file, config.n_layers * config.dim * config.dim)?;
            let b_q = read_floats(&mut file, config.n_layers * config.dim)?;
            let w_k = read_floats(&mut file, config.n_layers * config.dim * config.dim)?;
            let b_k = read_floats(&mut file, config.n_layers * config.dim)?;
            let w_v = read_floats(&mut file, config.n_layers * config.dim * config.dim)?;
            let b_v = read_floats(&mut file, config.n_layers * config.dim)?;
            let w_o = read_floats(&mut file, config.n_layers * config.dim * config.dim)?;
            let b_o = read_floats(&mut file, config.n_layers * config.dim)?;
            let ln_2_w = read_floats(&mut file, config.n_layers * config.dim)?;
            let ln_2_b = read_floats(&mut file, config.n_layers * config.dim)?;
            let fc1_w = read_floats(&mut file, config.n_layers * config.dim * config.hidden_dim)?;
            let fc1_b = read_floats(&mut file, config.n_layers * config.hidden_dim)?;
            let fc2_w = read_floats(&mut file, config.n_layers * config.hidden_dim * config.dim)?;
            let fc2_b = read_floats(&mut file, config.n_layers * config.dim)?;
            let ln_f_w = read_floats(&mut file, config.dim)?;
            let ln_f_b = read_floats(&mut file, config.dim)?;
            let classifier_w = read_floats(&mut file, lm_head_len)?;

            (
                token_emb, ln_1_w, ln_1_b, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
                ln_2_w, ln_2_b, fc1_w, fc1_b, fc2_w, fc2_b, ln_f_w, ln_f_b, classifier_w,
            )
        } else if remaining_floats == sequential_hf_floats {
            // Sequential HuggingFace / fair-esm dump layout
            let token_emb = read_floats(&mut file, config.vocab_size * config.dim)?;
            let _inv_freq = read_floats(&mut file, inv_freq_len)?;

            let mut ln_1_w = vec![0.0f32; config.n_layers * config.dim];
            let mut ln_1_b = vec![0.0f32; config.n_layers * config.dim];
            let mut w_q = vec![0.0f32; config.n_layers * config.dim * config.dim];
            let mut b_q = vec![0.0f32; config.n_layers * config.dim];
            let mut w_k = vec![0.0f32; config.n_layers * config.dim * config.dim];
            let mut b_k = vec![0.0f32; config.n_layers * config.dim];
            let mut w_v = vec![0.0f32; config.n_layers * config.dim * config.dim];
            let mut b_v = vec![0.0f32; config.n_layers * config.dim];
            let mut w_o = vec![0.0f32; config.n_layers * config.dim * config.dim];
            let mut b_o = vec![0.0f32; config.n_layers * config.dim];
            let mut ln_2_w = vec![0.0f32; config.n_layers * config.dim];
            let mut ln_2_b = vec![0.0f32; config.n_layers * config.dim];
            let mut fc1_w = vec![0.0f32; config.n_layers * config.dim * config.hidden_dim];
            let mut fc1_b = vec![0.0f32; config.n_layers * config.hidden_dim];
            let mut fc2_w = vec![0.0f32; config.n_layers * config.hidden_dim * config.dim];
            let mut fc2_b = vec![0.0f32; config.n_layers * config.dim];

            for l in 0..config.n_layers {
                let dim = config.dim;
                let hidden_dim = config.hidden_dim;
                let l_matrix_off = l * dim * dim;
                let l_dim_off = l * dim;
                let l_fc1_off = l * dim * hidden_dim;
                let l_fc1_b_off = l * hidden_dim;
                let l_fc2_off = l * hidden_dim * dim;

                let layer_wq = read_floats(&mut file, dim * dim)?;
                w_q[l_matrix_off..l_matrix_off + dim * dim].copy_from_slice(&layer_wq);

                let layer_bq = read_floats(&mut file, dim)?;
                b_q[l_dim_off..l_dim_off + dim].copy_from_slice(&layer_bq);

                let layer_wk = read_floats(&mut file, dim * dim)?;
                w_k[l_matrix_off..l_matrix_off + dim * dim].copy_from_slice(&layer_wk);

                let layer_bk = read_floats(&mut file, dim)?;
                b_k[l_dim_off..l_dim_off + dim].copy_from_slice(&layer_bk);

                let layer_wv = read_floats(&mut file, dim * dim)?;
                w_v[l_matrix_off..l_matrix_off + dim * dim].copy_from_slice(&layer_wv);

                let layer_bv = read_floats(&mut file, dim)?;
                b_v[l_dim_off..l_dim_off + dim].copy_from_slice(&layer_bv);

                let layer_wo = read_floats(&mut file, dim * dim)?;
                w_o[l_matrix_off..l_matrix_off + dim * dim].copy_from_slice(&layer_wo);

                let layer_bo = read_floats(&mut file, dim)?;
                b_o[l_dim_off..l_dim_off + dim].copy_from_slice(&layer_bo);

                let layer_ln1_w = read_floats(&mut file, dim)?;
                ln_1_w[l_dim_off..l_dim_off + dim].copy_from_slice(&layer_ln1_w);

                let layer_ln1_b = read_floats(&mut file, dim)?;
                ln_1_b[l_dim_off..l_dim_off + dim].copy_from_slice(&layer_ln1_b);

                let layer_fc1_w = read_floats(&mut file, hidden_dim * dim)?;
                fc1_w[l_fc1_off..l_fc1_off + hidden_dim * dim].copy_from_slice(&layer_fc1_w);

                let layer_fc1_b = read_floats(&mut file, hidden_dim)?;
                fc1_b[l_fc1_b_off..l_fc1_b_off + hidden_dim].copy_from_slice(&layer_fc1_b);

                let layer_fc2_w = read_floats(&mut file, dim * hidden_dim)?;
                fc2_w[l_fc2_off..l_fc2_off + dim * hidden_dim].copy_from_slice(&layer_fc2_w);

                let layer_fc2_b = read_floats(&mut file, dim)?;
                fc2_b[l_dim_off..l_dim_off + dim].copy_from_slice(&layer_fc2_b);

                let layer_ln2_w = read_floats(&mut file, dim)?;
                ln_2_w[l_dim_off..l_dim_off + dim].copy_from_slice(&layer_ln2_w);

                let layer_ln2_b = read_floats(&mut file, dim)?;
                ln_2_b[l_dim_off..l_dim_off + dim].copy_from_slice(&layer_ln2_b);
            }

            let ln_f_w = read_floats(&mut file, config.dim)?;
            let ln_f_b = read_floats(&mut file, config.dim)?;

            let _contact_head = read_floats(&mut file, contact_head_len)?;

            let lm_bias = read_floats(&mut file, config.vocab_size)?;
            let lm_dense_w = read_floats(&mut file, config.dim * config.dim)?;
            let lm_dense_b = read_floats(&mut file, config.dim)?;
            let lm_ln_w = read_floats(&mut file, config.dim)?;
            let lm_ln_b = read_floats(&mut file, config.dim)?;
            let lm_dec_w = read_floats(&mut file, config.vocab_size * config.dim)?;

            let mut classifier_w = Vec::with_capacity(lm_head_len);
            classifier_w.extend_from_slice(&lm_ln_w);
            classifier_w.extend_from_slice(&lm_ln_b);
            classifier_w.extend_from_slice(&lm_dense_w);
            classifier_w.extend_from_slice(&lm_dense_b);
            classifier_w.extend_from_slice(&lm_dec_w);
            classifier_w.extend_from_slice(&lm_bias);

            (
                token_emb, ln_1_w, ln_1_b, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
                ln_2_w, ln_2_b, fc1_w, fc1_b, fc2_w, fc2_b, ln_f_w, ln_f_b, classifier_w,
            )
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid FP32 weights size for {} ('{}'): found {} bytes ({} floats), expected exactly {} bytes ({} floats, grouped_canonical_v1) or {} bytes ({} floats, sequential_hf).",
                    config.model_name(),
                    model_path,
                    remaining_bytes_u64,
                    remaining_floats,
                    config.grouped_canonical_bytes(),
                    grouped_canonical_floats,
                    config.sequential_hf_bytes(),
                    sequential_hf_floats,
                ),
            ));
        };

        let final_pos = file.stream_position()?;
        if final_pos != file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Weights file underread: read {} of {} bytes", final_pos, file_len),
            ));
        }

        Ok(Self {
            token_embedding_table,
            ln_1_weight,
            ln_1_bias,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wo,
            bo,
            ln_2_weight,
            ln_2_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            ln_f_weight,
            ln_f_bias,
            classifier_weights,
        })
    }

    /// Loads and initializes ESM-2 configuration and weights from file path with automatic architecture detection.
    pub fn load_from_file(model_path: &str) -> io::Result<(EsmConfig, Self)> {
        let config = match EsmConfig::detect_from_file(model_path) {
            Ok(cfg) => cfg,
            Err(_) => EsmConfig::esm2_8m(),
        };
        let weights = Self::from_file(model_path, &config)?;
        Ok((config, weights))
    }

    pub fn quantize_q8(&self) -> EsmWeightsQ8 {
        EsmWeightsQ8 {
            token_embedding_table: quantize_slice_to_q8(&self.token_embedding_table),
            ln_1_weight: self.ln_1_weight.clone(),
            ln_1_bias: self.ln_1_bias.clone(),
            wq: quantize_slice_to_q8(&self.wq),
            bq: self.bq.clone(),
            wk: quantize_slice_to_q8(&self.wk),
            bk: self.bk.clone(),
            wv: quantize_slice_to_q8(&self.wv),
            bv: self.bv.clone(),
            wo: quantize_slice_to_q8(&self.wo),
            bo: self.bo.clone(),
            ln_2_weight: self.ln_2_weight.clone(),
            ln_2_bias: self.ln_2_bias.clone(),
            fc1_weight: quantize_slice_to_q8(&self.fc1_weight),
            fc1_bias: self.fc1_bias.clone(),
            fc2_weight: quantize_slice_to_q8(&self.fc2_weight),
            fc2_bias: self.fc2_bias.clone(),
            ln_f_weight: self.ln_f_weight.clone(),
            ln_f_bias: self.ln_f_bias.clone(),
            classifier_weights: self.classifier_weights.clone(),
        }
    }

    pub fn quantize_q4(&self) -> EsmWeightsQ4 {
        EsmWeightsQ4 {
            token_embedding_table: self.token_embedding_table.clone(),
            ln_1_weight: self.ln_1_weight.clone(),
            ln_1_bias: self.ln_1_bias.clone(),
            wq: quantize_slice_to_q4(&self.wq),
            bq: self.bq.clone(),
            wk: quantize_slice_to_q4(&self.wk),
            bk: self.bk.clone(),
            wv: quantize_slice_to_q4(&self.wv),
            bv: self.bv.clone(),
            wo: quantize_slice_to_q4(&self.wo),
            bo: self.bo.clone(),
            ln_2_weight: self.ln_2_weight.clone(),
            ln_2_bias: self.ln_2_bias.clone(),
            fc1_weight: quantize_slice_to_q4(&self.fc1_weight),
            fc1_bias: self.fc1_bias.clone(),
            fc2_weight: quantize_slice_to_q4(&self.fc2_weight),
            fc2_bias: self.fc2_bias.clone(),
            ln_f_weight: self.ln_f_weight.clone(),
            ln_f_bias: self.ln_f_bias.clone(),
            classifier_weights: self.classifier_weights.clone(),
        }
    }
}

/// Loads and initializes ESM-2 configuration and weights from file path with automatic architecture detection.
pub fn load_model_from_file(model_path: &str) -> io::Result<(EsmConfig, EsmWeights)> {
    EsmWeights::load_from_file(model_path)
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: f32,          // Delta (escala)
    pub qs: [i8; 32],    // 32 cuantizaciones en int8
}

impl BlockQ8_0 {
    pub fn quantize(src: &[f32]) -> Self {
        let mut max_val = 0.0f32;
        for &v in src {
            if v.abs() > max_val { max_val = v.abs(); }
        }
        let d = max_val / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = (src[i] * id).round().clamp(-127.0, 127.0) as i8;
        }
        Self { d, qs }
    }

    #[inline]
    pub fn dequantize_to(&self, dst: &mut [f32]) {
        for i in 0..32 {
            dst[i] = self.qs[i] as f32 * self.d;
        }
    }
}

pub fn quantize_slice_to_q8(src: &[f32]) -> Vec<BlockQ8_0> {
    assert!(src.len() % 32 == 0, "Source length must be a multiple of 32");
    src.chunks_exact(32).map(BlockQ8_0::quantize).collect()
}

pub struct EsmWeightsQ8 {
    pub token_embedding_table: Vec<BlockQ8_0>,
    pub ln_1_weight: Vec<f32>, pub ln_1_bias: Vec<f32>,
    pub wq: Vec<BlockQ8_0>, pub bq: Vec<f32>,
    pub wk: Vec<BlockQ8_0>, pub bk: Vec<f32>,
    pub wv: Vec<BlockQ8_0>, pub bv: Vec<f32>,
    pub wo: Vec<BlockQ8_0>, pub bo: Vec<f32>,
    pub ln_2_weight: Vec<f32>, pub ln_2_bias: Vec<f32>,
    pub fc1_weight: Vec<BlockQ8_0>, pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<BlockQ8_0>, pub fc2_bias: Vec<f32>,
    pub ln_f_weight: Vec<f32>, pub ln_f_bias: Vec<f32>,
    pub classifier_weights: Vec<f32>,
}

impl EsmWeightsQ8 {
    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        let mut file = File::create(path)?;
        write_q8_blocks(&mut file, &self.token_embedding_table)?;
        write_floats(&mut file, &self.ln_1_weight)?;
        write_floats(&mut file, &self.ln_1_bias)?;
        write_q8_blocks(&mut file, &self.wq)?;
        write_floats(&mut file, &self.bq)?;
        write_q8_blocks(&mut file, &self.wk)?;
        write_floats(&mut file, &self.bk)?;
        write_q8_blocks(&mut file, &self.wv)?;
        write_floats(&mut file, &self.bv)?;
        write_q8_blocks(&mut file, &self.wo)?;
        write_floats(&mut file, &self.bo)?;
        write_floats(&mut file, &self.ln_2_weight)?;
        write_floats(&mut file, &self.ln_2_bias)?;
        write_q8_blocks(&mut file, &self.fc1_weight)?;
        write_floats(&mut file, &self.fc1_bias)?;
        write_q8_blocks(&mut file, &self.fc2_weight)?;
        write_floats(&mut file, &self.fc2_bias)?;
        write_floats(&mut file, &self.ln_f_weight)?;
        write_floats(&mut file, &self.ln_f_bias)?;
        write_floats(&mut file, &self.classifier_weights)?;
        file.flush()?;
        Ok(())
    }

    pub fn save_to_file_with_header(&self, path: &str, config: &EsmConfig) -> io::Result<()> {
        let mut file = File::create(path)?;
        let header = CodqHeader::new(DTYPE_Q8_0, config);
        header.write_to(&mut file)?;
        write_q8_blocks(&mut file, &self.token_embedding_table)?;
        write_floats(&mut file, &self.ln_1_weight)?;
        write_floats(&mut file, &self.ln_1_bias)?;
        write_q8_blocks(&mut file, &self.wq)?;
        write_floats(&mut file, &self.bq)?;
        write_q8_blocks(&mut file, &self.wk)?;
        write_floats(&mut file, &self.bk)?;
        write_q8_blocks(&mut file, &self.wv)?;
        write_floats(&mut file, &self.bv)?;
        write_q8_blocks(&mut file, &self.wo)?;
        write_floats(&mut file, &self.bo)?;
        write_floats(&mut file, &self.ln_2_weight)?;
        write_floats(&mut file, &self.ln_2_bias)?;
        write_q8_blocks(&mut file, &self.fc1_weight)?;
        write_floats(&mut file, &self.fc1_bias)?;
        write_q8_blocks(&mut file, &self.fc2_weight)?;
        write_floats(&mut file, &self.fc2_bias)?;
        write_floats(&mut file, &self.ln_f_weight)?;
        write_floats(&mut file, &self.ln_f_bias)?;
        write_floats(&mut file, &self.classifier_weights)?;
        file.flush()?;
        Ok(())
    }

    pub fn from_file(path: &str, config: &EsmConfig) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len >= 64 {
            let mut magic = [0u8; 4];
            if file.read_exact(&mut magic).is_ok() && magic == CODQ_MAGIC {
                file.seek(io::SeekFrom::Start(0))?;
                let header = CodqHeader::read_from(&mut file)?;
                if header.dtype != DTYPE_Q8_0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Expected Q8_0 dtype ({}), got {}", DTYPE_Q8_0, header.dtype),
                    ));
                }
                header.validate(config)?;
            } else {
                file.seek(io::SeekFrom::Start(0))?;
            }
        } else {
            file.seek(io::SeekFrom::Start(0))?;
        }
        let token_embedding_table = read_q8_blocks(&mut file, config.vocab_size * (config.dim / 32))?;
        let ln_1_weight = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_1_bias = read_floats(&mut file, config.n_layers * config.dim)?;
        let wq = read_q8_blocks(&mut file, config.n_layers * config.dim * (config.dim / 32))?;
        let bq = read_floats(&mut file, config.n_layers * config.dim)?;
        let wk = read_q8_blocks(&mut file, config.n_layers * config.dim * (config.dim / 32))?;
        let bk = read_floats(&mut file, config.n_layers * config.dim)?;
        let wv = read_q8_blocks(&mut file, config.n_layers * config.dim * (config.dim / 32))?;
        let bv = read_floats(&mut file, config.n_layers * config.dim)?;
        let wo = read_q8_blocks(&mut file, config.n_layers * config.dim * (config.dim / 32))?;
        let bo = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_2_weight = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_2_bias = read_floats(&mut file, config.n_layers * config.dim)?;
        let fc1_weight = read_q8_blocks(&mut file, config.n_layers * config.hidden_dim * (config.dim / 32))?;
        let fc1_bias = read_floats(&mut file, config.n_layers * config.hidden_dim)?;
        let fc2_weight = read_q8_blocks(&mut file, config.n_layers * config.dim * (config.hidden_dim / 32))?;
        let fc2_bias = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_f_weight = read_floats(&mut file, config.dim)?;
        let ln_f_bias = read_floats(&mut file, config.dim)?;

        let current_pos = file.stream_position()?;
        let file_len = file.metadata()?.len();
        if current_pos > file_len {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Stream position exceeds file length"));
        }
        let remaining_bytes = (file_len - current_pos) as usize;
        if remaining_bytes % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unaligned trailing bytes in Q8: {} is not a multiple of 4", remaining_bytes),
            ));
        }
        let remaining_floats = remaining_bytes / 4;
        let classifier_weights = read_floats(&mut file, remaining_floats)?;

        let final_pos = file.stream_position()?;
        if final_pos != file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Q8 Weights file underread: read {} of {} bytes", final_pos, file_len),
            ));
        }

        Ok(Self {
            token_embedding_table,
            ln_1_weight,
            ln_1_bias,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wo,
            bo,
            ln_2_weight,
            ln_2_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            ln_f_weight,
            ln_f_bias,
            classifier_weights,
        })
    }

    pub fn load_from_file(model_path: &str) -> io::Result<(EsmConfig, Self)> {
        let config = match EsmConfig::detect_from_file(model_path) {
            Ok(cfg) => cfg,
            Err(_) => EsmConfig::esm2_8m(),
        };
        let weights = Self::from_file(model_path, &config)?;
        Ok((config, weights))
    }

    pub fn load_or_convert(model_path: &str, config: &EsmConfig) -> io::Result<Self> {
        let is_q8 = if let Ok(mut f) = File::open(model_path) {
            let mut magic = [0u8; 4];
            if f.read_exact(&mut magic).is_ok() && magic == CODQ_MAGIC {
                let _ = f.rewind();
                if let Ok(hdr) = CodqHeader::read_from(&mut f) {
                    hdr.dtype == DTYPE_Q8_0
                } else {
                    false
                }
            } else {
                model_path.ends_with(".q8") || model_path.ends_with(".q8.bin") || model_path.contains("q8")
            }
        } else {
            model_path.ends_with(".q8") || model_path.ends_with(".q8.bin") || model_path.contains("q8")
        };

        if is_q8 {
            Self::from_file(model_path, config)
        } else {
            let fp32 = EsmWeights::from_file(model_path, config)?;
            Ok(fp32.quantize_q8())
        }
    }
}

// ============================================================================
// CODON QUANTIZED BINARY HEADER (CODQ) SPECIFICATION (v1.6.0)
// ============================================================================

pub const CODQ_MAGIC: [u8; 4] = *b"CODQ";
pub const CODQ_VERSION: u32 = 1;

pub const DTYPE_FP32: u32 = 0;
pub const DTYPE_Q8_0: u32 = 1;
pub const DTYPE_Q4_0: u32 = 2;

pub const ARCH_8M: u32 = 0;
pub const ARCH_35M: u32 = 1;
pub const ARCH_150M: u32 = 2;
pub const ARCH_650M: u32 = 3;
pub const ARCH_3B: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodqHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub dtype: u32,
    pub arch: u32,
    pub n_layers: u32,
    pub dim: u32,
    pub hidden_dim: u32,
    pub vocab_size: u32,
    pub reserved: [u8; 32],
}

const _: () = assert!(core::mem::size_of::<CodqHeader>() == 64);
const _: () = assert!(core::mem::align_of::<CodqHeader>() == 4);

impl CodqHeader {
    pub fn new(dtype: u32, config: &EsmConfig) -> Self {
        let arch = match config.arch {
            EsmModelArch::Esm2_8M => ARCH_8M,
            EsmModelArch::Esm2_35M => ARCH_35M,
            EsmModelArch::Esm2_150M => ARCH_150M,
            EsmModelArch::Esm2_650M => ARCH_650M,
            EsmModelArch::Esm2_3B => ARCH_3B,
        };
        Self {
            magic: CODQ_MAGIC,
            version: CODQ_VERSION,
            dtype,
            arch,
            n_layers: config.n_layers as u32,
            dim: config.dim as u32,
            hidden_dim: config.hidden_dim as u32,
            vocab_size: config.vocab_size as u32,
            reserved: [0u8; 32],
        }
    }

    pub fn write_to(&self, file: &mut File) -> io::Result<()> {
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        };
        file.write_all(byte_slice)
    }

    pub fn read_from(file: &mut File) -> io::Result<Self> {
        let mut header = std::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            let byte_slice = std::slice::from_raw_parts_mut(
                header.as_mut_ptr() as *mut u8,
                std::mem::size_of::<Self>(),
            );
            file.read_exact(byte_slice)?;
            let header = header.assume_init();
            if header.magic != CODQ_MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid CODQ magic: {:?}", header.magic),
                ));
            }
            Ok(header)
        }
    }

    pub fn validate(&self, config: &EsmConfig) -> io::Result<()> {
        if self.magic != CODQ_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid CODQ magic: {:?}", self.magic),
            ));
        }
        if self.version != CODQ_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported CODQ version: {}, expected {}", self.version, CODQ_VERSION),
            ));
        }
        let expected_arch = match config.arch {
            EsmModelArch::Esm2_8M => ARCH_8M,
            EsmModelArch::Esm2_35M => ARCH_35M,
            EsmModelArch::Esm2_150M => ARCH_150M,
            EsmModelArch::Esm2_650M => ARCH_650M,
            EsmModelArch::Esm2_3B => ARCH_3B,
        };
        if self.arch != expected_arch {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("CODQ arch mismatch: header has {}, expected {}", self.arch, expected_arch),
            ));
        }
        if self.n_layers as usize != config.n_layers
            || self.dim as usize != config.dim
            || self.hidden_dim as usize != config.hidden_dim
            || self.vocab_size as usize != config.vocab_size
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "CODQ dimension mismatch: header (layers={}, dim={}, hidden={}, vocab={}) vs config (layers={}, dim={}, hidden={}, vocab={})",
                    self.n_layers, self.dim, self.hidden_dim, self.vocab_size,
                    config.n_layers, config.dim, config.hidden_dim, config.vocab_size
                ),
            ));
        }
        Ok(())
    }

    pub fn to_config(&self) -> io::Result<EsmConfig> {
        if self.magic != CODQ_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid CODQ magic: {:?}", self.magic),
            ));
        }
        if self.version != CODQ_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported CODQ version: {}, expected {}", self.version, CODQ_VERSION),
            ));
        }
        let cfg = match self.arch {
            ARCH_8M => EsmConfig::esm2_8m(),
            ARCH_35M => EsmConfig::esm2_35m(),
            ARCH_150M => EsmConfig::esm2_150m(),
            ARCH_650M => EsmConfig::esm2_650m(),
            ARCH_3B => EsmConfig::esm2_3b(),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown architecture ID in CODQ header: {}", self.arch),
                ));
            }
        };
        if self.n_layers as usize != cfg.n_layers
            || self.dim as usize != cfg.dim
            || self.hidden_dim as usize != cfg.hidden_dim
            || self.vocab_size as usize != cfg.vocab_size
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Incoherent dimensions in CODQ header for arch {}: layers={}, dim={}, hidden={}, vocab={}",
                    self.arch, self.n_layers, self.dim, self.hidden_dim, self.vocab_size
                ),
            ));
        }
        Ok(cfg)
    }
}

// ============================================================================
// BLOCK Q4_0 DATA STRUCTURE & QUANTIZATION SKELETON
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0 {
    pub d: f32,          // Delta (escala float32)
    pub qs: [u8; 16],    // 16 bytes = 32 valores cuantizados a 4 bits
}

const _: () = assert!(core::mem::size_of::<BlockQ4_0>() == 20);
const _: () = assert!(core::mem::align_of::<BlockQ4_0>() == 4);

impl BlockQ4_0 {
    pub fn quantize(src: &[f32]) -> Self {
        assert!(src.len() >= 32, "Source length must be at least 32 elements");
        let mut max = 0.0f32;
        for &v in &src[0..32] {
            if v.abs() > max.abs() {
                max = v;
            }
        }
        let d = if max == 0.0 { 0.0 } else { max / -8.0 };
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let q_lo = ((src[j] * id).round() as i32 + 8).clamp(0, 15) as u8;
            let q_hi = ((src[j + 16] * id).round() as i32 + 8).clamp(0, 15) as u8;
            qs[j] = (q_lo & 0x0F) | ((q_hi & 0x0F) << 4);
        }
        Self { d, qs }
    }

    #[inline]
    pub fn dequantize_to(&self, dst: &mut [f32]) {
        assert!(dst.len() >= 32, "Destination length must be at least 32 elements");
        for j in 0..16 {
            let q_lo = (self.qs[j] & 0x0F) as i32 - 8;
            dst[j] = q_lo as f32 * self.d;
            let q_hi = ((self.qs[j] >> 4) & 0x0F) as i32 - 8;
            dst[j + 16] = q_hi as f32 * self.d;
        }
    }
}

pub fn quantize_slice_to_q4(src: &[f32]) -> Vec<BlockQ4_0> {
    assert!(src.len() % 32 == 0, "Source length must be a multiple of 32");
    src.chunks_exact(32).map(BlockQ4_0::quantize).collect()
}

pub fn read_q4_blocks(file: &mut File, num_blocks: usize) -> io::Result<Vec<BlockQ4_0>> {
    let mut blocks = vec![BlockQ4_0 { d: 0.0, qs: [0; 16] }; num_blocks];
    unsafe {
        let byte_slice = std::slice::from_raw_parts_mut(
            blocks.as_mut_ptr() as *mut u8,
            num_blocks * std::mem::size_of::<BlockQ4_0>(),
        );
        file.read_exact(byte_slice)?;
    }
    Ok(blocks)
}

#[allow(clippy::manual_slice_size_calculation)]
pub fn write_q4_blocks(file: &mut File, blocks: &[BlockQ4_0]) -> io::Result<()> {
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            blocks.len() * std::mem::size_of::<BlockQ4_0>(),
        )
    };
    file.write_all(byte_slice)
}

pub struct EsmWeightsQ4 {
    pub token_embedding_table: Vec<f32>,
    pub ln_1_weight: Vec<f32>, pub ln_1_bias: Vec<f32>,
    pub wq: Vec<BlockQ4_0>, pub bq: Vec<f32>,
    pub wk: Vec<BlockQ4_0>, pub bk: Vec<f32>,
    pub wv: Vec<BlockQ4_0>, pub bv: Vec<f32>,
    pub wo: Vec<BlockQ4_0>, pub bo: Vec<f32>,
    pub ln_2_weight: Vec<f32>, pub ln_2_bias: Vec<f32>,
    pub fc1_weight: Vec<BlockQ4_0>, pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<BlockQ4_0>, pub fc2_bias: Vec<f32>,
    pub ln_f_weight: Vec<f32>, pub ln_f_bias: Vec<f32>,
    pub classifier_weights: Vec<f32>,
}

impl EsmWeightsQ4 {
    pub fn save_to_file(&self, path: &str, config: &EsmConfig) -> io::Result<()> {
        let mut file = File::create(path)?;
        let header = CodqHeader::new(DTYPE_Q4_0, config);
        header.write_to(&mut file)?;
        write_floats(&mut file, &self.token_embedding_table)?;
        write_floats(&mut file, &self.ln_1_weight)?;
        write_floats(&mut file, &self.ln_1_bias)?;
        write_q4_blocks(&mut file, &self.wq)?;
        write_floats(&mut file, &self.bq)?;
        write_q4_blocks(&mut file, &self.wk)?;
        write_floats(&mut file, &self.bk)?;
        write_q4_blocks(&mut file, &self.wv)?;
        write_floats(&mut file, &self.bv)?;
        write_q4_blocks(&mut file, &self.wo)?;
        write_floats(&mut file, &self.bo)?;
        write_floats(&mut file, &self.ln_2_weight)?;
        write_floats(&mut file, &self.ln_2_bias)?;
        write_q4_blocks(&mut file, &self.fc1_weight)?;
        write_floats(&mut file, &self.fc1_bias)?;
        write_q4_blocks(&mut file, &self.fc2_weight)?;
        write_floats(&mut file, &self.fc2_bias)?;
        write_floats(&mut file, &self.ln_f_weight)?;
        write_floats(&mut file, &self.ln_f_bias)?;
        write_floats(&mut file, &self.classifier_weights)?;
        file.flush()?;
        Ok(())
    }

    pub fn from_file(path: &str, config: &EsmConfig) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len >= 64 {
            let mut magic = [0u8; 4];
            if file.read_exact(&mut magic).is_ok() && magic == CODQ_MAGIC {
                file.seek(io::SeekFrom::Start(0))?;
                let header = CodqHeader::read_from(&mut file)?;
                if header.dtype != DTYPE_Q4_0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Expected Q4_0 dtype ({}), got {}", DTYPE_Q4_0, header.dtype),
                    ));
                }
                header.validate(config)?;
            } else {
                file.seek(io::SeekFrom::Start(0))?;
            }
        } else {
            file.seek(io::SeekFrom::Start(0))?;
        }

        let token_embedding_table = read_floats(&mut file, config.vocab_size * config.dim)?;
        let ln_1_weight = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_1_bias = read_floats(&mut file, config.n_layers * config.dim)?;
        let wq = read_q4_blocks(&mut file, config.n_layers * config.dim * (config.dim / 32))?;
        let bq = read_floats(&mut file, config.n_layers * config.dim)?;
        let wk = read_q4_blocks(&mut file, config.n_layers * config.dim * (config.dim / 32))?;
        let bk = read_floats(&mut file, config.n_layers * config.dim)?;
        let wv = read_q4_blocks(&mut file, config.n_layers * config.dim * (config.dim / 32))?;
        let bv = read_floats(&mut file, config.n_layers * config.dim)?;
        let wo = read_q4_blocks(&mut file, config.n_layers * config.dim * (config.dim / 32))?;
        let bo = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_2_weight = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_2_bias = read_floats(&mut file, config.n_layers * config.dim)?;
        let fc1_weight = read_q4_blocks(&mut file, config.n_layers * config.hidden_dim * (config.dim / 32))?;
        let fc1_bias = read_floats(&mut file, config.n_layers * config.hidden_dim)?;
        let fc2_weight = read_q4_blocks(&mut file, config.n_layers * config.dim * (config.hidden_dim / 32))?;
        let fc2_bias = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_f_weight = read_floats(&mut file, config.dim)?;
        let ln_f_bias = read_floats(&mut file, config.dim)?;

        let current_pos = file.stream_position()?;
        if current_pos > file_len {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Stream position exceeds file length"));
        }
        let remaining_bytes = (file_len - current_pos) as usize;
        if remaining_bytes % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unaligned trailing bytes in Q4: {} is not a multiple of 4", remaining_bytes),
            ));
        }
        let remaining_floats = remaining_bytes / 4;
        let classifier_weights = read_floats(&mut file, remaining_floats)?;

        let final_pos = file.stream_position()?;
        if final_pos != file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Q4 Weights file underread: read {} of {} bytes", final_pos, file_len),
            ));
        }

        Ok(Self {
            token_embedding_table,
            ln_1_weight,
            ln_1_bias,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wo,
            bo,
            ln_2_weight,
            ln_2_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            ln_f_weight,
            ln_f_bias,
            classifier_weights,
        })
    }

    pub fn load_from_file(model_path: &str) -> io::Result<(EsmConfig, Self)> {
        let config = match EsmConfig::detect_from_file(model_path) {
            Ok(cfg) => cfg,
            Err(_) => EsmConfig::esm2_8m(),
        };
        let weights = Self::from_file(model_path, &config)?;
        Ok((config, weights))
    }

    pub fn load_or_convert(model_path: &str, config: &EsmConfig) -> io::Result<Self> {
        let is_q4 = if let Ok(mut f) = File::open(model_path) {
            let mut magic = [0u8; 4];
            if f.read_exact(&mut magic).is_ok() && magic == CODQ_MAGIC {
                let _ = f.rewind();
                if let Ok(hdr) = CodqHeader::read_from(&mut f) {
                    hdr.dtype == DTYPE_Q4_0
                } else {
                    false
                }
            } else {
                model_path.ends_with(".q4")
                    || model_path.ends_with(".q4.bin")
                    || model_path.ends_with(".q4_0")
                    || model_path.ends_with(".q4_0.bin")
                    || model_path.contains("q4")
            }
        } else {
            model_path.ends_with(".q4")
                || model_path.ends_with(".q4.bin")
                || model_path.ends_with(".q4_0")
                || model_path.ends_with(".q4_0.bin")
                || model_path.contains("q4")
        };

        if is_q4 {
            Self::from_file(model_path, config)
        } else {
            let fp32 = EsmWeights::from_file(model_path, config)?;
            Ok(fp32.quantize_q4())
        }
    }
}

fn stream_copy_floats<R: Read, W: Write>(reader: &mut R, writer: &mut W, num_floats: usize) -> io::Result<()> {
    let mut rem_bytes = (num_floats as u64) * 4;
    let mut buf = [0u8; 65536];
    while rem_bytes > 0 {
        let chunk = (rem_bytes.min(buf.len() as u64)) as usize;
        reader.read_exact(&mut buf[..chunk])?;
        writer.write_all(&buf[..chunk])?;
        rem_bytes -= chunk as u64;
    }
    Ok(())
}

/// Streams a `grouped_canonical_v1` FP32 weights file and writes a `CODQ` `Q8_0` binary
/// block-by-block using `BufReader` and `BufWriter` (< 10 MB RAM, safe for 3B on 11 GiB machines).
pub fn quantize_file_streaming_q8(src_path: &str, dst_path: &str, config: &EsmConfig) -> io::Result<u64> {
    let mut src_file = File::open(src_path)?;
    let file_len = src_file.metadata()?.len();
    let mut magic = [0u8; 4];
    let start_pos = if file_len >= 64 && src_file.read_exact(&mut magic).is_ok() && magic == CODQ_MAGIC {
        src_file.seek(io::SeekFrom::Start(0))?;
        let hdr = CodqHeader::read_from(&mut src_file)?;
        hdr.validate(config)?;
        if hdr.dtype != DTYPE_FP32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Expected FP32 source in quantize_file_streaming_q8, found dtype={}", hdr.dtype),
            ));
        }
        64u64
    } else {
        src_file.seek(io::SeekFrom::Start(0))?;
        0u64
    };

    let rem_bytes = file_len.saturating_sub(start_pos);
    let expected_bytes = config.grouped_canonical_bytes();
    if rem_bytes != expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "quantize_file_streaming_q8 requires grouped_canonical_v1 layout ({} bytes) for {}, found {} bytes in '{}'",
                expected_bytes,
                config.model_name(),
                rem_bytes,
                src_path
            ),
        ));
    }

    let mut reader = io::BufReader::with_capacity(4 * 1024 * 1024, src_file);
    let dst_file = File::create(dst_path)?;
    let mut writer = io::BufWriter::with_capacity(4 * 1024 * 1024, dst_file);

    let header = CodqHeader::new(DTYPE_Q8_0, config);
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const CodqHeader as *const u8,
            std::mem::size_of::<CodqHeader>(),
        )
    };
    writer.write_all(header_bytes)?;

    let schedule: [(usize, bool); 20] = [
        (config.vocab_size * config.dim, true),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim * config.dim, true),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim * config.dim, true),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim * config.dim, true),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim * config.dim, true),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim * config.hidden_dim, true),
        (config.n_layers * config.hidden_dim, false),
        (config.n_layers * config.hidden_dim * config.dim, true),
        (config.n_layers * config.dim, false),
        (config.dim, false),
        (config.dim, false),
        (config.lm_head_floats(), false),
    ];

    const BATCH_BLOCKS: usize = 2048;
    let mut in_buf = vec![0u8; BATCH_BLOCKS * 32 * 4];
    let mut out_buf = vec![0u8; BATCH_BLOCKS * 36];

    for (num_floats, quantize) in schedule {
        if !quantize {
            stream_copy_floats(&mut reader, &mut writer, num_floats)?;
        } else {
            if num_floats % 32 != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Quantized tensor length {} is not a multiple of 32", num_floats),
                ));
            }
            let mut rem_blocks = num_floats / 32;
            while rem_blocks > 0 {
                let batch = rem_blocks.min(BATCH_BLOCKS);
                let in_len = batch * 32 * 4;
                reader.read_exact(&mut in_buf[..in_len])?;
                for b in 0..batch {
                    let mut vals = [0.0f32; 32];
                    let b_in = &in_buf[b * 128..(b + 1) * 128];
                    for i in 0..32 {
                        vals[i] = f32::from_le_bytes([
                            b_in[i * 4],
                            b_in[i * 4 + 1],
                            b_in[i * 4 + 2],
                            b_in[i * 4 + 3],
                        ]);
                    }
                    let blk = BlockQ8_0::quantize(&vals);
                    let b_out = &mut out_buf[b * 36..(b + 1) * 36];
                    b_out[0..4].copy_from_slice(&blk.d.to_le_bytes());
                    for i in 0..32 {
                        b_out[4 + i] = blk.qs[i] as u8;
                    }
                }
                writer.write_all(&out_buf[..batch * 36])?;
                rem_blocks -= batch;
            }
        }
    }

    writer.flush()?;
    let out_file = writer.into_inner().map_err(|e| e.into_error())?;
    out_file.metadata().map(|m| m.len())
}

/// Streams a `grouped_canonical_v1` FP32 weights file and writes a `CODQ` `Q4_0` binary
/// block-by-block using `BufReader` and `BufWriter` (< 10 MB RAM, safe for 3B on 11 GiB machines).
pub fn quantize_file_streaming_q4(src_path: &str, dst_path: &str, config: &EsmConfig) -> io::Result<u64> {
    let mut src_file = File::open(src_path)?;
    let file_len = src_file.metadata()?.len();
    let mut magic = [0u8; 4];
    let start_pos = if file_len >= 64 && src_file.read_exact(&mut magic).is_ok() && magic == CODQ_MAGIC {
        src_file.seek(io::SeekFrom::Start(0))?;
        let hdr = CodqHeader::read_from(&mut src_file)?;
        hdr.validate(config)?;
        if hdr.dtype != DTYPE_FP32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Expected FP32 source in quantize_file_streaming_q4, found dtype={}", hdr.dtype),
            ));
        }
        64u64
    } else {
        src_file.seek(io::SeekFrom::Start(0))?;
        0u64
    };

    let rem_bytes = file_len.saturating_sub(start_pos);
    let expected_bytes = config.grouped_canonical_bytes();
    if rem_bytes != expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "quantize_file_streaming_q4 requires grouped_canonical_v1 layout ({} bytes) for {}, found {} bytes in '{}'",
                expected_bytes,
                config.model_name(),
                rem_bytes,
                src_path
            ),
        ));
    }

    let mut reader = io::BufReader::with_capacity(4 * 1024 * 1024, src_file);
    let dst_file = File::create(dst_path)?;
    let mut writer = io::BufWriter::with_capacity(4 * 1024 * 1024, dst_file);

    let header = CodqHeader::new(DTYPE_Q4_0, config);
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const CodqHeader as *const u8,
            std::mem::size_of::<CodqHeader>(),
        )
    };
    writer.write_all(header_bytes)?;

    let schedule: [(usize, bool); 20] = [
        (config.vocab_size * config.dim, false),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim * config.dim, true),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim * config.dim, true),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim * config.dim, true),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim * config.dim, true),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim, false),
        (config.n_layers * config.dim * config.hidden_dim, true),
        (config.n_layers * config.hidden_dim, false),
        (config.n_layers * config.hidden_dim * config.dim, true),
        (config.n_layers * config.dim, false),
        (config.dim, false),
        (config.dim, false),
        (config.lm_head_floats(), false),
    ];

    const BATCH_BLOCKS: usize = 2048;
    let mut in_buf = vec![0u8; BATCH_BLOCKS * 32 * 4];
    let mut out_buf = vec![0u8; BATCH_BLOCKS * 20];

    for (num_floats, quantize) in schedule {
        if !quantize {
            stream_copy_floats(&mut reader, &mut writer, num_floats)?;
        } else {
            if num_floats % 32 != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Quantized tensor length {} is not a multiple of 32", num_floats),
                ));
            }
            let mut rem_blocks = num_floats / 32;
            while rem_blocks > 0 {
                let batch = rem_blocks.min(BATCH_BLOCKS);
                let in_len = batch * 32 * 4;
                reader.read_exact(&mut in_buf[..in_len])?;
                for b in 0..batch {
                    let mut vals = [0.0f32; 32];
                    let b_in = &in_buf[b * 128..(b + 1) * 128];
                    for i in 0..32 {
                        vals[i] = f32::from_le_bytes([
                            b_in[i * 4],
                            b_in[i * 4 + 1],
                            b_in[i * 4 + 2],
                            b_in[i * 4 + 3],
                        ]);
                    }
                    let blk = BlockQ4_0::quantize(&vals);
                    let b_out = &mut out_buf[b * 20..(b + 1) * 20];
                    b_out[0..4].copy_from_slice(&blk.d.to_le_bytes());
                    b_out[4..20].copy_from_slice(&blk.qs);
                }
                writer.write_all(&out_buf[..batch * 20])?;
                rem_blocks -= batch;
            }
        }
    }

    writer.flush()?;
    let out_file = writer.into_inner().map_err(|e| e.into_error())?;
    out_file.metadata().map(|m| m.len())
}

/// Reads a float32 `.npy` array from an uncompressed `.npz` archive (`np.savez` STORE format)
/// by parsing the ZIP Local File Header (including ZIP64 if present) and the `.npy` shape header
/// to compute the exact byte length without scanning float data for `b"PK"`.
pub fn read_npz_floats(npz_path: &str, target_name: &str) -> io::Result<Vec<f32>> {
    let data = std::fs::read(npz_path)?;
    let target_bytes = target_name.as_bytes();
    let mut pos = 0usize;
    while pos + 30 <= data.len() && &data[pos..pos + 4] == b"PK\x03\x04" {
        let comp_method = u16::from_le_bytes([data[pos + 8], data[pos + 9]]);
        let mut comp_size = u32::from_le_bytes([data[pos + 18], data[pos + 19], data[pos + 20], data[pos + 21]]) as usize;
        let mut uncomp_size = u32::from_le_bytes([data[pos + 22], data[pos + 23], data[pos + 24], data[pos + 25]]) as usize;
        let fn_len = u16::from_le_bytes([data[pos + 26], data[pos + 27]]) as usize;
        let extra_len = u16::from_le_bytes([data[pos + 28], data[pos + 29]]) as usize;
        let fn_start = pos + 30;
        let fn_end = fn_start + fn_len;
        let data_start = fn_end + extra_len;
        if data_start > data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated ZIP local header in .npz"));
        }

        // Check ZIP64 extra field (0x0001) if sizes are 0xFFFFFFFF
        if uncomp_size == 0xFFFF_FFFF || comp_size == 0xFFFF_FFFF {
            let extra = &data[fn_end..data_start];
            let mut epos = 0usize;
            while epos + 4 <= extra.len() {
                let tag = u16::from_le_bytes([extra[epos], extra[epos + 1]]);
                let sz = u16::from_le_bytes([extra[epos + 2], extra[epos + 3]]) as usize;
                if tag == 0x0001 && epos + 4 + sz <= extra.len() && sz >= 16 {
                    let u64_uncomp = u64::from_le_bytes(extra[epos + 4..epos + 12].try_into().unwrap()) as usize;
                    let u64_comp = u64::from_le_bytes(extra[epos + 12..epos + 20].try_into().unwrap()) as usize;
                    uncomp_size = u64_uncomp;
                    comp_size = u64_comp;
                    break;
                }
                epos += 4 + sz;
            }
        }

        // Parse .npy header to determine exact payload size even if ZIP local header has sizes in data descriptor
        if data_start + 10 > data.len() || &data[data_start..data_start + 6] != b"\x93NUMPY" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid or compressed .npy entry in '{}' (method={})", npz_path, comp_method),
            ));
        }
        let major = data[data_start + 6];
        let (prefix_len, header_len) = if major == 1 {
            (10usize, u16::from_le_bytes([data[data_start + 8], data[data_start + 9]]) as usize)
        } else {
            if data_start + 12 > data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated .npy v2 header"));
            }
            (
                12usize,
                u32::from_le_bytes([
                    data[data_start + 8],
                    data[data_start + 9],
                    data[data_start + 10],
                    data[data_start + 11],
                ]) as usize,
            )
        };
        let raw_start = data_start + prefix_len + header_len;
        if raw_start > data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated .npy dictionary header"));
        }

        let hdr_str = std::str::from_utf8(&data[data_start + prefix_len..raw_start]).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid UTF-8 in .npy header: {}", e))
        })?;

        // Parse shape tuple from "'shape': (...)"
        let shape_idx = hdr_str.find("'shape':").ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Missing 'shape' in .npy header")
        })?;
        let after_shape = &hdr_str[shape_idx..];
        let open_paren = after_shape.find('(').ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Missing '(' in .npy shape")
        })?;
        let close_paren = after_shape[open_paren..].find(')').ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Missing ')' in .npy shape")
        })?;
        let shape_inner = &after_shape[open_paren + 1..open_paren + close_paren];
        let mut num_elements = 1usize;
        for part in shape_inner.split(',') {
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                let dim = trimmed.parse::<usize>().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid dimension '{}': {}", trimmed, e))
                })?;
                num_elements = num_elements.saturating_mul(dim);
            }
        }

        // Determine element byte size from "'descr':" or ZIP uncomp_size
        let payload_bytes = if uncomp_size > prefix_len + header_len {
            uncomp_size - (prefix_len + header_len)
        } else {
            num_elements * 4
        };

        if &data[fn_start..fn_end] == target_bytes {
            if comp_method != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Entry '{}' in '{}' is compressed (method={}); repackage with np.savez", target_name, npz_path, comp_method),
                ));
            }
            let n_floats = num_elements;
            let expected_float_bytes = n_floats * 4;
            let raw_end = raw_start + expected_float_bytes;
            if raw_end > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("Entry '{}' in '{}' truncated: need {}..{}, file len {}", target_name, npz_path, raw_start, raw_end, data.len()),
                ));
            }
            let float_bytes = &data[raw_start..raw_end];
            let mut floats = Vec::with_capacity(n_floats);
            for chunk in float_bytes.chunks_exact(4) {
                floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            return Ok(floats);
        }

        let entry_total = if comp_size > 0 {
            comp_size
        } else {
            prefix_len + header_len + payload_bytes
        };
        pos = data_start + entry_total;
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("{} not found in {}", target_name, npz_path),
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    Fp32,
    Q8_0,
    Q4_0,
}

#[derive(Clone, Copy)]
pub enum ModelWeightsRef<'a> {
    Fp32(&'a EsmWeights),
    Q8(&'a EsmWeightsQ8),
    Q4(&'a EsmWeightsQ4),
}

pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    let len = a.len().min(b.len());
    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 1e-12 {
        dot / denom
    } else {
        1.0
    }
}

pub fn compute_rmse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 { return 0.0; }
    let mut sum_sq = 0.0f64;
    for i in 0..len {
        let diff = (a[i] - b[i]) as f64;
        sum_sq += diff * diff;
    }
    (sum_sq / len as f64).sqrt() as f32
}

pub fn compute_top1_match_rate(logits_ref: &[f32], logits_test: &[f32], vocab_size: usize) -> f32 {
    if vocab_size == 0 { return 1.0; }
    let n_tokens = logits_ref.len().min(logits_test.len()) / vocab_size;
    if n_tokens == 0 { return 1.0; }
    let mut matches = 0;
    for t in 0..n_tokens {
        let slice_ref = &logits_ref[t * vocab_size..(t + 1) * vocab_size];
        let slice_test = &logits_test[t * vocab_size..(t + 1) * vocab_size];
        let arg_ref = slice_ref.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal)).map(|x| x.0).unwrap_or(0);
        let arg_test = slice_test.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal)).map(|x| x.0).unwrap_or(0);
        if arg_ref == arg_test {
            matches += 1;
        }
    }
    matches as f32 / n_tokens as f32
}

pub fn compute_kl_divergence(logits_ref: &[f32], logits_test: &[f32], vocab_size: usize) -> f32 {
    if vocab_size == 0 { return 0.0; }
    let n_tokens = logits_ref.len().min(logits_test.len()) / vocab_size;
    if n_tokens == 0 { return 0.0; }
    let mut total_kl = 0.0f32;

    for t in 0..n_tokens {
        let slice_ref = &logits_ref[t * vocab_size..(t + 1) * vocab_size];
        let slice_test = &logits_test[t * vocab_size..(t + 1) * vocab_size];

        let max_ref = slice_ref.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_test = slice_test.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp_ref = 0.0f32;
        let mut sum_exp_test = 0.0f32;
        for i in 0..vocab_size {
            sum_exp_ref += (slice_ref[i] - max_ref).exp();
            sum_exp_test += (slice_test[i] - max_test).exp();
        }

        let mut kl_t = 0.0f32;
        for i in 0..vocab_size {
            let p = ((slice_ref[i] - max_ref).exp() / sum_exp_ref).max(1e-12);
            let q = ((slice_test[i] - max_test).exp() / sum_exp_test).max(1e-12);
            kl_t += p * (p.ln() - q.ln());
        }
        total_kl += kl_t;
    }
    total_kl / n_tokens as f32
}

pub fn compute_centered_cosine_similarity(a: &[f32], b: &[f32], vocab_size: usize) -> f32 {
    if vocab_size == 0 { return 1.0; }
    let n_tokens = a.len().min(b.len()) / vocab_size;
    if n_tokens == 0 { return 1.0; }
    let mut sum_cos = 0.0f32;
    for t in 0..n_tokens {
        let sa = &a[t * vocab_size..(t + 1) * vocab_size];
        let sb = &b[t * vocab_size..(t + 1) * vocab_size];
        let mean_a = sa.iter().sum::<f32>() / vocab_size as f32;
        let mean_b = sb.iter().sum::<f32>() / vocab_size as f32;

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..vocab_size {
            let ca = sa[i] - mean_a;
            let cb = sb[i] - mean_b;
            dot += ca * cb;
            norm_a += ca * ca;
            norm_b += cb * cb;
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom > 1e-12 {
            sum_cos += dot / denom;
        } else {
            sum_cos += 1.0;
        }
    }
    sum_cos / n_tokens as f32
}

pub fn dequantize_q4_to_vec(blocks: &[BlockQ4_0]) -> Vec<f32> {
    let mut out = vec![0.0f32; blocks.len() * 32];
    for (i, b) in blocks.iter().enumerate() {
        b.dequantize_to(&mut out[i * 32..(i + 1) * 32]);
    }
    out
}

pub fn matmul_q8(out: &mut [f32], x: &[f32], w: &[BlockQ8_0], bias: &[f32], in_dim: usize, out_dim: usize) {
    debug_assert!(out.len() >= out_dim && bias.len() >= out_dim && x.len() >= in_dim);
    let blocks_per_row = in_dim / 32;
    for i in 0..out_dim {
        let row_blocks = &w[i * blocks_per_row..(i + 1) * blocks_per_row];
        let mut sum = 0.0f32;
        for (b_idx, block) in row_blocks.iter().enumerate() {
            let x_chunk = &x[b_idx * 32..(b_idx + 1) * 32];
            let mut block_sum = 0.0f32;
            for j in 0..32 {
                block_sum += x_chunk[j] * (block.qs[j] as f32);
            }
            sum += block_sum * block.d;
        }
        out[i] = sum + bias[i];
    }
}

#[allow(clippy::manual_slice_size_calculation)]
fn read_floats(file: &mut File, num_floats: usize) -> io::Result<Vec<f32>> {
    let mut floats = vec![0.0f32; num_floats];
    unsafe {
        let byte_slice = std::slice::from_raw_parts_mut(
            floats.as_mut_ptr() as *mut u8,
            num_floats * std::mem::size_of::<f32>(),
        );
        file.read_exact(byte_slice)?;
    }
    #[cfg(target_endian = "big")]
    {
        for f in floats.iter_mut() {
            *f = f32::from_bits(f.to_bits().swap_bytes());
        }
    }
    Ok(floats)
}

#[allow(clippy::manual_slice_size_calculation)]
fn write_floats(file: &mut File, floats: &[f32]) -> io::Result<()> {
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            floats.as_ptr() as *const u8,
            floats.len() * std::mem::size_of::<f32>(),
        )
    };
    file.write_all(byte_slice)
}

#[allow(clippy::manual_slice_size_calculation)]
fn read_q8_blocks(file: &mut File, num_blocks: usize) -> io::Result<Vec<BlockQ8_0>> {
    let mut blocks = vec![BlockQ8_0 { d: 0.0, qs: [0; 32] }; num_blocks];
    unsafe {
        let byte_slice = std::slice::from_raw_parts_mut(
            blocks.as_mut_ptr() as *mut u8,
            num_blocks * std::mem::size_of::<BlockQ8_0>(),
        );
        file.read_exact(byte_slice)?;
    }
    Ok(blocks)
}

#[allow(clippy::manual_slice_size_calculation)]
fn write_q8_blocks(file: &mut File, blocks: &[BlockQ8_0]) -> io::Result<()> {
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            blocks.len() * std::mem::size_of::<BlockQ8_0>(),
        )
    };
    file.write_all(byte_slice)
}

pub struct RunState {
    pub x: Vec<f32>, pub xb: Vec<f32>, pub xb2: Vec<f32>, pub hb: Vec<f32>, pub hb2: Vec<f32>,
    pub q: Vec<f32>, pub k: Vec<f32>, pub v: Vec<f32>, pub att: Vec<f32>, pub logits: Vec<f32>,
    pub key_cache: Vec<f32>, pub value_cache: Vec<f32>, pub rope_cos: Vec<f32>, pub rope_sin: Vec<f32>,
}
impl RunState {
    pub fn new(config: &EsmConfig) -> Self {
        let head_dim = config.dim / config.n_heads; let half_head = head_dim / 2;
        let mut rope_cos = vec![0.0; config.max_seq_len * half_head];
        let mut rope_sin = vec![0.0; config.max_seq_len * half_head];
        for pos in 0..config.max_seq_len {
            for j in 0..half_head {
                let freq = 1.0 / 10000.0f32.powf((j * 2) as f32 / head_dim as f32);
                let val = (pos as f32) * freq;
                rope_cos[pos * half_head + j] = val.cos(); rope_sin[pos * half_head + j] = val.sin();
            }
        }
        Self {
            x: vec![0.0; config.dim], xb: vec![0.0; config.dim], xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim], hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim], k: vec![0.0; config.dim], v: vec![0.0; config.dim],
            att: vec![0.0; config.n_heads * config.max_seq_len], logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; config.n_layers * config.max_seq_len * config.dim],
            value_cache: vec![0.0; config.n_layers * config.max_seq_len * config.dim],
            rope_cos, rope_sin,
        }
    }
}

pub struct BidirectionalRunState {
    pub max_len: usize,
    pub x: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub att: Vec<f32>,
    pub out_att: Vec<f32>,
    pub xb: Vec<f32>,
    pub xb2: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub logits: Vec<f32>,
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
    pub final_layer_att: Vec<f32>,
}

impl BidirectionalRunState {
    pub fn new(config: &EsmConfig, max_len: usize) -> Self {
        let head_dim = config.dim / config.n_heads;
        let half_head = head_dim / 2;
        let mut rope_cos = vec![0.0; max_len * half_head];
        let mut rope_sin = vec![0.0; max_len * half_head];
        for pos in 0..max_len {
            for j in 0..half_head {
                let freq = 1.0 / 10000.0f32.powf((j * 2) as f32 / head_dim as f32);
                let val = (pos as f32) * freq;
                rope_cos[pos * half_head + j] = val.cos();
                rope_sin[pos * half_head + j] = val.sin();
            }
        }
        Self {
            max_len,
            x: vec![0.0; max_len * config.dim],
            q: vec![0.0; max_len * config.dim],
            k: vec![0.0; max_len * config.dim],
            v: vec![0.0; max_len * config.dim],
            att: vec![0.0; config.n_heads * max_len],
            out_att: vec![0.0; max_len * config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            logits: vec![0.0; max_len * config.vocab_size],
            rope_cos,
            rope_sin,
            final_layer_att: vec![0.0; max_len * max_len],
        }
    }
}

pub fn active_simd_backend() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") { return "AVX-512 (512-bit Vector Engine)"; }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") { return "AVX2+FMA (256-bit Fused Multiply-Add)"; }
        if std::is_x86_feature_detected!("avx") { return "AVX1 (256-bit Sandy Bridge Vector Engine)"; }
    }
    "Scalar Fallback"
}

pub fn layer_norm(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32]) {
    let dim = x.len(); const EPSILON: f32 = 1e-5;
    let mean = x.iter().sum::<f32>() / dim as f32;
    let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
    let inv_stddev = 1.0 / (variance + EPSILON).sqrt();
    for i in 0..dim { out[i] = weight[i] * (x[i] - mean) * inv_stddev + bias[i]; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn hsum256_ps_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let vlow = _mm256_castps256_ps128(v); let vhigh = _mm256_extractf128_ps(v, 1);
        let v128 = _mm_add_ps(vlow, vhigh); let hi64 = _mm_movehl_ps(v128, v128); let sum64 = _mm_add_ps(v128, hi64);
        let hi32 = _mm_shuffle_ps(sum64, sum64, 0x55); let sum32 = _mm_add_ss(sum64, hi32); _mm_cvtss_f32(sum32)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_avx2_fma(out: &mut [f32], x: &[f32], w: &[f32], bias: &[f32], in_dim: usize, out_dim: usize) {
    debug_assert!(out.len() >= out_dim && bias.len() >= out_dim && x.len() >= in_dim && w.len() >= out_dim * in_dim);
    use std::arch::x86_64::*; unsafe {
        let chunks32 = in_dim / 32; let rem_start = chunks32 * 32; let x_ptr = x.as_ptr();
        for i in 0..out_dim {
            let w_row = w.as_ptr().add(i * in_dim);
            let mut acc0 = _mm256_setzero_ps(); let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps(); let mut acc3 = _mm256_setzero_ps();
            for c in 0..chunks32 {
                let offset = c * 32;
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w_row.add(offset)), _mm256_loadu_ps(x_ptr.add(offset)), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w_row.add(offset + 8)), _mm256_loadu_ps(x_ptr.add(offset + 8)), acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(w_row.add(offset + 16)), _mm256_loadu_ps(x_ptr.add(offset + 16)), acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(w_row.add(offset + 24)), _mm256_loadu_ps(x_ptr.add(offset + 24)), acc3);
            }
            let sum01 = _mm256_add_ps(acc0, acc1); let sum23 = _mm256_add_ps(acc2, acc3);
            let mut val = hsum256_ps_avx(_mm256_add_ps(sum01, sum23));
            for j in rem_start..in_dim { val += *w_row.add(j) * *x_ptr.add(j); }
            *out.get_unchecked_mut(i) = val + *bias.get_unchecked(i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn matmul_avx1(out: &mut [f32], x: &[f32], w: &[f32], bias: &[f32], in_dim: usize, out_dim: usize) {
    debug_assert!(out.len() >= out_dim && bias.len() >= out_dim && x.len() >= in_dim && w.len() >= out_dim * in_dim);
    use std::arch::x86_64::*; unsafe {
        let chunks32 = in_dim / 32; let rem_start = chunks32 * 32; let x_ptr = x.as_ptr();
        for i in 0..out_dim {
            let w_row = w.as_ptr().add(i * in_dim);
            let mut acc0 = _mm256_setzero_ps(); let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps(); let mut acc3 = _mm256_setzero_ps();
            for c in 0..chunks32 {
                let offset = c * 32;
                let prod0 = _mm256_mul_ps(_mm256_loadu_ps(w_row.add(offset)), _mm256_loadu_ps(x_ptr.add(offset)));
                acc0 = _mm256_add_ps(acc0, prod0);
                let prod1 = _mm256_mul_ps(_mm256_loadu_ps(w_row.add(offset + 8)), _mm256_loadu_ps(x_ptr.add(offset + 8)));
                acc1 = _mm256_add_ps(acc1, prod1);
                let prod2 = _mm256_mul_ps(_mm256_loadu_ps(w_row.add(offset + 16)), _mm256_loadu_ps(x_ptr.add(offset + 16)));
                acc2 = _mm256_add_ps(acc2, prod2);
                let prod3 = _mm256_mul_ps(_mm256_loadu_ps(w_row.add(offset + 24)), _mm256_loadu_ps(x_ptr.add(offset + 24)));
                acc3 = _mm256_add_ps(acc3, prod3);
            }
            let sum01 = _mm256_add_ps(acc0, acc1); let sum23 = _mm256_add_ps(acc2, acc3);
            let mut val = hsum256_ps_avx(_mm256_add_ps(sum01, sum23));
            for j in rem_start..in_dim { val += *w_row.add(j) * *x_ptr.add(j); }
            *out.get_unchecked_mut(i) = val + *bias.get_unchecked(i);
        }
    }
}

pub fn matmul_scalar(out: &mut [f32], x: &[f32], w: &[f32], bias: &[f32], in_dim: usize, out_dim: usize) {
    for i in 0..out_dim {
        let w_row = &w[i * in_dim..(i + 1) * in_dim]; let mut val = 0.0f32;
        for j in 0..in_dim { val += w_row[j] * x[j]; } out[i] = val + bias[i];
    }
}

#[inline]
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: &[f32], in_dim: usize, out_dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") { unsafe { matmul_avx2_fma(out, x, w, bias, in_dim, out_dim); } return; }
        if std::is_x86_feature_detected!("avx") { unsafe { matmul_avx1(out, x, w, bias, in_dim, out_dim); } return; }
    }
    matmul_scalar(out, x, w, bias, in_dim, out_dim);
}

// ==========================================
// PERSISTENT WORKER THREAD POOL (ZERO OS SPAWNING)
// ==========================================

struct SharedJob {
    job_fn: Option<unsafe fn(*const (), usize, usize)>,
    context: *const (),
    num_threads: usize,
    generation: usize,
    shutdown: bool,
}

unsafe impl Send for SharedJob {}
unsafe impl Sync for SharedJob {}

struct ThreadPoolInner {
    job_state: Mutex<SharedJob>,
    job_condvar: Condvar,
    done_count: AtomicUsize,
    done_mutex: Mutex<()>,
    done_condvar: Condvar,
}

pub struct ThreadPool {
    run_lock: Mutex<()>,
    inner: Arc<ThreadPoolInner>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new(num_workers: usize) -> Self {
        let inner = Arc::new(ThreadPoolInner {
            job_state: Mutex::new(SharedJob {
                job_fn: None,
                context: std::ptr::null(),
                num_threads: 1,
                generation: 0,
                shutdown: false,
            }),
            job_condvar: Condvar::new(),
            done_count: AtomicUsize::new(0),
            done_mutex: Mutex::new(()),
            done_condvar: Condvar::new(),
        });

        let mut handles = Vec::with_capacity(num_workers);
        for worker_id in 1..=num_workers {
            let inner_clone = Arc::clone(&inner);
            let handle = std::thread::Builder::new()
                .name(format!("codon-worker-{}", worker_id))
                .spawn(move || {
                    let mut my_generation = 0;
                    loop {
                        let (job_fn, context, num_threads) = {
                            let mut state = inner_clone.job_state.lock().unwrap();
                            while state.generation == my_generation && !state.shutdown {
                                state = inner_clone.job_condvar.wait(state).unwrap();
                            }
                            if state.shutdown {
                                break;
                            }
                            my_generation = state.generation;
                            (state.job_fn, state.context, state.num_threads)
                        };

                        if worker_id < num_threads {
                            if let Some(f) = job_fn {
                                unsafe {
                                    f(context, worker_id, num_threads);
                                }
                            }

                            let needed = num_threads - 1;
                            let prev = inner_clone.done_count.fetch_add(1, Ordering::AcqRel);
                            if prev + 1 >= needed {
                                let _lock = inner_clone.done_mutex.lock().unwrap();
                                inner_clone.done_condvar.notify_one();
                            }
                        }
                    }
                })
                .expect("Failed to spawn persistent worker thread");
            handles.push(handle);
        }

        Self {
            run_lock: Mutex::new(()),
            inner,
            handles,
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn run_parallel(
        &self,
        num_threads: usize,
        job_fn: unsafe fn(*const (), usize, usize),
        context: *const (),
    ) {
        if num_threads <= 1 {
            unsafe { job_fn(context, 0, 1); }
            return;
        }

        let _run_guard = self.run_lock.lock().unwrap();
        let effective_threads = num_threads.min(self.handles.len() + 1);
        let needed_workers = effective_threads - 1;
        self.inner.done_count.store(0, Ordering::Release);

        {
            let mut state = self.inner.job_state.lock().unwrap();
            state.job_fn = Some(job_fn);
            state.context = context;
            state.num_threads = effective_threads;
            state.generation = state.generation.wrapping_add(1);
            self.inner.job_condvar.notify_all();
        }

        // Calling thread executes thread 0's slice!
        unsafe {
            job_fn(context, 0, effective_threads);
        }

        // Wait for workers to finish
        let mut spins = 0;
        while self.inner.done_count.load(Ordering::Acquire) < needed_workers {
            if spins < 500 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                let mut guard = self.inner.done_mutex.lock().unwrap();
                while self.inner.done_count.load(Ordering::Acquire) < needed_workers {
                    guard = self.inner.done_condvar.wait(guard).unwrap();
                }
                break;
            }
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        {
            let mut state = self.inner.job_state.lock().unwrap();
            state.shutdown = true;
            state.generation = state.generation.wrapping_add(1);
            self.inner.job_condvar.notify_all();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

static GLOBAL_POOL: OnceLock<ThreadPool> = OnceLock::new();

pub fn get_global_pool() -> &'static ThreadPool {
    GLOBAL_POOL.get_or_init(|| {
        let max_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(4);
        ThreadPool::new(max_workers)
    })
}

/// Detects optimal thread count for HPC vector workloads.
/// On CPUs with SMT (Hyper-Threading), physical cores avoid AVX vector unit port contention
/// and L1/L2 cache pressure, achieving higher sustained throughput than oversubscribed logical threads.
pub fn detect_optimal_threads() -> usize {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    #[cfg(target_os = "linux")]
    {
        if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in info.lines() {
                if line.starts_with("cpu cores") {
                    if let Some(val_str) = line.split(':').nth(1) {
                        if let Ok(cores) = val_str.trim().parse::<usize>() {
                            if cores > 0 && cores <= logical {
                                return cores;
                            }
                        }
                    }
                }
            }
        }
    }
    logical
}

struct Fp32Job<'a> {
    out: *mut f32,
    x: &'a [f32],
    w: &'a [f32],
    bias: &'a [f32],
    in_dim: usize,
    out_dim: usize,
}

unsafe fn run_fp32_slice(ctx: *const (), t_idx: usize, n_threads: usize) {
    let job = unsafe { &*(ctx as *const Fp32Job) };
    let chunk_size = (job.out_dim + n_threads - 1) / n_threads;
    let start_row = t_idx * chunk_size;
    if start_row < job.out_dim {
        let rows_in_chunk = (job.out_dim - start_row).min(chunk_size);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(job.out.add(start_row), rows_in_chunk) };
        let bias_chunk = &job.bias[start_row..start_row + rows_in_chunk];
        let w_start = start_row * job.in_dim;
        let w_end = w_start + rows_in_chunk * job.in_dim;
        let w_chunk = &job.w[w_start..w_end];
        matmul(out_slice, job.x, w_chunk, bias_chunk, job.in_dim, rows_in_chunk);
    }
}

pub fn matmul_parallel(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
    num_threads: usize,
) {
    if num_threads <= 1 || out_dim < 32 {
        matmul(out, x, w, bias, in_dim, out_dim);
        return;
    }

    let job = Fp32Job {
        out: out.as_mut_ptr(),
        x,
        w,
        bias,
        in_dim,
        out_dim,
    };
    get_global_pool().run_parallel(num_threads, run_fp32_slice, &job as *const _ as *const ());
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn matmul_q8_avx1(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ8_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(out.len() >= out_dim && bias.len() >= out_dim && x.len() >= in_dim);
    let blocks_per_row = in_dim / 32;
    let x_ptr = x.as_ptr();

    for i in 0..out_dim {
        let row_blocks = &w[i * blocks_per_row..(i + 1) * blocks_per_row];
        let mut row_sum = 0.0f32;

        for (b_idx, block) in row_blocks.iter().enumerate() {
            let block_x = x_ptr.add(b_idx * 32);
            let qs_ptr = block.qs.as_ptr();

            let mut acc = _mm256_setzero_ps();

            // 4 chunks of 8 elements = 32 elements
            for c in 0..4 {
                let offset = c * 8;
                let vx = _mm256_loadu_ps(block_x.add(offset));
                let raw_8 = _mm_loadl_epi64(qs_ptr.add(offset) as *const __m128i);
                let epi32_lo = _mm_cvtepi8_epi32(raw_8);
                let epi32_hi = _mm_cvtepi8_epi32(_mm_srli_si128(raw_8, 4));
                let ps_lo = _mm_cvtepi32_ps(epi32_lo);
                let ps_hi = _mm_cvtepi32_ps(epi32_hi);
                let vq = _mm256_set_m128(ps_hi, ps_lo);
                let prod = _mm256_mul_ps(vq, vx);
                acc = _mm256_add_ps(acc, prod);
            }

            row_sum += hsum256_ps_avx(acc) * block.d;
        }

        *out.get_unchecked_mut(i) = row_sum + *bias.get_unchecked(i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_q8_avx2_fma(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ8_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(out.len() >= out_dim && bias.len() >= out_dim && x.len() >= in_dim);
    let blocks_per_row = in_dim / 32;
    let x_ptr = x.as_ptr();

    for i in 0..out_dim {
        let row_blocks = &w[i * blocks_per_row..(i + 1) * blocks_per_row];
        let mut row_sum = 0.0f32;

        for (b_idx, block) in row_blocks.iter().enumerate() {
            let block_x = x_ptr.add(b_idx * 32);
            let qs_ptr = block.qs.as_ptr();

            let mut acc = _mm256_setzero_ps();

            // 4 chunks of 8 elements = 32 elements
            for c in 0..4 {
                let offset = c * 8;
                let vx = _mm256_loadu_ps(block_x.add(offset));
                let raw_8 = _mm_loadl_epi64(qs_ptr.add(offset) as *const __m128i);
                let epi32 = _mm256_cvtepi8_epi32(raw_8);
                let vq = _mm256_cvtepi32_ps(epi32);
                acc = _mm256_fmadd_ps(vq, vx, acc);
            }

            row_sum += hsum256_ps_avx(acc) * block.d;
        }

        *out.get_unchecked_mut(i) = row_sum + *bias.get_unchecked(i);
    }
}

#[inline]
pub fn matmul_q8_simd(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ8_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) {
    assert!(out.len() >= out_dim, "out len {} < out_dim {}", out.len(), out_dim);
    assert!(bias.len() >= out_dim, "bias len {} < out_dim {}", bias.len(), out_dim);
    assert!(x.len() >= in_dim, "x len {} < in_dim {}", x.len(), in_dim);
    assert!(in_dim % 32 == 0, "in_dim {} must be multiple of 32", in_dim);
    let blocks_per_row = in_dim / 32;
    assert!(w.len() >= out_dim * blocks_per_row, "w len {} < expected {}", w.len(), out_dim * blocks_per_row);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { matmul_q8_avx2_fma(out, x, w, bias, in_dim, out_dim) };
            return;
        }
        if std::is_x86_feature_detected!("avx") {
            unsafe { matmul_q8_avx1(out, x, w, bias, in_dim, out_dim) };
            return;
        }
    }
    matmul_q8(out, x, w, bias, in_dim, out_dim);
}

struct Q8Job<'a> {
    out: *mut f32,
    x: &'a [f32],
    w: &'a [BlockQ8_0],
    bias: &'a [f32],
    in_dim: usize,
    out_dim: usize,
}

unsafe fn run_q8_slice(ctx: *const (), t_idx: usize, n_threads: usize) {
    let job = unsafe { &*(ctx as *const Q8Job) };
    let blocks_per_row = job.in_dim / 32;
    let chunk_size = (job.out_dim + n_threads - 1) / n_threads;
    let start_row = t_idx * chunk_size;
    if start_row < job.out_dim {
        let rows_in_chunk = (job.out_dim - start_row).min(chunk_size);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(job.out.add(start_row), rows_in_chunk) };
        let bias_chunk = &job.bias[start_row..start_row + rows_in_chunk];
        let w_start = start_row * blocks_per_row;
        let w_end = w_start + rows_in_chunk * blocks_per_row;
        let w_chunk = &job.w[w_start..w_end];
        matmul_q8_simd(out_slice, job.x, w_chunk, bias_chunk, job.in_dim, rows_in_chunk);
    }
}

pub fn matmul_q8_parallel(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ8_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
    num_threads: usize,
) {
    if num_threads <= 1 || out_dim < 32 {
        matmul_q8_simd(out, x, w, bias, in_dim, out_dim);
        return;
    }

    let job = Q8Job {
        out: out.as_mut_ptr(),
        x,
        w,
        bias,
        in_dim,
        out_dim,
    };
    get_global_pool().run_parallel(num_threads, run_q8_slice, &job as *const _ as *const ());
}

// ============================================================================
// MATMUL Q4_0 KERNELS (SCALAR, AVX1, AVX2+FMA, SIMD DISPATCH, PARALLEL)
// ============================================================================

/// Scalar reference kernel for Q4_0 matrix multiplication (Ground Truth).
pub fn matmul_q4(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ4_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert!(out.len() >= out_dim && bias.len() >= out_dim && x.len() >= in_dim);
    let blocks_per_row = in_dim / 32;
    for i in 0..out_dim {
        let row_blocks = &w[i * blocks_per_row..(i + 1) * blocks_per_row];
        let mut sum = 0.0f32;
        for (b_idx, block) in row_blocks.iter().enumerate() {
            let x_chunk = &x[b_idx * 32..(b_idx + 1) * 32];
            let mut block_sum = 0.0f32;
            for j in 0..16 {
                let q_lo = (block.qs[j] & 0x0F) as i32 - 8;
                block_sum += x_chunk[j] * (q_lo as f32);
                let q_hi = ((block.qs[j] >> 4) & 0x0F) as i32 - 8;
                block_sum += x_chunk[j + 16] * (q_hi as f32);
            }
            sum += block_sum * block.d;
        }
        out[i] = sum + bias[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn matmul_q4_avx1(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ4_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(out.len() >= out_dim && bias.len() >= out_dim && x.len() >= in_dim);
    let blocks_per_row = in_dim / 32;
    let x_ptr = x.as_ptr();

    let mask_low = _mm_set1_epi8(0x0F);
    let c8 = _mm_set1_epi8(8);

    for i in 0..out_dim {
        let row_blocks = &w[i * blocks_per_row..(i + 1) * blocks_per_row];
        let mut row_sum = 0.0f32;

        for (b_idx, block) in row_blocks.iter().enumerate() {
            let block_x = x_ptr.add(b_idx * 32);
            let raw16 = _mm_loadu_si128(block.qs.as_ptr() as *const __m128i);

            let lo_i8 = _mm_sub_epi8(_mm_and_si128(raw16, mask_low), c8);
            let hi_i8 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw16, 4), mask_low), c8);

            let mut acc = _mm256_setzero_ps();

            // 0..7
            let lo_0_3 = _mm_cvtepi8_epi32(lo_i8);
            let lo_4_7 = _mm_cvtepi8_epi32(_mm_srli_si128(lo_i8, 4));
            let vq0 = _mm256_set_m128(_mm_cvtepi32_ps(lo_4_7), _mm_cvtepi32_ps(lo_0_3));
            let vx0 = _mm256_loadu_ps(block_x);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(vq0, vx0));

            // 8..15
            let lo_hi64 = _mm_srli_si128(lo_i8, 8);
            let lo_8_11 = _mm_cvtepi8_epi32(lo_hi64);
            let lo_12_15 = _mm_cvtepi8_epi32(_mm_srli_si128(lo_hi64, 4));
            let vq1 = _mm256_set_m128(_mm_cvtepi32_ps(lo_12_15), _mm_cvtepi32_ps(lo_8_11));
            let vx1 = _mm256_loadu_ps(block_x.add(8));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(vq1, vx1));

            // 16..23
            let hi_0_3 = _mm_cvtepi8_epi32(hi_i8);
            let hi_4_7 = _mm_cvtepi8_epi32(_mm_srli_si128(hi_i8, 4));
            let vq2 = _mm256_set_m128(_mm_cvtepi32_ps(hi_4_7), _mm_cvtepi32_ps(hi_0_3));
            let vx2 = _mm256_loadu_ps(block_x.add(16));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(vq2, vx2));

            // 24..31
            let hi_hi64 = _mm_srli_si128(hi_i8, 8);
            let hi_8_11 = _mm_cvtepi8_epi32(hi_hi64);
            let hi_12_15 = _mm_cvtepi8_epi32(_mm_srli_si128(hi_hi64, 4));
            let vq3 = _mm256_set_m128(_mm_cvtepi32_ps(hi_12_15), _mm_cvtepi32_ps(hi_8_11));
            let vx3 = _mm256_loadu_ps(block_x.add(24));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(vq3, vx3));

            row_sum += hsum256_ps_avx(acc) * block.d;
        }

        *out.get_unchecked_mut(i) = row_sum + *bias.get_unchecked(i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_q4_avx2_fma(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ4_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(out.len() >= out_dim && bias.len() >= out_dim && x.len() >= in_dim);
    let blocks_per_row = in_dim / 32;
    let x_ptr = x.as_ptr();

    let mask_low = _mm_set1_epi8(0x0F);
    let c8 = _mm_set1_epi8(8);

    for i in 0..out_dim {
        let row_blocks = &w[i * blocks_per_row..(i + 1) * blocks_per_row];
        let mut row_sum = 0.0f32;

        for (b_idx, block) in row_blocks.iter().enumerate() {
            let block_x = x_ptr.add(b_idx * 32);
            let raw16 = _mm_loadu_si128(block.qs.as_ptr() as *const __m128i);

            let lo_i8 = _mm_sub_epi8(_mm_and_si128(raw16, mask_low), c8);
            let hi_i8 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw16, 4), mask_low), c8);

            let mut acc = _mm256_setzero_ps();

            // 0..7
            let vq0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_i8));
            let vx0 = _mm256_loadu_ps(block_x);
            acc = _mm256_fmadd_ps(vq0, vx0, acc);

            // 8..15
            let vq1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo_i8, 8)));
            let vx1 = _mm256_loadu_ps(block_x.add(8));
            acc = _mm256_fmadd_ps(vq1, vx1, acc);

            // 16..23
            let vq2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_i8));
            let vx2 = _mm256_loadu_ps(block_x.add(16));
            acc = _mm256_fmadd_ps(vq2, vx2, acc);

            // 24..31
            let vq3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi_i8, 8)));
            let vx3 = _mm256_loadu_ps(block_x.add(24));
            acc = _mm256_fmadd_ps(vq3, vx3, acc);

            row_sum += hsum256_ps_avx(acc) * block.d;
        }

        *out.get_unchecked_mut(i) = row_sum + *bias.get_unchecked(i);
    }
}

#[inline]
pub fn matmul_q4_simd(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ4_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) {
    assert!(out.len() >= out_dim, "out len {} < out_dim {}", out.len(), out_dim);
    assert!(bias.len() >= out_dim, "bias len {} < out_dim {}", bias.len(), out_dim);
    assert!(x.len() >= in_dim, "x len {} < in_dim {}", x.len(), in_dim);
    assert!(in_dim % 32 == 0, "in_dim {} must be multiple of 32", in_dim);
    let blocks_per_row = in_dim / 32;
    assert!(w.len() >= out_dim * blocks_per_row, "w len {} < expected {}", w.len(), out_dim * blocks_per_row);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { matmul_q4_avx2_fma(out, x, w, bias, in_dim, out_dim) };
            return;
        }
        if std::is_x86_feature_detected!("avx") {
            unsafe { matmul_q4_avx1(out, x, w, bias, in_dim, out_dim) };
            return;
        }
    }
    matmul_q4(out, x, w, bias, in_dim, out_dim);
}

struct Q4Job<'a> {
    out: *mut f32,
    x: &'a [f32],
    w: &'a [BlockQ4_0],
    bias: &'a [f32],
    in_dim: usize,
    out_dim: usize,
}

unsafe fn run_q4_slice(ctx: *const (), t_idx: usize, n_threads: usize) {
    let job = unsafe { &*(ctx as *const Q4Job) };
    let blocks_per_row = job.in_dim / 32;
    let chunk_size = (job.out_dim + n_threads - 1) / n_threads;
    let start_row = t_idx * chunk_size;
    if start_row < job.out_dim {
        let rows_in_chunk = (job.out_dim - start_row).min(chunk_size);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(job.out.add(start_row), rows_in_chunk) };
        let bias_chunk = &job.bias[start_row..start_row + rows_in_chunk];
        let w_start = start_row * blocks_per_row;
        let w_end = w_start + rows_in_chunk * blocks_per_row;
        let w_chunk = &job.w[w_start..w_end];
        matmul_q4_simd(out_slice, job.x, w_chunk, bias_chunk, job.in_dim, rows_in_chunk);
    }
}

pub fn matmul_q4_parallel(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ4_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
    num_threads: usize,
) {
    if num_threads <= 1 || out_dim < 32 {
        matmul_q4_simd(out, x, w, bias, in_dim, out_dim);
        return;
    }

    let job = Q4Job {
        out: out.as_mut_ptr(),
        x,
        w,
        bias,
        in_dim,
        out_dim,
    };
    get_global_pool().run_parallel(num_threads, run_q4_slice, &job as *const _ as *const ());
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn dot_product_head_avx(q: *const f32, k: *const f32, head_dim: usize) -> f32 {
    use std::arch::x86_64::*;
    match head_dim {
        16 => {
            let s0 = _mm256_mul_ps(_mm256_loadu_ps(q), _mm256_loadu_ps(k));
            let s1 = _mm256_mul_ps(_mm256_loadu_ps(q.add(8)), _mm256_loadu_ps(k.add(8)));
            hsum256_ps_avx(_mm256_add_ps(s0, s1))
        }
        24 => {
            let s0 = _mm256_mul_ps(_mm256_loadu_ps(q), _mm256_loadu_ps(k));
            let s1 = _mm256_mul_ps(_mm256_loadu_ps(q.add(8)), _mm256_loadu_ps(k.add(8)));
            let s2 = _mm256_mul_ps(_mm256_loadu_ps(q.add(16)), _mm256_loadu_ps(k.add(16)));
            hsum256_ps_avx(_mm256_add_ps(_mm256_add_ps(s0, s1), s2))
        }
        32 => {
            let s0 = _mm256_mul_ps(_mm256_loadu_ps(q), _mm256_loadu_ps(k));
            let s1 = _mm256_mul_ps(_mm256_loadu_ps(q.add(8)), _mm256_loadu_ps(k.add(8)));
            let s2 = _mm256_mul_ps(_mm256_loadu_ps(q.add(16)), _mm256_loadu_ps(k.add(16)));
            let s3 = _mm256_mul_ps(_mm256_loadu_ps(q.add(24)), _mm256_loadu_ps(k.add(24)));
            hsum256_ps_avx(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)))
        }
        64 => {
            let s0 = _mm256_mul_ps(_mm256_loadu_ps(q), _mm256_loadu_ps(k));
            let s1 = _mm256_mul_ps(_mm256_loadu_ps(q.add(8)), _mm256_loadu_ps(k.add(8)));
            let s2 = _mm256_mul_ps(_mm256_loadu_ps(q.add(16)), _mm256_loadu_ps(k.add(16)));
            let s3 = _mm256_mul_ps(_mm256_loadu_ps(q.add(24)), _mm256_loadu_ps(k.add(24)));
            let s4 = _mm256_mul_ps(_mm256_loadu_ps(q.add(32)), _mm256_loadu_ps(k.add(32)));
            let s5 = _mm256_mul_ps(_mm256_loadu_ps(q.add(40)), _mm256_loadu_ps(k.add(40)));
            let s6 = _mm256_mul_ps(_mm256_loadu_ps(q.add(48)), _mm256_loadu_ps(k.add(48)));
            let s7 = _mm256_mul_ps(_mm256_loadu_ps(q.add(56)), _mm256_loadu_ps(k.add(56)));
            let sum01 = _mm256_add_ps(s0, s1);
            let sum23 = _mm256_add_ps(s2, s3);
            let sum45 = _mm256_add_ps(s4, s5);
            let sum67 = _mm256_add_ps(s6, s7);
            let sum0123 = _mm256_add_ps(sum01, sum23);
            let sum4567 = _mm256_add_ps(sum45, sum67);
            hsum256_ps_avx(_mm256_add_ps(sum0123, sum4567))
        }
        _ => {
            let mut sum = 0.0f32;
            for i in 0..head_dim {
                sum += *q.add(i) * *k.add(i);
            }
            sum
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn accumulate_value_head_avx(out: *mut f32, v: *const f32, prob: f32, head_dim: usize) {
    use std::arch::x86_64::*;
    let p_vec = _mm256_set1_ps(prob);
    match head_dim {
        16 => {
            let o0 = _mm256_add_ps(_mm256_loadu_ps(out), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v)));
            let o1 = _mm256_add_ps(_mm256_loadu_ps(out.add(8)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(8))));
            _mm256_storeu_ps(out, o0);
            _mm256_storeu_ps(out.add(8), o1);
        }
        24 => {
            let o0 = _mm256_add_ps(_mm256_loadu_ps(out), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v)));
            let o1 = _mm256_add_ps(_mm256_loadu_ps(out.add(8)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(8))));
            let o2 = _mm256_add_ps(_mm256_loadu_ps(out.add(16)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(16))));
            _mm256_storeu_ps(out, o0);
            _mm256_storeu_ps(out.add(8), o1);
            _mm256_storeu_ps(out.add(16), o2);
        }
        32 => {
            let o0 = _mm256_add_ps(_mm256_loadu_ps(out), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v)));
            let o1 = _mm256_add_ps(_mm256_loadu_ps(out.add(8)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(8))));
            let o2 = _mm256_add_ps(_mm256_loadu_ps(out.add(16)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(16))));
            let o3 = _mm256_add_ps(_mm256_loadu_ps(out.add(24)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(24))));
            _mm256_storeu_ps(out, o0);
            _mm256_storeu_ps(out.add(8), o1);
            _mm256_storeu_ps(out.add(16), o2);
            _mm256_storeu_ps(out.add(24), o3);
        }
        64 => {
            let o0 = _mm256_add_ps(_mm256_loadu_ps(out), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v)));
            let o1 = _mm256_add_ps(_mm256_loadu_ps(out.add(8)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(8))));
            let o2 = _mm256_add_ps(_mm256_loadu_ps(out.add(16)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(16))));
            let o3 = _mm256_add_ps(_mm256_loadu_ps(out.add(24)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(24))));
            let o4 = _mm256_add_ps(_mm256_loadu_ps(out.add(32)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(32))));
            let o5 = _mm256_add_ps(_mm256_loadu_ps(out.add(40)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(40))));
            let o6 = _mm256_add_ps(_mm256_loadu_ps(out.add(48)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(48))));
            let o7 = _mm256_add_ps(_mm256_loadu_ps(out.add(56)), _mm256_mul_ps(p_vec, _mm256_loadu_ps(v.add(56))));
            _mm256_storeu_ps(out, o0);
            _mm256_storeu_ps(out.add(8), o1);
            _mm256_storeu_ps(out.add(16), o2);
            _mm256_storeu_ps(out.add(24), o3);
            _mm256_storeu_ps(out.add(32), o4);
            _mm256_storeu_ps(out.add(40), o5);
            _mm256_storeu_ps(out.add(48), o6);
            _mm256_storeu_ps(out.add(56), o7);
        }
        _ => {
            for i in 0..head_dim {
                *out.add(i) += prob * *v.add(i);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product_head_avx512_d64(q: *const f32, k: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let q0 = _mm512_loadu_ps(q);
    let k0 = _mm512_loadu_ps(k);
    let mut acc0 = _mm512_mul_ps(q0, k0);

    let q1 = _mm512_loadu_ps(q.add(16));
    let k1 = _mm512_loadu_ps(k.add(16));
    acc0 = _mm512_fmadd_ps(q1, k1, acc0);

    let q2 = _mm512_loadu_ps(q.add(32));
    let k2 = _mm512_loadu_ps(k.add(32));
    acc0 = _mm512_fmadd_ps(q2, k2, acc0);

    let q3 = _mm512_loadu_ps(q.add(48));
    let k3 = _mm512_loadu_ps(k.add(48));
    acc0 = _mm512_fmadd_ps(q3, k3, acc0);

    _mm512_reduce_add_ps(acc0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn accumulate_value_head_avx512_d64(out: *mut f32, v: *const f32, prob: f32) {
    use std::arch::x86_64::*;
    let p_vec = _mm512_set1_ps(prob);

    let o0 = _mm512_fmadd_ps(p_vec, _mm512_loadu_ps(v), _mm512_loadu_ps(out));
    let o1 = _mm512_fmadd_ps(p_vec, _mm512_loadu_ps(v.add(16)), _mm512_loadu_ps(out.add(16)));
    let o2 = _mm512_fmadd_ps(p_vec, _mm512_loadu_ps(v.add(32)), _mm512_loadu_ps(out.add(32)));
    let o3 = _mm512_fmadd_ps(p_vec, _mm512_loadu_ps(v.add(48)), _mm512_loadu_ps(out.add(48)));

    _mm512_storeu_ps(out, o0);
    _mm512_storeu_ps(out.add(16), o1);
    _mm512_storeu_ps(out.add(32), o2);
    _mm512_storeu_ps(out.add(48), o3);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn attention_head_avx1(
    q_head: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    att_head: &mut [f32],
    xb2_head: &mut [f32],
    layer_kv_base: usize,
    q_offset: usize,
    pos: usize,
    dim: usize,
    head_dim: usize,
    inv_scale: f32,
) {
    let q_ptr = q_head.as_ptr();
    let has_avx512 = head_dim == 64 && std::is_x86_feature_detected!("avx512f");
    if has_avx512 {
        for t in 0..=pos {
            let k_ptr = key_cache.as_ptr().add(layer_kv_base + t * dim + q_offset);
            let score = dot_product_head_avx512_d64(q_ptr, k_ptr);
            *att_head.get_unchecked_mut(t) = score * inv_scale;
        }
    } else {
        for t in 0..=pos {
            let k_ptr = key_cache.as_ptr().add(layer_kv_base + t * dim + q_offset);
            let score = dot_product_head_avx(q_ptr, k_ptr, head_dim);
            *att_head.get_unchecked_mut(t) = score * inv_scale;
        }
    }
    softmax(&mut att_head[0..=pos]);
    xb2_head.fill(0.0);
    let out_ptr = xb2_head.as_mut_ptr();
    if has_avx512 {
        for t in 0..=pos {
            let prob = *att_head.get_unchecked(t);
            let v_ptr = value_cache.as_ptr().add(layer_kv_base + t * dim + q_offset);
            accumulate_value_head_avx512_d64(out_ptr, v_ptr, prob);
        }
    } else {
        for t in 0..=pos {
            let prob = *att_head.get_unchecked(t);
            let v_ptr = value_cache.as_ptr().add(layer_kv_base + t * dim + q_offset);
            accumulate_value_head_avx(out_ptr, v_ptr, prob, head_dim);
        }
    }
}

#[inline]
fn erf_f32(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-ax * ax).exp();
    sign * y
}

pub fn gelu(out: &mut [f32], x: &[f32]) {
    const INV_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for (o, &x0) in out.iter_mut().zip(x.iter()) {
        *o = 0.5 * x0 * (1.0 + erf_f32(x0 * INV_SQRT_2));
    }
}

pub fn softmax(x: &mut [f32]) {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32; for val in x.iter_mut() { *val = (*val - max_val).exp(); sum += *val; }
    let inv_sum = 1.0 / sum; for val in x.iter_mut() { *val *= inv_sum; }
}

pub struct SimpleRng { state: u64 }
impl SimpleRng {
    pub fn new(seed: u64) -> Self { Self { state: if seed == 0 { 0x123456789abcdef } else { seed } } }
    pub fn next_u32(&mut self) -> u32 { let mut x = self.state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.state = x; (x & 0xffffffff) as u32 }
    pub fn next_f32(&mut self) -> f32 { (self.next_u32() as f32) / (u32::MAX as f32) }
}

pub fn sample(logits: &mut [f32], temperature: f32, rng: &mut SimpleRng) -> usize {
    let _vocab_size = logits.len();
    for (i, logit) in logits.iter_mut().enumerate() {
        let is_standard = (4..=23).contains(&i); let is_eos = i == 2;
        if !is_standard && !is_eos { *logit = -f32::INFINITY; }
    }
    if temperature <= 0.0 {
        let mut max_val = logits[0]; let mut max_idx = 0;
        for (i, &val) in logits.iter().enumerate().skip(1) { if val > max_val { max_val = val; max_idx = i; } }
        return max_idx;
    }
    let inv_temp = 1.0 / temperature;
    for val in logits.iter_mut() { if *val != -f32::INFINITY { *val *= inv_temp; } }
    softmax(logits);
    let r = rng.next_f32(); let mut cumulative = 0.0;
    for (i, &prob) in logits.iter().enumerate() { if prob.is_nan() || prob == 0.0 { continue; } cumulative += prob; if r <= cumulative { return i; } }
    4
}

#[inline]
pub fn apply_rope(q: &mut [f32], k: &mut [f32], rope_cos: &[f32], rope_sin: &[f32], pos: usize, dim: usize, n_heads: usize) {
    let head_dim = dim / n_heads; let half_head = head_dim / 2;
    let table_offset = pos * half_head;
    let cos_slice = &rope_cos[table_offset..table_offset + half_head];
    let sin_slice = &rope_sin[table_offset..table_offset + half_head];
    for h in 0..n_heads {
        let head_offset = h * head_dim;
        for j in 0..half_head {
            let fcr = cos_slice[j]; let fci = sin_slice[j];
            let idx0 = head_offset + j;
            let idx1 = head_offset + j + half_head;
            let q0 = q[idx0]; let q1 = q[idx1];
            let k0 = k[idx0]; let k1 = k[idx1];
            q[idx0] = q0 * fcr - q1 * fci;
            q[idx1] = q1 * fcr + q0 * fci;
            k[idx0] = k0 * fcr - k1 * fci;
            k[idx1] = k1 * fcr + k0 * fci;
        }
    }
}

pub fn forward(token: usize, pos: usize, config: &EsmConfig, weights: &EsmWeights, state: &mut RunState) {
    forward_parallel(token, pos, config, weights, state, 1);
}

pub fn forward_parallel(
    token: usize,
    pos: usize,
    config: &EsmConfig,
    weights: &EsmWeights,
    state: &mut RunState,
    num_threads: usize,
) {
    let dim = config.dim;
    let embed_scale = 1.0 - 0.12;
    if token == 32 {
        state.x.fill(0.0);
    } else {
        let embed_offset = token * dim;
        for i in 0..dim {
            state.x[i] = weights.token_embedding_table[embed_offset + i] * embed_scale;
        }
    }
    let head_dim = dim / config.n_heads;
    let inv_scale = 1.0 / (head_dim as f32).sqrt();

    #[cfg(target_arch = "x86_64")]
    let has_avx = std::is_x86_feature_detected!("avx") && (head_dim == 16 || head_dim == 24 || head_dim == 32 || head_dim == 64);

    for l in 0..config.n_layers {
        let layer_dim_offset = l * dim;
        let layer_matrix_offset = l * dim * dim;

        layer_norm(
            &mut state.xb,
            &state.x,
            &weights.ln_1_weight[layer_dim_offset..layer_dim_offset + dim],
            &weights.ln_1_bias[layer_dim_offset..layer_dim_offset + dim],
        );

        matmul_parallel(
            &mut state.q,
            &state.xb,
            &weights.wq[layer_matrix_offset..layer_matrix_offset + (dim * dim)],
            &weights.bq[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );
        matmul_parallel(
            &mut state.k,
            &state.xb,
            &weights.wk[layer_matrix_offset..layer_matrix_offset + (dim * dim)],
            &weights.bk[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );
        matmul_parallel(
            &mut state.v,
            &state.xb,
            &weights.wv[layer_matrix_offset..layer_matrix_offset + (dim * dim)],
            &weights.bv[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );

        apply_rope(&mut state.q, &mut state.k, &state.rope_cos, &state.rope_sin, pos, dim, config.n_heads);

        let layer_kv_base = l * config.max_seq_len * dim;
        let kv_offset = layer_kv_base + pos * dim;
        state.key_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.k);
        state.value_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.v);

        for h in 0..config.n_heads {
            let q_offset = h * head_dim;
            let q_head = &state.q[q_offset..q_offset + head_dim];
            let att_offset = h * config.max_seq_len;
            let att_head = &mut state.att[att_offset..att_offset + config.max_seq_len];
            let xb2_head = &mut state.xb2[q_offset..q_offset + head_dim];

            #[cfg(target_arch = "x86_64")]
            if has_avx {
                unsafe {
                    attention_head_avx1(
                        q_head,
                        &state.key_cache,
                        &state.value_cache,
                        att_head,
                        xb2_head,
                        layer_kv_base,
                        q_offset,
                        pos,
                        dim,
                        head_dim,
                        inv_scale,
                    );
                }
                continue;
            }

            for t in 0..=pos {
                let cache_offset = layer_kv_base + t * dim + q_offset;
                let k_head = &state.key_cache[cache_offset..cache_offset + head_dim];
                let mut score = 0.0f32;
                for i in 0..head_dim {
                    score += q_head[i] * k_head[i];
                }
                att_head[t] = score * inv_scale;
            }
            softmax(&mut att_head[0..=pos]);
            xb2_head.fill(0.0);
            for t in 0..=pos {
                let prob = att_head[t];
                let cache_offset = layer_kv_base + t * dim + q_offset;
                let v_head = &state.value_cache[cache_offset..cache_offset + head_dim];
                for i in 0..head_dim {
                    xb2_head[i] += prob * v_head[i];
                }
            }
        }

        matmul_parallel(
            &mut state.xb,
            &state.xb2,
            &weights.wo[layer_matrix_offset..layer_matrix_offset + (dim * dim)],
            &weights.bo[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );

        for i in 0..dim {
            state.x[i] += state.xb[i];
        }

        layer_norm(
            &mut state.xb,
            &state.x,
            &weights.ln_2_weight[layer_dim_offset..layer_dim_offset + dim],
            &weights.ln_2_bias[layer_dim_offset..layer_dim_offset + dim],
        );

        let hidden_dim = config.hidden_dim;
        let fc1_w_offset = l * dim * hidden_dim;
        let fc1_b_offset = l * hidden_dim;
        matmul_parallel(
            &mut state.hb,
            &state.xb,
            &weights.fc1_weight[fc1_w_offset..fc1_w_offset + (dim * hidden_dim)],
            &weights.fc1_bias[fc1_b_offset..fc1_b_offset + hidden_dim],
            dim,
            hidden_dim,
            num_threads,
        );

        gelu(&mut state.hb2, &state.hb);

        let fc2_w_offset = l * hidden_dim * dim;
        matmul_parallel(
            &mut state.xb,
            &state.hb2,
            &weights.fc2_weight[fc2_w_offset..fc2_w_offset + (hidden_dim * dim)],
            &weights.fc2_bias[layer_dim_offset..layer_dim_offset + dim],
            hidden_dim,
            dim,
            num_threads,
        );

        for i in 0..dim {
            state.x[i] += state.xb[i];
        }
    }

    layer_norm(&mut state.xb, &state.x, &weights.ln_f_weight, &weights.ln_f_bias);
    project_lm_head_parallel(
        &mut state.logits,
        &state.xb,
        &weights.classifier_weights,
        &weights.token_embedding_table,
        config,
        &mut state.hb,
        &mut state.hb2,
        num_threads,
    );
}

pub fn forward_q8(
    token: usize,
    pos: usize,
    config: &EsmConfig,
    weights: &EsmWeightsQ8,
    state: &mut RunState,
    num_threads: usize,
) {
    let dim = config.dim;
    let head_dim = dim / config.n_heads;
    let inv_scale = 1.0 / (head_dim as f32).sqrt();

    let embed_scale = 1.0 - 0.12;
    if token == 32 {
        state.x.fill(0.0);
    } else {
        let embed_blocks_per_row = dim / 32;
        let embed_row = &weights.token_embedding_table[token * embed_blocks_per_row..(token + 1) * embed_blocks_per_row];
        for (b_idx, block) in embed_row.iter().enumerate() {
            block.dequantize_to(&mut state.x[b_idx * 32..(b_idx + 1) * 32]);
        }
        for i in 0..dim {
            state.x[i] *= embed_scale;
        }
    }

    let q8_matrix_blocks = dim * (dim / 32);
    let q8_fc1_blocks = config.hidden_dim * (dim / 32);
    let q8_fc2_blocks = dim * (config.hidden_dim / 32);

    #[cfg(target_arch = "x86_64")]
    let has_avx = std::is_x86_feature_detected!("avx") && (head_dim == 16 || head_dim == 24 || head_dim == 32 || head_dim == 64);

    for l in 0..config.n_layers {
        let layer_dim_offset = l * dim;
        let layer_matrix_q8 = l * q8_matrix_blocks;

        layer_norm(
            &mut state.xb,
            &state.x,
            &weights.ln_1_weight[layer_dim_offset..layer_dim_offset + dim],
            &weights.ln_1_bias[layer_dim_offset..layer_dim_offset + dim],
        );

        matmul_q8_parallel(
            &mut state.q,
            &state.xb,
            &weights.wq[layer_matrix_q8..layer_matrix_q8 + q8_matrix_blocks],
            &weights.bq[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );
        matmul_q8_parallel(
            &mut state.k,
            &state.xb,
            &weights.wk[layer_matrix_q8..layer_matrix_q8 + q8_matrix_blocks],
            &weights.bk[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );
        matmul_q8_parallel(
            &mut state.v,
            &state.xb,
            &weights.wv[layer_matrix_q8..layer_matrix_q8 + q8_matrix_blocks],
            &weights.bv[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );

        apply_rope(&mut state.q, &mut state.k, &state.rope_cos, &state.rope_sin, pos, dim, config.n_heads);

        let layer_kv_base = l * config.max_seq_len * dim;
        let kv_offset = layer_kv_base + pos * dim;
        state.key_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.k);
        state.value_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.v);

        for h in 0..config.n_heads {
            let q_offset = h * head_dim;
            let q_head = &state.q[q_offset..q_offset + head_dim];
            let att_offset = h * config.max_seq_len;
            let att_head = &mut state.att[att_offset..att_offset + config.max_seq_len];
            let xb2_head = &mut state.xb2[q_offset..q_offset + head_dim];

            #[cfg(target_arch = "x86_64")]
            if has_avx {
                unsafe {
                    attention_head_avx1(
                        q_head,
                        &state.key_cache,
                        &state.value_cache,
                        att_head,
                        xb2_head,
                        layer_kv_base,
                        q_offset,
                        pos,
                        dim,
                        head_dim,
                        inv_scale,
                    );
                }
                continue;
            }

            for t in 0..=pos {
                let cache_offset = layer_kv_base + t * dim + q_offset;
                let k_head = &state.key_cache[cache_offset..cache_offset + head_dim];
                let mut score = 0.0f32;
                for i in 0..head_dim {
                    score += q_head[i] * k_head[i];
                }
                att_head[t] = score * inv_scale;
            }
            softmax(&mut att_head[0..=pos]);
            xb2_head.fill(0.0);
            for t in 0..=pos {
                let prob = att_head[t];
                let cache_offset = layer_kv_base + t * dim + q_offset;
                let v_head = &state.value_cache[cache_offset..cache_offset + head_dim];
                for i in 0..head_dim {
                    xb2_head[i] += prob * v_head[i];
                }
            }
        }

        matmul_q8_parallel(
            &mut state.xb,
            &state.xb2,
            &weights.wo[layer_matrix_q8..layer_matrix_q8 + q8_matrix_blocks],
            &weights.bo[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );

        for i in 0..dim {
            state.x[i] += state.xb[i];
        }

        layer_norm(
            &mut state.xb,
            &state.x,
            &weights.ln_2_weight[layer_dim_offset..layer_dim_offset + dim],
            &weights.ln_2_bias[layer_dim_offset..layer_dim_offset + dim],
        );

        let fc1_q8 = l * q8_fc1_blocks;
        let fc1_b = l * config.hidden_dim;
        matmul_q8_parallel(
            &mut state.hb,
            &state.xb,
            &weights.fc1_weight[fc1_q8..fc1_q8 + q8_fc1_blocks],
            &weights.fc1_bias[fc1_b..fc1_b + config.hidden_dim],
            dim,
            config.hidden_dim,
            num_threads,
        );

        gelu(&mut state.hb2, &state.hb);

        let fc2_q8 = l * q8_fc2_blocks;
        matmul_q8_parallel(
            &mut state.xb,
            &state.hb2,
            &weights.fc2_weight[fc2_q8..fc2_q8 + q8_fc2_blocks],
            &weights.fc2_bias[layer_dim_offset..layer_dim_offset + dim],
            config.hidden_dim,
            dim,
            num_threads,
        );

        for i in 0..dim {
            state.x[i] += state.xb[i];
        }
    }

    layer_norm(&mut state.xb, &state.x, &weights.ln_f_weight, &weights.ln_f_bias);
    project_lm_head_q8(
        &mut state.logits,
        &state.xb,
        &weights.classifier_weights,
        &weights.token_embedding_table,
        config,
        &mut state.hb,
        &mut state.hb2,
        num_threads,
    );
}

pub fn project_lm_head(
    logits: &mut [f32],
    x_norm: &[f32],
    weights: &EsmWeights,
    config: &EsmConfig,
    dense_buf: &mut [f32],
    ln_buf: &mut [f32],
) {
    project_lm_head_parallel(
        logits,
        x_norm,
        &weights.classifier_weights,
        &weights.token_embedding_table,
        config,
        dense_buf,
        ln_buf,
        1,
    );
}

pub fn project_lm_head_parallel(
    logits: &mut [f32],
    x_norm: &[f32],
    classifier_weights: &[f32],
    embedding_table: &[f32],
    config: &EsmConfig,
    dense_buf: &mut [f32],
    ln_buf: &mut [f32],
    num_threads: usize,
) {
    let dim = config.dim;
    let required_len = 3 * dim + dim * dim + config.vocab_size * dim + config.vocab_size;
    if classifier_weights.len() >= required_len {
        let lm_ln_w = &classifier_weights[0..dim];
        let lm_ln_b = &classifier_weights[dim..2 * dim];
        let lm_dense_w = &classifier_weights[2 * dim..2 * dim + dim * dim];
        let lm_dense_b = &classifier_weights[2 * dim + dim * dim..3 * dim + dim * dim];
        let lm_dec_w = &classifier_weights[3 * dim + dim * dim..3 * dim + dim * dim + config.vocab_size * dim];
        let lm_dec_b = &classifier_weights[3 * dim + dim * dim + config.vocab_size * dim..3 * dim + dim * dim + config.vocab_size * dim + config.vocab_size];

        let dense_out = &mut dense_buf[0..dim];
        matmul_parallel(dense_out, x_norm, lm_dense_w, lm_dense_b, dim, dim, num_threads);
        let gelu_out = &mut ln_buf[0..dim];
        gelu(gelu_out, dense_out);
        layer_norm(dense_out, gelu_out, lm_ln_w, lm_ln_b);
        matmul_parallel(logits, dense_out, lm_dec_w, lm_dec_b, dim, config.vocab_size, num_threads);
    } else {
        const ZERO_BIAS: [f32; 33] = [0.0; 33];
        matmul_parallel(logits, x_norm, embedding_table, &ZERO_BIAS, dim, config.vocab_size, num_threads);
    }
}

pub fn project_lm_head_q8(
    logits: &mut [f32],
    x_norm: &[f32],
    classifier_weights: &[f32],
    embedding_table: &[BlockQ8_0],
    config: &EsmConfig,
    dense_buf: &mut [f32],
    ln_buf: &mut [f32],
    num_threads: usize,
) {
    let dim = config.dim;
    let required_len = 3 * dim + dim * dim + config.vocab_size * dim + config.vocab_size;
    if classifier_weights.len() >= required_len {
        let lm_ln_w = &classifier_weights[0..dim];
        let lm_ln_b = &classifier_weights[dim..2 * dim];
        let lm_dense_w = &classifier_weights[2 * dim..2 * dim + dim * dim];
        let lm_dense_b = &classifier_weights[2 * dim + dim * dim..3 * dim + dim * dim];
        let lm_dec_w = &classifier_weights[3 * dim + dim * dim..3 * dim + dim * dim + config.vocab_size * dim];
        let lm_dec_b = &classifier_weights[3 * dim + dim * dim + config.vocab_size * dim..3 * dim + dim * dim + config.vocab_size * dim + config.vocab_size];

        let dense_out = &mut dense_buf[0..dim];
        matmul_parallel(dense_out, x_norm, lm_dense_w, lm_dense_b, dim, dim, num_threads);
        let gelu_out = &mut ln_buf[0..dim];
        gelu(gelu_out, dense_out);
        layer_norm(dense_out, gelu_out, lm_ln_w, lm_ln_b);
        matmul_parallel(logits, dense_out, lm_dec_w, lm_dec_b, dim, config.vocab_size, num_threads);
    } else {
        const ZERO_BIAS: [f32; 33] = [0.0; 33];
        matmul_q8_parallel(logits, x_norm, embedding_table, &ZERO_BIAS, dim, config.vocab_size, num_threads);
    }
}

pub fn forward_q4(
    token: usize,
    pos: usize,
    config: &EsmConfig,
    weights: &EsmWeightsQ4,
    state: &mut RunState,
    num_threads: usize,
) {
    let dim = config.dim;
    let head_dim = dim / config.n_heads;
    let inv_scale = 1.0 / (head_dim as f32).sqrt();

    let embed_scale = 1.0 - 0.12;
    if token == 32 {
        state.x[0..dim].fill(0.0);
    } else {
        let embed_row = &weights.token_embedding_table[token * dim..(token + 1) * dim];
        for i in 0..dim {
            state.x[i] = embed_row[i] * embed_scale;
        }
    }

    let q4_matrix_blocks = dim * (dim / 32);
    let q4_fc1_blocks = config.hidden_dim * (dim / 32);
    let q4_fc2_blocks = dim * (config.hidden_dim / 32);

    #[cfg(target_arch = "x86_64")]
    let has_avx = std::is_x86_feature_detected!("avx") && (head_dim == 16 || head_dim == 24 || head_dim == 32 || head_dim == 64);

    for l in 0..config.n_layers {
        let layer_dim_offset = l * dim;
        let layer_matrix_q4 = l * q4_matrix_blocks;

        layer_norm(
            &mut state.xb,
            &state.x,
            &weights.ln_1_weight[layer_dim_offset..layer_dim_offset + dim],
            &weights.ln_1_bias[layer_dim_offset..layer_dim_offset + dim],
        );

        matmul_q4_parallel(
            &mut state.q,
            &state.xb,
            &weights.wq[layer_matrix_q4..layer_matrix_q4 + q4_matrix_blocks],
            &weights.bq[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );
        matmul_q4_parallel(
            &mut state.k,
            &state.xb,
            &weights.wk[layer_matrix_q4..layer_matrix_q4 + q4_matrix_blocks],
            &weights.bk[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );
        matmul_q4_parallel(
            &mut state.v,
            &state.xb,
            &weights.wv[layer_matrix_q4..layer_matrix_q4 + q4_matrix_blocks],
            &weights.bv[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );

        apply_rope(&mut state.q, &mut state.k, &state.rope_cos, &state.rope_sin, pos, dim, config.n_heads);

        let layer_kv_base = l * config.max_seq_len * dim;
        let kv_offset = layer_kv_base + pos * dim;
        state.key_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.k);
        state.value_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.v);

        for h in 0..config.n_heads {
            let q_offset = h * head_dim;
            let q_head = &state.q[q_offset..q_offset + head_dim];
            let att_offset = h * config.max_seq_len;
            let att_head = &mut state.att[att_offset..att_offset + config.max_seq_len];
            let xb2_head = &mut state.xb2[q_offset..q_offset + head_dim];

            #[cfg(target_arch = "x86_64")]
            if has_avx {
                unsafe {
                    attention_head_avx1(
                        q_head,
                        &state.key_cache,
                        &state.value_cache,
                        att_head,
                        xb2_head,
                        layer_kv_base,
                        q_offset,
                        pos,
                        dim,
                        head_dim,
                        inv_scale,
                    );
                }
                continue;
            }

            for t in 0..=pos {
                let cache_offset = layer_kv_base + t * dim + q_offset;
                let k_head = &state.key_cache[cache_offset..cache_offset + head_dim];
                let mut score = 0.0f32;
                for i in 0..head_dim {
                    score += q_head[i] * k_head[i];
                }
                att_head[t] = score * inv_scale;
            }
            softmax(&mut att_head[0..=pos]);
            xb2_head.fill(0.0);
            for t in 0..=pos {
                let prob = att_head[t];
                let cache_offset = layer_kv_base + t * dim + q_offset;
                let v_head = &state.value_cache[cache_offset..cache_offset + head_dim];
                for i in 0..head_dim {
                    xb2_head[i] += prob * v_head[i];
                }
            }
        }

        matmul_q4_parallel(
            &mut state.xb,
            &state.xb2,
            &weights.wo[layer_matrix_q4..layer_matrix_q4 + q4_matrix_blocks],
            &weights.bo[layer_dim_offset..layer_dim_offset + dim],
            dim,
            dim,
            num_threads,
        );

        for i in 0..dim {
            state.x[i] += state.xb[i];
        }

        layer_norm(
            &mut state.xb,
            &state.x,
            &weights.ln_2_weight[layer_dim_offset..layer_dim_offset + dim],
            &weights.ln_2_bias[layer_dim_offset..layer_dim_offset + dim],
        );

        let fc1_q4 = l * q4_fc1_blocks;
        let fc1_b = l * config.hidden_dim;
        matmul_q4_parallel(
            &mut state.hb,
            &state.xb,
            &weights.fc1_weight[fc1_q4..fc1_q4 + q4_fc1_blocks],
            &weights.fc1_bias[fc1_b..fc1_b + config.hidden_dim],
            dim,
            config.hidden_dim,
            num_threads,
        );

        gelu(&mut state.hb2, &state.hb);

        let fc2_q4 = l * q4_fc2_blocks;
        matmul_q4_parallel(
            &mut state.xb,
            &state.hb2,
            &weights.fc2_weight[fc2_q4..fc2_q4 + q4_fc2_blocks],
            &weights.fc2_bias[layer_dim_offset..layer_dim_offset + dim],
            config.hidden_dim,
            dim,
            num_threads,
        );

        for i in 0..dim {
            state.x[i] += state.xb[i];
        }
    }

    layer_norm(&mut state.xb, &state.x, &weights.ln_f_weight, &weights.ln_f_bias);
    project_lm_head_q4(
        &mut state.logits,
        &state.xb,
        &weights.classifier_weights,
        &weights.token_embedding_table,
        config,
        &mut state.hb,
        &mut state.hb2,
        num_threads,
    );
}

pub fn project_lm_head_q4(
    logits: &mut [f32],
    x_norm: &[f32],
    classifier_weights: &[f32],
    embedding_table: &[f32],
    config: &EsmConfig,
    dense_buf: &mut [f32],
    ln_buf: &mut [f32],
    num_threads: usize,
) {
    let dim = config.dim;
    let required_len = 3 * dim + dim * dim + config.vocab_size * dim + config.vocab_size;
    if classifier_weights.len() >= required_len {
        let lm_ln_w = &classifier_weights[0..dim];
        let lm_ln_b = &classifier_weights[dim..2 * dim];
        let lm_dense_w = &classifier_weights[2 * dim..2 * dim + dim * dim];
        let lm_dense_b = &classifier_weights[2 * dim + dim * dim..3 * dim + dim * dim];
        let lm_dec_w = &classifier_weights[3 * dim + dim * dim..3 * dim + dim * dim + config.vocab_size * dim];
        let lm_dec_b = &classifier_weights[3 * dim + dim * dim + config.vocab_size * dim..3 * dim + dim * dim + config.vocab_size * dim + config.vocab_size];

        let dense_out = &mut dense_buf[0..dim];
        matmul_parallel(dense_out, x_norm, lm_dense_w, lm_dense_b, dim, dim, num_threads);
        let gelu_out = &mut ln_buf[0..dim];
        gelu(gelu_out, dense_out);
        layer_norm(dense_out, gelu_out, lm_ln_w, lm_ln_b);
        matmul_parallel(logits, dense_out, lm_dec_w, lm_dec_b, dim, config.vocab_size, num_threads);
    } else {
        const ZERO_BIAS: [f32; 33] = [0.0; 33];
        matmul_parallel(logits, x_norm, embedding_table, &ZERO_BIAS, dim, config.vocab_size, num_threads);
    }
}


pub fn log_softmax_canonical(logits: &[f32]) -> [f32; 33] {
    let mut out = [-f32::INFINITY; 33];
    let mut max_val = f32::NEG_INFINITY;
    for i in 4..=23 {
        if logits[i] > max_val { max_val = logits[i]; }
    }
    let mut sum_exp = 0.0f32;
    for i in 4..=23 {
        sum_exp += (logits[i] - max_val).exp();
    }
    let log_sum_exp = max_val + sum_exp.ln();
    for i in 4..=23 {
        out[i] = logits[i] - log_sum_exp;
    }
    out
}

pub fn forward_bidirectional(
    tokens: &[usize],
    config: &EsmConfig,
    weights: &EsmWeights,
    state: &mut BidirectionalRunState,
) {
    forward_bidirectional_parallel(tokens, config, weights, state, 1);
}

pub fn forward_bidirectional_parallel(
    tokens: &[usize],
    config: &EsmConfig,
    weights: &EsmWeights,
    state: &mut BidirectionalRunState,
    num_threads: usize,
) {
    forward_bidirectional_parallel_ext(tokens, config, weights, state, num_threads, None);
}

pub fn forward_bidirectional_parallel_ext(
    tokens: &[usize],
    config: &EsmConfig,
    weights: &EsmWeights,
    state: &mut BidirectionalRunState,
    num_threads: usize,
    mut on_layer: Option<&mut dyn FnMut(usize, &[f32])>,
) {
    let seq_len = tokens.len();
    assert!(seq_len <= state.max_len, "Sequence length {} exceeds max_len {}", seq_len, state.max_len);
    let dim = config.dim;
    let head_dim = dim / config.n_heads;
    let inv_scale = 1.0 / (head_dim as f32).sqrt();
    let hidden_dim = config.hidden_dim;

    #[cfg(target_arch = "x86_64")]
    let has_avx512 = head_dim == 64 && std::is_x86_feature_detected!("avx512f");
    #[cfg(target_arch = "x86_64")]
    let has_avx = std::is_x86_feature_detected!("avx") && (head_dim == 16 || head_dim == 24 || head_dim == 32 || head_dim == 64);

    let mask_count = tokens.iter().filter(|&&t| t == 32).count();
    let mask_ratio_observed = mask_count as f32 / seq_len as f32;
    let embed_scale = if mask_ratio_observed >= 1.0 { 0.0 } else { (1.0 - 0.12) / (1.0 - mask_ratio_observed) };

    for p in 0..seq_len {
        let token = tokens[p];
        if token == 32 {
            state.x[p * dim..(p + 1) * dim].fill(0.0);
        } else {
            let embed_offset = token * dim;
            for i in 0..dim {
                state.x[p * dim + i] = weights.token_embedding_table[embed_offset + i] * embed_scale;
            }
        }
    }

    for l in 0..config.n_layers {
        let layer_dim_offset = l * dim;
        let layer_matrix_offset = l * dim * dim;
        let fc1_w_offset = l * dim * hidden_dim;
        let fc1_b_offset = l * hidden_dim;
        let fc2_w_offset = l * hidden_dim * dim;

        for p in 0..seq_len {
            let xp = &state.x[p * dim..(p + 1) * dim];
            layer_norm(&mut state.xb, xp, &weights.ln_1_weight[layer_dim_offset..layer_dim_offset + dim], &weights.ln_1_bias[layer_dim_offset..layer_dim_offset + dim]);
            let q_slice = &mut state.q[p * dim..(p + 1) * dim];
            matmul_parallel(q_slice, &state.xb, &weights.wq[layer_matrix_offset..layer_matrix_offset + dim * dim], &weights.bq[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            let k_slice = &mut state.k[p * dim..(p + 1) * dim];
            matmul_parallel(k_slice, &state.xb, &weights.wk[layer_matrix_offset..layer_matrix_offset + dim * dim], &weights.bk[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            let v_slice = &mut state.v[p * dim..(p + 1) * dim];
            matmul_parallel(v_slice, &state.xb, &weights.wv[layer_matrix_offset..layer_matrix_offset + dim * dim], &weights.bv[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            apply_rope(q_slice, k_slice, &state.rope_cos, &state.rope_sin, p, dim, config.n_heads);
        }

        let is_last_layer = l == config.n_layers - 1;
        if is_last_layer {
            state.final_layer_att[0..seq_len * seq_len].fill(0.0);
        }

        for p in 0..seq_len {
            for h in 0..config.n_heads {
                let q_offset = p * dim + h * head_dim;
                let q_head = &state.q[q_offset..q_offset + head_dim];
                let att_head = &mut state.att[h * state.max_len..h * state.max_len + seq_len];

                #[cfg(target_arch = "x86_64")]
                {
                    if has_avx512 {
                        let q_ptr = q_head.as_ptr();
                        for t in 0..seq_len {
                            let k_offset = t * dim + h * head_dim;
                            let k_ptr = unsafe { state.k.as_ptr().add(k_offset) };
                            let score = unsafe { dot_product_head_avx512_d64(q_ptr, k_ptr) };
                            att_head[t] = score * inv_scale;
                        }
                    } else if has_avx {
                        let q_ptr = q_head.as_ptr();
                        for t in 0..seq_len {
                            let k_offset = t * dim + h * head_dim;
                            let k_ptr = unsafe { state.k.as_ptr().add(k_offset) };
                            let score = unsafe { dot_product_head_avx(q_ptr, k_ptr, head_dim) };
                            att_head[t] = score * inv_scale;
                        }
                    } else {
                        for t in 0..seq_len {
                            let k_offset = t * dim + h * head_dim;
                            let k_head = &state.k[k_offset..k_offset + head_dim];
                            let mut score = 0.0f32;
                            for i in 0..head_dim { score += q_head[i] * k_head[i]; }
                            att_head[t] = score * inv_scale;
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for t in 0..seq_len {
                        let k_offset = t * dim + h * head_dim;
                        let k_head = &state.k[k_offset..k_offset + head_dim];
                        let mut score = 0.0f32;
                        for i in 0..head_dim { score += q_head[i] * k_head[i]; }
                        att_head[t] = score * inv_scale;
                    }
                }
                softmax(att_head);

                if is_last_layer {
                    let head_weight = 1.0 / config.n_heads as f32;
                    for t in 0..seq_len {
                        state.final_layer_att[p * seq_len + t] += att_head[t] * head_weight;
                    }
                }

                let out_head = &mut state.out_att[p * dim + h * head_dim..p * dim + (h + 1) * head_dim];
                out_head.fill(0.0);
                #[cfg(target_arch = "x86_64")]
                {
                    if has_avx512 {
                        let out_ptr = out_head.as_mut_ptr();
                        for t in 0..seq_len {
                            let prob = att_head[t];
                            let v_offset = t * dim + h * head_dim;
                            let v_ptr = unsafe { state.v.as_ptr().add(v_offset) };
                            unsafe { accumulate_value_head_avx512_d64(out_ptr, v_ptr, prob); }
                        }
                    } else if has_avx {
                        let out_ptr = out_head.as_mut_ptr();
                        for t in 0..seq_len {
                            let prob = att_head[t];
                            let v_offset = t * dim + h * head_dim;
                            let v_ptr = unsafe { state.v.as_ptr().add(v_offset) };
                            unsafe { accumulate_value_head_avx(out_ptr, v_ptr, prob, head_dim); }
                        }
                    } else {
                        for t in 0..seq_len {
                            let prob = att_head[t];
                            let v_offset = t * dim + h * head_dim;
                            let v_head = &state.v[v_offset..v_offset + head_dim];
                            for i in 0..head_dim { out_head[i] += prob * v_head[i]; }
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for t in 0..seq_len {
                        let prob = att_head[t];
                        let v_offset = t * dim + h * head_dim;
                        let v_head = &state.v[v_offset..v_offset + head_dim];
                        for i in 0..head_dim { out_head[i] += prob * v_head[i]; }
                    }
                }
            }

            let out_slice = &state.out_att[p * dim..(p + 1) * dim];
            matmul_parallel(&mut state.xb, out_slice, &weights.wo[layer_matrix_offset..layer_matrix_offset + dim * dim], &weights.bo[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            for i in 0..dim { state.x[p * dim + i] += state.xb[i]; }

            let xp = &state.x[p * dim..(p + 1) * dim];
            layer_norm(&mut state.xb, xp, &weights.ln_2_weight[layer_dim_offset..layer_dim_offset + dim], &weights.ln_2_bias[layer_dim_offset..layer_dim_offset + dim]);
            matmul_parallel(&mut state.hb, &state.xb, &weights.fc1_weight[fc1_w_offset..fc1_w_offset + dim * hidden_dim], &weights.fc1_bias[fc1_b_offset..fc1_b_offset + hidden_dim], dim, hidden_dim, num_threads);
            gelu(&mut state.hb2, &state.hb);
            matmul_parallel(&mut state.xb, &state.hb2, &weights.fc2_weight[fc2_w_offset..fc2_w_offset + hidden_dim * dim], &weights.fc2_bias[layer_dim_offset..layer_dim_offset + dim], hidden_dim, dim, num_threads);
            for i in 0..dim { state.x[p * dim + i] += state.xb[i]; }
        }

        if let Some(ref mut cb) = on_layer {
            cb(l, &state.x[0..seq_len * dim]);
        }
    }

    for p in 0..seq_len {
        let xp = &state.x[p * dim..(p + 1) * dim];
        layer_norm(&mut state.xb, xp, &weights.ln_f_weight, &weights.ln_f_bias);
        state.x[p * dim..(p + 1) * dim].copy_from_slice(&state.xb);
        let logits_p = &mut state.logits[p * config.vocab_size..(p + 1) * config.vocab_size];
        project_lm_head_parallel(logits_p, &state.xb, &weights.classifier_weights, &weights.token_embedding_table, config, &mut state.hb, &mut state.hb2, num_threads);
    }
}

pub fn forward_bidirectional_q8(
    tokens: &[usize],
    config: &EsmConfig,
    weights: &EsmWeightsQ8,
    state: &mut BidirectionalRunState,
    num_threads: usize,
) {
    let seq_len = tokens.len();
    assert!(seq_len <= state.max_len, "Sequence length {} exceeds max_len {}", seq_len, state.max_len);
    let dim = config.dim;
    let head_dim = dim / config.n_heads;
    let inv_scale = 1.0 / (head_dim as f32).sqrt();
    let hidden_dim = config.hidden_dim;

    #[cfg(target_arch = "x86_64")]
    let has_avx512 = head_dim == 64 && std::is_x86_feature_detected!("avx512f");
    #[cfg(target_arch = "x86_64")]
    let has_avx = std::is_x86_feature_detected!("avx") && (head_dim == 16 || head_dim == 24 || head_dim == 32 || head_dim == 64);

    let mask_count = tokens.iter().filter(|&&t| t == 32).count();
    let mask_ratio_observed = mask_count as f32 / seq_len as f32;
    let embed_scale = if mask_ratio_observed >= 1.0 { 0.0 } else { (1.0 - 0.12) / (1.0 - mask_ratio_observed) };

    let embed_blocks_per_row = dim / 32;
    for p in 0..seq_len {
        let token = tokens[p];
        if token == 32 {
            state.x[p * dim..(p + 1) * dim].fill(0.0);
        } else {
            let row_blocks = &weights.token_embedding_table[token * embed_blocks_per_row..(token + 1) * embed_blocks_per_row];
            for (b_idx, block) in row_blocks.iter().enumerate() {
                block.dequantize_to(&mut state.x[p * dim + b_idx * 32..p * dim + (b_idx + 1) * 32]);
            }
            for i in 0..dim {
                state.x[p * dim + i] *= embed_scale;
            }
        }
    }

    let q8_matrix_blocks = dim * (dim / 32);
    let q8_fc1_blocks = hidden_dim * (dim / 32);
    let q8_fc2_blocks = dim * (hidden_dim / 32);

    for l in 0..config.n_layers {
        let layer_dim_offset = l * dim;
        let layer_matrix_q8 = l * q8_matrix_blocks;
        let fc1_q8 = l * q8_fc1_blocks;
        let fc1_b = l * hidden_dim;
        let fc2_q8 = l * q8_fc2_blocks;

        for p in 0..seq_len {
            let xp = &state.x[p * dim..(p + 1) * dim];
            layer_norm(&mut state.xb, xp, &weights.ln_1_weight[layer_dim_offset..layer_dim_offset + dim], &weights.ln_1_bias[layer_dim_offset..layer_dim_offset + dim]);
            let q_slice = &mut state.q[p * dim..(p + 1) * dim];
            matmul_q8_parallel(q_slice, &state.xb, &weights.wq[layer_matrix_q8..layer_matrix_q8 + q8_matrix_blocks], &weights.bq[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            let k_slice = &mut state.k[p * dim..(p + 1) * dim];
            matmul_q8_parallel(k_slice, &state.xb, &weights.wk[layer_matrix_q8..layer_matrix_q8 + q8_matrix_blocks], &weights.bk[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            let v_slice = &mut state.v[p * dim..(p + 1) * dim];
            matmul_q8_parallel(v_slice, &state.xb, &weights.wv[layer_matrix_q8..layer_matrix_q8 + q8_matrix_blocks], &weights.bv[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            apply_rope(q_slice, k_slice, &state.rope_cos, &state.rope_sin, p, dim, config.n_heads);
        }

        let is_last_layer = l == config.n_layers - 1;
        if is_last_layer {
            state.final_layer_att[0..seq_len * seq_len].fill(0.0);
        }

        for p in 0..seq_len {
            for h in 0..config.n_heads {
                let q_offset = p * dim + h * head_dim;
                let q_head = &state.q[q_offset..q_offset + head_dim];
                let att_head = &mut state.att[h * state.max_len..h * state.max_len + seq_len];

                #[cfg(target_arch = "x86_64")]
                {
                    if has_avx512 {
                        let q_ptr = q_head.as_ptr();
                        for t in 0..seq_len {
                            let k_offset = t * dim + h * head_dim;
                            let k_ptr = unsafe { state.k.as_ptr().add(k_offset) };
                            let score = unsafe { dot_product_head_avx512_d64(q_ptr, k_ptr) };
                            att_head[t] = score * inv_scale;
                        }
                    } else if has_avx {
                        let q_ptr = q_head.as_ptr();
                        for t in 0..seq_len {
                            let k_offset = t * dim + h * head_dim;
                            let k_ptr = unsafe { state.k.as_ptr().add(k_offset) };
                            let score = unsafe { dot_product_head_avx(q_ptr, k_ptr, head_dim) };
                            att_head[t] = score * inv_scale;
                        }
                    } else {
                        for t in 0..seq_len {
                            let k_offset = t * dim + h * head_dim;
                            let k_head = &state.k[k_offset..k_offset + head_dim];
                            let mut score = 0.0f32;
                            for i in 0..head_dim { score += q_head[i] * k_head[i]; }
                            att_head[t] = score * inv_scale;
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for t in 0..seq_len {
                        let k_offset = t * dim + h * head_dim;
                        let k_head = &state.k[k_offset..k_offset + head_dim];
                        let mut score = 0.0f32;
                        for i in 0..head_dim { score += q_head[i] * k_head[i]; }
                        att_head[t] = score * inv_scale;
                    }
                }
                softmax(att_head);

                if is_last_layer {
                    let head_weight = 1.0 / config.n_heads as f32;
                    for t in 0..seq_len {
                        state.final_layer_att[p * seq_len + t] += att_head[t] * head_weight;
                    }
                }

                let out_head = &mut state.out_att[p * dim + h * head_dim..p * dim + (h + 1) * head_dim];
                out_head.fill(0.0);
                #[cfg(target_arch = "x86_64")]
                {
                    if has_avx512 {
                        let out_ptr = out_head.as_mut_ptr();
                        for t in 0..seq_len {
                            let prob = att_head[t];
                            let v_offset = t * dim + h * head_dim;
                            let v_ptr = unsafe { state.v.as_ptr().add(v_offset) };
                            unsafe { accumulate_value_head_avx512_d64(out_ptr, v_ptr, prob); }
                        }
                    } else if has_avx {
                        let out_ptr = out_head.as_mut_ptr();
                        for t in 0..seq_len {
                            let prob = att_head[t];
                            let v_offset = t * dim + h * head_dim;
                            let v_ptr = unsafe { state.v.as_ptr().add(v_offset) };
                            unsafe { accumulate_value_head_avx(out_ptr, v_ptr, prob, head_dim); }
                        }
                    } else {
                        for t in 0..seq_len {
                            let prob = att_head[t];
                            let v_offset = t * dim + h * head_dim;
                            let v_head = &state.v[v_offset..v_offset + head_dim];
                            for i in 0..head_dim { out_head[i] += prob * v_head[i]; }
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for t in 0..seq_len {
                        let prob = att_head[t];
                        let v_offset = t * dim + h * head_dim;
                        let v_head = &state.v[v_offset..v_offset + head_dim];
                        for i in 0..head_dim { out_head[i] += prob * v_head[i]; }
                    }
                }
            }

            let out_slice = &state.out_att[p * dim..(p + 1) * dim];
            matmul_q8_parallel(&mut state.xb, out_slice, &weights.wo[layer_matrix_q8..layer_matrix_q8 + q8_matrix_blocks], &weights.bo[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            for i in 0..dim { state.x[p * dim + i] += state.xb[i]; }

            let xp = &state.x[p * dim..(p + 1) * dim];
            layer_norm(&mut state.xb, xp, &weights.ln_2_weight[layer_dim_offset..layer_dim_offset + dim], &weights.ln_2_bias[layer_dim_offset..layer_dim_offset + dim]);
            matmul_q8_parallel(&mut state.hb, &state.xb, &weights.fc1_weight[fc1_q8..fc1_q8 + q8_fc1_blocks], &weights.fc1_bias[fc1_b..fc1_b + hidden_dim], dim, hidden_dim, num_threads);
            gelu(&mut state.hb2, &state.hb);
            matmul_q8_parallel(&mut state.xb, &state.hb2, &weights.fc2_weight[fc2_q8..fc2_q8 + q8_fc2_blocks], &weights.fc2_bias[layer_dim_offset..layer_dim_offset + dim], hidden_dim, dim, num_threads);
            for i in 0..dim { state.x[p * dim + i] += state.xb[i]; }
        }
    }

    for p in 0..seq_len {
        let xp = &state.x[p * dim..(p + 1) * dim];
        layer_norm(&mut state.xb, xp, &weights.ln_f_weight, &weights.ln_f_bias);
        state.x[p * dim..(p + 1) * dim].copy_from_slice(&state.xb);
        let logits_p = &mut state.logits[p * config.vocab_size..(p + 1) * config.vocab_size];
        project_lm_head_q8(logits_p, &state.xb, &weights.classifier_weights, &weights.token_embedding_table, config, &mut state.hb, &mut state.hb2, num_threads);
    }
}

pub fn forward_bidirectional_q4(
    tokens: &[usize],
    config: &EsmConfig,
    weights: &EsmWeightsQ4,
    state: &mut BidirectionalRunState,
    num_threads: usize,
) {
    forward_bidirectional_q4_ext(tokens, config, weights, state, num_threads, None);
}

pub fn forward_bidirectional_q4_ext(
    tokens: &[usize],
    config: &EsmConfig,
    weights: &EsmWeightsQ4,
    state: &mut BidirectionalRunState,
    num_threads: usize,
    mut on_layer: Option<&mut dyn FnMut(usize, &[f32])>,
) {
    let seq_len = tokens.len();
    assert!(seq_len <= state.max_len, "Sequence length {} exceeds max_len {}", seq_len, state.max_len);
    let dim = config.dim;
    let head_dim = dim / config.n_heads;
    let inv_scale = 1.0 / (head_dim as f32).sqrt();
    let hidden_dim = config.hidden_dim;

    #[cfg(target_arch = "x86_64")]
    let has_avx512 = head_dim == 64 && std::is_x86_feature_detected!("avx512f");
    #[cfg(target_arch = "x86_64")]
    let has_avx = std::is_x86_feature_detected!("avx") && (head_dim == 16 || head_dim == 24 || head_dim == 32 || head_dim == 64);

    let mask_count = tokens.iter().filter(|&&t| t == 32).count();
    let mask_ratio_observed = mask_count as f32 / seq_len as f32;
    let embed_scale = if mask_ratio_observed >= 1.0 { 0.0 } else { (1.0 - 0.12) / (1.0 - mask_ratio_observed) };

    for p in 0..seq_len {
        let token = tokens[p];
        if token == 32 {
            state.x[p * dim..(p + 1) * dim].fill(0.0);
        } else {
            let embed_row = &weights.token_embedding_table[token * dim..(token + 1) * dim];
            for i in 0..dim {
                state.x[p * dim + i] = embed_row[i] * embed_scale;
            }
        }
    }

    let q4_matrix_blocks = dim * (dim / 32);
    let q4_fc1_blocks = hidden_dim * (dim / 32);
    let q4_fc2_blocks = dim * (hidden_dim / 32);

    for l in 0..config.n_layers {
        let layer_dim_offset = l * dim;
        let layer_matrix_q4 = l * q4_matrix_blocks;
        let fc1_q4 = l * q4_fc1_blocks;
        let fc1_b = l * hidden_dim;
        let fc2_q4 = l * q4_fc2_blocks;

        for p in 0..seq_len {
            let xp = &state.x[p * dim..(p + 1) * dim];
            layer_norm(&mut state.xb, xp, &weights.ln_1_weight[layer_dim_offset..layer_dim_offset + dim], &weights.ln_1_bias[layer_dim_offset..layer_dim_offset + dim]);
            let q_slice = &mut state.q[p * dim..(p + 1) * dim];
            matmul_q4_parallel(q_slice, &state.xb, &weights.wq[layer_matrix_q4..layer_matrix_q4 + q4_matrix_blocks], &weights.bq[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            let k_slice = &mut state.k[p * dim..(p + 1) * dim];
            matmul_q4_parallel(k_slice, &state.xb, &weights.wk[layer_matrix_q4..layer_matrix_q4 + q4_matrix_blocks], &weights.bk[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            let v_slice = &mut state.v[p * dim..(p + 1) * dim];
            matmul_q4_parallel(v_slice, &state.xb, &weights.wv[layer_matrix_q4..layer_matrix_q4 + q4_matrix_blocks], &weights.bv[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            apply_rope(q_slice, k_slice, &state.rope_cos, &state.rope_sin, p, dim, config.n_heads);
        }

        let is_last_layer = l == config.n_layers - 1;
        if is_last_layer {
            state.final_layer_att[0..seq_len * seq_len].fill(0.0);
        }

        for p in 0..seq_len {
            for h in 0..config.n_heads {
                let q_offset = p * dim + h * head_dim;
                let q_head = &state.q[q_offset..q_offset + head_dim];
                let att_head = &mut state.att[h * state.max_len..h * state.max_len + seq_len];

                #[cfg(target_arch = "x86_64")]
                {
                    if has_avx512 {
                        let q_ptr = q_head.as_ptr();
                        for t in 0..seq_len {
                            let k_offset = t * dim + h * head_dim;
                            let k_ptr = unsafe { state.k.as_ptr().add(k_offset) };
                            let score = unsafe { dot_product_head_avx512_d64(q_ptr, k_ptr) };
                            att_head[t] = score * inv_scale;
                        }
                    } else if has_avx {
                        let q_ptr = q_head.as_ptr();
                        for t in 0..seq_len {
                            let k_offset = t * dim + h * head_dim;
                            let k_ptr = unsafe { state.k.as_ptr().add(k_offset) };
                            let score = unsafe { dot_product_head_avx(q_ptr, k_ptr, head_dim) };
                            att_head[t] = score * inv_scale;
                        }
                    } else {
                        for t in 0..seq_len {
                            let k_offset = t * dim + h * head_dim;
                            let k_head = &state.k[k_offset..k_offset + head_dim];
                            let mut score = 0.0f32;
                            for i in 0..head_dim { score += q_head[i] * k_head[i]; }
                            att_head[t] = score * inv_scale;
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for t in 0..seq_len {
                        let k_offset = t * dim + h * head_dim;
                        let k_head = &state.k[k_offset..k_offset + head_dim];
                        let mut score = 0.0f32;
                        for i in 0..head_dim { score += q_head[i] * k_head[i]; }
                        att_head[t] = score * inv_scale;
                    }
                }
                softmax(att_head);

                if is_last_layer {
                    let head_weight = 1.0 / config.n_heads as f32;
                    for t in 0..seq_len {
                        state.final_layer_att[p * seq_len + t] += att_head[t] * head_weight;
                    }
                }

                let out_head = &mut state.out_att[p * dim + h * head_dim..p * dim + (h + 1) * head_dim];
                out_head.fill(0.0);
                #[cfg(target_arch = "x86_64")]
                {
                    if has_avx512 {
                        let out_ptr = out_head.as_mut_ptr();
                        for t in 0..seq_len {
                            let prob = att_head[t];
                            let v_offset = t * dim + h * head_dim;
                            let v_ptr = unsafe { state.v.as_ptr().add(v_offset) };
                            unsafe { accumulate_value_head_avx512_d64(out_ptr, v_ptr, prob); }
                        }
                    } else if has_avx {
                        let out_ptr = out_head.as_mut_ptr();
                        for t in 0..seq_len {
                            let prob = att_head[t];
                            let v_offset = t * dim + h * head_dim;
                            let v_ptr = unsafe { state.v.as_ptr().add(v_offset) };
                            unsafe { accumulate_value_head_avx(out_ptr, v_ptr, prob, head_dim); }
                        }
                    } else {
                        for t in 0..seq_len {
                            let prob = att_head[t];
                            let v_offset = t * dim + h * head_dim;
                            let v_head = &state.v[v_offset..v_offset + head_dim];
                            for i in 0..head_dim { out_head[i] += prob * v_head[i]; }
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for t in 0..seq_len {
                        let prob = att_head[t];
                        let v_offset = t * dim + h * head_dim;
                        let v_head = &state.v[v_offset..v_offset + head_dim];
                        for i in 0..head_dim { out_head[i] += prob * v_head[i]; }
                    }
                }
            }

            let out_slice = &state.out_att[p * dim..(p + 1) * dim];
            matmul_q4_parallel(&mut state.xb, out_slice, &weights.wo[layer_matrix_q4..layer_matrix_q4 + q4_matrix_blocks], &weights.bo[layer_dim_offset..layer_dim_offset + dim], dim, dim, num_threads);
            for i in 0..dim { state.x[p * dim + i] += state.xb[i]; }

            let xp = &state.x[p * dim..(p + 1) * dim];
            layer_norm(&mut state.xb, xp, &weights.ln_2_weight[layer_dim_offset..layer_dim_offset + dim], &weights.ln_2_bias[layer_dim_offset..layer_dim_offset + dim]);
            matmul_q4_parallel(&mut state.hb, &state.xb, &weights.fc1_weight[fc1_q4..fc1_q4 + q4_fc1_blocks], &weights.fc1_bias[fc1_b..fc1_b + hidden_dim], dim, hidden_dim, num_threads);
            gelu(&mut state.hb2, &state.hb);
            matmul_q4_parallel(&mut state.xb, &state.hb2, &weights.fc2_weight[fc2_q4..fc2_q4 + q4_fc2_blocks], &weights.fc2_bias[layer_dim_offset..layer_dim_offset + dim], hidden_dim, dim, num_threads);
            for i in 0..dim { state.x[p * dim + i] += state.xb[i]; }
        }

        if let Some(ref mut cb) = on_layer {
            cb(l, &state.x[0..seq_len * dim]);
        }
    }

    for p in 0..seq_len {
        let xp = &state.x[p * dim..(p + 1) * dim];
        layer_norm(&mut state.xb, xp, &weights.ln_f_weight, &weights.ln_f_bias);
        state.x[p * dim..(p + 1) * dim].copy_from_slice(&state.xb);
        let logits_p = &mut state.logits[p * config.vocab_size..(p + 1) * config.vocab_size];
        project_lm_head_q4(logits_p, &state.xb, &weights.classifier_weights, &weights.token_embedding_table, config, &mut state.hb, &mut state.hb2, num_threads);
    }
}


pub struct Tokenizer;
impl Tokenizer {
    pub fn encode(text: &str) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(text.len() + 2); tokens.push(0);
        let normalized = text.replace("<mask_1>", "*").replace("<mask_1>", "*").replace("<mask>", "*");
        for c in normalized.chars() {
            let id = match c {
                'L'=>4,'A'=>5,'G'=>6,'V'=>7,'S'=>8,'E'=>9,'R'=>10,'T'=>11,'I'=>12,'D'=>13,
                'P'=>14,'K'=>15,'Q'=>16,'N'=>17,'F'=>18,'Y'=>19,'M'=>20,'H'=>21,'W'=>22,'C'=>23,
                'X'=>24,'B'=>25,'U'=>26,'Z'=>27,'O'=>28,'.'=>29,'-'=>30,'*'=>32, _=>3,
            }; tokens.push(id);
        }
        tokens.push(2); tokens
    }
    pub fn decode(token_id: usize) -> Option<char> {
        match token_id {
            4=>Some('L'),5=>Some('A'),6=>Some('G'),7=>Some('V'),8=>Some('S'),9=>Some('E'),10=>Some('R'),
            11=>Some('T'),12=>Some('I'),13=>Some('D'),14=>Some('P'),15=>Some('K'),16=>Some('Q'),17=>Some('N'),
            18=>Some('F'),19=>Some('Y'),20=>Some('M'),21=>Some('H'),22=>Some('W'),23=>Some('C'),24=>Some('X'),
            25=>Some('B'),26=>Some('U'),27=>Some('Z'),28=>Some('O'),29=>Some('.'),30=>Some('-'),32=>Some('*'),_=>None,
        }
    }
}

pub struct GenerationMetrics {
    pub full_sequence: String, pub generated_suffix: String, pub tokens_generated: usize,
    pub prefill_ms: f64, pub generation_ms: f64, pub throughput_tps: f32,
}

pub fn generate_sequence(
    prompt: &str,
    temperature: f32,
    max_new_tokens: Option<usize>,
    seed: u64,
    config: &EsmConfig,
    weights: &EsmWeights,
    state: &mut RunState,
    on_token: Option<&mut dyn FnMut(char)>,
) -> GenerationMetrics {
    generate_sequence_parallel(prompt, temperature, max_new_tokens, seed, config, weights, state, 1, on_token)
}

pub fn generate_sequence_parallel(
    prompt: &str,
    temperature: f32,
    max_new_tokens: Option<usize>,
    seed: u64,
    config: &EsmConfig,
    weights: &EsmWeights,
    state: &mut RunState,
    num_threads: usize,
    mut on_token: Option<&mut dyn FnMut(char)>,
) -> GenerationMetrics {
    let mut prompt_tokens = Tokenizer::encode(prompt);
    prompt_tokens.pop();
    if prompt_tokens.is_empty() {
        prompt_tokens.push(0);
    }
    let mut pos = 0;
    let prefill_timer = Instant::now();
    for &token in &prompt_tokens {
        if pos >= config.max_seq_len {
            break;
        }
        forward_parallel(token, pos, config, weights, state, num_threads);
        pos += 1;
    }
    let prefill_duration = prefill_timer.elapsed();
    let mut rng = SimpleRng::new(seed);
    let mut generated_tokens_count = 0;
    let mut generated_suffix = String::new();
    let max_limit = max_new_tokens
        .map(|m| (pos + m).min(config.max_seq_len))
        .unwrap_or(config.max_seq_len);
    let gen_timer = Instant::now();
    while pos < max_limit {
        let next_token = sample(&mut state.logits, temperature, &mut rng);
        if next_token == 2 {
            break;
        }
        generated_tokens_count += 1;
        if let Some(c) = Tokenizer::decode(next_token) {
            generated_suffix.push(c);
            if let Some(cb) = on_token.as_mut() {
                cb(c);
            }
        }
        forward_parallel(next_token, pos, config, weights, state, num_threads);
        pos += 1;
    }
    let gen_duration = gen_timer.elapsed();
    let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;
    let generation_ms = gen_duration.as_secs_f64() * 1000.0;
    let throughput_tps = if gen_duration.as_secs_f32() > 0.0 {
        generated_tokens_count as f32 / gen_duration.as_secs_f32()
    } else {
        0.0
    };
    let mut full_sequence = String::with_capacity(prompt.len() + generated_suffix.len());
    full_sequence.push_str(prompt);
    full_sequence.push_str(&generated_suffix);
    GenerationMetrics {
        full_sequence,
        generated_suffix,
        tokens_generated: generated_tokens_count,
        prefill_ms,
        generation_ms,
        throughput_tps,
    }
}

pub fn generate_sequence_q8(
    prompt: &str,
    temperature: f32,
    max_new_tokens: Option<usize>,
    seed: u64,
    config: &EsmConfig,
    weights: &EsmWeightsQ8,
    state: &mut RunState,
    num_threads: usize,
    mut on_token: Option<&mut dyn FnMut(char)>,
) -> GenerationMetrics {
    let mut prompt_tokens = Tokenizer::encode(prompt);
    prompt_tokens.pop();
    if prompt_tokens.is_empty() {
        prompt_tokens.push(0);
    }
    let mut pos = 0;
    let prefill_timer = Instant::now();
    for &token in &prompt_tokens {
        if pos >= config.max_seq_len {
            break;
        }
        forward_q8(token, pos, config, weights, state, num_threads);
        pos += 1;
    }
    let prefill_duration = prefill_timer.elapsed();
    let mut rng = SimpleRng::new(seed);
    let mut generated_tokens_count = 0;
    let mut generated_suffix = String::new();
    let max_limit = max_new_tokens
        .map(|m| (pos + m).min(config.max_seq_len))
        .unwrap_or(config.max_seq_len);
    let gen_timer = Instant::now();
    while pos < max_limit {
        let next_token = sample(&mut state.logits, temperature, &mut rng);
        if next_token == 2 {
            break;
        }
        generated_tokens_count += 1;
        if let Some(c) = Tokenizer::decode(next_token) {
            generated_suffix.push(c);
            if let Some(cb) = on_token.as_mut() {
                cb(c);
            }
        }
        forward_q8(next_token, pos, config, weights, state, num_threads);
        pos += 1;
    }
    let gen_duration = gen_timer.elapsed();
    let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;
    let generation_ms = gen_duration.as_secs_f64() * 1000.0;
    let throughput_tps = if gen_duration.as_secs_f32() > 0.0 {
        generated_tokens_count as f32 / gen_duration.as_secs_f32()
    } else {
        0.0
    };
    let mut full_sequence = String::with_capacity(prompt.len() + generated_suffix.len());
    full_sequence.push_str(prompt);
    full_sequence.push_str(&generated_suffix);
    GenerationMetrics {
        full_sequence,
        generated_suffix,
        tokens_generated: generated_tokens_count,
        prefill_ms,
        generation_ms,
        throughput_tps,
    }
}

pub fn generate_sequence_q4(
    prompt: &str,
    temperature: f32,
    max_new_tokens: Option<usize>,
    seed: u64,
    config: &EsmConfig,
    weights: &EsmWeightsQ4,
    state: &mut RunState,
    num_threads: usize,
    mut on_token: Option<&mut dyn FnMut(char)>,
) -> GenerationMetrics {
    let mut prompt_tokens = Tokenizer::encode(prompt);
    prompt_tokens.pop();
    if prompt_tokens.is_empty() {
        prompt_tokens.push(0);
    }
    let mut pos = 0;
    let prefill_timer = Instant::now();
    for &token in &prompt_tokens {
        if pos >= config.max_seq_len {
            break;
        }
        forward_q4(token, pos, config, weights, state, num_threads);
        pos += 1;
    }
    let prefill_duration = prefill_timer.elapsed();
    let mut rng = SimpleRng::new(seed);
    let mut generated_tokens_count = 0;
    let mut generated_suffix = String::new();
    let max_limit = max_new_tokens
        .map(|m| (pos + m).min(config.max_seq_len))
        .unwrap_or(config.max_seq_len);
    let gen_timer = Instant::now();
    while pos < max_limit {
        let next_token = sample(&mut state.logits, temperature, &mut rng);
        if next_token == 2 {
            break;
        }
        generated_tokens_count += 1;
        if let Some(c) = Tokenizer::decode(next_token) {
            generated_suffix.push(c);
            if let Some(cb) = on_token.as_mut() {
                cb(c);
            }
        }
        forward_q4(next_token, pos, config, weights, state, num_threads);
        pos += 1;
    }
    let gen_duration = gen_timer.elapsed();
    let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;
    let generation_ms = gen_duration.as_secs_f64() * 1000.0;
    let throughput_tps = if gen_duration.as_secs_f32() > 0.0 {
        generated_tokens_count as f32 / gen_duration.as_secs_f32()
    } else {
        0.0
    };
    let mut full_sequence = String::with_capacity(prompt.len() + generated_suffix.len());
    full_sequence.push_str(prompt);
    full_sequence.push_str(&generated_suffix);
    GenerationMetrics {
        full_sequence,
        generated_suffix,
        tokens_generated: generated_tokens_count,
        prefill_ms,
        generation_ms,
        throughput_tps,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_0_quantization_roundtrip() {
        let mut src = [0.0f32; 32];
        for i in 0..32 {
            src[i] = ((i as f32) / 31.0) * 2.0 - 1.0;
        }
        let block = BlockQ8_0::quantize(&src);
        let mut dst = [0.0f32; 32];
        block.dequantize_to(&mut dst);
        let mut mse = 0.0f32;
        for i in 0..32 {
            let diff = src[i] - dst[i];
            mse += diff * diff;
        }
        mse /= 32.0;
        assert!(mse < 0.02, "MSE {} is not < 0.02", mse);
    }

    #[test] fn test_layer_norm_stability() {
        let mut out = vec![0.0;4]; layer_norm(&mut out, &[1.0,2.0,3.0,4.0], &[1.0;4], &[0.0;4]); assert!(!out[0].is_nan());
    }
    #[test] fn test_softmax_sum_to_one() {
        let mut logits = vec![1.0,2.0,3.0]; softmax(&mut logits); assert!((logits.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
    #[test] fn test_tokenizer_boundaries() { assert_eq!(Tokenizer::encode("LA"), vec![0,4,5,2]); }
    #[test]
    fn test_evaluation_metrics_unit() {
        let logits_a = [10.0, 5.0, 1.0, 2.0, 8.0, 0.0];
        // Exact affine scaling: b = 2.0 * a + 3.0 -> centered correlation is 1.0
        let logits_b = [23.0, 13.0, 5.0, 7.0, 19.0, 3.0];
        let vocab_size = 3;
        // Two tokens. Token 1 top is class 0 for both (10.0 vs 23.0). Token 2 top is class 1 for both (8.0 vs 19.0).
        let top1 = compute_top1_match_rate(&logits_a, &logits_b, vocab_size);
        assert_eq!(top1, 1.0);

        let kl = compute_kl_divergence(&logits_a, &logits_a, vocab_size);
        assert!(kl.abs() < 1e-6);

        let cos_cent = compute_centered_cosine_similarity(&logits_a, &logits_b, vocab_size);
        assert!((cos_cent - 1.0).abs() < 1e-4);
    }
    #[test] fn test_matmul_avx1_and_dispatch_parity() {
        let in_dim = 320; let out_dim = 64;
        let x: Vec<f32> = (0..in_dim).map(|i|(i as f32)*0.01 - 1.5).collect();
        let w: Vec<f32> = (0..in_dim*out_dim).map(|i|((i%97) as f32)*0.005 - 0.25).collect();
        let bias: Vec<f32> = (0..out_dim).map(|i|(i as f32)*0.02).collect();
        let mut out_scalar = vec![0.0; out_dim]; let mut out_dispatch = vec![0.0; out_dim];
        matmul_scalar(&mut out_scalar, &x, &w, &bias, in_dim, out_dim);
        matmul(&mut out_dispatch, &x, &w, &bias, in_dim, out_dim);
        for i in 0..out_dim { assert!((out_scalar[i]-out_dispatch[i]).abs() < 1e-4, "Dispatch mismatch at {}", i); }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx") {
            let mut out_avx1 = vec![0.0; out_dim]; unsafe { matmul_avx1(&mut out_avx1, &x, &w, &bias, in_dim, out_dim); }
            for i in 0..out_dim { assert!((out_scalar[i]-out_avx1[i]).abs() < 1e-4, "AVX1 mismatch at {}", i); }
        }
    }

    #[test]
    fn test_forward_pass_with_weights_if_available() {
        let model_path = if std::path::Path::new("esm2_8m_weights.bin").exists() {
            "esm2_8m_weights.bin"
        } else if std::path::Path::new("esm2_weights.bin").exists() {
            "esm2_weights.bin"
        } else {
            return;
        };
        let config = EsmConfig::esm2_8m();
        let weights = EsmWeights::from_file(model_path, &config).expect("weights should load");
        let mut state = RunState::new(&config);
        forward(1, 0, &config, &weights, &mut state);
        assert!(!state.logits[0].is_nan());
    }

    #[test]
    fn test_mask_token_encoding_and_decoding() {
        assert_eq!(Tokenizer::encode("M*C"), vec![0, 20, 32, 23, 2]);
        assert_eq!(Tokenizer::encode("M<mask_1>C"), vec![0, 20, 32, 23, 2]);
        assert_eq!(Tokenizer::decode(32), Some('*'));
    }

    #[test]
    fn test_log_softmax_canonical() {
        let logits = [1.0f32; 33];
        let log_probs = log_softmax_canonical(&logits);
        let mut sum_prob = 0.0f32;
        for i in 4..=23 {
            sum_prob += log_probs[i].exp();
        }
        assert!((sum_prob - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_forward_bidirectional_if_available() {
        let model_path = if std::path::Path::new("esm2_8m_weights.bin").exists() {
            "esm2_8m_weights.bin"
        } else if std::path::Path::new("esm2_weights.bin").exists() {
            "esm2_weights.bin"
        } else {
            return;
        };
        let config = EsmConfig::esm2_8m();
        let weights = EsmWeights::from_file(model_path, &config).expect("weights should load");
        let mut bi_state = BidirectionalRunState::new(&config, 32);
        let tokens = Tokenizer::encode("MCMP*TTD");
        forward_bidirectional(&tokens, &config, &weights, &mut bi_state);
        let mask_idx = 5;
        let mask_logits = &bi_state.logits[mask_idx * config.vocab_size..(mask_idx + 1) * config.vocab_size];
        assert!(!mask_logits[18].is_nan()); // Phenylalanine (F) logit
    }

    #[test]
    fn test_bidirectional_logits_entropy() {
        let model_path = if std::path::Path::new("esm2_8m_weights.bin").exists() {
            "esm2_8m_weights.bin"
        } else if std::path::Path::new("esm2_weights.bin").exists() {
            "esm2_weights.bin"
        } else {
            return;
        };
        let config = EsmConfig::esm2_8m();
        let weights = EsmWeights::from_file(model_path, &config).expect("weights should load");
        let mut bi_state = BidirectionalRunState::new(&config, 32);
        let tokens = Tokenizer::encode("MCMP*TTD");
        forward_bidirectional(&tokens, &config, &weights, &mut bi_state);
        let mask_idx = 5;
        let mask_logits = &bi_state.logits[mask_idx * config.vocab_size..(mask_idx + 1) * config.vocab_size];
        let log_probs = log_softmax_canonical(mask_logits);

        let mut entropy = 0.0f32;
        for i in 4..=23 {
            let p = log_probs[i].exp();
            if p > 1e-12 {
                entropy -= p * log_probs[i];
            }
        }
        assert!(entropy > 0.1, "Logits collapsed to a single token! Shannon entropy H = {}", entropy);
    }

    #[test]
    fn test_simd_benchmark_parity_run() {
        let rep = crate::benchmark::run_simd_benchmark(20);
        assert!(rep.rmse_parity < 1e-4);
    }

    #[test]
    fn test_esm_config_autodetection() {
        // 8M ~ FP32 (30,091,652 B grouped_canonical_v1) and Q8_0 (~8.5-12MB)
        let cfg_8m = EsmConfig::detect_from_file_len(30_091_652).expect("8M FP32 detection");
        assert_eq!(cfg_8m.arch, EsmModelArch::Esm2_8M);
        assert_eq!(cfg_8m.dim, 320);
        assert_eq!(cfg_8m.hidden_dim, 1280);
        assert_eq!(cfg_8m.n_layers, 6);
        assert_eq!(cfg_8m.n_heads, 20);
        assert_eq!(cfg_8m.head_dim(), 16);
        let cfg_8m_q8 = EsmConfig::detect_from_file_len(8_500_000).expect("8M Q8_0 detection");
        assert_eq!(cfg_8m_q8.arch, EsmModelArch::Esm2_8M);

        // 35M ~ FP32 (134,067,972 B grouped_canonical_v1) and Q8_0 (~38-48MB)
        let cfg_35m = EsmConfig::detect_from_file_len(134_067_972).expect("35M FP32 detection");
        assert_eq!(cfg_35m.arch, EsmModelArch::Esm2_35M);
        assert_eq!(cfg_35m.dim, 480);
        assert_eq!(cfg_35m.hidden_dim, 1920);
        assert_eq!(cfg_35m.n_layers, 12);
        assert_eq!(cfg_35m.n_heads, 20);
        assert_eq!(cfg_35m.head_dim(), 24);
        let cfg_35m_q8 = EsmConfig::detect_from_file_len(38_000_000).expect("35M Q8_0 detection");
        assert_eq!(cfg_35m_q8.arch, EsmModelArch::Esm2_35M);

        // 150M ~ FP32 (592,642,692 B grouped_canonical_v1) and Q8_0 (~166-178MB)
        let cfg_150m = EsmConfig::detect_from_file_len(592_642_692).expect("150M FP32 detection");
        assert_eq!(cfg_150m.arch, EsmModelArch::Esm2_150M);
        assert_eq!(cfg_150m.dim, 640);
        assert_eq!(cfg_150m.hidden_dim, 2560);
        assert_eq!(cfg_150m.n_layers, 30);
        assert_eq!(cfg_150m.n_heads, 20);
        assert_eq!(cfg_150m.head_dim(), 32);
        let cfg_150m_q8 = EsmConfig::detect_from_file_len(170_000_000).expect("150M Q8_0 detection");
        assert_eq!(cfg_150m_q8.arch, EsmModelArch::Esm2_150M);

        // 650M ~ FP32 (2,604,339,332 B grouped_canonical_v1) and Q8_0 (~732-780MB)
        let cfg_650m = EsmConfig::detect_from_file_len(2_604_339_332).expect("650M FP32 detection");
        assert_eq!(cfg_650m.arch, EsmModelArch::Esm2_650M);
        assert_eq!(cfg_650m.dim, 1280);
        assert_eq!(cfg_650m.hidden_dim, 5120);
        assert_eq!(cfg_650m.n_layers, 33);
        assert_eq!(cfg_650m.n_heads, 20);
        assert_eq!(cfg_650m.head_dim(), 64);
        let cfg_650m_q8 = EsmConfig::detect_from_file_len(750_000_000).expect("650M Q8_0 detection");
        assert_eq!(cfg_650m_q8.arch, EsmModelArch::Esm2_650M);

        // 3B ~ FP32 (11,356,354,692 B grouped_canonical_v1) and Q8_0 (~3.19-3.4GB)
        let cfg_3b = EsmConfig::detect_from_file_len(11_356_354_692).expect("3B FP32 detection");
        assert_eq!(cfg_3b.arch, EsmModelArch::Esm2_3B);
        assert_eq!(cfg_3b.dim, 2560);
        assert_eq!(cfg_3b.hidden_dim, 10240);
        assert_eq!(cfg_3b.n_layers, 36);
        assert_eq!(cfg_3b.n_heads, 40);
        assert_eq!(cfg_3b.head_dim(), 64);
        let cfg_3b_q8 = EsmConfig::detect_from_file_len(3_300_000_000).expect("3B Q8_0 detection");
        assert_eq!(cfg_3b_q8.arch, EsmModelArch::Esm2_3B);

        // Invalid length boundaries
        assert!(EsmConfig::detect_from_file_len(100_000).is_err());
        assert!(EsmConfig::detect_from_file_len(5_999_999).is_err());
    }

    #[test]
    fn test_attention_simd_parity_dk_16_24_32() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx") {
            for &dk in &[16, 24, 32, 64] {
                let q: Vec<f32> = (0..dk).map(|i| (i as f32) * 0.1 - 0.5).collect();
                let k: Vec<f32> = (0..dk).map(|i| ((i * 7) as f32) * 0.05 - 0.3).collect();
                let v: Vec<f32> = (0..dk).map(|i| ((i * 13) as f32) * 0.03 - 0.2).collect();

                // 1. Dot product parity
                let mut scalar_dot = 0.0f32;
                for i in 0..dk {
                    scalar_dot += q[i] * k[i];
                }
                let avx_dot = unsafe { dot_product_head_avx(q.as_ptr(), k.as_ptr(), dk) };
                assert!(
                    (scalar_dot - avx_dot).abs() < 1e-5,
                    "Dot product mismatch for dk={}: scalar={} vs avx={}",
                    dk,
                    scalar_dot,
                    avx_dot
                );

                // 2. Accumulate value parity
                let prob = 0.42f32;
                let mut scalar_out = vec![1.5f32; dk];
                let mut avx_out = vec![1.5f32; dk];

                for i in 0..dk {
                    scalar_out[i] += prob * v[i];
                }
                unsafe {
                    accumulate_value_head_avx(avx_out.as_mut_ptr(), v.as_ptr(), prob, dk);
                }

                for i in 0..dk {
                    assert!(
                        (scalar_out[i] - avx_out[i]).abs() < 1e-5,
                        "Accumulate mismatch for dk={} at idx {}: scalar={} vs avx={}",
                        dk,
                        i,
                        scalar_out[i],
                        avx_out[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_attention_simd_parity_dk_64() {
        let dk = 64;
        let q: Vec<f32> = (0..dk).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let k: Vec<f32> = (0..dk).map(|i| ((i * 3) as f32) * 0.01 - 0.2).collect();
        let v: Vec<f32> = (0..dk).map(|i| ((i * 5) as f32) * 0.01 - 0.25).collect();

        // 1. Scalar Reference Dot Product
        let mut scalar_dot = 0.0f32;
        for i in 0..dk {
            scalar_dot += q[i] * k[i];
        }

        // 2. Scalar Reference Accumulate Value
        let prob = 0.618f32;
        let mut scalar_out = vec![std::f32::consts::E; dk];
        for i in 0..dk {
            scalar_out[i] += prob * v[i];
        }

        #[cfg(target_arch = "x86_64")]
        {
            // Verify AVX / AVX2 (8 x __m256)
            if std::is_x86_feature_detected!("avx") {
                let avx_dot = unsafe { dot_product_head_avx(q.as_ptr(), k.as_ptr(), dk) };
                assert!(
                    (scalar_dot - avx_dot).abs() < 1e-5,
                    "AVX dot product mismatch for dk=64: scalar={} vs avx={}, diff={}",
                    scalar_dot,
                    avx_dot,
                    (scalar_dot - avx_dot).abs()
                );

                let mut avx_out = vec![std::f32::consts::E; dk];
                unsafe {
                    accumulate_value_head_avx(avx_out.as_mut_ptr(), v.as_ptr(), prob, dk);
                }
                for i in 0..dk {
                    assert!(
                        (scalar_out[i] - avx_out[i]).abs() < 1e-5,
                        "AVX accumulate mismatch for dk=64 at idx {}: scalar={} vs avx={}",
                        i,
                        scalar_out[i],
                        avx_out[i]
                    );
                }
            }

            // Verify AVX-512 (4 x __m512)
            if std::is_x86_feature_detected!("avx512f") {
                let avx512_dot = unsafe { dot_product_head_avx512_d64(q.as_ptr(), k.as_ptr()) };
                assert!(
                    (scalar_dot - avx512_dot).abs() < 1e-5,
                    "AVX-512 dot product mismatch for dk=64: scalar={} vs avx512={}, diff={}",
                    scalar_dot,
                    avx512_dot,
                    (scalar_dot - avx512_dot).abs()
                );

                let mut avx512_out = vec![std::f32::consts::E; dk];
                unsafe {
                    accumulate_value_head_avx512_d64(avx512_out.as_mut_ptr(), v.as_ptr(), prob);
                }
                for i in 0..dk {
                    assert!(
                        (scalar_out[i] - avx512_out[i]).abs() < 1e-5,
                        "AVX-512 accumulate mismatch for dk=64 at idx {}: scalar={} vs avx512={}",
                        i,
                        scalar_out[i],
                        avx512_out[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_forward_650m_if_available() {
        let _guard = crate::HEAVY_MODEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model_path = "esm2_650m_weights.bin";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let (config, weights) = load_model_from_file(model_path).expect("650M weights must load");
        assert_eq!(config.arch, EsmModelArch::Esm2_650M);
        assert_eq!(config.dim, 1280);
        assert_eq!(config.n_layers, 33);
        assert_eq!(config.n_heads, 20);
        assert_eq!(config.head_dim(), 64);

        let mut run_state = RunState::new(&config);
        forward(20, 0, &config, &weights, &mut run_state); // token 20 = Met (M)
        assert!(!run_state.logits[0].is_nan());
        assert!(!run_state.logits[20].is_nan());

        let mut bi_state = BidirectionalRunState::new(&config, 8);
        let tokens = Tokenizer::encode("M");
        forward_bidirectional(&tokens, &config, &weights, &mut bi_state);
        assert!(!bi_state.logits[0].is_nan());
        assert!(!bi_state.logits[config.vocab_size].is_nan());
    }

    #[test]
    fn test_matmul_fp32_parallel_parity() {
        let in_dim = 320;
        let out_dim = 128;
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01 - 1.2).collect();
        let w: Vec<f32> = (0..in_dim * out_dim).map(|i| ((i % 89) as f32) * 0.005 - 0.22).collect();
        let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.03).collect();

        let mut out_ref = vec![0.0f32; out_dim];
        matmul(&mut out_ref, &x, &w, &bias, in_dim, out_dim);

        for threads in [1, 2, 4] {
            let mut out_par = vec![0.0f32; out_dim];
            matmul_parallel(&mut out_par, &x, &w, &bias, in_dim, out_dim, threads);
            for i in 0..out_dim {
                assert!(
                    (out_ref[i] - out_par[i]).abs() < 1e-4,
                    "Thread {} mismatch at idx {}: ref={} vs par={}",
                    threads,
                    i,
                    out_ref[i],
                    out_par[i]
                );
            }
        }
    }

    #[test]
    fn test_matmul_q8_simd_parity() {
        let in_dim = 320;
        let out_dim = 128;
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01 - 1.5).collect();
        let raw_w: Vec<f32> = (0..in_dim * out_dim).map(|i| ((i % 97) as f32) * 0.005 - 0.25).collect();
        let q8_blocks = quantize_slice_to_q8(&raw_w);
        let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.02).collect();

        let mut out_scalar = vec![0.0f32; out_dim];
        matmul_q8(&mut out_scalar, &x, &q8_blocks, &bias, in_dim, out_dim);

        let mut out_simd = vec![0.0f32; out_dim];
        matmul_q8_simd(&mut out_simd, &x, &q8_blocks, &bias, in_dim, out_dim);

        for i in 0..out_dim {
            assert!(
                (out_scalar[i] - out_simd[i]).abs() < 1e-4,
                "Q8 SIMD mismatch at idx {}: scalar={} vs simd={}",
                i,
                out_scalar[i],
                out_simd[i]
            );
        }

        for threads in [1, 2, 4] {
            let mut out_par = vec![0.0f32; out_dim];
            matmul_q8_parallel(&mut out_par, &x, &q8_blocks, &bias, in_dim, out_dim, threads);
            for i in 0..out_dim {
                assert!(
                    (out_scalar[i] - out_par[i]).abs() < 1e-4,
                    "Q8 parallel (threads={}) mismatch at idx {}: scalar={} vs par={}",
                    threads,
                    i,
                    out_scalar[i],
                    out_par[i]
                );
            }
        }
    }

    #[test]
    fn test_forward_q8_cosine_similarity_8m() {
        let model_path = if std::path::Path::new("esm2_8m_weights.bin").exists() {
            "esm2_8m_weights.bin"
        } else if std::path::Path::new("esm2_weights.bin").exists() {
            "esm2_weights.bin"
        } else {
            return;
        };

        let config = EsmConfig::esm2_8m();
        let weights = EsmWeights::from_file(model_path, &config).expect("weights must load");

        // 1. Run FP32 forward pass
        let mut state_fp32 = RunState::new(&config);
        forward(20, 0, &config, &weights, &mut state_fp32);

        // 2. Quantize to Q8_0 in memory
        let weights_q8 = weights.quantize_q8();
        let mut state_q8 = RunState::new(&config);
        forward_q8(20, 0, &config, &weights_q8, &mut state_q8, 4);

        // 3. Compute Cosine Similarity between logits across standard vocabulary (indices 4..=23)
        let mut dot = 0.0f32;
        let mut norm_fp32 = 0.0f32;
        let mut norm_q8 = 0.0f32;
        for i in 4..=23 {
            let l_fp = state_fp32.logits[i];
            let l_q8 = state_q8.logits[i];
            dot += l_fp * l_q8;
            norm_fp32 += l_fp * l_fp;
            norm_q8 += l_q8 * l_q8;
        }
        let cos_sim = dot / (norm_fp32.sqrt() * norm_q8.sqrt());
        assert!(
            cos_sim > 0.995,
            "Cosine similarity between FP32 and Q8_0 logits is {:.5}, expected > 0.995",
            cos_sim
        );
    }

    #[test]
    fn test_forward_q4_cosine_similarity_8m() {
        let model_path = if std::path::Path::new("esm2_8m_weights.bin").exists() {
            "esm2_8m_weights.bin"
        } else if std::path::Path::new("esm2_weights.bin").exists() {
            "esm2_weights.bin"
        } else {
            return;
        };

        let config = EsmConfig::esm2_8m();
        let weights = EsmWeights::from_file(model_path, &config).expect("weights must load");

        // 1. Run FP32 forward pass
        let mut state_fp32 = RunState::new(&config);
        forward(20, 0, &config, &weights, &mut state_fp32);

        // 2. Quantize to Q4_0 in memory
        let weights_q4 = weights.quantize_q4();
        let mut state_q4 = RunState::new(&config);
        forward_q4(20, 0, &config, &weights_q4, &mut state_q4, 4);

        // 3. Compute Cosine Similarity between logits across standard vocabulary (indices 4..=23)
        let mut dot = 0.0f32;
        let mut norm_fp32 = 0.0f32;
        let mut norm_q4 = 0.0f32;
        for i in 4..=23 {
            let l_fp = state_fp32.logits[i];
            let l_q4 = state_q4.logits[i];
            dot += l_fp * l_q4;
            norm_fp32 += l_fp * l_fp;
            norm_q4 += l_q4 * l_q4;
        }
        let cos_sim = dot / (norm_fp32.sqrt() * norm_q4.sqrt());
        println!("8M Cosine similarity FP32 vs Q4_0: {:.5}", cos_sim);
        assert!(
            cos_sim > 0.850,
            "Cosine similarity between FP32 and Q4_0 logits is {:.5}, expected > 0.850",
            cos_sim
        );
    }

    #[test]
    fn test_q8_weights_save_load_roundtrip() {
        let model_path = if std::path::Path::new("esm2_8m_weights.bin").exists() {
            "esm2_8m_weights.bin"
        } else if std::path::Path::new("esm2_weights.bin").exists() {
            "esm2_weights.bin"
        } else {
            return;
        };

        let config = EsmConfig::esm2_8m();
        let weights = EsmWeights::from_file(model_path, &config).expect("weights must load");
        let weights_q8 = weights.quantize_q8();

        let temp_q8_path = "target/test_esm2_8m_roundtrip.q8_0";
        weights_q8.save_to_file(temp_q8_path).expect("Q8 save must succeed");

        let loaded_q8 = EsmWeightsQ8::from_file(temp_q8_path, &config).expect("Q8 load must succeed");
        assert_eq!(weights_q8.token_embedding_table.len(), loaded_q8.token_embedding_table.len());
        assert_eq!(weights_q8.wq.len(), loaded_q8.wq.len());
        assert_eq!(weights_q8.fc1_weight.len(), loaded_q8.fc1_weight.len());

        let mut state_orig = RunState::new(&config);
        forward_q8(20, 0, &config, &weights_q8, &mut state_orig, 2);

        let mut state_loaded = RunState::new(&config);
        forward_q8(20, 0, &config, &loaded_q8, &mut state_loaded, 2);

        for i in 0..config.vocab_size {
            assert_eq!(state_orig.logits[i], state_loaded.logits[i], "Mismatch in loaded Q8 logits at idx {}", i);
        }

        let _ = std::fs::remove_file(temp_q8_path);
    }

    #[test]
    fn test_forward_650m_q8_and_multithread_if_available() {
        let _guard = crate::HEAVY_MODEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model_path = "esm2_650m_weights.bin";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let (config, weights) = load_model_from_file(model_path).expect("650M weights must load");

        // 1. Test FP32 with 4 threads
        let mut state_fp32 = RunState::new(&config);
        forward_parallel(20, 0, &config, &weights, &mut state_fp32, 4);
        assert!(!state_fp32.logits[0].is_nan());
        assert!(!state_fp32.logits[20].is_nan());

        // 2. Test Q8 quantization and 4-thread execution
        let weights_q8 = weights.quantize_q8();
        let mut state_q8 = RunState::new(&config);
        forward_q8(20, 0, &config, &weights_q8, &mut state_q8, 4);
        assert!(!state_q8.logits[0].is_nan());
        assert!(!state_q8.logits[20].is_nan());

        // 3. Check Cosine similarity on 650M
        let mut dot = 0.0f32;
        let mut norm_fp32 = 0.0f32;
        let mut norm_q8 = 0.0f32;
        for i in 4..=23 {
            let l_fp = state_fp32.logits[i];
            let l_q8 = state_q8.logits[i];
            dot += l_fp * l_q8;
            norm_fp32 += l_fp * l_fp;
            norm_q8 += l_q8 * l_q8;
        }
        let cos_sim = dot / (norm_fp32.sqrt() * norm_q8.sqrt());
        assert!(
            cos_sim > 0.990,
            "650M Cosine similarity between FP32 and Q8_0 logits is {:.5}, expected > 0.990",
            cos_sim
        );
    }
}

#[cfg(test)]
pub(crate) static HEAVY_MODEL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests_q4_format {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip_quantize_dequantize_bound() {
        let mut test_vectors: Vec<[f32; 32]> = Vec::new();

        let mut v1 = [0.0f32; 32];
        for i in 0..32 {
            v1[i] = -8.0 + (i as f32) * (15.0 / 31.0);
        }
        test_vectors.push(v1);

        let scales = [0.05f32, 1.0f32, 2.5f32, 10.0f32];
        for &scale in &scales {
            let mut v = [0.0f32; 32];
            v[0] = -8.0 * scale;
            for i in 1..32 {
                let q_level = ((i % 16) as i32 - 8) as f32;
                let noise = (((i * 11) % 9) as f32 - 4.0) / 10.0 * 0.45 * scale;
                v[i] = q_level * scale + noise;
            }
            test_vectors.push(v);
        }

        for &scale in &scales {
            let mut v = [0.0f32; 32];
            v[0] = 8.0 * scale;
            for i in 1..32 {
                let q_level = ((i % 16) as i32 - 8) as f32;
                let noise = (((i * 7) % 9) as f32 - 4.0) / 10.0 * 0.45 * scale;
                v[i] = q_level * (-scale) + noise;
            }
            test_vectors.push(v);
        }

        for (idx, src) in test_vectors.iter().enumerate() {
            let block = BlockQ4_0::quantize(src);
            let mut dst = [0.0f32; 32];
            block.dequantize_to(&mut dst);

            let bound = block.d.abs() / 2.0 + 1e-5;
            for i in 0..32 {
                let err = (src[i] - dst[i]).abs();
                assert!(
                    err <= bound,
                    "Vector {} elem {} (src={}, dst={}, d={}): err {} > bound {}",
                    idx, i, src[i], dst[i], block.d, err, bound
                );
            }
        }
    }

    #[test]
    fn test_q4_0_exact_nibble_packing() {
        let mut src = [0.0f32; 32];
        for j in 0..16 {
            let j_i32 = j as i32;
            src[j] = (8 - j_i32) as f32;
            src[j + 16] = (j_i32 - 7) as f32;
        }

        let block = BlockQ4_0::quantize(&src);
        assert_eq!(block.d, -1.0);

        for j in 0..16 {
            let expected_lo = j as u8;
            let expected_hi = (15 - j) as u8;
            let expected_byte = expected_lo | (expected_hi << 4);
            assert_eq!(
                block.qs[j], expected_byte,
                "Nibble packing mismatch at byte {}: got 0x{:02X}, expected 0x{:02X}",
                j, block.qs[j], expected_byte
            );
        }

        let mut dst = [0.0f32; 32];
        block.dequantize_to(&mut dst);
        for i in 0..32 {
            assert_eq!(dst[i], src[i], "Dequantized mismatch at index {}", i);
        }
    }

    #[test]
    fn test_q4_0_zero_block() {
        let zeros = [0.0f32; 32];
        let block = BlockQ4_0::quantize(&zeros);
        assert_eq!(block.d, 0.0f32);
        assert_eq!(block.qs, [0x88; 16]);

        let mut dst = [999.0f32; 32];
        block.dequantize_to(&mut dst);
        for i in 0..32 {
            assert_eq!(dst[i], 0.0f32, "Zero block dequantization mismatch at {}", i);
        }
    }

    #[test]
    fn test_codq_header_valid_corrupt_incoherent() {
        let config = EsmConfig::esm2_650m();
        let header = CodqHeader::new(DTYPE_Q4_0, &config);

        assert_eq!(header.magic, CODQ_MAGIC);
        assert_eq!(header.version, CODQ_VERSION);
        assert_eq!(header.dtype, DTYPE_Q4_0);
        assert_eq!(header.arch, ARCH_650M);
        assert_eq!(header.n_layers, 33);
        assert_eq!(header.dim, 1280);
        assert_eq!(header.hidden_dim, 5120);
        assert_eq!(header.vocab_size, 33);
        assert!(header.validate(&config).is_ok());

        let cfg_detected = header.to_config().expect("Should extract valid config");
        assert_eq!(cfg_detected, config);

        let mut corrupt_magic = header;
        corrupt_magic.magic = *b"NOPE";
        assert!(corrupt_magic.validate(&config).is_err());
        assert!(corrupt_magic.to_config().is_err());

        let mut corrupt_version = header;
        corrupt_version.version = 99;
        assert!(corrupt_version.validate(&config).is_err());
        assert!(corrupt_version.to_config().is_err());

        let mut incoherent_dim = header;
        incoherent_dim.dim = 999;
        assert!(incoherent_dim.validate(&config).is_err());
        assert!(incoherent_dim.to_config().is_err());

        let mut incoherent_layers = header;
        incoherent_layers.n_layers = 42;
        assert!(incoherent_layers.validate(&config).is_err());
        assert!(incoherent_layers.to_config().is_err());
    }

    #[test]
    fn test_first_block_alignment_at_offset_64() {
        let temp_path = "target/test_align_q4.bin";
        let config = EsmConfig::esm2_8m();

        let w = EsmWeightsQ4 {
            token_embedding_table: vec![0.5f32; config.vocab_size * config.dim],
            ln_1_weight: vec![1.0; config.n_layers * config.dim],
            ln_1_bias: vec![0.0; config.n_layers * config.dim],
            wq: quantize_slice_to_q4(&vec![0.1; config.n_layers * config.dim * config.dim]),
            bq: vec![0.0; config.n_layers * config.dim],
            wk: quantize_slice_to_q4(&vec![0.1; config.n_layers * config.dim * config.dim]),
            bk: vec![0.0; config.n_layers * config.dim],
            wv: quantize_slice_to_q4(&vec![0.1; config.n_layers * config.dim * config.dim]),
            bv: vec![0.0; config.n_layers * config.dim],
            wo: quantize_slice_to_q4(&vec![0.1; config.n_layers * config.dim * config.dim]),
            bo: vec![0.0; config.n_layers * config.dim],
            ln_2_weight: vec![1.0; config.n_layers * config.dim],
            ln_2_bias: vec![0.0; config.n_layers * config.dim],
            fc1_weight: quantize_slice_to_q4(&vec![0.1; config.n_layers * config.dim * config.hidden_dim]),
            fc1_bias: vec![0.0; config.n_layers * config.hidden_dim],
            fc2_weight: quantize_slice_to_q4(&vec![0.1; config.n_layers * config.hidden_dim * config.dim]),
            fc2_bias: vec![0.0; config.n_layers * config.dim],
            ln_f_weight: vec![1.0; config.dim],
            ln_f_bias: vec![0.0; config.dim],
            classifier_weights: vec![0.0; config.vocab_size * config.dim],
        };

        w.save_to_file(temp_path, &config).expect("Must save Q4 file");

        let mut file = File::open(temp_path).expect("Must open saved file");
        let header = CodqHeader::read_from(&mut file).expect("Must read header");
        assert_eq!(header.magic, CODQ_MAGIC);

        let stream_pos = file.stream_position().expect("Stream pos");
        assert_eq!(stream_pos, 64, "First weight block MUST start at offset 64");
        assert_eq!(stream_pos % 64, 0, "Offset 64 is 64-byte aligned");
        assert_eq!(stream_pos % (core::mem::align_of::<BlockQ4_0>() as u64), 0, "Aligned to BlockQ4_0");

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_no_ambiguity_650m_q4_vs_150m() {
        let file_len_410mb = 410_000_000u64;
        let legacy_detected = EsmConfig::detect_from_file_len(file_len_410mb).unwrap();
        assert_eq!(legacy_detected.arch, EsmModelArch::Esm2_150M, "Legacy size detection maps 410MB to 150M");

        let temp_path = "target/test_no_ambiguity_650m_q4.bin";
        {
            let mut file = File::create(temp_path).expect("Create file");
            let config_650m = EsmConfig::esm2_650m();
            let header = CodqHeader::new(DTYPE_Q4_0, &config_650m);
            header.write_to(&mut file).expect("Write header");
            file.set_len(file_len_410mb).expect("Set len");
        }

        let detected = EsmConfig::detect_from_file(temp_path).expect("Must detect from header");
        assert_eq!(detected.arch, EsmModelArch::Esm2_650M, "CODQ header correctly detects 650M despite 410MB length");
        assert_ne!(detected.arch, EsmModelArch::Esm2_150M, "CODQ header prevents false 150M detection");

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_q4_weights_save_load_roundtrip_8m() {
        let model_path = if std::path::Path::new("esm2_8m_weights.bin").exists() {
            "esm2_8m_weights.bin"
        } else if std::path::Path::new("esm2_weights.bin").exists() {
            "esm2_weights.bin"
        } else {
            return;
        };

        let config = EsmConfig::esm2_8m();
        let weights = EsmWeights::from_file(model_path, &config).expect("weights must load");
        let weights_q4 = weights.quantize_q4();

        let temp_q4_path = "target/test_esm2_8m_roundtrip.q4_0";
        weights_q4.save_to_file(temp_q4_path, &config).expect("Q4 save must succeed");

        let (loaded_config, loaded_q4) = EsmWeightsQ4::load_from_file(temp_q4_path).expect("Q4 load must succeed");
        assert_eq!(loaded_config, config);
        assert_eq!(weights_q4.token_embedding_table.len(), loaded_q4.token_embedding_table.len());
        assert_eq!(weights_q4.wq.len(), loaded_q4.wq.len());
        assert_eq!(weights_q4.fc1_weight.len(), loaded_q4.fc1_weight.len());
        assert_eq!(weights_q4.classifier_weights.len(), loaded_q4.classifier_weights.len());

        let _ = std::fs::remove_file(temp_q4_path);
    }
}

#[cfg(test)]
mod tests_q4_simd {
    use super::*;

    fn create_test_block_q4(src: &[f32; 32]) -> BlockQ4_0 {
        let mut max_abs = 0.0f32;
        let mut max_val = 0.0f32;
        for &val in src.iter() {
            let abs = val.abs();
            if abs > max_abs {
                max_abs = abs;
                max_val = val;
            }
        }

        let d = max_val / -8.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0u8; 16];
        for j in 0..16 {
            let x0 = src[j];
            let x1 = src[j + 16];
            let q0 = ((x0 * id).round() + 8.0).clamp(0.0, 15.0) as u8;
            let q1 = ((x1 * id).round() + 8.0).clamp(0.0, 15.0) as u8;
            qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        BlockQ4_0 { d, qs }
    }

    fn create_test_matrix_q4(in_dim: usize, out_dim: usize, seed_offset: f32) -> Vec<BlockQ4_0> {
        assert_eq!(in_dim % 32, 0);
        let num_blocks = out_dim * (in_dim / 32);
        let mut blocks = Vec::with_capacity(num_blocks);

        for b in 0..num_blocks {
            let mut src = [0.0f32; 32];
            for (k, elem) in src.iter_mut().enumerate() {
                let idx = b * 32 + k;
                *elem = ((idx as f32 * 1.5 + seed_offset).sin()) * 1.75;
            }
            blocks.push(create_test_block_q4(&src));
        }
        blocks
    }

    fn check_rel_parity(scalar: &[f32], simd: &[f32], tol: f32, label: &str) {
        assert_eq!(scalar.len(), simd.len());
        let mut sum_sq_diff = 0.0f32;
        let mut sum_sq_ref = 0.0f32;
        let mut max_diff = 0.0f32;
        let mut max_diff_idx = 0;

        for i in 0..scalar.len() {
            let s = scalar[i];
            let v = simd[i];
            let diff = (s - v).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
            sum_sq_diff += diff * diff;
            sum_sq_ref += s * s;
        }

        let l2_diff = sum_sq_diff.sqrt();
        let l2_ref = sum_sq_ref.sqrt();
        let l2_rel_err = if l2_ref > 0.0 { l2_diff / l2_ref } else { l2_diff };

        println!(
            "{}: L2 rel err = {:.2e}, max diff = {:.2e} at idx {} (scalar={}, simd={})",
            label, l2_rel_err, max_diff, max_diff_idx, scalar[max_diff_idx], simd[max_diff_idx]
        );

        assert!(
            l2_rel_err <= tol,
            "{}: L2 relative error {:.2e} exceeded tolerance {:.2e}",
            label, l2_rel_err, tol
        );
    }

    #[test]
    fn test_q4_simd_parity_esm_dimensions() {
        let test_cases: &[(usize, usize)] = &[
            (320, 320),
            (480, 480),
            (640, 640),
            (1280, 320),
            (2560, 64),
            (5120, 64),
            (320, 1280),
            (640, 2560),
            (320, 5120),
        ];

        for &(in_dim, out_dim) in test_cases {
            let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.03).cos() * 0.8).collect();
            let blocks = create_test_matrix_q4(in_dim, out_dim, 0.42);
            let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32 * 0.01) - 0.2).collect();

            let mut out_scalar = vec![0.0f32; out_dim];
            matmul_q4(&mut out_scalar, &x, &blocks, &bias, in_dim, out_dim);

            let mut out_simd = vec![0.0f32; out_dim];
            matmul_q4_simd(&mut out_simd, &x, &blocks, &bias, in_dim, out_dim);
            check_rel_parity(
                &out_scalar,
                &out_simd,
                1e-5,
                &format!("SIMD dispatch ({in_dim}x{out_dim})"),
            );

            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx") {
                let mut out_avx1 = vec![0.0f32; out_dim];
                unsafe { matmul_q4_avx1(&mut out_avx1, &x, &blocks, &bias, in_dim, out_dim) };
                check_rel_parity(
                    &out_scalar,
                    &out_avx1,
                    1e-5,
                    &format!("AVX1 ({in_dim}x{out_dim})"),
                );
            }

            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                let mut out_avx2 = vec![0.0f32; out_dim];
                unsafe { matmul_q4_avx2_fma(&mut out_avx2, &x, &blocks, &bias, in_dim, out_dim) };
                check_rel_parity(
                    &out_scalar,
                    &out_avx2,
                    1e-5,
                    &format!("AVX2+FMA ({in_dim}x{out_dim})"),
                );
            }
        }
    }

    #[test]
    fn test_q4_parallel_vs_sequential_identity_threads() {
        let test_dims = &[(640, 128), (1280, 64)];

        for &(in_dim, out_dim) in test_dims {
            let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.05).sin()).collect();
            let blocks = create_test_matrix_q4(in_dim, out_dim, 0.77);
            let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32 * 0.01) - 0.5).collect();

            let mut out_seq = vec![0.0f32; out_dim];
            matmul_q4_simd(&mut out_seq, &x, &blocks, &bias, in_dim, out_dim);

            for &threads in &[1, 2, 4, 8] {
                let mut out_par = vec![0.0f32; out_dim];
                matmul_q4_parallel(&mut out_par, &x, &blocks, &bias, in_dim, out_dim, threads);

                for i in 0..out_dim {
                    assert_eq!(
                        out_seq[i].to_bits(),
                        out_par[i].to_bits(),
                        "threads={} mismatch at row {}: seq={} vs par={}",
                        threads, i, out_seq[i], out_par[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_q4_edge_case_small_out_dim() {
        let in_dim = 320;
        let small_out_dims = [1, 2, 7, 15, 16, 31];

        for &out_dim in &small_out_dims {
            let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.02).cos()).collect();
            let blocks = create_test_matrix_q4(in_dim, out_dim, 1.23);
            let bias: Vec<f32> = (0..out_dim).map(|i| i as f32 * 0.1).collect();

            let mut out_scalar = vec![0.0f32; out_dim];
            matmul_q4(&mut out_scalar, &x, &blocks, &bias, in_dim, out_dim);

            let mut out_simd = vec![0.0f32; out_dim];
            matmul_q4_simd(&mut out_simd, &x, &blocks, &bias, in_dim, out_dim);
            check_rel_parity(&out_scalar, &out_simd, 1e-5, &format!("small out_dim={out_dim} simd"));

            let mut out_par = vec![0.0f32; out_dim];
            matmul_q4_parallel(&mut out_par, &x, &blocks, &bias, in_dim, out_dim, 4);
            for i in 0..out_dim {
                assert_eq!(
                    out_simd[i].to_bits(),
                    out_par[i].to_bits(),
                    "small out_dim={out_dim} parallel fallback mismatch at {i}"
                );
            }
        }
    }

    #[test]
    fn test_q4_edge_case_zero_blocks() {
        let in_dim = 64;
        let out_dim = 32;
        let blocks_per_row = in_dim / 32;

        let zero_blocks = vec![BlockQ4_0 { d: 0.0, qs: [0x88; 16] }; out_dim * blocks_per_row];
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) + 1.0).collect();
        let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 2.5).collect();

        let mut out_scalar = vec![0.0f32; out_dim];
        matmul_q4(&mut out_scalar, &x, &zero_blocks, &bias, in_dim, out_dim);

        let mut out_simd = vec![0.0f32; out_dim];
        matmul_q4_simd(&mut out_simd, &x, &zero_blocks, &bias, in_dim, out_dim);

        for i in 0..out_dim {
            assert_eq!(out_scalar[i], bias[i], "Scalar zero-weights should equal bias");
            assert_eq!(out_simd[i], bias[i], "SIMD zero-weights should equal bias");
        }

        let x_zero = vec![0.0f32; in_dim];
        let test_blocks = create_test_matrix_q4(in_dim, out_dim, std::f32::consts::PI);
        let mut out_from_zero_x = vec![0.0f32; out_dim];
        matmul_q4_simd(&mut out_from_zero_x, &x_zero, &test_blocks, &bias, in_dim, out_dim);

        for i in 0..out_dim {
            assert_eq!(out_from_zero_x[i], bias[i], "Zero x should equal bias");
        }
    }

    #[test]
    fn test_q4_edge_case_extreme_values() {
        let in_dim = 64;
        let out_dim = 32;
        let blocks_per_row = in_dim / 32;

        let pos_extreme_blocks = vec![BlockQ4_0 { d: 5000.0, qs: [0xFF; 16] }; out_dim * blocks_per_row];
        let neg_extreme_blocks = vec![BlockQ4_0 { d: 5000.0, qs: [0x00; 16] }; out_dim * blocks_per_row];

        let x_extreme: Vec<f32> = (0..in_dim).map(|i| if i % 2 == 0 { 100.0 } else { -100.0 }).collect();
        let bias = vec![0.0f32; out_dim];

        let mut out_pos_scalar = vec![0.0f32; out_dim];
        matmul_q4(&mut out_pos_scalar, &x_extreme, &pos_extreme_blocks, &bias, in_dim, out_dim);
        let mut out_pos_simd = vec![0.0f32; out_dim];
        matmul_q4_simd(&mut out_pos_simd, &x_extreme, &pos_extreme_blocks, &bias, in_dim, out_dim);
        check_rel_parity(&out_pos_scalar, &out_pos_simd, 1e-5, "extreme positive values");

        let mut out_neg_scalar = vec![0.0f32; out_dim];
        matmul_q4(&mut out_neg_scalar, &x_extreme, &neg_extreme_blocks, &bias, in_dim, out_dim);
        let mut out_neg_simd = vec![0.0f32; out_dim];
        matmul_q4_simd(&mut out_neg_simd, &x_extreme, &neg_extreme_blocks, &bias, in_dim, out_dim);
        check_rel_parity(&out_neg_scalar, &out_neg_simd, 1e-5, "extreme negative values");

        for i in 0..out_dim {
            assert!(out_pos_simd[i].is_finite());
            assert!(out_neg_simd[i].is_finite());
        }
    }
}

#[cfg(test)]
mod tests_q4_integration {
    use super::*;

    #[test]
    fn test_forward_q4_integration_8m() {
        let q4_path = "esm2_8m_q4_0.bin";
        if !std::path::Path::new(q4_path).exists() {
            println!("Skipping test_forward_q4_integration_8m: file not found");
            return;
        }

        let (config, weights_q4) = EsmWeightsQ4::load_from_file(q4_path).expect("8M Q4 weights must load");
        assert_eq!(config.arch, EsmModelArch::Esm2_8M);

        let mut state_q4 = RunState::new(&config);
        forward_q4(20, 0, &config, &weights_q4, &mut state_q4, 4);

        for &l in &state_q4.logits[..config.vocab_size] {
            assert!(!l.is_nan() && !l.is_infinite(), "8M Q4 logits must be finite");
        }

        let fp32_path = "esm2_8m_weights.bin";
        if std::path::Path::new(fp32_path).exists() {
            let weights_fp = EsmWeights::from_file(fp32_path, &config).expect("FP32 weights must load");
            let mut state_fp = RunState::new(&config);
            forward(20, 0, &config, &weights_fp, &mut state_fp);

            let cos_sim = compute_cosine_similarity(&state_fp.logits[4..=23], &state_q4.logits[4..=23]);
            println!("8M Integration: Cosine similarity FP32 vs Q4_0 = {:.5}", cos_sim);
            assert!(cos_sim >= 0.850, "8M Q4 cosine similarity {:.5} < 0.850", cos_sim);
        }
    }

    #[test]
    fn test_forward_q4_integration_35m() {
        let q4_path = "esm2_35m_q4_0.bin";
        if !std::path::Path::new(q4_path).exists() {
            println!("Skipping test_forward_q4_integration_35m: file not found");
            return;
        }

        let (config, weights_q4) = EsmWeightsQ4::load_from_file(q4_path).expect("35M Q4 weights must load");
        assert_eq!(config.arch, EsmModelArch::Esm2_35M);

        let mut state_q4 = RunState::new(&config);
        forward_q4(20, 0, &config, &weights_q4, &mut state_q4, 4);

        for &l in &state_q4.logits[..config.vocab_size] {
            assert!(!l.is_nan() && !l.is_infinite(), "35M Q4 logits must be finite");
        }

        let fp32_path = "esm2_35m_weights.bin";
        if std::path::Path::new(fp32_path).exists() {
            let weights_fp = EsmWeights::from_file(fp32_path, &config).expect("FP32 weights must load");
            let mut state_fp = RunState::new(&config);
            forward_parallel(20, 0, &config, &weights_fp, &mut state_fp, 4);

            let cos_sim = compute_cosine_similarity(&state_fp.logits[4..=23], &state_q4.logits[4..=23]);
            println!("35M Integration: Cosine similarity FP32 vs Q4_0 = {:.5}", cos_sim);
            assert!(cos_sim >= 0.850, "35M Q4 cosine similarity {:.5} < 0.850", cos_sim);
        }
    }

    #[test]
    fn test_forward_q4_integration_150m() {
        let _guard = crate::HEAVY_MODEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let q4_path = "esm2_150m_q4_0.bin";
        if !std::path::Path::new(q4_path).exists() {
            println!("Skipping test_forward_q4_integration_150m: file not found");
            return;
        }

        let (config, weights_q4) = EsmWeightsQ4::load_from_file(q4_path).expect("150M Q4 weights must load");
        assert_eq!(config.arch, EsmModelArch::Esm2_150M);

        let mut state_q4 = RunState::new(&config);
        forward_q4(20, 0, &config, &weights_q4, &mut state_q4, 4);

        for &l in &state_q4.logits[..config.vocab_size] {
            assert!(!l.is_nan() && !l.is_infinite(), "150M Q4 logits must be finite");
        }

        let fp32_path = "esm2_150m_weights.bin";
        if std::path::Path::new(fp32_path).exists() {
            let weights_fp = EsmWeights::from_file(fp32_path, &config).expect("FP32 weights must load");
            let mut state_fp = RunState::new(&config);
            forward_parallel(20, 0, &config, &weights_fp, &mut state_fp, 4);

            let cos_sim = compute_cosine_similarity(&state_fp.logits[4..=23], &state_q4.logits[4..=23]);
            println!("150M Integration: Cosine similarity FP32 vs Q4_0 = {:.5}", cos_sim);
            assert!(cos_sim >= 0.950, "150M Q4 cosine similarity {:.5} < 0.950", cos_sim);
        }
    }

    #[test]
    fn test_forward_q4_integration_650m() {
        let _guard = crate::HEAVY_MODEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let q4_path = "esm2_650m_q4_0.bin";
        if !std::path::Path::new(q4_path).exists() {
            println!("Skipping test_forward_q4_integration_650m: file not found");
            return;
        }

        let (config, weights_q4) = EsmWeightsQ4::load_from_file(q4_path).expect("650M Q4 weights must load");
        assert_eq!(config.arch, EsmModelArch::Esm2_650M);

        let mut state_q4 = RunState::new(&config);
        forward_q4(20, 0, &config, &weights_q4, &mut state_q4, 4);

        for &l in &state_q4.logits[..config.vocab_size] {
            assert!(!l.is_nan() && !l.is_infinite(), "650M Q4 logits must be finite");
        }

        let fp32_path = "esm2_650m_weights.bin";
        if std::path::Path::new(fp32_path).exists() {
            let weights_fp = EsmWeights::from_file(fp32_path, &config).expect("FP32 weights must load");
            let mut state_fp = RunState::new(&config);
            forward_parallel(20, 0, &config, &weights_fp, &mut state_fp, 4);

            let cos_sim = compute_cosine_similarity(&state_fp.logits[4..=23], &state_q4.logits[4..=23]);
            println!("650M Integration: Cosine similarity FP32 vs Q4_0 = {:.5}", cos_sim);
            assert!(cos_sim >= 0.940, "650M Q4 cosine similarity {:.5} < 0.940", cos_sim);
        }
    }

    #[test]
    fn test_forward_q8_codq_header_integration() {
        let _guard = crate::HEAVY_MODEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let models = [
            ("esm2_8m_q8_0.bin", EsmModelArch::Esm2_8M),
            ("esm2_35m_q8_0.bin", EsmModelArch::Esm2_35M),
            ("esm2_150m_q8_0.bin", EsmModelArch::Esm2_150M),
            ("esm2_650m_q8_0.bin", EsmModelArch::Esm2_650M),
        ];

        for (path, expected_arch) in &models {
            if !std::path::Path::new(path).exists() {
                println!("Skipping {} integration test: file not found", path);
                continue;
            }

            let (config, weights_q8) = EsmWeightsQ8::load_from_file(path).expect("Q8 weights must load");
            assert_eq!(config.arch, *expected_arch, "Architecture mismatch for {}", path);

            let mut state_q8 = RunState::new(&config);
            forward_q8(20, 0, &config, &weights_q8, &mut state_q8, 4);

            for &l in &state_q8.logits[..config.vocab_size] {
                assert!(!l.is_nan() && !l.is_infinite(), "Q8 logits for {} must be finite", path);
            }
            println!("Validated CODQ Q8_0 forward pass for '{}'", path);
        }
    }

    #[test]
    #[ignore]
    fn test_generate_validation_report() {
        use std::time::Instant;

        let scales = [
            ("8M", "esm2_8m_weights.bin", "esm2_8m_q8_0.bin", "esm2_8m_q4_0.bin", EsmConfig::esm2_8m()),
            ("35M", "esm2_35m_weights.bin", "esm2_35m_q8_0.bin", "esm2_35m_q4_0.bin", EsmConfig::esm2_35m()),
            ("150M", "esm2_150m_weights.bin", "esm2_150m_q8_0.bin", "esm2_150m_q4_0.bin", EsmConfig::esm2_150m()),
            ("650M", "esm2_650m_weights.bin", "esm2_650m_q8_0.bin", "esm2_650m_q4_0.bin", EsmConfig::esm2_650m()),
        ];

        let seq = "AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS";
        let tokens = Tokenizer::encode(seq);
        let seq_len = tokens.len();
        let num_threads = 4;

        println!("\n=========================================================================================");
        println!("               CODON-INFER v1.6.0 Q4_0 & Q8_0 MULTI-SCALE VALIDATION REPORT              ");
        println!("=========================================================================================\n");

        for (scale_name, fp32_path, q8_path, q4_path, config) in &scales {
            println!("#########################################################################################");
            println!(" SCALE: {} ({})", scale_name, config.model_name());
            println!("#########################################################################################");

            if !std::path::Path::new(fp32_path).exists() || !std::path::Path::new(q4_path).exists() || !std::path::Path::new(q8_path).exists() {
                println!("Skipping {}: one or more weights files not found.", scale_name);
                continue;
            }

            let fp32_size = std::fs::metadata(fp32_path).unwrap().len();
            let q8_size = std::fs::metadata(q8_path).unwrap().len();
            let q4_size = std::fs::metadata(q4_path).unwrap().len();

            println!("\n--- [1] TAMAÑO EN DISCO Y COMPRESIÓN ---");
            println!(" * FP32:  {:>12} bytes ({:>8.2} MB)", fp32_size, fp32_size as f64 / 1_048_576.0);
            println!(" * Q8_0:  {:>12} bytes ({:>8.2} MB) | Compresión real: {:.2}x (Teórica matriz: 3.56x)",
                q8_size, q8_size as f64 / 1_048_576.0, fp32_size as f64 / q8_size as f64);
            println!(" * Q4_0:  {:>12} bytes ({:>8.2} MB) | Compresión real: {:.2}x (Teórica matriz: 6.40x)",
                q4_size, q4_size as f64 / 1_048_576.0, fp32_size as f64 / q4_size as f64);

            let t_load = Instant::now();
            let weights_fp = EsmWeights::from_file(fp32_path, config).expect("FP32 weights must load");
            let weights_q4 = EsmWeightsQ4::load_from_file(q4_path).expect("Q4 weights must load").1;
            let weights_q8 = EsmWeightsQ8::load_from_file(q8_path).expect("Q8 weights must load").1;
            println!(" * Carga en memoria completada en {:.2} ms", t_load.elapsed().as_secs_f64() * 1000.0);

            // Matrix-by-matrix fidelity
            println!("\n--- [2] FIDELIDAD NUMÉRICA POR MATRIZ (FP32 vs Q4_0) ---");
            let mut worst_mat_cos = 1.0f32;
            let mut worst_mat_rmse = 0.0f32;
            let mut worst_mat_name = String::new();

            // token embedding (kept in FP32)
            let emb_cos = compute_cosine_similarity(&weights_fp.token_embedding_table, &weights_q4.token_embedding_table);
            let emb_rmse = compute_rmse(&weights_fp.token_embedding_table, &weights_q4.token_embedding_table);
            println!(" * token_embedding_table (FP32): Cosine = {:.6}, RMSE = {:.6}", emb_cos, emb_rmse);
            if emb_cos < worst_mat_cos { worst_mat_cos = emb_cos; worst_mat_rmse = emb_rmse; worst_mat_name = "token_embedding_table".to_string(); }

            let dim = config.dim;
            let hidden_dim = config.hidden_dim;
            let q4_matrix_blocks = dim * (dim / 32);
            let q4_fc1_blocks = hidden_dim * (dim / 32);
            let q4_fc2_blocks = dim * (hidden_dim / 32);

            let mut sum_cos_wq = 0.0f32;
            let mut sum_cos_fc1 = 0.0f32;
            let mut sum_cos_fc2 = 0.0f32;

            for l in 0..config.n_layers {
                let layer_mat_fp = l * dim * dim;
                let layer_mat_q4 = l * q4_matrix_blocks;

                // wq
                let dequant_wq = dequantize_q4_to_vec(&weights_q4.wq[layer_mat_q4..layer_mat_q4 + q4_matrix_blocks]);
                let cos_wq = compute_cosine_similarity(&weights_fp.wq[layer_mat_fp..layer_mat_fp + dim * dim], &dequant_wq);
                let rmse_wq = compute_rmse(&weights_fp.wq[layer_mat_fp..layer_mat_fp + dim * dim], &dequant_wq);
                sum_cos_wq += cos_wq;
                if cos_wq < worst_mat_cos { worst_mat_cos = cos_wq; worst_mat_rmse = rmse_wq; worst_mat_name = format!("layer_{}.wq", l); }

                // wk
                let dequant_wk = dequantize_q4_to_vec(&weights_q4.wk[layer_mat_q4..layer_mat_q4 + q4_matrix_blocks]);
                let cos_wk = compute_cosine_similarity(&weights_fp.wk[layer_mat_fp..layer_mat_fp + dim * dim], &dequant_wk);
                let rmse_wk = compute_rmse(&weights_fp.wk[layer_mat_fp..layer_mat_fp + dim * dim], &dequant_wk);
                if cos_wk < worst_mat_cos { worst_mat_cos = cos_wk; worst_mat_rmse = rmse_wk; worst_mat_name = format!("layer_{}.wk", l); }

                // wv
                let dequant_wv = dequantize_q4_to_vec(&weights_q4.wv[layer_mat_q4..layer_mat_q4 + q4_matrix_blocks]);
                let cos_wv = compute_cosine_similarity(&weights_fp.wv[layer_mat_fp..layer_mat_fp + dim * dim], &dequant_wv);
                let rmse_wv = compute_rmse(&weights_fp.wv[layer_mat_fp..layer_mat_fp + dim * dim], &dequant_wv);
                if cos_wv < worst_mat_cos { worst_mat_cos = cos_wv; worst_mat_rmse = rmse_wv; worst_mat_name = format!("layer_{}.wv", l); }

                // wo
                let dequant_wo = dequantize_q4_to_vec(&weights_q4.wo[layer_mat_q4..layer_mat_q4 + q4_matrix_blocks]);
                let cos_wo = compute_cosine_similarity(&weights_fp.wo[layer_mat_fp..layer_mat_fp + dim * dim], &dequant_wo);
                let rmse_wo = compute_rmse(&weights_fp.wo[layer_mat_fp..layer_mat_fp + dim * dim], &dequant_wo);
                if cos_wo < worst_mat_cos { worst_mat_cos = cos_wo; worst_mat_rmse = rmse_wo; worst_mat_name = format!("layer_{}.wo", l); }

                // fc1
                let fc1_fp = l * dim * hidden_dim;
                let fc1_q4 = l * q4_fc1_blocks;
                let dequant_fc1 = dequantize_q4_to_vec(&weights_q4.fc1_weight[fc1_q4..fc1_q4 + q4_fc1_blocks]);
                let cos_fc1 = compute_cosine_similarity(&weights_fp.fc1_weight[fc1_fp..fc1_fp + dim * hidden_dim], &dequant_fc1);
                let rmse_fc1 = compute_rmse(&weights_fp.fc1_weight[fc1_fp..fc1_fp + dim * hidden_dim], &dequant_fc1);
                sum_cos_fc1 += cos_fc1;
                if cos_fc1 < worst_mat_cos { worst_mat_cos = cos_fc1; worst_mat_rmse = rmse_fc1; worst_mat_name = format!("layer_{}.fc1", l); }

                // fc2
                let fc2_fp = l * hidden_dim * dim;
                let fc2_q4 = l * q4_fc2_blocks;
                let dequant_fc2 = dequantize_q4_to_vec(&weights_q4.fc2_weight[fc2_q4..fc2_q4 + q4_fc2_blocks]);
                let cos_fc2 = compute_cosine_similarity(&weights_fp.fc2_weight[fc2_fp..fc2_fp + hidden_dim * dim], &dequant_fc2);
                let rmse_fc2 = compute_rmse(&weights_fp.fc2_weight[fc2_fp..fc2_fp + hidden_dim * dim], &dequant_fc2);
                sum_cos_fc2 += cos_fc2;
                if cos_fc2 < worst_mat_cos { worst_mat_cos = cos_fc2; worst_mat_rmse = rmse_fc2; worst_mat_name = format!("layer_{}.fc2", l); }
            }

            println!(" * Promedio Coseno Atencion (wq): {:.6}", sum_cos_wq / config.n_layers as f32);
            println!(" * Promedio Coseno FFN1 (fc1):    {:.6}", sum_cos_fc1 / config.n_layers as f32);
            println!(" * Promedio Coseno FFN2 (fc2):    {:.6}", sum_cos_fc2 / config.n_layers as f32);
            println!(" * PEOR CASO MATRIZ: {} -> Coseno = {:.6}, RMSE = {:.6}", worst_mat_name, worst_mat_cos, worst_mat_rmse);

            // Forward pass with layer tracking
            println!("\n--- [3] INFERENCIA EN test_peptidase.fasta (SeqLen={}) ---", seq_len);
            let mut state_fp = BidirectionalRunState::new(config, seq_len + 8);
            let mut fp_layers: Vec<Vec<f32>> = Vec::new();
            forward_bidirectional_parallel_ext(&tokens, config, &weights_fp, &mut state_fp, num_threads, Some(&mut |_l, x_slice| {
                fp_layers.push(x_slice.to_vec());
            }));

            let mut state_q4 = BidirectionalRunState::new(config, seq_len + 8);
            let mut q4_layers: Vec<Vec<f32>> = Vec::new();
            forward_bidirectional_q4_ext(&tokens, config, &weights_q4, &mut state_q4, num_threads, Some(&mut |_l, x_slice| {
                q4_layers.push(x_slice.to_vec());
            }));

            let mut state_q8 = BidirectionalRunState::new(config, seq_len + 8);
            forward_bidirectional_q8(&tokens, config, &weights_q8, &mut state_q8, num_threads);

            // Final embeddings fidelity
            let emb_len = seq_len * dim;
            let cos_emb_q4 = compute_cosine_similarity(&state_fp.x[0..emb_len], &state_q4.x[0..emb_len]);
            let rmse_emb_q4 = compute_rmse(&state_fp.x[0..emb_len], &state_q4.x[0..emb_len]);
            let cos_emb_q8 = compute_cosine_similarity(&state_fp.x[0..emb_len], &state_q8.x[0..emb_len]);
            let rmse_emb_q8 = compute_rmse(&state_fp.x[0..emb_len], &state_q8.x[0..emb_len]);

            println!(" * Embeddings Finales FP32 vs Q8_0: Coseno = {:.6}, RMSE = {:.6}", cos_emb_q8, rmse_emb_q8);
            println!(" * Embeddings Finales FP32 vs Q4_0: Coseno = {:.6}, RMSE = {:.6}", cos_emb_q4, rmse_emb_q4);

            // Final logits fidelity
            let log_len = seq_len * config.vocab_size;
            let cos_log_q4 = compute_cosine_similarity(&state_fp.logits[0..log_len], &state_q4.logits[0..log_len]);
            let rmse_log_q4 = compute_rmse(&state_fp.logits[0..log_len], &state_q4.logits[0..log_len]);
            let cos_log_q8 = compute_cosine_similarity(&state_fp.logits[0..log_len], &state_q8.logits[0..log_len]);
            let rmse_log_q8 = compute_rmse(&state_fp.logits[0..log_len], &state_q8.logits[0..log_len]);

            println!(" * Logits Finales FP32 vs Q8_0:     Coseno = {:.6}, RMSE = {:.6}", cos_log_q8, rmse_log_q8);
            println!(" * Logits Finales FP32 vs Q4_0:     Coseno = {:.6}, RMSE = {:.6}", cos_log_q4, rmse_log_q4);

            println!("\n * DESGLOSE CAPA POR CAPA (Coseno FP32 vs Q4_0 de Hidden States):");
            let step = if config.n_layers > 12 { config.n_layers / 6 } else { 1 };
            for l in 0..config.n_layers {
                let cos_l = compute_cosine_similarity(&fp_layers[l], &q4_layers[l]);
                let rmse_l = compute_rmse(&fp_layers[l], &q4_layers[l]);
                if l == 0 || l == config.n_layers - 1 || l % step == 0 {
                    println!("    - Capa {:>2}/{:>2}: Coseno = {:.6}, RMSE = {:.6}", l + 1, config.n_layers, cos_l, rmse_l);
                }
            }

            // Latency benchmark
            println!("\n--- [4] LATENCIA DE INFERENCIA ({} hilos, promedio 5 iteraciones) ---", num_threads);
            // FP32 latency
            let iters = 5;
            for _ in 0..2 { forward_bidirectional_parallel(&tokens, config, &weights_fp, &mut state_fp, num_threads); }
            let t0_fp = Instant::now();
            for _ in 0..iters { forward_bidirectional_parallel(&tokens, config, &weights_fp, &mut state_fp, num_threads); }
            let lat_fp = (t0_fp.elapsed().as_secs_f64() * 1000.0) / iters as f64;

            // Q8_0 latency
            for _ in 0..2 { forward_bidirectional_q8(&tokens, config, &weights_q8, &mut state_q8, num_threads); }
            let t0_q8 = Instant::now();
            for _ in 0..iters { forward_bidirectional_q8(&tokens, config, &weights_q8, &mut state_q8, num_threads); }
            let lat_q8 = (t0_q8.elapsed().as_secs_f64() * 1000.0) / iters as f64;

            // Q4_0 latency
            for _ in 0..2 { forward_bidirectional_q4(&tokens, config, &weights_q4, &mut state_q4, num_threads); }
            let t0_q4 = Instant::now();
            for _ in 0..iters { forward_bidirectional_q4(&tokens, config, &weights_q4, &mut state_q4, num_threads); }
            let lat_q4 = (t0_q4.elapsed().as_secs_f64() * 1000.0) / iters as f64;

            println!(" * Latencia FP32:  {:>8.2} ms", lat_fp);
            println!(" * Latencia Q8_0:  {:>8.2} ms (Speedup: {:.2}x)", lat_q8, lat_fp / lat_q8);
            println!(" * Latencia Q4_0:  {:>8.2} ms (Speedup: {:.2}x)", lat_q4, lat_fp / lat_q4);
            println!();
        }
    }
}

#[cfg(test)]
mod tests_regression {
    use super::*;

    pub fn read_npz_floats(npz_path: &str, target_name: &str) -> io::Result<Vec<f32>> {
        crate::read_npz_floats(npz_path, target_name)
    }

    #[test]
    fn test_strict_fp32_loader_rejects_corrupted_size() {
        let temp_path = "target/test_corrupt_fp32_size.bin";
        {
            let file = File::create(temp_path).expect("Create temp file");
            file.set_len(30_091_656).expect("Set corrupted length (expected 30,091,652)");
        }
        let cfg = EsmConfig::esm2_8m();
        let err = EsmWeights::from_file(temp_path, &cfg)
            .err()
            .expect("Must reject non-canonical FP32 length");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_regression_esm2_fp32_all_scales_against_ref_npz() {
        let _guard = crate::HEAVY_MODEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let fasta_path = "test_peptidase.fasta";
        let raw_seq = if std::path::Path::new(fasta_path).exists() {
            let content = std::fs::read_to_string(fasta_path).expect("Read fasta");
            content.lines().filter(|l| !l.is_empty() && !l.starts_with('>')).collect::<Vec<_>>().join("")
        } else {
            "AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS".to_string()
        };

        let tokens = Tokenizer::encode(&raw_seq);
        let seq_len = tokens.len();
        assert_eq!(seq_len, 41, "Sequence must be 39 aa + BOS/EOS = 41 tokens");

        let scales = [
            ("8M", "esm2_8m_weights.bin", "scripts/ref_8m.npz", EsmConfig::esm2_8m()),
            ("35M", "esm2_35m_weights.bin", "scripts/ref_35m.npz", EsmConfig::esm2_35m()),
            ("150M", "esm2_150m_weights.bin", "scripts/ref_150m.npz", EsmConfig::esm2_150m()),
            ("650M", "esm2_650m_weights.bin", "scripts/ref_650m.npz", EsmConfig::esm2_650m()),
        ];

        for (name, model_path, ref_path, config) in scales {
            if !std::path::Path::new(model_path).exists() || !std::path::Path::new(ref_path).exists() {
                println!("Skipping {} FP32 regression: {} or {} not found", name, model_path, ref_path);
                continue;
            }

            let ref_logits = read_npz_floats(ref_path, "logits.npy").expect("Read ref logits.npy");
            let ref_hs = read_npz_floats(ref_path, "hidden_states.npy").expect("Read ref hidden_states.npy");
            assert_eq!(ref_logits.len(), seq_len * 33, "{} ref logits length mismatch", name);
            assert_eq!(
                ref_hs.len(),
                (config.n_layers + 1) * seq_len * config.dim,
                "{} ref hidden_states length mismatch",
                name
            );

            let weights = EsmWeights::from_file(model_path, &config).expect("Load FP32 weights");
            let mut state = BidirectionalRunState::new(&config, seq_len);
            let mut layer_outputs: Vec<Vec<f32>> = Vec::with_capacity(config.n_layers);
            forward_bidirectional_parallel_ext(
                &tokens,
                &config,
                &weights,
                &mut state,
                4,
                Some(&mut |_l, x_slice| {
                    layer_outputs.push(x_slice.to_vec());
                }),
            );

            // Check intermediate layers (0..n_layers-1) and final post-LN hidden state against hidden_states
            let hs_stride = seq_len * config.dim;
            let mut min_layer_cos = 1.0f32;
            for l in 0..config.n_layers.saturating_sub(1) {
                let ref_l = &ref_hs[(l + 1) * hs_stride..(l + 2) * hs_stride];
                let cos_l = compute_cosine_similarity(&layer_outputs[l], ref_l);
                if cos_l < min_layer_cos {
                    min_layer_cos = cos_l;
                }
            }
            let ref_final_hs = &ref_hs[config.n_layers * hs_stride..(config.n_layers + 1) * hs_stride];
            let rust_final_hs = &state.x[0..hs_stride];
            let hs_cos = compute_cosine_similarity(rust_final_hs, ref_final_hs);

            let rust_logits = &state.logits[0..seq_len * 33];
            let logits_cos = compute_cosine_similarity(rust_logits, &ref_logits);
            let top1_vs_hf = compute_top1_match_rate(&ref_logits, rust_logits, 33);

            println!(
                "{} FP32 Regression vs HF: Min Layer Cos = {:.7}, Final HS Cos = {:.7}, Logits Cos = {:.7}, Top-1 vs HF = {:.2}%",
                name,
                min_layer_cos,
                hs_cos,
                logits_cos,
                top1_vs_hf * 100.0
            );

            assert!(
                min_layer_cos >= 0.99999,
                "{} Min intermediate layer HS cosine ({:.7}) must be >= 0.99999",
                name,
                min_layer_cos
            );
            assert!(
                hs_cos >= 0.99999,
                "{} Final HS cosine ({:.7}) must be >= 0.99999",
                name,
                hs_cos
            );
            assert!(
                logits_cos >= 0.99999,
                "{} Logits cosine ({:.7}) must be >= 0.99999",
                name,
                logits_cos
            );
            assert!(
                (top1_vs_hf - 1.0).abs() < 1e-6,
                "{} Top-1 match vs HF ({:.2}%) must be 100%",
                name,
                top1_vs_hf * 100.0
            );
        }
    }

    #[test]
    fn test_regression_quantized_q8_q4_against_ref_npz_including_3b() {
        let _guard = crate::HEAVY_MODEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let fasta_path = "test_peptidase.fasta";
        let raw_seq = if std::path::Path::new(fasta_path).exists() {
            let content = std::fs::read_to_string(fasta_path).expect("Read fasta");
            content.lines().filter(|l| !l.is_empty() && !l.starts_with('>')).collect::<Vec<_>>().join("")
        } else {
            "AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS".to_string()
        };
        let tokens = Tokenizer::encode(&raw_seq);
        let seq_len = tokens.len();

        let scales = [
            ("8M", "esm2_8m_q8_0.bin", "esm2_8m_q4_0.bin", "scripts/ref_8m.npz", EsmConfig::esm2_8m()),
            ("35M", "esm2_35m_q8_0.bin", "esm2_35m_q4_0.bin", "scripts/ref_35m.npz", EsmConfig::esm2_35m()),
            ("150M", "esm2_150m_q8_0.bin", "esm2_150m_q4_0.bin", "scripts/ref_150m.npz", EsmConfig::esm2_150m()),
            ("650M", "esm2_650m_q8_0.bin", "esm2_650m_q4_0.bin", "scripts/ref_650m.npz", EsmConfig::esm2_650m()),
            ("3B", "esm2_3b_q8_0.bin", "esm2_3b_q4_0.bin", "scripts/ref_3b.npz", EsmConfig::esm2_3b()),
        ];

        for (name, q8_path, q4_path, ref_path, config) in scales {
            if !std::path::Path::new(ref_path).exists() {
                continue;
            }
            let ref_logits = read_npz_floats(ref_path, "logits.npy").expect("Read ref logits");
            let ref_hs = read_npz_floats(ref_path, "hidden_states.npy").expect("Read ref hidden_states");
            let hs_stride = seq_len * config.dim;
            let ref_final_hs = &ref_hs[config.n_layers * hs_stride..(config.n_layers + 1) * hs_stride];

            if std::path::Path::new(q8_path).exists() {
                let (_, weights_q8) = EsmWeightsQ8::load_from_file(q8_path).expect("Load Q8_0");
                let mut state_q8 = BidirectionalRunState::new(&config, seq_len);
                forward_bidirectional_q8(&tokens, &config, &weights_q8, &mut state_q8, 4);
                let cos_hs = compute_cosine_similarity(&state_q8.x[0..hs_stride], ref_final_hs);
                let cos_log = compute_cosine_similarity(&state_q8.logits[0..seq_len * 33], &ref_logits);
                println!("{} Q8_0 vs HF: HS Cos = {:.6}, Logits Cos = {:.6}", name, cos_hs, cos_log);
                assert!(cos_hs >= 0.990, "{} Q8_0 HS cosine {:.6} < 0.990", name, cos_hs);
                assert!(cos_log >= 0.990, "{} Q8_0 Logits cosine {:.6} < 0.990", name, cos_log);
            }

            if std::path::Path::new(q4_path).exists() {
                let (_, weights_q4) = EsmWeightsQ4::load_from_file(q4_path).expect("Load Q4_0");
                let mut state_q4 = BidirectionalRunState::new(&config, seq_len);
                forward_bidirectional_q4(&tokens, &config, &weights_q4, &mut state_q4, 4);
                let cos_hs = compute_cosine_similarity(&state_q4.x[0..hs_stride], ref_final_hs);
                let cos_log = compute_cosine_similarity(&state_q4.logits[0..seq_len * 33], &ref_logits);
                println!("{} Q4_0 vs HF: HS Cos = {:.6}, Logits Cos = {:.6}", name, cos_hs, cos_log);
                assert!(cos_hs >= 0.900, "{} Q4_0 HS cosine {:.6} < 0.900", name, cos_hs);
                assert!(cos_log >= 0.900, "{} Q4_0 Logits cosine {:.6} < 0.900", name, cos_log);
            }
        }
    }
}


