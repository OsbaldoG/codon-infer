#![allow(unused_unsafe, unsafe_op_in_unsafe_fn)]
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

    /// Autodetects ESM-2 model architecture from weights file length in bytes.
    /// Deterministically supports both FP32 (~4 bytes/param) and Q8_0 (~1.125 bytes/param)
    /// representations across all 5 scales (8M, 35M, 150M, 650M, 3B) without collisions or gaps.
    pub fn detect_from_file_len(file_len: u64) -> io::Result<Self> {
        if file_len >= 3_000_000_000 {
            // ESM-2 3B: ~11.4 GB (FP32) or ~3.2 GB - 3.4 GB (Q8_0)
            Ok(Self::esm2_3b())
        } else if file_len >= 2_000_000_000 {
            // ESM-2 650M: ~2.60 GB (FP32)
            Ok(Self::esm2_650m())
        } else if file_len >= 600_000_000 {
            // ESM-2 650M: ~698.5 MB - 780 MB (Q8_0)
            Ok(Self::esm2_650m())
        } else if file_len >= 450_000_000 {
            // ESM-2 150M: ~592.5 MB (FP32)
            Ok(Self::esm2_150m())
        } else if file_len >= 140_000_000 {
            // ESM-2 150M: ~166 MB - 178 MB (Q8_0)
            Ok(Self::esm2_150m())
        } else if file_len >= 80_000_000 {
            // ESM-2 35M: ~134 MB (FP32)
            Ok(Self::esm2_35m())
        } else if file_len >= 30_500_000 {
            // ESM-2 35M: ~38 MB - 48 MB (Q8_0)
            Ok(Self::esm2_35m())
        } else if file_len >= 20_000_000 {
            // ESM-2 8M: ~30 MB (FP32)
            Ok(Self::esm2_8m())
        } else if file_len >= 6_000_000 {
            // ESM-2 8M: ~8.5 MB - 12 MB (Q8_0)
            Ok(Self::esm2_8m())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Unrecognized ESM-2 weights size: {} bytes. Expected ~30MB/9MB (8M), ~134MB/38MB (35M), ~592MB/167MB (150M), ~2.6GB/698MB (650M), or ~11.4GB/3.2GB (3B).",
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
        let token_embedding_table = read_floats(&mut file, config.vocab_size * config.dim)?;
        let ln_1_weight = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_1_bias = read_floats(&mut file, config.n_layers * config.dim)?;
        let wq = read_floats(&mut file, config.n_layers * config.dim * config.dim)?;
        let bq = read_floats(&mut file, config.n_layers * config.dim)?;
        let wk = read_floats(&mut file, config.n_layers * config.dim * config.dim)?;
        let bk = read_floats(&mut file, config.n_layers * config.dim)?;
        let wv = read_floats(&mut file, config.n_layers * config.dim * config.dim)?;
        let bv = read_floats(&mut file, config.n_layers * config.dim)?;
        let wo = read_floats(&mut file, config.n_layers * config.dim * config.dim)?;
        let bo = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_2_weight = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_2_bias = read_floats(&mut file, config.n_layers * config.dim)?;
        let fc1_weight = read_floats(&mut file, config.n_layers * config.dim * config.hidden_dim)?;
        let fc1_bias = read_floats(&mut file, config.n_layers * config.hidden_dim)?;
        let fc2_weight = read_floats(&mut file, config.n_layers * config.hidden_dim * config.dim)?;
        let fc2_bias = read_floats(&mut file, config.n_layers * config.dim)?;
        let ln_f_weight = read_floats(&mut file, config.dim)?;
        let ln_f_bias = read_floats(&mut file, config.dim)?;

        // Dynamically read remaining bytes until EOF for classifier_weights (EsmLMHead)
        let current_pos = file.stream_position()?;
        let file_len = file.metadata()?.len();
        if current_pos > file_len {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Stream position exceeds file length"));
        }
        let remaining_bytes = (file_len - current_pos) as usize;
        if remaining_bytes % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unaligned trailing bytes: {} is not a multiple of 4", remaining_bytes),
            ));
        }
        let remaining_floats = remaining_bytes / 4;
        let classifier_weights = read_floats(&mut file, remaining_floats)?;

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
            token_embedding_table: quantize_slice_to_q4(&self.token_embedding_table),
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
        let is_q8_file = model_path.ends_with(".q8") || model_path.ends_with(".q8.bin");
        if is_q8_file {
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
        for &v in src {
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

#[allow(clippy::uninit_vec)]
pub fn read_q4_blocks(file: &mut File, num_blocks: usize) -> io::Result<Vec<BlockQ4_0>> {
    let mut blocks = Vec::<BlockQ4_0>::with_capacity(num_blocks);
    unsafe {
        blocks.set_len(num_blocks);
        let byte_slice = std::slice::from_raw_parts_mut(
            blocks.as_mut_ptr() as *mut u8,
            num_blocks * std::mem::size_of::<BlockQ4_0>(),
        );
        file.read_exact(byte_slice)?;
    }
    Ok(blocks)
}

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
    pub token_embedding_table: Vec<BlockQ4_0>,
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
        write_q4_blocks(&mut file, &self.token_embedding_table)?;
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

        let token_embedding_table = read_q4_blocks(&mut file, config.vocab_size * (config.dim / 32))?;
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
        let is_q4_file = model_path.ends_with(".q4")
            || model_path.ends_with(".q4.bin")
            || model_path.ends_with(".q4_0")
            || model_path.ends_with(".q4_0.bin");
        if is_q4_file {
            Self::from_file(model_path, config)
        } else {
            let fp32 = EsmWeights::from_file(model_path, config)?;
            Ok(fp32.quantize_q4())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    Fp32,
    Q8_0,
}

#[derive(Clone, Copy)]
pub enum ModelWeightsRef<'a> {
    Fp32(&'a EsmWeights),
    Q8(&'a EsmWeightsQ8),
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

fn read_floats(file: &mut File, num_floats: usize) -> io::Result<Vec<f32>> {
    let mut floats = Vec::<f32>::with_capacity(num_floats);
    unsafe {
        floats.set_len(num_floats);
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

fn write_floats(file: &mut File, floats: &[f32]) -> io::Result<()> {
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            floats.as_ptr() as *const u8,
            floats.len() * std::mem::size_of::<f32>(),
        )
    };
    file.write_all(byte_slice)
}

fn read_q8_blocks(file: &mut File, num_blocks: usize) -> io::Result<Vec<BlockQ8_0>> {
    let mut blocks = Vec::<BlockQ8_0>::with_capacity(num_blocks);
    unsafe {
        blocks.set_len(num_blocks);
        let byte_slice = std::slice::from_raw_parts_mut(
            blocks.as_mut_ptr() as *mut u8,
            num_blocks * std::mem::size_of::<BlockQ8_0>(),
        );
        file.read_exact(byte_slice)?;
    }
    Ok(blocks)
}

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
// MATMUL Q4_0 KERNELS SKELETON
// ============================================================================

pub fn matmul_q4(
    _out: &mut [f32],
    _x: &[f32],
    _w: &[BlockQ4_0],
    _bias: &[f32],
    _in_dim: usize,
    _out_dim: usize,
) {
    todo!("Subagent B: matmul_q4 scalar reference kernel")
}

pub fn matmul_q4_simd(
    _out: &mut [f32],
    _x: &[f32],
    _w: &[BlockQ4_0],
    _bias: &[f32],
    _in_dim: usize,
    _out_dim: usize,
) {
    todo!("Subagent B: matmul_q4_simd runtime dispatch")
}

pub fn matmul_q4_parallel(
    _out: &mut [f32],
    _x: &[f32],
    _w: &[BlockQ4_0],
    _bias: &[f32],
    _in_dim: usize,
    _out_dim: usize,
    _num_threads: usize,
) {
    todo!("Subagent B: matmul_q4_parallel ThreadPool dispatch")
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
    let embed_offset = token * dim;
    state.x.copy_from_slice(&weights.token_embedding_table[embed_offset..embed_offset + dim]);
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

    // Dequantize embedding for current token
    let embed_blocks_per_row = dim / 32;
    let embed_row = &weights.token_embedding_table[token * embed_blocks_per_row..(token + 1) * embed_blocks_per_row];
    for (b_idx, block) in embed_row.iter().enumerate() {
        block.dequantize_to(&mut state.x[b_idx * 32..(b_idx + 1) * 32]);
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

    for p in 0..seq_len {
        let token = tokens[p];
        let embed_offset = token * dim;
        state.x[p * dim..(p + 1) * dim].copy_from_slice(&weights.token_embedding_table[embed_offset..embed_offset + dim]);
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
    }

    for p in 0..seq_len {
        let xp = &state.x[p * dim..(p + 1) * dim];
        layer_norm(&mut state.xb, xp, &weights.ln_f_weight, &weights.ln_f_bias);
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

    let embed_blocks_per_row = dim / 32;
    for p in 0..seq_len {
        let token = tokens[p];
        let row_blocks = &weights.token_embedding_table[token * embed_blocks_per_row..(token + 1) * embed_blocks_per_row];
        for (b_idx, block) in row_blocks.iter().enumerate() {
            block.dequantize_to(&mut state.x[p * dim + b_idx * 32..p * dim + (b_idx + 1) * 32]);
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
        let logits_p = &mut state.logits[p * config.vocab_size..(p + 1) * config.vocab_size];
        project_lm_head_q8(logits_p, &state.xb, &weights.classifier_weights, &weights.token_embedding_table, config, &mut state.hb, &mut state.hb2, num_threads);
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
        // 8M ~ FP32 (~30MB) and Q8_0 (~8.5-12MB)
        let cfg_8m = EsmConfig::detect_from_file_len(30_092_168).expect("8M FP32 detection");
        assert_eq!(cfg_8m.arch, EsmModelArch::Esm2_8M);
        assert_eq!(cfg_8m.dim, 320);
        assert_eq!(cfg_8m.hidden_dim, 1280);
        assert_eq!(cfg_8m.n_layers, 6);
        assert_eq!(cfg_8m.n_heads, 20);
        assert_eq!(cfg_8m.head_dim(), 16);
        let cfg_8m_q8 = EsmConfig::detect_from_file_len(8_500_000).expect("8M Q8_0 detection");
        assert_eq!(cfg_8m_q8.arch, EsmModelArch::Esm2_8M);

        // 35M ~ FP32 (~134MB) and Q8_0 (~38-48MB)
        let cfg_35m = EsmConfig::detect_from_file_len(134_068_984).expect("35M FP32 detection");
        assert_eq!(cfg_35m.arch, EsmModelArch::Esm2_35M);
        assert_eq!(cfg_35m.dim, 480);
        assert_eq!(cfg_35m.hidden_dim, 1920);
        assert_eq!(cfg_35m.n_layers, 12);
        assert_eq!(cfg_35m.n_heads, 20);
        assert_eq!(cfg_35m.head_dim(), 24);
        let cfg_35m_q8 = EsmConfig::detect_from_file_len(38_000_000).expect("35M Q8_0 detection");
        assert_eq!(cfg_35m_q8.arch, EsmModelArch::Esm2_35M);

        // 150M ~ FP32 (~592MB) and Q8_0 (~166-178MB)
        let cfg_150m = EsmConfig::detect_from_file_len(592_645_160).expect("150M FP32 detection");
        assert_eq!(cfg_150m.arch, EsmModelArch::Esm2_150M);
        assert_eq!(cfg_150m.dim, 640);
        assert_eq!(cfg_150m.hidden_dim, 2560);
        assert_eq!(cfg_150m.n_layers, 30);
        assert_eq!(cfg_150m.n_heads, 20);
        assert_eq!(cfg_150m.head_dim(), 32);
        let cfg_150m_q8 = EsmConfig::detect_from_file_len(170_000_000).expect("150M Q8_0 detection");
        assert_eq!(cfg_150m_q8.arch, EsmModelArch::Esm2_150M);

        // 650M ~ FP32 (~2.60GB) and Q8_0 (~732-780MB)
        let cfg_650m = EsmConfig::detect_from_file_len(2_604_342_104).expect("650M FP32 detection");
        assert_eq!(cfg_650m.arch, EsmModelArch::Esm2_650M);
        assert_eq!(cfg_650m.dim, 1280);
        assert_eq!(cfg_650m.hidden_dim, 5120);
        assert_eq!(cfg_650m.n_layers, 33);
        assert_eq!(cfg_650m.n_heads, 20);
        assert_eq!(cfg_650m.head_dim(), 64);
        let cfg_650m_q8 = EsmConfig::detect_from_file_len(750_000_000).expect("650M Q8_0 detection");
        assert_eq!(cfg_650m_q8.arch, EsmModelArch::Esm2_650M);

        // 3B ~ FP32 (~11.4GB) and Q8_0 (~3.19-3.4GB)
        let cfg_3b = EsmConfig::detect_from_file_len(11_357_091_972).expect("3B FP32 detection");
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
mod tests_q4_format {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip_quantize_dequantize_bound() {
        // Test varying patterns: linear, sinusoidal, and perturbed grid
        let mut test_vectors: Vec<[f32; 32]> = Vec::new();

        // Vector 1: Linear sweep [-8.0 .. 7.0]
        let mut v1 = [0.0f32; 32];
        for i in 0..32 {
            v1[i] = -8.0 + (i as f32) * (15.0 / 31.0);
        }
        test_vectors.push(v1);

        // Vector 2: Exact grid points with controlled noise
        let scales = [0.05f32, 1.0f32, 2.5f32, 10.0f32];
        for &scale in &scales {
            let mut v = [0.0f32; 32];
            // Ensure max produces scale d = scale (so max = -8.0 * scale)
            v[0] = -8.0 * scale;
            for i in 1..32 {
                let q_level = ((i % 16) as i32 - 8) as f32; // -8 .. 7
                let noise = (((i * 11) % 9) as f32 - 4.0) / 10.0 * 0.45 * scale; // in [-0.45*d, 0.45*d]
                v[i] = q_level * scale + noise;
            }
            test_vectors.push(v);
        }

        // Vector 3: Positive max sweep
        for &scale in &scales {
            let mut v = [0.0f32; 32];
            // Positive max: max = 8.0 * scale => d = -scale
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
        // Construct exact known values:
        // Let max = 8.0, so d = 8.0 / -8.0 = -1.0, id = -1.0.
        // For j in 0..16:
        // q_lo = j  => src[j] = 8 - j
        // q_hi = 15 - j => src[j + 16] = j - 7
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

        // Verify exact dequantization back to source floats
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

        // 1. Valid header
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

        // 2. Corrupt magic
        let mut corrupt_magic = header;
        corrupt_magic.magic = *b"NOPE";
        assert!(corrupt_magic.validate(&config).is_err());
        assert!(corrupt_magic.to_config().is_err());

        // 3. Corrupt version
        let mut corrupt_version = header;
        corrupt_version.version = 99;
        assert!(corrupt_version.validate(&config).is_err());
        assert!(corrupt_version.to_config().is_err());

        // 4. Incoherent dimensions
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

        let dummy_q4 = quantize_slice_to_q4(&vec![0.5f32; config.vocab_size * config.dim]);
        let w = EsmWeightsQ4 {
            token_embedding_table: dummy_q4.clone(),
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
        // A 650M model in Q4_0 is ~410MB.
        // Legacy file length detection would map 410MB into 150M range:
        // [140MB, 450MB) -> Esm2_150M.
        let file_len_410mb = 410_000_000u64;
        let legacy_detected = EsmConfig::detect_from_file_len(file_len_410mb).unwrap();
        assert_eq!(legacy_detected.arch, EsmModelArch::Esm2_150M, "Legacy size detection maps 410MB to 150M");

        // Now create a simulated 410MB file with CODQ header for 650M
        let temp_path = "target/test_no_ambiguity_650m_q4.bin";
        {
            let mut file = File::create(temp_path).expect("Create file");
            let config_650m = EsmConfig::esm2_650m();
            let header = CodqHeader::new(DTYPE_Q4_0, &config_650m);
            header.write_to(&mut file).expect("Write header");
            // Set file length to 410MB sparsely without disk cost
            file.set_len(file_len_410mb).expect("Set len");
        }

        // Autodetection with CODQ header: MUST detect 650M, NOT 150M!
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


