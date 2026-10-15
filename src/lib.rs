#![allow(unused_unsafe, unsafe_op_in_unsafe_fn)]
use std::fs::File;
use std::io::{self, Read};
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
    run_pipeline_dms, run_pipeline_inpaint, run_pipeline_contacts, render_dms_heatmap,
    compute_pipeline_inpaint, compute_pipeline_contacts, InpaintPipelineResult, InpaintSite,
    InpaintCandidate, ContactsPipelineResult,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EsmModelArch {
    Esm2_8M,
    Esm2_35M,
    Esm2_150M,
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

    #[inline]
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn model_name(&self) -> &'static str {
        match self.arch {
            EsmModelArch::Esm2_8M => "ESM-2 8M (esm2_t6_8M_UR50D)",
            EsmModelArch::Esm2_35M => "ESM-2 35M (esm2_t12_35M_UR50D)",
            EsmModelArch::Esm2_150M => "ESM-2 150M (esm2_t30_150M_UR50D)",
        }
    }

    /// Autodetects ESM-2 model architecture from weights file length in bytes.
    pub fn detect_from_file_len(file_len: u64) -> io::Result<Self> {
        let floats = file_len / 4;
        if floats >= 140_000_000 {
            Ok(Self::esm2_150m())
        } else if floats >= 30_000_000 {
            Ok(Self::esm2_35m())
        } else if floats >= 7_000_000 {
            Ok(Self::esm2_8m())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Unrecognized ESM-2 weights size: {} bytes ({} floats). Expected ~30MB (8M), ~133MB (35M), or ~591MB (150M).",
                    file_len, floats
                ),
            ))
        }
    }

    /// Opens file on disk and autodetects ESM-2 configuration.
    pub fn detect_from_file(path: &str) -> io::Result<Self> {
        let metadata = std::fs::metadata(path)?;
        Self::detect_from_file_len(metadata.len())
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
        let mut remaining_bytes = Vec::new();
        file.read_to_end(&mut remaining_bytes)?;
        let classifier_weights: Vec<f32> = remaining_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

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
    let mut buffer = vec![0u8; num_floats * 4]; file.read_exact(&mut buffer)?;
    let mut floats = Vec::with_capacity(num_floats);
    for chunk in buffer.chunks_exact(4) { floats.push(f32::from_le_bytes(chunk.try_into().unwrap())); }
    Ok(floats)
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
        _ => {
            for i in 0..head_dim {
                *out.add(i) += prob * *v.add(i);
            }
        }
    }
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
    for t in 0..=pos {
        let k_ptr = key_cache.as_ptr().add(layer_kv_base + t * dim + q_offset);
        let score = dot_product_head_avx(q_ptr, k_ptr, head_dim);
        *att_head.get_unchecked_mut(t) = score * inv_scale;
    }
    softmax(&mut att_head[0..=pos]);
    xb2_head.fill(0.0);
    let out_ptr = xb2_head.as_mut_ptr();
    for t in 0..=pos {
        let prob = *att_head.get_unchecked(t);
        let v_ptr = value_cache.as_ptr().add(layer_kv_base + t * dim + q_offset);
        accumulate_value_head_avx(out_ptr, v_ptr, prob, head_dim);
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
    let dim = config.dim; let embed_offset = token * dim;
    state.x.copy_from_slice(&weights.token_embedding_table[embed_offset..embed_offset + dim]);
    let head_dim = dim / config.n_heads; let inv_scale = 1.0 / (head_dim as f32).sqrt();
    #[cfg(target_arch = "x86_64")]
    let has_avx = std::is_x86_feature_detected!("avx") && (head_dim == 16 || head_dim == 24 || head_dim == 32);
    for l in 0..config.n_layers {
        let layer_dim_offset = l * dim; let layer_matrix_offset = l * dim * dim;
        layer_norm(&mut state.xb, &state.x, &weights.ln_1_weight[layer_dim_offset..layer_dim_offset + dim], &weights.ln_1_bias[layer_dim_offset..layer_dim_offset + dim]);
        matmul(&mut state.q, &state.xb, &weights.wq[layer_matrix_offset..layer_matrix_offset + (dim * dim)], &weights.bq[layer_dim_offset..layer_dim_offset + dim], dim, dim);
        matmul(&mut state.k, &state.xb, &weights.wk[layer_matrix_offset..layer_matrix_offset + (dim * dim)], &weights.bk[layer_dim_offset..layer_dim_offset + dim], dim, dim);
        matmul(&mut state.v, &state.xb, &weights.wv[layer_matrix_offset..layer_matrix_offset + (dim * dim)], &weights.bv[layer_dim_offset..layer_dim_offset + dim], dim, dim);
        apply_rope(&mut state.q, &mut state.k, &state.rope_cos, &state.rope_sin, pos, dim, config.n_heads);
        let layer_kv_base = l * config.max_seq_len * dim; let kv_offset = layer_kv_base + pos * dim;
        state.key_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.k);
        state.value_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.v);
        for h in 0..config.n_heads {
            let q_offset = h * head_dim; let q_head = &state.q[q_offset..q_offset + head_dim];
            let att_offset = h * config.max_seq_len; let att_head = &mut state.att[att_offset..att_offset + config.max_seq_len];
            let xb2_head = &mut state.xb2[q_offset..q_offset + head_dim];
            #[cfg(target_arch = "x86_64")]
            if has_avx {
                unsafe { attention_head_avx1(q_head, &state.key_cache, &state.value_cache, att_head, xb2_head, layer_kv_base, q_offset, pos, dim, head_dim, inv_scale); }
                continue;
            }
            for t in 0..=pos {
                let cache_offset = layer_kv_base + t * dim + q_offset;
                let k_head = &state.key_cache[cache_offset..cache_offset + head_dim];
                let mut score = 0.0f32; for i in 0..head_dim { score += q_head[i] * k_head[i]; }
                att_head[t] = score * inv_scale;
            }
            softmax(&mut att_head[0..=pos]); xb2_head.fill(0.0);
            for t in 0..=pos {
                let prob = att_head[t]; let cache_offset = layer_kv_base + t * dim + q_offset;
                let v_head = &state.value_cache[cache_offset..cache_offset + head_dim];
                for i in 0..head_dim { xb2_head[i] += prob * v_head[i]; }
            }
        }
        matmul(&mut state.xb, &state.xb2, &weights.wo[layer_matrix_offset..layer_matrix_offset + (dim * dim)], &weights.bo[layer_dim_offset..layer_dim_offset + dim], dim, dim);
        for i in 0..dim { state.x[i] += state.xb[i]; }
        layer_norm(&mut state.xb, &state.x, &weights.ln_2_weight[layer_dim_offset..layer_dim_offset + dim], &weights.ln_2_bias[layer_dim_offset..layer_dim_offset + dim]);
        let hidden_dim = config.hidden_dim; let fc1_w_offset = l * dim * hidden_dim; let fc1_b_offset = l * hidden_dim;
        matmul(&mut state.hb, &state.xb, &weights.fc1_weight[fc1_w_offset..fc1_w_offset + (dim * hidden_dim)], &weights.fc1_bias[fc1_b_offset..fc1_b_offset + hidden_dim], dim, hidden_dim);
        gelu(&mut state.hb2, &state.hb);
        let fc2_w_offset = l * hidden_dim * dim;
        matmul(&mut state.xb, &state.hb2, &weights.fc2_weight[fc2_w_offset..fc2_w_offset + (hidden_dim * dim)], &weights.fc2_bias[layer_dim_offset..layer_dim_offset + dim], hidden_dim, dim);
        for i in 0..dim { state.x[i] += state.xb[i]; }
    }
    layer_norm(&mut state.xb, &state.x, &weights.ln_f_weight, &weights.ln_f_bias);
    const ZERO_BIAS: [f32; 33] = [0.0; 33];
    matmul(&mut state.logits, &state.xb, &weights.token_embedding_table, &ZERO_BIAS, dim, config.vocab_size);
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
    let seq_len = tokens.len();
    assert!(seq_len <= state.max_len, "Sequence length {} exceeds max_len {}", seq_len, state.max_len);
    let dim = config.dim;
    let head_dim = dim / config.n_heads;
    let inv_scale = 1.0 / (head_dim as f32).sqrt();
    let hidden_dim = config.hidden_dim;

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
            matmul(q_slice, &state.xb, &weights.wq[layer_matrix_offset..layer_matrix_offset + dim * dim], &weights.bq[layer_dim_offset..layer_dim_offset + dim], dim, dim);
            let k_slice = &mut state.k[p * dim..(p + 1) * dim];
            matmul(k_slice, &state.xb, &weights.wk[layer_matrix_offset..layer_matrix_offset + dim * dim], &weights.bk[layer_dim_offset..layer_dim_offset + dim], dim, dim);
            let v_slice = &mut state.v[p * dim..(p + 1) * dim];
            matmul(v_slice, &state.xb, &weights.wv[layer_matrix_offset..layer_matrix_offset + dim * dim], &weights.bv[layer_dim_offset..layer_dim_offset + dim], dim, dim);
            apply_rope(q_slice, k_slice, &state.rope_cos, &state.rope_sin, p, dim, config.n_heads);
        }

        let is_last_layer = l == config.n_layers - 1;
        if is_last_layer {
            state.final_layer_att[0..seq_len * seq_len].fill(0.0);
        }

        #[cfg(target_arch = "x86_64")]
        let has_avx = std::is_x86_feature_detected!("avx") && (head_dim == 16 || head_dim == 24 || head_dim == 32);

        for p in 0..seq_len {
            for h in 0..config.n_heads {
                let q_offset = p * dim + h * head_dim;
                let q_head = &state.q[q_offset..q_offset + head_dim];
                let att_head = &mut state.att[h * state.max_len..h * state.max_len + seq_len];

                #[cfg(target_arch = "x86_64")]
                if has_avx {
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
                if has_avx {
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

            let out_slice = &state.out_att[p * dim..(p + 1) * dim];
            matmul(&mut state.xb, out_slice, &weights.wo[layer_matrix_offset..layer_matrix_offset + dim * dim], &weights.bo[layer_dim_offset..layer_dim_offset + dim], dim, dim);
            for i in 0..dim { state.x[p * dim + i] += state.xb[i]; }

            let xp = &state.x[p * dim..(p + 1) * dim];
            layer_norm(&mut state.xb, xp, &weights.ln_2_weight[layer_dim_offset..layer_dim_offset + dim], &weights.ln_2_bias[layer_dim_offset..layer_dim_offset + dim]);
            matmul(&mut state.hb, &state.xb, &weights.fc1_weight[fc1_w_offset..fc1_w_offset + dim * hidden_dim], &weights.fc1_bias[fc1_b_offset..fc1_b_offset + hidden_dim], dim, hidden_dim);
            gelu(&mut state.hb2, &state.hb);
            matmul(&mut state.xb, &state.hb2, &weights.fc2_weight[fc2_w_offset..fc2_w_offset + hidden_dim * dim], &weights.fc2_bias[layer_dim_offset..layer_dim_offset + dim], hidden_dim, dim);
            for i in 0..dim { state.x[p * dim + i] += state.xb[i]; }
        }
    }

    const ZERO_BIAS: [f32; 33] = [0.0; 33];
    for p in 0..seq_len {
        let xp = &state.x[p * dim..(p + 1) * dim];
        layer_norm(&mut state.xb, xp, &weights.ln_f_weight, &weights.ln_f_bias);
        let logits_p = &mut state.logits[p * config.vocab_size..(p + 1) * config.vocab_size];
        matmul(logits_p, &state.xb, &weights.token_embedding_table, &ZERO_BIAS, dim, config.vocab_size);
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

pub fn generate_sequence(prompt: &str, temperature: f32, max_new_tokens: Option<usize>, seed: u64, config: &EsmConfig, weights: &EsmWeights, state: &mut RunState, mut on_token: Option<&mut dyn FnMut(char)>) -> GenerationMetrics {
    let mut prompt_tokens = Tokenizer::encode(prompt); prompt_tokens.pop();
    if prompt_tokens.is_empty() { prompt_tokens.push(0); }
    let mut pos = 0; let prefill_timer = Instant::now();
    for &token in &prompt_tokens { if pos >= config.max_seq_len { break; } forward(token, pos, config, weights, state); pos += 1; }
    let prefill_duration = prefill_timer.elapsed();
    let mut rng = SimpleRng::new(seed); let mut generated_tokens_count = 0; let mut generated_suffix = String::new();
    let max_limit = max_new_tokens.map(|m| (pos + m).min(config.max_seq_len)).unwrap_or(config.max_seq_len);
    let gen_timer = Instant::now();
    while pos < max_limit {
        let next_token = sample(&mut state.logits, temperature, &mut rng);
        if next_token == 2 { break; }
        generated_tokens_count += 1;
        if let Some(c) = Tokenizer::decode(next_token) { generated_suffix.push(c); if let Some(cb) = on_token.as_mut() { cb(c); } }
        forward(next_token, pos, config, weights, state); pos += 1;
    }
    let gen_duration = gen_timer.elapsed();
    let prefill_ms = prefill_duration.as_secs_f64() * 1000.0; let generation_ms = gen_duration.as_secs_f64() * 1000.0;
    let throughput_tps = if gen_duration.as_secs_f32() > 0.0 { generated_tokens_count as f32 / gen_duration.as_secs_f32() } else { 0.0 };
    let mut full_sequence = String::with_capacity(prompt.len() + generated_suffix.len()); full_sequence.push_str(prompt); full_sequence.push_str(&generated_suffix);
    GenerationMetrics { full_sequence, generated_suffix, tokens_generated: generated_tokens_count, prefill_ms, generation_ms, throughput_tps }
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
        // 8M ~ 7.5M floats (30MB)
        let cfg_8m = EsmConfig::detect_from_file_len(30_000_000).expect("8M config detection");
        assert_eq!(cfg_8m.arch, EsmModelArch::Esm2_8M);
        assert_eq!(cfg_8m.dim, 320);
        assert_eq!(cfg_8m.n_layers, 6);
        assert_eq!(cfg_8m.head_dim(), 16);

        // 35M ~ 33.3M floats (133MB)
        let cfg_35m = EsmConfig::detect_from_file_len(133_200_000).expect("35M config detection");
        assert_eq!(cfg_35m.arch, EsmModelArch::Esm2_35M);
        assert_eq!(cfg_35m.dim, 480);
        assert_eq!(cfg_35m.n_layers, 12);
        assert_eq!(cfg_35m.head_dim(), 24);

        // 150M ~ 147.8M floats (591MB)
        let cfg_150m = EsmConfig::detect_from_file_len(591_200_000).expect("150M config detection");
        assert_eq!(cfg_150m.arch, EsmModelArch::Esm2_150M);
        assert_eq!(cfg_150m.dim, 640);
        assert_eq!(cfg_150m.n_layers, 30);
        assert_eq!(cfg_150m.head_dim(), 32);

        // Invalid length
        assert!(EsmConfig::detect_from_file_len(100_000).is_err());
    }

    #[test]
    fn test_attention_simd_parity_dk_16_24_32() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx") {
            for &dk in &[16, 24, 32] {
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
}

