use std::fs::File;
use std::io::{self, Read};

pub mod mcp;

// ==========================================
// 1. CONFIGURACIÓN Y ESTRUCTURAS DE MEMORIA
// ==========================================

#[derive(Debug, Clone)]
pub struct EsmConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

pub struct EsmWeights {
    pub token_embedding_table: Vec<f32>,
    pub emb_ln_weight: Vec<f32>,
    pub emb_ln_bias: Vec<f32>,
    pub ln_1_weight: Vec<f32>,
    pub ln_1_bias: Vec<f32>,
    pub wq: Vec<f32>,
    pub bq: Vec<f32>,
    pub wk: Vec<f32>,
    pub bk: Vec<f32>,
    pub wv: Vec<f32>,
    pub bv: Vec<f32>,
    pub wo: Vec<f32>,
    pub bo: Vec<f32>,
    pub ln_2_weight: Vec<f32>,
    pub ln_2_bias: Vec<f32>,
    pub fc1_weight: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<f32>,
    pub fc2_bias: Vec<f32>,
    pub lm_head_ln_weight: Vec<f32>,
    pub lm_head_ln_bias: Vec<f32>,
    pub lm_head_dense_weight: Vec<f32>,
    pub lm_head_dense_bias: Vec<f32>,
    pub lm_head_weight: Vec<f32>,
    pub lm_head_bias: Vec<f32>,
}

pub struct RunState {
    pub x: Vec<f32>,
    pub xb: Vec<f32>,
    pub xb2: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub att: Vec<f32>,
    pub logits: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
}

impl RunState {
    pub fn new(cfg: &EsmConfig) -> Self {
        let context_len = cfg.max_seq_len;
        Self {
            x: vec![0.0; cfg.dim],
            xb: vec![0.0; cfg.dim],
            xb2: vec![0.0; cfg.dim],
            hb: vec![0.0; cfg.hidden_dim],
            hb2: vec![0.0; cfg.hidden_dim],
            q: vec![0.0; cfg.dim],
            k: vec![0.0; cfg.dim],
            v: vec![0.0; cfg.dim],
            att: vec![0.0; cfg.n_heads * context_len],
            logits: vec![0.0; cfg.vocab_size],
            key_cache: vec![0.0; cfg.n_layers * context_len * cfg.dim],
            value_cache: vec![0.0; cfg.n_layers * context_len * cfg.dim],
        }
    }
}

// ==========================================
// 2. LECTURA DE ARCHIVO BINARIO (I/O)
// ==========================================

fn read_floats(file: &mut File, size: usize) -> io::Result<Vec<f32>> {
    let mut buffer = vec![0u8; size * 4];
    file.read_exact(&mut buffer)?;
    let mut floats = Vec::with_capacity(size);
    for chunk in buffer.chunks(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(floats)
}

impl EsmWeights {
    pub fn load(file: &mut File, config: &EsmConfig) -> io::Result<Self> {
        Ok(Self {
            token_embedding_table: read_floats(file, config.vocab_size * config.dim)?,
            emb_ln_weight: read_floats(file, config.dim)?,
            emb_ln_bias: read_floats(file, config.dim)?,
            ln_1_weight: read_floats(file, config.n_layers * config.dim)?,
            ln_1_bias: read_floats(file, config.n_layers * config.dim)?,
            wq: read_floats(file, config.n_layers * config.dim * config.dim)?,
            bq: read_floats(file, config.n_layers * config.dim)?,
            wk: read_floats(file, config.n_layers * config.dim * config.dim)?,
            bk: read_floats(file, config.n_layers * config.dim)?,
            wv: read_floats(file, config.n_layers * config.dim * config.dim)?,
            bv: read_floats(file, config.n_layers * config.dim)?,
            wo: read_floats(file, config.n_layers * config.dim * config.dim)?,
            bo: read_floats(file, config.n_layers * config.dim)?,
            ln_2_weight: read_floats(file, config.n_layers * config.dim)?,
            ln_2_bias: read_floats(file, config.n_layers * config.dim)?,
            fc1_weight: read_floats(file, config.n_layers * config.hidden_dim * config.dim)?,
            fc1_bias: read_floats(file, config.n_layers * config.hidden_dim)?,
            fc2_weight: read_floats(file, config.n_layers * config.dim * config.hidden_dim)?,
            fc2_bias: read_floats(file, config.n_layers * config.dim)?,
            lm_head_ln_weight: read_floats(file, config.dim)?,
            lm_head_ln_bias: read_floats(file, config.dim)?,
            lm_head_dense_weight: read_floats(file, config.dim * config.dim)?,
            lm_head_dense_bias: read_floats(file, config.dim)?,
            lm_head_weight: read_floats(file, config.vocab_size * config.dim)?,
            lm_head_bias: read_floats(file, config.vocab_size)?,
        })
    }
}

// ==========================================
// 3. PRIMITIVAS MATEMÁTICAS (CERO-ASIGNACIÓN)
// ==========================================

pub fn layer_norm(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32]) {
    let dim = x.len();
    let mut mean = 0.0;
    for &val in x {
        mean += val;
    }
    mean /= dim as f32;

    let mut var = 0.0;
    for &val in x {
        let diff = val - mean;
        var += diff * diff;
    }
    var /= dim as f32;

    let std_dev = (var + 1e-5).sqrt();
    for i in 0..dim {
        out[i] = weight[i] * (x[i] - mean) / std_dev + bias[i];
    }
}

pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: &[f32], in_dim: usize, out_dim: usize) {
    for i in 0..out_dim {
        let mut sum = 0.0;
        let row_offset = i * in_dim;
        for j in 0..in_dim {
            sum += w[row_offset + j] * x[j];
        }
        out[i] = sum + bias[i];
    }
}

pub fn gelu(out: &mut [f32], x: &[f32]) {
    for i in 0..x.len() {
        let val = x[i];
        let erf = libm::erff(val / std::f32::consts::SQRT_2);
        out[i] = 0.5 * val * (1.0 + erf);
    }
}

pub fn softmax(x: &mut [f32]) {
    let mut max_val = x[0];
    for &val in x.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }
    let mut sum = 0.0;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }
    for val in x.iter_mut() {
        *val /= sum;
    }
}

// Exact inv_freq values from PyTorch ESM-2 8M (float32 precision)
// Source: model.layers[0].self_attn.rot_emb.inv_freq
const INV_FREQ: [f32; 8] = [
    1.0,
    0.316_162_1,
    0.099_975_59,
    0.031_616_21,
    0.010_002_136,
    0.003_162_384,
    0.001_000_404_4,
    0.000_316_143,
];

pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, dim: usize, n_heads: usize) {
    let head_dim = dim / n_heads;
    let half_head_dim = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for (i, &freq) in INV_FREQ.iter().enumerate().take(half_head_dim) {
            let idx1 = base + i;
            let idx2 = base + i + half_head_dim;
            let val = pos as f32 * freq;
            let cos = val.cos();
            let sin = val.sin();

            let q1 = q[idx1];
            let q2 = q[idx2];
            q[idx1] = q1 * cos - q2 * sin;
            q[idx2] = q2 * cos + q1 * sin;

            let k1 = k[idx1];
            let k2 = k[idx2];
            k[idx1] = k1 * cos - k2 * sin;
            k[idx2] = k2 * cos + k1 * sin;
        }
    }
}

// ==========================================
// 4. FORWARD PASS Y TOKENIZER
// ==========================================

pub fn forward(token: usize, pos: usize, config: &EsmConfig, weights: &EsmWeights, state: &mut RunState) {
    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let n_heads = config.n_heads;
    let head_dim = dim / n_heads;

    // 1. Token Embedding (embed_scale = 1.0 in ESM-2 eval mode, not sqrt(dim))
    let emb_offset = token * dim;
    for i in 0..dim {
        state.x[i] = weights.token_embedding_table[emb_offset + i];
    }

    // 2. Capas del Transformer
    for l in 0..config.n_layers {
        // Layer Norm 1
        let ln1_w_offset = l * dim;
        layer_norm(
            &mut state.xb,
            &state.x,
            &weights.ln_1_weight[ln1_w_offset..ln1_w_offset + dim],
            &weights.ln_1_bias[ln1_w_offset..ln1_w_offset + dim],
        );

        // Q, K, V Projections
        let layer_offset = l * dim * dim;
        let bias_offset = l * dim;
        // Apply scaling to q (1/sqrt(head_dim)) BEFORE RoPE to match PyTorch exactly
        let scaling = 1.0 / (head_dim as f32).sqrt();
        matmul(
            &mut state.q,
            &state.xb,
            &weights.wq[layer_offset..layer_offset + dim * dim],
            &weights.bq[bias_offset..bias_offset + dim],
            dim,
            dim,
        );
        // Scale q in-place
        for i in 0..dim {
            state.q[i] *= scaling;
        }
        matmul(
            &mut state.k,
            &state.xb,
            &weights.wk[layer_offset..layer_offset + dim * dim],
            &weights.bk[bias_offset..bias_offset + dim],
            dim,
            dim,
        );
        matmul(
            &mut state.v,
            &state.xb,
            &weights.wv[layer_offset..layer_offset + dim * dim],
            &weights.bv[bias_offset..bias_offset + dim],
            dim,
            dim,
        );

        apply_rope(&mut state.q, &mut state.k, pos, dim, n_heads);

        // Guardar en KV Cache
        let cache_offset = l * config.max_seq_len * dim + pos * dim;
        for i in 0..dim {
            state.key_cache[cache_offset + i] = state.k[i];
            state.value_cache[cache_offset + i] = state.v[i];
        }

        // Multi-head Attention
        for h in 0..n_heads {
            let q_offset = h * head_dim;
            let att_offset = h * config.max_seq_len;

            for t in 0..=pos {
                let k_cache_off = l * config.max_seq_len * dim + t * dim + h * head_dim;
                let mut score = 0.0;
                for i in 0..head_dim {
                    score += state.q[q_offset + i] * state.key_cache[k_cache_off + i];
                }
                // Scaling already applied to q before RoPE
                state.att[att_offset + t] = score;
            }

            softmax(&mut state.att[att_offset..=att_offset + pos]);

            let xb_offset = h * head_dim;
            for i in 0..head_dim {
                state.xb[xb_offset + i] = 0.0;
            }
            for t in 0..=pos {
                let v_cache_off = l * config.max_seq_len * dim + t * dim + h * head_dim;
                let a = state.att[att_offset + t];
                for i in 0..head_dim {
                    state.xb[xb_offset + i] += a * state.value_cache[v_cache_off + i];
                }
            }
        }

        // Output Projection
        matmul(
            &mut state.xb2,
            &state.xb,
            &weights.wo[layer_offset..layer_offset + dim * dim],
            &weights.bo[bias_offset..bias_offset + dim],
            dim,
            dim,
        );
        for i in 0..dim {
            state.x[i] += state.xb2[i];
        }

        // Layer Norm 2
        layer_norm(
            &mut state.xb,
            &state.x,
            &weights.ln_2_weight[ln1_w_offset..ln1_w_offset + dim],
            &weights.ln_2_bias[ln1_w_offset..ln1_w_offset + dim],
        );

        // FeedForward (FFN)
        let ffn_w1_off = l * dim * hidden_dim;
        let ffn_b1_off = l * hidden_dim;
        matmul(
            &mut state.hb,
            &state.xb,
            &weights.fc1_weight[ffn_w1_off..ffn_w1_off + dim * hidden_dim],
            &weights.fc1_bias[ffn_b1_off..ffn_b1_off + hidden_dim],
            dim,
            hidden_dim,
        );
        gelu(&mut state.hb2, &state.hb);

        let ffn_w2_off = l * hidden_dim * dim;
        let ffn_b2_off = l * dim;
        matmul(
            &mut state.xb2,
            &state.hb2,
            &weights.fc2_weight[ffn_w2_off..ffn_w2_off + hidden_dim * dim],
            &weights.fc2_bias[ffn_b2_off..ffn_b2_off + dim],
            hidden_dim,
            dim,
        );

        for i in 0..dim {
            state.x[i] += state.xb2[i];
        }
    }

    // 3. Embedding Layer Norm (emb_layer_norm_after) - applied AFTER all transformer layers in ESM-2
    layer_norm(&mut state.xb, &state.x, &weights.emb_ln_weight, &weights.emb_ln_bias);
    for i in 0..dim {
        state.x[i] = state.xb[i];
    }

    // 4. lm_head.dense
    matmul(
        &mut state.xb,
        &state.x,
        &weights.lm_head_dense_weight,
        &weights.lm_head_dense_bias,
        dim,
        dim,
    );
    gelu(&mut state.xb2, &state.xb);

    // 5. lm_head.layer_norm
    layer_norm(
        &mut state.xb,
        &state.xb2,
        &weights.lm_head_ln_weight,
        &weights.lm_head_ln_bias,
    );

    // 6. Classifier (lm_head.out)
    matmul(
        &mut state.logits,
        &state.xb,
        &weights.lm_head_weight,
        &weights.lm_head_bias,
        dim,
        config.vocab_size,
    );
}

pub struct Tokenizer;

impl Tokenizer {
    pub fn encode(text: &str) -> Vec<usize> {
        let mut tokens = vec![0];
        for c in text.chars() {
            let token_id = match c {
                'L' => 4, 'A' => 5, 'G' => 6, 'V' => 7, 'S' => 8, 'E' => 9, 'R' => 10,
                'T' => 11, 'I' => 12, 'D' => 13, 'P' => 14, 'K' => 15, 'Q' => 16, 'N' => 17,
                'F' => 18, 'Y' => 19, 'M' => 20, 'H' => 21, 'W' => 22, 'C' => 23,
                'X' => 24, 'B' => 25, 'U' => 26, 'Z' => 27, 'O' => 28, '.' => 29, '-' => 30,
                _ => 3,
            };
            tokens.push(token_id);
        }
        tokens.push(2);
        tokens
    }

    pub fn decode(token_id: usize) -> Option<char> {
        match token_id {
            4 => Some('L'), 5 => Some('A'), 6 => Some('G'), 7 => Some('V'), 8 => Some('S'),
            9 => Some('E'), 10 => Some('R'), 11 => Some('T'), 12 => Some('I'), 13 => Some('D'),
            14 => Some('P'), 15 => Some('K'), 16 => Some('Q'), 17 => Some('N'), 18 => Some('F'),
            19 => Some('Y'), 20 => Some('M'), 21 => Some('H'), 22 => Some('W'), 23 => Some('C'),
            24 => Some('X'), 25 => Some('B'), 26 => Some('U'), 27 => Some('Z'), 28 => Some('O'),
            29 => Some('.'), 30 => Some('-'),
            _ => None,
        }
    }
}

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x123456789abcdef } else { seed } }
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x & 0xffffffff) as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

pub fn sample(logits: &mut [f32], temperature: f32, rng: &mut SimpleRng) -> usize {
    for (i, logit) in logits.iter_mut().enumerate() {
        let is_standard_amino_acid = (4..=23).contains(&i);
        let is_eos = i == 2;
        if !is_standard_amino_acid && !is_eos {
            *logit = -f32::INFINITY;
        }
    }

    if temperature <= 0.0 {
        let mut max_val = logits[0];
        let mut max_idx = 0;
        for (i, &logit) in logits.iter().enumerate().skip(1) {
            if logit > max_val {
                max_val = logit;
                max_idx = i;
            }
        }
        return max_idx;
    }

    for val in logits.iter_mut() {
        if *val != -f32::INFINITY {
            *val /= temperature;
        }
    }

    softmax(logits);

    let r = rng.next_f32();
    let mut cumulative = 0.0;
    for (i, &prob) in logits.iter().enumerate() {
        if prob.is_nan() || prob == 0.0 {
            continue;
        }
        cumulative += prob;
        if r <= cumulative {
            return i;
        }
    }

    4
}

// ==========================================
// 5. PRUEBAS UNITARIAS
// ==========================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_stability() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];

        layer_norm(&mut output, &input, &weight, &bias);
        assert_eq!(output.len(), 4);
        assert!(!output[0].is_nan());
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tokenizer_boundaries() {
        let seq = "LA";
        let tokens = Tokenizer::encode(seq);
        assert_eq!(tokens, vec![0, 4, 5, 2]);
    }

    #[test]
    fn test_gelu_shape() {
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 4];
        gelu(&mut output, &input);
        assert_eq!(output.len(), 4);
        assert!(!output[0].is_nan());
    }

    #[test]
    fn test_matmul_shape() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let bias = vec![0.0, 0.0];
        let mut out = vec![0.0; 2];
        matmul(&mut out, &x, &w, &bias, 3, 2);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_apply_rope_no_nan() {
        let mut q = vec![1.0; 320];
        let mut k = vec![1.0; 320];
        apply_rope(&mut q, &mut k, 0, 320, 20);
        assert!(!q.iter().any(|v| v.is_nan()));
        assert!(!k.iter().any(|v| v.is_nan()));
    }
}