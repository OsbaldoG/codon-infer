use std::fs::File;
use std::io::{self, Read};

// ==========================================
// 1. ESPECIFICACIÓN DE ARQUITECTURA
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
    pub ln_f_weight: Vec<f32>,
    pub ln_f_bias: Vec<f32>,
    pub classifier_weights: Vec<f32>,
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
    pub fn new(config: &EsmConfig) -> Self {
        Self {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim],
            k: vec![0.0; config.dim],
            v: vec![0.0; config.dim],
            att: vec![0.0; config.n_heads * config.max_seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; config.n_layers * config.max_seq_len * config.dim],
            value_cache: vec![0.0; config.n_layers * config.max_seq_len * config.dim],
        }
    }
}

// ==========================================
// 2. ESPECIFICACIÓN DE I/O (DATOS)
// ==========================================

fn read_floats(file: &mut File, num_floats: usize) -> io::Result<Vec<f32>> {
    let mut buffer = vec![0u8; num_floats * 4]; 
    file.read_exact(&mut buffer)?;
    
    let mut floats = Vec::with_capacity(num_floats);
    for chunk in buffer.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().unwrap();
        floats.push(f32::from_le_bytes(bytes)); 
    }
    Ok(floats)
}

// ==========================================
// 3. ESPECIFICACIÓN DE CÓMPUTO (PRIMITIVAS)
// ==========================================

pub fn layer_norm(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32]) {
    let dim = x.len();
    const EPSILON: f32 = 1e-5;

    let mut mean = 0.0;
    for i in 0..dim {
        mean += x[i];
    }
    mean /= dim as f32;

    let mut variance = 0.0;
    for i in 0..dim {
        let diff = x[i] - mean;
        variance += diff * diff;
    }
    variance /= dim as f32;

    let inv_stddev = 1.0 / (variance + EPSILON).sqrt();
    for i in 0..dim {
        out[i] = weight[i] * (x[i] - mean) * inv_stddev + bias[i];
    }
}

pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: &[f32], in_dim: usize, out_dim: usize) {
    for i in 0..out_dim {
        let mut val = 0.0;
        for j in 0..in_dim {
            val += w[i * in_dim + j] * x[j];
        }
        out[i] = val + bias[i];
    }
}

pub fn gelu(out: &mut [f32], x: &[f32]) {
    for i in 0..x.len() {
        let x0 = x[i];
        let cube = 0.044715 * x0 * x0 * x0;
        let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (x0 + cube);
        out[i] = 0.5 * x0 * (1.0 + inner.tanh());
    }
}

pub fn softmax(x: &mut [f32]) {
    let mut max_val = x[0];
    for &val in x.iter().skip(1) {
        if val > max_val { max_val = val; }
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

pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, dim: usize, n_heads: usize) {
    let head_dim = dim / n_heads;
    for h in 0..n_heads {
        let head_offset = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / 10000.0f32.powf(i as f32 / head_dim as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();

            let idx0 = head_offset + i;
            let idx1 = head_offset + i + 1;

            let q0 = q[idx0];
            let q1 = q[idx1];
            let k0 = k[idx0];
            let k1 = k[idx1];

            q[idx0] = q0 * fcr - q1 * fci;
            q[idx1] = q0 * fci + q1 * fcr;
            k[idx0] = k0 * fcr - k1 * fci;
            k[idx1] = k0 * fci + k1 * fcr;
        }
    }
}

// ==========================================
// 4. ORQUESTADOR PRINCIPAL (FORWARD PASS)
// ==========================================

pub fn forward(token: usize, pos: usize, config: &EsmConfig, weights: &EsmWeights, state: &mut RunState) {
    let dim = config.dim;
    
    // 1. Extracción del Embedding Inicial
    let embed_offset = token * dim;
    state.x.copy_from_slice(&weights.token_embedding_table[embed_offset..embed_offset + dim]);

    // 2. Iteración a través de las 6 capas del Transformer
    for l in 0..config.n_layers {
        let layer_dim_offset = l * dim;
        let layer_matrix_offset = l * dim * dim;

        // --- a) Normalización de entrada (Layer Norm 1) ---
        layer_norm(
            &mut state.xb, 
            &state.x, 
            &weights.ln_1_weight[layer_dim_offset..layer_dim_offset + dim], 
            &weights.ln_1_bias[layer_dim_offset..layer_dim_offset + dim]
        );

        // --- b) Proyecciones Q, K, V ---
        matmul(&mut state.q, &state.xb, &weights.wq[layer_matrix_offset..layer_matrix_offset + (dim * dim)], &weights.bq[layer_dim_offset..layer_dim_offset + dim], dim, dim);
        matmul(&mut state.k, &state.xb, &weights.wk[layer_matrix_offset..layer_matrix_offset + (dim * dim)], &weights.bk[layer_dim_offset..layer_dim_offset + dim], dim, dim);
        matmul(&mut state.v, &state.xb, &weights.wv[layer_matrix_offset..layer_matrix_offset + (dim * dim)], &weights.bv[layer_dim_offset..layer_dim_offset + dim], dim, dim);

        // --- c) Inyección Posicional Relativa ---
        apply_rope(&mut state.q, &mut state.k, pos, dim, config.n_heads);
        
        // --- d) Caché KV (Contexto histórico) ---
        let kv_offset = l * config.max_seq_len * dim + pos * dim;
        state.key_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.k);
        state.value_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.v);

        // --- e) Atención Multicabezal ---
        let head_dim = dim / config.n_heads;
        let scale = (head_dim as f32).sqrt();

        for h in 0..config.n_heads {
            let q_offset = h * head_dim;
            
            for t in 0..=pos {
                let cache_offset = l * config.max_seq_len * dim + t * dim + h * head_dim;
                let mut score = 0.0;
                for i in 0..head_dim {
                    score += state.q[q_offset + i] * state.key_cache[cache_offset + i];
                }
                score /= scale;
                state.att[h * config.max_seq_len + t] = score; 
            }
            
            softmax(&mut state.att[(h * config.max_seq_len)..(h * config.max_seq_len + pos + 1)]);
            
            for i in 0..head_dim {
                state.xb2[q_offset + i] = 0.0;
            }
            
            for t in 0..=pos {
                let prob = state.att[h * config.max_seq_len + t];
                let cache_offset = l * config.max_seq_len * dim + t * dim + h * head_dim;
                for i in 0..head_dim {
                    state.xb2[q_offset + i] += prob * state.value_cache[cache_offset + i];
                }
            }
        }

        // --- f) Proyección de Salida y Conexión Residual 1 ---
        matmul(&mut state.xb, &state.xb2, &weights.wo[layer_matrix_offset..layer_matrix_offset + (dim * dim)], &weights.bo[layer_dim_offset..layer_dim_offset + dim], dim, dim);
        for i in 0..dim {
            state.x[i] += state.xb[i];
        }

        // --- g) Bloque FeedForward ---
        layer_norm(&mut state.xb, &state.x, &weights.ln_2_weight[layer_dim_offset..layer_dim_offset + dim], &weights.ln_2_bias[layer_dim_offset..layer_dim_offset + dim]);

        let hidden_dim = config.hidden_dim;
        let fc1_w_offset = l * dim * hidden_dim;
        let fc1_b_offset = l * hidden_dim;
        matmul(&mut state.hb, &state.xb, &weights.fc1_weight[fc1_w_offset..fc1_w_offset + (dim * hidden_dim)], &weights.fc1_bias[fc1_b_offset..fc1_b_offset + hidden_dim], dim, hidden_dim);

        gelu(&mut state.hb2, &state.hb);

        let fc2_w_offset = l * hidden_dim * dim;
        matmul(&mut state.xb, &state.hb2, &weights.fc2_weight[fc2_w_offset..fc2_w_offset + (hidden_dim * dim)], &weights.fc2_bias[layer_dim_offset..layer_dim_offset + dim], hidden_dim, dim);

        for i in 0..dim {
            state.x[i] += state.xb[i];
        }
    }

    // --- 3. Clasificación Final (Language Modeling Head) ---
    layer_norm(&mut state.xb, &state.x, &weights.ln_f_weight, &weights.ln_f_bias);

    const ZERO_BIAS: [f32; 33] = [0.0; 33];
    matmul(&mut state.logits, &state.xb, &weights.token_embedding_table, &ZERO_BIAS, dim, config.vocab_size);
}

// ==========================================
// 5. TOKENIZER (TRADUCCIÓN BIOLÓGICA)
// ==========================================

pub struct Tokenizer;

impl Tokenizer {
    // Convierte un string biológico en un arreglo de Token IDs
    pub fn encode(text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        
        // 1. Regla de Frontera: Añadir Token <cls> al inicio
        tokens.push(0);

        // 2. Mapeo de caracteres
        for c in text.chars() {
            let token_id = match c {
                'L' => 4, 'A' => 5, 'G' => 6, 'V' => 7, 'S' => 8, 'E' => 9, 'R' => 10,
                'T' => 11, 'I' => 12, 'D' => 13, 'P' => 14, 'K' => 15, 'Q' => 16, 'N' => 17,
                'F' => 18, 'Y' => 19, 'M' => 20, 'H' => 21, 'W' => 22, 'C' => 23,
                'X' => 24, 'B' => 25, 'U' => 26, 'Z' => 27, 'O' => 28, '.' => 29, '-' => 30,
                _ => 3, // Token <unk> para caracteres no reconocidos
            };
            tokens.push(token_id);
        }

        // 3. Regla de Frontera: Añadir Token <eos> al final
        tokens.push(2);
        
        tokens
    }

    // Convierte un Token ID numérico de vuelta a un caracter biológico
    pub fn decode(token_id: usize) -> Option<char> {
        match token_id {
            4 => Some('L'), 5 => Some('A'), 6 => Some('G'), 7 => Some('V'), 8 => Some('S'),
            9 => Some('E'), 10 => Some('R'), 11 => Some('T'), 12 => Some('I'), 13 => Some('D'),
            14 => Some('P'), 15 => Some('K'), 16 => Some('Q'), 17 => Some('N'), 18 => Some('F'),
            19 => Some('Y'), 20 => Some('M'), 21 => Some('H'), 22 => Some('W'), 23 => Some('C'),
            24 => Some('X'), 25 => Some('B'), 26 => Some('U'), 27 => Some('Z'), 28 => Some('O'),
            29 => Some('.'), 30 => Some('-'),
            // Ignorar tokens de control (0, 1, 2, 3) y tokens especiales (31, 32) silenciosamente
            _ => None,
        }
    }
}

// ==========================================
// 6. EJECUCIÓN (MAIN)
// ==========================================

fn main() -> io::Result<()> {
    println!("--- Codon-Infer Engine V0.4.2 ---");
    println!("Iniciando inferencia estricta...\n");

    let config = EsmConfig {
        dim: 320,
        hidden_dim: 1280,
        n_layers: 6,
        n_heads: 20,
        vocab_size: 33,
        max_seq_len: 1024,
    };

    let mut file = File::open("esm2_pesos.bin")?;

    let weights = EsmWeights {
        token_embedding_table: read_floats(&mut file, config.vocab_size * config.dim)?,
        ln_1_weight: read_floats(&mut file, config.n_layers * config.dim)?,
        ln_1_bias:   read_floats(&mut file, config.n_layers * config.dim)?,
        wq: read_floats(&mut file, config.n_layers * config.dim * config.dim)?,
        bq: read_floats(&mut file, config.n_layers * config.dim)?,
        wk: read_floats(&mut file, config.n_layers * config.dim * config.dim)?,
        bk: read_floats(&mut file, config.n_layers * config.dim)?,
        wv: read_floats(&mut file, config.n_layers * config.dim * config.dim)?,
        bv: read_floats(&mut file, config.n_layers * config.dim)?,
        wo: read_floats(&mut file, config.n_layers * config.dim * config.dim)?,
        bo: read_floats(&mut file, config.n_layers * config.dim)?,
        ln_2_weight: read_floats(&mut file, config.n_layers * config.dim)?,
        ln_2_bias:   read_floats(&mut file, config.n_layers * config.dim)?,
        fc1_weight: read_floats(&mut file, config.n_layers * config.dim * config.hidden_dim)?,
        fc1_bias:   read_floats(&mut file, config.n_layers * config.hidden_dim)?,
        fc2_weight: read_floats(&mut file, config.n_layers * config.hidden_dim * config.dim)?,
        fc2_bias:   read_floats(&mut file, config.n_layers * config.dim)?,
        ln_f_weight: read_floats(&mut file, config.dim)?,
        ln_f_bias:   read_floats(&mut file, config.dim)?,
        classifier_weights: read_floats(&mut file, 102_849)?,
    };

    let mut run_state = RunState::new(&config);
    
    // Simulamos el paso del token 1 en la posición 0
    println!("\nPrueba del Tokenizer...");
    let seq = "MKTV";
    let tokens = Tokenizer::encode(seq);
    println!("Secuencia '{}' codificada: {:?}", seq, tokens);
    
    print!("Decodificación inversa: ");
    for &t in &tokens {
        if let Some(c) = Tokenizer::decode(t) {
            print!("{}", c);
        }
    }
    println!("\n(Nota: Los tokens 0 y 2 fueron filtrados por seguridad, como dictan las reglas).");

    Ok(())
}