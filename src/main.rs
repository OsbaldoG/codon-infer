use std::fs::File;
use std::io::{self, Read};

// 1. Estructura de Configuración
#[derive(Debug, Clone)]
pub struct EsmConfig {
    pub dim: usize,           
    pub hidden_dim: usize,    
    pub n_layers: usize,      
    pub n_heads: usize,       
    pub vocab_size: usize,    
    pub max_seq_len: usize,   
}

// 2. Estructura de Pesos (RoBERTa + Cabezas de Clasificación)
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

// 3. Estructura de Estado de Ejecución (Memoria de Trabajo Temporal)
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

// Implementación estricta para asignar la memoria de trabajo en un solo bloque
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
            
            // Caché KV: Esencial para evitar recalcular tokens anteriores
            key_cache: vec![0.0; config.n_layers * config.max_seq_len * config.dim],
            value_cache: vec![0.0; config.n_layers * config.max_seq_len * config.dim],
        }
    }
}

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

// --- Compute Spec: Funciones Matemáticas ---

pub fn layer_norm(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32]) {
    let dim = x.len();
    const EPSILON: f32 = 1e-5;

    // 1. Calcular la media
    let mut mean = 0.0;
    for i in 0..dim {
        mean += x[i];
    }
    mean /= dim as f32;

    // 2. Calcular la varianza
    let mut variance = 0.0;
    for i in 0..dim {
        let diff = x[i] - mean;
        variance += diff * diff;
    }
    variance /= dim as f32;

    // 3. Normalizar, escalar (weight) y desplazar (bias)
    let inv_stddev = 1.0 / (variance + EPSILON).sqrt();
    for i in 0..dim {
        out[i] = weight[i] * (x[i] - mean) * inv_stddev + bias[i];
    }
}

pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: &[f32], in_dim: usize, out_dim: usize) {
    // Iteramos sobre cada dimensión de salida (cada neurona/fila de la matriz)
    for i in 0..out_dim {
        let mut val = 0.0;
        // Producto punto entre el vector de entrada 'x' y la fila 'i' de la matriz 'w'
        for j in 0..in_dim {
            val += w[i * in_dim + j] * x[j];
        }
        // Asignamos el resultado al buffer de salida sumando el sesgo
        out[i] = val + bias[i];
    }
}

// --- Primitivas No Lineales ---

pub fn gelu(out: &mut [f32], x: &[f32]) {
    for i in 0..x.len() {
        let x0 = x[i];
        let cube = 0.044715 * x0 * x0 * x0;
        let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (x0 + cube);
        out[i] = 0.5 * x0 * (1.0 + inner.tanh());
    }
}

pub fn softmax(x: &mut [f32]) {
    // 1. Encontrar el valor máximo para estabilidad numérica (evitar que exp() explote a infinito)
    let mut max_val = x[0];
    for &val in x.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }
    
    // 2. Calcular los exponenciales y la suma total
    let mut sum = 0.0;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }
    
    // 3. Dividir cada elemento por la suma para obtener probabilidades
    for val in x.iter_mut() {
        *val /= sum;
    }
}

pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, dim: usize, n_heads: usize) {
    let head_dim = dim / n_heads;
    
    for h in 0..n_heads {
        // Encontramos el inicio del bloque de memoria para este cabezal
        let head_offset = h * head_dim;
        
        // RoPE opera en pares de dimensiones consecutivas
        for i in (0..head_dim).step_by(2) {
            // Calcular la frecuencia theta para el par de dimensiones actual
            let freq = 1.0 / 10000.0f32.powf(i as f32 / head_dim as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();

            // Índices absolutos en el vector completo (320 dimensiones)
            let idx0 = head_offset + i;
            let idx1 = head_offset + i + 1;

            // Leer valores originales
            let q0 = q[idx0];
            let q1 = q[idx1];
            let k0 = k[idx0];
            let k1 = k[idx1];

            // Rotar Q in-place
            q[idx0] = q0 * fcr - q1 * fci;
            q[idx1] = q0 * fci + q1 * fcr;

            // Rotar K in-place
            k[idx0] = k0 * fcr - k1 * fci;
            k[idx1] = k0 * fci + k1 * fcr;
        }
    }
}

// --- Propagación hacia adelante (Forward Pass) ---

pub fn forward(token: usize, pos: usize, config: &EsmConfig, weights: &EsmWeights, state: &mut RunState) {
    let dim = config.dim;
    
    // 1. Extracción del Embedding Inicial
    // Copiamos el vector biológico de 320 dimensiones correspondiente al token actual
    let embed_offset = token * dim;
    state.x.copy_from_slice(&weights.token_embedding_table[embed_offset..embed_offset + dim]);

    // 2. Iteración a través de las 6 capas del Transformer
    for l in 0..config.n_layers {
        // Calculamos los saltos de memoria para ubicar los tensores de esta capa específica
        let layer_dim_offset = l * dim;
        let layer_matrix_offset = l * dim * dim;

        // --- INICIO DEL BLOQUE DE ATENCIÓN ---
        
        // a) Normalización de entrada (Layer Norm 1)
        layer_norm(
            &mut state.xb, 
            &state.x, 
            &weights.ln_1_weight[layer_dim_offset..layer_dim_offset + dim], 
            &weights.ln_1_bias[layer_dim_offset..layer_dim_offset + dim]
        );

        // b) Proyecciones Q, K, V (Matmul)
        matmul(
            &mut state.q, 
            &state.xb, 
            &weights.wq[layer_matrix_offset..layer_matrix_offset + (dim * dim)], 
            &weights.bq[layer_dim_offset..layer_dim_offset + dim], 
            dim, dim
        );
        matmul(
            &mut state.k, 
            &state.xb, 
            &weights.wk[layer_matrix_offset..layer_matrix_offset + (dim * dim)], 
            &weights.bk[layer_dim_offset..layer_dim_offset + dim], 
            dim, dim
        );
        matmul(
            &mut state.v, 
            &state.xb, 
            &weights.wv[layer_matrix_offset..layer_matrix_offset + (dim * dim)], 
            &weights.bv[layer_dim_offset..layer_dim_offset + dim], 
            dim, dim
        );

        // c) Inyección Posicional Relativa
        apply_rope(&mut state.q, &mut state.k, pos, dim, config.n_heads);
        
        // TODO: d) Caché KV y Cálculo de Atención (Multi-Head Score)
        // --- d) Caché KV (Contexto histórico) ---
        let kv_offset = l * config.max_seq_len * dim + pos * dim;
        state.key_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.k);
        state.value_cache[kv_offset..kv_offset + dim].copy_from_slice(&state.v);

        // --- e) Atención Multicabezal (Multi-Head Attention) ---
        let head_dim = dim / config.n_heads;
        let scale = (head_dim as f32).sqrt();

        for h in 0..config.n_heads {
            let q_offset = h * head_dim;
            
            // 1. Calcular puntuación de atención contra todos los tokens en el tiempo (0 hasta pos)
            for t in 0..=pos {
                let cache_offset = l * config.max_seq_len * dim + t * dim + h * head_dim;
                let mut score = 0.0;
                for i in 0..head_dim {
                    score += state.q[q_offset + i] * state.key_cache[cache_offset + i];
                }
                score /= scale;
                state.att[h * config.max_seq_len + t] = score; // Guardar score crudo
            }
            
            // 2. Softmax: Transforma scores en probabilidades (Cero-asignación mediante slicing)
            softmax(&mut state.att[(h * config.max_seq_len)..(h * config.max_seq_len + pos + 1)]);
            
            // 3. Multiplicar probabilidades por Values (V) y acumular en búfer secundario xb2
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
        // Proyectar resultado consolidado de atención (xb2) hacia xb usando pesos de salida (wo)
        matmul(
            &mut state.xb, 
            &state.xb2, 
            &weights.wo[layer_matrix_offset..layer_matrix_offset + (dim * dim)], 
            &weights.bo[layer_dim_offset..layer_dim_offset + dim], 
            dim, dim
        );
        
        // Conexión Residual: Sumar al vector original
        for i in 0..dim {
            state.x[i] += state.xb[i];
        }

    // --- g) Bloque FeedForward (Expansión Biológica) ---
        // 1. Normalización previa al FFN (Layer Norm 2)
        layer_norm(
            &mut state.xb, 
            &state.x, 
            &weights.ln_2_weight[layer_dim_offset..layer_dim_offset + dim], 
            &weights.ln_2_bias[layer_dim_offset..layer_dim_offset + dim]
        );

        // 2. Expansión densa (fc1: 320 -> 1280)
        let hidden_dim = config.hidden_dim;
        let fc1_w_offset = l * dim * hidden_dim;
        let fc1_b_offset = l * hidden_dim;
        
        matmul(
            &mut state.hb, 
            &state.xb, 
            &weights.fc1_weight[fc1_w_offset..fc1_w_offset + (dim * hidden_dim)], 
            &weights.fc1_bias[fc1_b_offset..fc1_b_offset + hidden_dim], 
            dim, hidden_dim
        );

        // 3. Activación biológica no lineal
        gelu(&mut state.hb2, &state.hb);

        // 4. Compresión de vuelta (fc2: 1280 -> 320)
        let fc2_w_offset = l * hidden_dim * dim;
        
        matmul(
            &mut state.xb, 
            &state.hb2, 
            &weights.fc2_weight[fc2_w_offset..fc2_w_offset + (hidden_dim * dim)], 
            &weights.fc2_bias[layer_dim_offset..layer_dim_offset + dim], // ESM-2 usa dim para los sesgos de fc2
            hidden_dim, dim
        );

        // 5. Conexión Residual 2 (Sumar al vector original)
        for i in 0..dim {
            state.x[i] += state.xb[i];
        }

    }
}




fn main() -> io::Result<()> {
    println!("--- Codon-Infer Engine V0.4.2 ---");
    println!("Alineación final de memoria y prueba de RunState...\n");

    let config = EsmConfig {
        dim: 320,
        hidden_dim: 1280,
        n_layers: 6,
        n_heads: 20,
        vocab_size: 33,
        max_seq_len: 1024,
    };

    let mut file = File::open("esm2_pesos.bin")?;

    let _weights = EsmWeights {
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

    println!("¡Éxito Absoluto! Anatomía base mapeada.");

    // --- ESTO SE QUEDA (Nota el 'mut' añadido a run_state) ---
    println!("\nIniciando reserva estricta de memoria de trabajo (RunState)...");
    let mut run_state = RunState::new(&config);
    println!("Reserva en el Heap completada.");
    
    let kv_cache_size = run_state.key_cache.len() + run_state.value_cache.len();
    println!("- Memoria KV Cache asignada: {} parámetros (flotantes)", kv_cache_size);

    // --- ESTO ES LO NUEVO (Reemplaza a todas las pruebas matemáticas anteriores) ---
    
    println!("\nSometiendo a prueba el bucle Forward (Fase 3: Ciclo completo por capas con FFN)...");
    forward(1, 0, &config, &_weights, &mut run_state);
    println!("Fase 3 exitosa. Primer valor del vector estado X al salir de las 6 capas: {:.6}", run_state.x[0]);

    Ok(())
}