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

    // Prueba de esfuerzo de memoria RAM
    println!("\nIniciando reserva estricta de memoria de trabajo (RunState)...");
    let run_state = RunState::new(&config);
    println!("Reserva en el Heap completada.");
    
    let kv_cache_size = run_state.key_cache.len() + run_state.value_cache.len();
    println!("- Memoria KV Cache asignada: {} parámetros (flotantes)", kv_cache_size);

    // Prueba de esfuerzo matemático: Normalización de Capa
    println!("\nSometiendo a prueba matemática: layer_norm...");
    let dummy_input = vec![1.0; config.dim];
    let mut dummy_output = vec![0.0; config.dim];
    
    layer_norm(
        &mut dummy_output, 
        &dummy_input, 
        // Extraemos solo los pesos correspondientes a la capa 0
        &_weights.ln_1_weight[0..config.dim], 
        &_weights.ln_1_bias[0..config.dim]
    );

    println!("Operación exitosa. Primer valor normalizado: {:.6}", dummy_output[0]);

    // Prueba de esfuerzo matemático: Multiplicación de Matrices (Q Projection)
    println!("\nSometiendo a prueba matemática: matmul (Proyección Query)...");
    
    // Usamos el resultado de la normalización anterior como entrada
    let mut q_out = vec![0.0; config.dim];
    
    matmul(
        &mut q_out,
        &dummy_output, // El vector x que normalizamos en el paso anterior
        &_weights.wq[0..(config.dim * config.dim)], // Matriz de pesos W (320x320)
        &_weights.bq[0..config.dim],                // Vector de sesgo B (320)
        config.dim,
        config.dim
    );

    println!("Operación exitosa. Primer valor proyectado (Q): {:.6}", q_out[0]);
    
    // Prueba de esfuerzo matemático: Activación GELU
    println!("\nSometiendo a prueba matemática: gelu (Activación no lineal)...");
    
    let mut gelu_out = vec![0.0; config.dim];
    gelu(&mut gelu_out, &q_out);

    println!("Operación exitosa. Primer valor tras GELU: {:.6}", gelu_out[0]);
    
    // Prueba de esfuerzo matemático: Softmax (Atención simulada)
    println!("\nSometiendo a prueba matemática: softmax (Distribución de probabilidad)...");
    let mut softmax_test = vec![1.0, 2.0, 3.0];
    softmax(&mut softmax_test);
    println!("Operación exitosa. Probabilidades resultantes: [{:.4}, {:.4}, {:.4}]", 
             softmax_test[0], softmax_test[1], softmax_test[2]);

    Ok(())
}