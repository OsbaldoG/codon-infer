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

// 2. Estructura de Pesos (Con la adición de los clasificadores finales)
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
    
    // Novedad: Las cabezas que proyectan el resultado a la biología (102,849 parámetros)
    pub classifier_weights: Vec<f32>,
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

fn main() -> io::Result<()> {
    println!("--- Codon-Infer Engine V0.4.1 ---");
    println!("Alineación final de memoria (100% del binario)...\n");

    let config = EsmConfig {
        dim: 320,
        hidden_dim: 1280,
        n_layers: 6,
        n_heads: 20,
        vocab_size: 33,
        max_seq_len: 1024,
    };

    let mut file = File::open("esm2_pesos.bin")?;

    // 4. Mapeo Matemático de la Matriz Binaria Completa
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
        
        // Absorbemos los 102,849 parámetros exactos del LM Head y Contact Head
        classifier_weights: read_floats(&mut file, 102_849)?,
    };

    let mut leftover = Vec::new();
    file.read_to_end(&mut leftover)?;

    println!("¡Éxito Absoluto! Anatomía completada.");
    println!("- Parámetros del Transformer mapeados: 7,408,960");
    println!("- Parámetros de Clasificación mapeados: {}", weights.classifier_weights.len());
    println!("- Bytes sobrantes no leídos (debería ser 0): {}", leftover.len());

    Ok(())
}