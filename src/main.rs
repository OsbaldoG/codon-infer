use std::fs::File;
use std::io::{self, Read};

// 1. Traemos la estructura definida en tu Architecture Spec
#[derive(Debug, Clone)]
pub struct EsmConfig {
    pub dim: usize,           
    pub hidden_dim: usize,    
    pub n_layers: usize,      
    pub n_heads: usize,       
    pub vocab_size: usize,    
    pub max_seq_len: usize,   
}

fn main() -> io::Result<()> {
    println!("--- Codon-Infer Engine V0.2 (Rust Edition) ---");
    println!("Inicializando lectura bajo el Data Spec estricto...\n");

    // Inicializamos la configuración hiper-arquitectónica del ESM-2 (8M)
    let _config = EsmConfig {
        dim: 320,
        hidden_dim: 1280,
        n_layers: 6,
        n_heads: 20,
        vocab_size: 33,
        max_seq_len: 1024,
    };

    // 2. Abrimos el archivo binario crudo
    let mut file = File::open("esm2_pesos.bin")?;

    // 3. Aplicamos la Estrategia de Lectura (Data Spec)
    let num_pesos = 10;
    let mut buffer = vec![0u8; num_pesos * 4]; 
    
    // Usamos read_exact para evitar reasignaciones de memoria como dicta el Spec
    file.read_exact(&mut buffer)?;

    // 4. Convertimos los bytes crudos (Little Endian) a Float32
    let mut pesos_iniciales = Vec::with_capacity(num_pesos);
    for chunk in buffer.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().unwrap();
        pesos_iniciales.push(f32::from_le_bytes(bytes));
    }

    println!("¡Éxito! Conectado con la matriz de pesos desde Rust.");
    println!("Los primeros {} valores matemáticos del modelo son:", num_pesos);
    for (i, peso) in pesos_iniciales.iter().enumerate() {
        println!("Peso [{}]: {:.6}", i, peso);
    }

    println!("\nPrueba de lectura segura en Rust superada.");
    Ok(())
}