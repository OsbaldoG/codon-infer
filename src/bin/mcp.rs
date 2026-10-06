use codon_infer::{EsmConfig, EsmWeights};
use codon_infer::mcp::server::run_server;
use std::fs::File;
use std::io;

fn main() -> io::Result<()> {
    eprintln!("Codon-Infer MCP Server starting...");
    
    let config = EsmConfig {
        dim: 320,
        hidden_dim: 1280,
        n_layers: 6,
        n_heads: 20,
        vocab_size: 33,
        max_seq_len: 1024,
    };

    let mut file = File::open("esm2_8m_weights.bin")?;
    let weights = EsmWeights::load(&mut file, &config)?;
    
    eprintln!("Model loaded successfully");
    
    run_server(config, weights)
}