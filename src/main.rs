use std::io::{self, Write};
use std::time::Instant;
use codon_infer::{EsmConfig, EsmWeights, RunState, active_simd_backend, generate_sequence};

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = if std::path::Path::new("esm2_weights.bin").exists() { "esm2_weights.bin".to_string() } else { "esm2_pesos.bin".to_string() };

    // Subcommand: benchmark
    if args.len() > 1 && args[1] == "benchmark" {
        let mut suite = "full".to_string();
        let mut i = 2;
        while i < args.len() {
            if (args[i] == "--suite" || args[i] == "-s") && i + 1 < args.len() {
                suite = args[i + 1].clone();
                i += 2;
            } else if args[i] == "--model" && i + 1 < args.len() {
                model_path = args[i + 1].clone();
                i += 2;
            } else {
                i += 1;
            }
        }
        let config = EsmConfig::esm2_8m();
        let load_t = Instant::now();
        println!("Loading ESM-2 weights from '{}' for benchmark suite...", model_path);
        let weights = EsmWeights::from_file(&model_path, &config)?;
        println!("Weights mapped successfully in {:.2} ms.", load_t.elapsed().as_secs_f64() * 1000.0);
        codon_infer::benchmark::run_full_benchmark_suite(&weights, &config, &suite);
        return Ok(());
    }

    let mut prompt = "M".to_string(); let mut custom_seed: Option<u64> = None; let mut temperature: f32 = 0.7; let mut max_tokens: Option<usize> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() { model_path = args[i+1].clone(); i += 2; }
        else if args[i] == "--sequence" && i + 1 < args.len() { prompt = args[i+1].clone(); i += 2; }
        else if args[i] == "--seed" && i + 1 < args.len() { if let Ok(s) = args[i+1].parse::<u64>() { custom_seed = Some(s); } i += 2; }
        else if args[i] == "--temperature" && i + 1 < args.len() { if let Ok(t) = args[i+1].parse::<f32>() { temperature = t; } i += 2; }
        else if args[i] == "--max-tokens" && i + 1 < args.len() { if let Ok(m) = args[i+1].parse::<usize>() { max_tokens = Some(m); } i += 2; }
        else { i += 1; }
    }
    println!("--- Codon-Infer Engine V1.2 (Zero-Deps SIMD Mode) ---"); println!("Active SIMD Backend: {}", active_simd_backend());
    let load_timer = Instant::now(); println!("Loading model weights from '{}'...", model_path);
    let config = EsmConfig::esm2_8m(); let weights = EsmWeights::from_file(&model_path, &config)?;
    println!("Model weights loaded successfully in {:?}!\n", load_timer.elapsed());
    let mut run_state = RunState::new(&config);
    println!("Initial seed prompt: '{}'", prompt); println!("Starting autoregressive generation with stochastic sampling...");
    print!("Generation: {}", prompt); io::stdout().flush().unwrap();
    let seed = custom_seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0x123456789abcdef));
    let mut print_cb = |c: char| { print!("{}", c); io::stdout().flush().unwrap(); };
    let metrics = generate_sequence(&prompt, temperature, max_tokens, seed, &config, &weights, &mut run_state, Some(&mut print_cb));
    println!("\n\n--- Performance & Validation Report ---");
    println!(" * Prompt Pre-fill Latency:            {:.2} ms", metrics.prefill_ms);
    println!(" * Token Generation Time:              {:.2} ms", metrics.generation_ms);
    println!(" * Tokens Generated:                   {} tokens", metrics.tokens_generated);
    println!(" * Throughput (Speed):                 {:.2} tokens/second", metrics.throughput_tps);
    println!("[Execution Completed Successfully]");
    Ok(())
}
