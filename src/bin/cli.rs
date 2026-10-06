use codon_infer::{EsmConfig, EsmWeights, RunState, forward, Tokenizer, SimpleRng, sample};
use std::fs::File;
use std::io::{self, Write};

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = "esm2_8m_weights.bin".to_string();
    let mut prompt = "M".to_string();

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() {
            model_path = args[i + 1].clone();
            i += 2;
        } else if args[i] == "--sequence" && i + 1 < args.len() {
            prompt = args[i + 1].clone();
            i += 2;
        } else {
            i += 1;
        }
    }

    println!("--- Codon-Infer Engine V1.0 (Benchmarking Mode) ---");

    let load_timer = std::time::Instant::now();
    println!("Cargando modelo desde '{}'...", model_path);
    let config = EsmConfig {
        dim: 320,
        hidden_dim: 1280,
        n_layers: 6,
        n_heads: 20,
        vocab_size: 33,
        max_seq_len: 1024,
    };

    let mut file = File::open(&model_path)?;

    let weights = EsmWeights::load(&mut file, &config)?;
    println!("¡Modelo cargado exitosamente en {:?}!\n", load_timer.elapsed());

    let mut run_state = RunState::new(&config);

    println!("Secuencia inicial (Prompt): '{}'", prompt);
    println!("Iniciando Generación Auto-Regresiva con Muestreo Estocástico...");

    let mut prompt_tokens = Tokenizer::encode(&prompt);
    prompt_tokens.pop();

    print!("Generación: {}", prompt);
    io::stdout().flush().unwrap();

    let mut pos = 0;

    let prefill_timer = std::time::Instant::now();
    for &token in prompt_tokens.iter().take(prompt_tokens.len().saturating_sub(1)) {
        forward(token, pos, &config, &weights, &mut run_state);
        pos += 1;
    }
    let prefill_duration = prefill_timer.elapsed();

    let mut next_token = prompt_tokens[prompt_tokens.len() - 1];
    let mut rng = SimpleRng::new(42);
    let temperature = 0.7;

    let mut generated_tokens_count = 0;
    let gen_timer = std::time::Instant::now();

    while pos < config.max_seq_len {
        forward(next_token, pos, &config, &weights, &mut run_state);

        if let Some(c) = Tokenizer::decode(next_token) {
            print!("{}", c);
            io::stdout().flush().unwrap();
        }

        next_token = sample(&mut run_state.logits, temperature, &mut rng);
        pos += 1;
        generated_tokens_count += 1;

        if next_token == 2 {
            break;
        }
    }
    let gen_duration = gen_timer.elapsed();

    println!("\n\n--- Reporte de Validación y Rendimiento ---");
    println!(" * Tiempo de Pre-fill (Prompt):     {:?}", prefill_duration);
    println!(" * Tiempo de Generación de Tokens:  {:?}", gen_duration);
    println!(" * Tokens Generados:                {} tokens", generated_tokens_count);
    if gen_duration.as_secs_f32() > 0.0 {
        let throughput = generated_tokens_count as f32 / gen_duration.as_secs_f32();
        println!(" * Throughput (Velocidad):          {:.2} tokens/segundo", throughput);
    }
    println!("[Validación de Ejecución Finalizada con Éxito]");

    Ok(())
}