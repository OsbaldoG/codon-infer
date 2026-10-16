use std::time::Instant;
use codon_infer::{
    Tokenizer, EsmConfig, EsmWeights, EsmWeightsQ8, EsmWeightsQ4,
    BidirectionalRunState, forward_bidirectional_parallel,
    forward_bidirectional_q8, forward_bidirectional_q4,
    CodqHeader, CODQ_MAGIC, DTYPE_Q4_0, DTYPE_Q8_0,
    read_npz_floats, compute_cosine_similarity, compute_centered_cosine_similarity,
    compute_kl_divergence, compute_top1_match_rate,
};

fn print_ref_metrics(
    ref_npz: &str,
    tokens: &[usize],
    config: &EsmConfig,
    state: &BidirectionalRunState,
) {
    if ref_npz.is_empty() || !std::path::Path::new(ref_npz).exists() {
        return;
    }
    let seq_len = tokens.len();
    let ref_logits = read_npz_floats(ref_npz, "logits.npy").expect("Read ref logits.npy");
    let ref_hs = read_npz_floats(ref_npz, "hidden_states.npy").expect("Read ref hidden_states.npy");
    let hs_stride = seq_len * config.dim;
    let ref_final_hs = &ref_hs[config.n_layers * hs_stride..(config.n_layers + 1) * hs_stride];
    let rust_final_hs = &state.x[0..hs_stride];
    let rust_logits = &state.logits[0..seq_len * config.vocab_size];

    let hs_cos = compute_cosine_similarity(rust_final_hs, ref_final_hs);
    let logits_cos = compute_cosine_similarity(rust_logits, &ref_logits);
    let centered_cos = compute_centered_cosine_similarity(rust_logits, &ref_logits, config.vocab_size);
    let kl_div = compute_kl_divergence(&ref_logits, rust_logits, config.vocab_size);
    let top1_vs_hf = compute_top1_match_rate(&ref_logits, rust_logits, config.vocab_size);

    let mut unmasked_matches = 0usize;
    let aa_tokens = &tokens[1..seq_len - 1];
    for (idx, &expected_tok) in aa_tokens.iter().enumerate() {
        let p = idx + 1;
        let row = &rust_logits[p * config.vocab_size..(p + 1) * config.vocab_size];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|x| x.0)
            .unwrap_or(0);
        if pred == expected_tok {
            unmasked_matches += 1;
        }
    }
    let unmasked_acc = unmasked_matches as f32 / aa_tokens.len() as f32;

    println!("RESULT_HS_COS: {:.6}", hs_cos);
    println!("RESULT_LOGITS_COS: {:.6}", logits_cos);
    println!("RESULT_CENTERED_COS: {:.6}", centered_cos);
    println!("RESULT_KL_DIV: {:.6}", kl_div);
    println!("RESULT_TOP1_VS_HF: {:.4}", top1_vs_hf * 100.0);
    println!("RESULT_TOP1_UNMASKED: {}/{} ({:.2}%)", unmasked_matches, aa_tokens.len(), unmasked_acc * 100.0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut ref_npz = String::new();
    let mut num_threads = 4;
    let mut iters = 5usize;
    let mut force_format = String::from("auto");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                model_path = args[i + 1].clone();
                i += 1;
            }
            "--ref" if i + 1 < args.len() => {
                ref_npz = args[i + 1].clone();
                i += 1;
            }
            "--threads" if i + 1 < args.len() => {
                num_threads = args[i + 1].parse().unwrap_or(4);
                i += 1;
            }
            "--iters" if i + 1 < args.len() => {
                iters = args[i + 1].parse().unwrap_or(5).max(1);
                i += 1;
            }
            "--format" if i + 1 < args.len() => {
                force_format = args[i + 1].clone();
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    if model_path.is_empty() {
        eprintln!("Usage: bench_metrics --model <path> [--ref <npz>] [--threads <N>] [--iters <N>] [--format <auto|fp32|q8_0|q4_0>]");
        std::process::exit(1);
    }

    let file_len = match std::fs::metadata(&model_path) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("Error reading file {}: {}", model_path, e);
            std::process::exit(1);
        }
    };

    let seq = "AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS";
    let tokens = Tokenizer::encode(seq);
    let seq_len = tokens.len();

    // Determine format
    let is_q4 = if force_format == "q4_0" {
        true
    } else if force_format == "q8_0" || force_format == "fp32" {
        false
    } else if let Ok(mut file) = std::fs::File::open(&model_path) {
        if let Ok(header) = CodqHeader::read_from(&mut file) {
            header.magic == CODQ_MAGIC && header.dtype == DTYPE_Q4_0
        } else {
            model_path.contains("q4_0") || model_path.contains("q4")
        }
    } else {
        false
    };

    let is_q8 = if force_format == "q8_0" {
        true
    } else if force_format == "q4_0" || force_format == "fp32" {
        false
    } else if !is_q4 {
        if let Ok(mut file) = std::fs::File::open(&model_path) {
            if let Ok(header) = CodqHeader::read_from(&mut file) {
                header.magic == CODQ_MAGIC && header.dtype == DTYPE_Q8_0
            } else {
                model_path.contains("q8_0") || model_path.contains("q8")
            }
        } else {
            false
        }
    } else {
        false
    };

    let detected_config = EsmConfig::detect_from_file(&model_path)
        .unwrap_or_else(|_| EsmConfig::detect_from_file_len(file_len).expect("Autodetect failed"));

    println!("============================================================");
    println!(" BENCHMARK RUN: {}", model_path);
    println!(" Model Arch:    {}", detected_config.model_name());
    println!(" File Size:     {} bytes ({:.2} MB)", file_len, file_len as f64 / 1_048_576.0);
    println!(" Format:        {}", if is_q4 { "Q4_0" } else if is_q8 { "Q8_0" } else { "FP32" });
    println!(" Threads:       {}", num_threads);
    println!(" Sequence:      {} ({} tokens)", seq, seq_len);
    println!("============================================================");

    if is_q4 {
        let t_load = Instant::now();
        let (config, weights) = EsmWeightsQ4::load_from_file(&model_path).expect("Load Q4");
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        let mut state = BidirectionalRunState::new(&config, seq_len + 8);

        // Warm-up run
        forward_bidirectional_q4(&tokens, &config, &weights, &mut state, num_threads);
        print_ref_metrics(&ref_npz, &tokens, &config, &state);

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            forward_bidirectional_q4(&tokens, &config, &weights, &mut state, num_threads);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = times[iters / 2];

        println!("RESULT_LOAD_MS: {:.3}", load_ms);
        println!("RESULT_FORWARD_RUNS_MS: {:?}", times);
        println!("RESULT_FORWARD_MEDIAN_MS: {:.3}", median_ms);
        println!("RESULT_TOKENS_PER_SEC: {:.2}", (seq_len as f64) / (median_ms / 1000.0));
    } else if is_q8 {
        let t_load = Instant::now();
        let (config, weights) = EsmWeightsQ8::load_from_file(&model_path).expect("Load Q8");
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        let mut state = BidirectionalRunState::new(&config, seq_len + 8);

        // Warm-up run
        forward_bidirectional_q8(&tokens, &config, &weights, &mut state, num_threads);
        print_ref_metrics(&ref_npz, &tokens, &config, &state);

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            forward_bidirectional_q8(&tokens, &config, &weights, &mut state, num_threads);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = times[iters / 2];

        println!("RESULT_LOAD_MS: {:.3}", load_ms);
        println!("RESULT_FORWARD_RUNS_MS: {:?}", times);
        println!("RESULT_FORWARD_MEDIAN_MS: {:.3}", median_ms);
        println!("RESULT_TOKENS_PER_SEC: {:.2}", (seq_len as f64) / (median_ms / 1000.0));
    } else {
        let t_load = Instant::now();
        let weights = EsmWeights::from_file(&model_path, &detected_config).expect("Load FP32");
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        let mut state = BidirectionalRunState::new(&detected_config, seq_len + 8);

        // Warm-up run
        forward_bidirectional_parallel(&tokens, &detected_config, &weights, &mut state, num_threads);
        print_ref_metrics(&ref_npz, &tokens, &detected_config, &state);

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            forward_bidirectional_parallel(&tokens, &detected_config, &weights, &mut state, num_threads);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = times[iters / 2];

        println!("RESULT_LOAD_MS: {:.3}", load_ms);
        println!("RESULT_FORWARD_RUNS_MS: {:?}", times);
        println!("RESULT_FORWARD_MEDIAN_MS: {:.3}", median_ms);
        println!("RESULT_TOKENS_PER_SEC: {:.2}", (seq_len as f64) / (median_ms / 1000.0));
    }
}
