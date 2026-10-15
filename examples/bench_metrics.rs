use std::time::Instant;
use codon_infer::{
    Tokenizer, EsmConfig, EsmWeights, EsmWeightsQ8, EsmWeightsQ4,
    BidirectionalRunState, forward_bidirectional_parallel,
    forward_bidirectional_q8, forward_bidirectional_q4,
    CodqHeader, CODQ_MAGIC, DTYPE_Q4_0, DTYPE_Q8_0,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut num_threads = 4;
    let mut force_format = String::from("auto");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--threads" => {
                if i + 1 < args.len() {
                    num_threads = args[i + 1].parse().unwrap_or(4);
                    i += 1;
                }
            }
            "--format" => {
                if i + 1 < args.len() {
                    force_format = args[i + 1].clone();
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    if model_path.is_empty() {
        eprintln!("Usage: bench_metrics --model <path> [--threads <N>] [--format <auto|fp32|q8_0|q4_0>]");
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
    } else {
        // Auto detect
        if let Ok(mut file) = std::fs::File::open(&model_path) {
            if let Ok(header) = CodqHeader::read_from(&mut file) {
                header.magic == CODQ_MAGIC && header.dtype == DTYPE_Q4_0
            } else {
                model_path.contains("q4_0") || model_path.contains("q4")
            }
        } else {
            false
        }
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

        // 5 timed iterations
        let mut times = Vec::with_capacity(5);
        for _ in 0..5 {
            let t0 = Instant::now();
            forward_bidirectional_q4(&tokens, &config, &weights, &mut state, num_threads);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = times[2];

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

        // 5 timed iterations
        let mut times = Vec::with_capacity(5);
        for _ in 0..5 {
            let t0 = Instant::now();
            forward_bidirectional_q8(&tokens, &config, &weights, &mut state, num_threads);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = times[2];

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

        // 5 timed iterations
        let mut times = Vec::with_capacity(5);
        for _ in 0..5 {
            let t0 = Instant::now();
            forward_bidirectional_parallel(&tokens, &detected_config, &weights, &mut state, num_threads);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = times[2];

        println!("RESULT_LOAD_MS: {:.3}", load_ms);
        println!("RESULT_FORWARD_RUNS_MS: {:?}", times);
        println!("RESULT_FORWARD_MEDIAN_MS: {:.3}", median_ms);
        println!("RESULT_TOKENS_PER_SEC: {:.2}", (seq_len as f64) / (median_ms / 1000.0));
    }
}
