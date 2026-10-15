use std::io::{self, Write};
use std::time::Instant;
use codon_infer::{
    active_simd_backend, export_contacts_csv, export_contacts_json,
    export_dms_csv, export_dms_json, generate_sequence_parallel,
    generate_sequence_q8, parse_fasta_file, parse_raw_sequence, render_dms_heatmap,
    run_pipeline_contacts_parallel, run_pipeline_dms_parallel,
    run_pipeline_inpaint_parallel, EsmConfig, EsmWeights, EsmWeightsQ8,
    FastaRecord, ModelWeightsRef, RunState, TerminalTheme,
};

fn print_version() {
    println!("codon-infer v{}", env!("CARGO_PKG_VERSION"));
    println!("Engine: Pure Rust Zero-Dependencies (std)");
    println!("Vector Dispatch: {}", active_simd_backend());
    println!("Concurrency: std::thread::scope (zero runtime overhead)");
    println!("Quantization: Q8_0 In-Memory Block Engine (32-float scale block)");
    println!("Target Model: ESM-2 Multi-Scale [8M, 35M, 150M, 650M, 3B] (Meta AI / UniRef50)");
}

fn print_help(theme: &TerminalTheme) {
    let logo = if theme.enabled {
        "\x1b[38;2;139;92;246m╔═══════════════════════════════════════════════════════════════════════════════╗\n\
         ║   \x1b[38;2;59;130;246mCODON-INFER\x1b[38;2;139;92;246m: High-Performance Zero-Allocation ESM-2 Engine in Pure Rust     ║\n\
         ║   \x1b[38;2;16;185;129mVersion 1.5.0-dev\x1b[38;2;139;92;246m | Q8_0 & Scoped Multi-Thread R&D Suite (8M to 3B)       ║\n\
         ╚═══════════════════════════════════════════════════════════════════════════════╝\x1b[0m"
    } else {
        "╔═══════════════════════════════════════════════════════════════════════════════╗\n\
         ║   CODON-INFER: High-Performance Zero-Allocation ESM-2 Engine in Pure Rust     ║\n\
         ║   Version 1.5.0-dev | Q8_0 & Scoped Multi-Thread R&D Suite (8M to 3B)       ║\n\
         ╚═══════════════════════════════════════════════════════════════════════════════╝"
    };
    println!("{}", logo);
    println!("\nUSAGE:");
    println!("    codon-infer <SUBCOMMAND> [OPTIONS]");
    println!("\nSUBCOMMANDS:");
    println!("    generate     Autoregressive de novo sequence generation (causal mode)");
    println!("    inpaint      Masked sequence infilling & functional active-site reconstruction");
    println!("    dms          In silico Deep Mutational Scanning with TrueColor thermal heatmap");
    println!("    contacts     Zero-shot 3D contact map prediction and disulfide bridge analysis");
    println!("    benchmark    Hardware validation suite (SIMD parity, cache thrashing, scaling)");
    println!("\nGLOBAL OPTIONS:");
    println!("    -h, --help               Print help information");
    println!("    -V, --version            Print engine version and active vector acceleration");
    println!("    -t, --threads <N>        Number of CPU threads (default: available logical cores)");
    println!("    --quantize <q8_0|fp32>   Quantization mode for execution (default: fp32)");
    println!("    --q8                     Shortcut flag for '--quantize q8_0'");
    println!("    --model <PATH>           Path to weights binary (or 'all' for multi-scale benchmark)");
    println!("    --export-json <PATH>     Export benchmark or analysis metrics to JSON");
    println!("    --color <auto|always|never> Control ANSI TrueColor terminal styling");
    println!("    --no-color               Disable all colored terminal escapes (NO_COLOR standard)");
    println!("\nSUBCOMMAND EXAMPLES:");
    println!("    codon-infer generate --sequence \"M\" --temperature 0.7 --max-tokens 50 --threads 4");
    println!("    codon-infer generate --sequence \"M\" --quantize q8_0 --threads 4");
    println!("    codon-infer inpaint --sequence \"MCMPC*TTDHQMARKCDD\" --threads 4");
    println!("    codon-infer dms --fasta protein.fasta --window 128 --export-csv dms.csv --threads 4");
    println!("    codon-infer contacts --sequence \"MCMPCFTTDHQMARKCDDCCGGKGRGKCYGPQCLCR\" --threads 4");
    println!("    codon-infer benchmark --suite full --threads 4");
    println!("    codon-infer benchmark --model esm2_650m_weights.bin --suite full --threads 4");
    println!("    codon-infer benchmark --model esm2_650m_weights.bin --suite full --threads 4 --quantize q8_0");
}

fn load_input_record(
    fasta_path: Option<&str>,
    sequence_arg: Option<&str>,
    default_id: &str,
) -> io::Result<FastaRecord> {
    if let Some(fpath) = fasta_path {
        let records = parse_fasta_file(fpath)?;
        if records.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("FASTA file '{}' contains no sequence records", fpath),
            ));
        }
        Ok(records.into_iter().next().unwrap())
    } else if let Some(seq) = sequence_arg {
        Ok(parse_raw_sequence(seq, default_id))
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No input protein provided. Use --sequence <SEQ> or --fasta <FILE> (or '-' for stdin).",
        ))
    }
}

pub enum LoadedWeights {
    Fp32(EsmWeights),
    Q8(EsmWeightsQ8),
}

impl LoadedWeights {
    pub fn as_ref(&self) -> ModelWeightsRef<'_> {
        match *self {
            LoadedWeights::Fp32(ref w) => ModelWeightsRef::Fp32(w),
            LoadedWeights::Q8(ref w) => ModelWeightsRef::Q8(w),
        }
    }
}

pub fn load_model_weights(model_path: &str, quantize_q8: bool) -> io::Result<(EsmConfig, LoadedWeights)> {
    let config = match EsmConfig::detect_from_file(model_path) {
        Ok(cfg) => cfg,
        Err(_) => EsmConfig::esm2_8m(),
    };
    if quantize_q8 {
        let is_q8_file = model_path.ends_with(".q8") || model_path.ends_with(".q8.bin");
        if is_q8_file {
            let weights_q8 = EsmWeightsQ8::from_file(model_path, &config)?;
            Ok((config, LoadedWeights::Q8(weights_q8)))
        } else {
            let fp32 = EsmWeights::from_file(model_path, &config)?;
            let weights_q8 = fp32.quantize_q8();
            Ok((config, LoadedWeights::Q8(weights_q8)))
        }
    } else {
        let weights = EsmWeights::from_file(model_path, &config)?;
        Ok((config, LoadedWeights::Fp32(weights)))
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Check early help and version flags
    if args.len() > 1 && (args[1] == "-h" || args[1] == "--help" || args[1] == "help") {
        let theme = TerminalTheme::detect(None, false);
        print_help(&theme);
        return Ok(());
    }
    if args.len() > 1 && (args[1] == "-V" || args[1] == "--version") {
        print_version();
        return Ok(());
    }

    // Detect hardware parallelism for default threading
    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Detect global preferences early
    let mut color_arg: Option<String> = None;
    let mut no_color_flag = false;
    let mut global_threads: Option<usize> = None;
    let mut global_quantize: Option<bool> = None;

    for (i, a) in args.iter().enumerate() {
        if a == "--no-color" {
            no_color_flag = true;
        } else if a == "--color" && i + 1 < args.len() {
            color_arg = Some(args[i + 1].clone());
        } else if (a == "--threads" || a == "-t") && i + 1 < args.len() {
            if let Ok(t) = args[i + 1].parse::<usize>() {
                global_threads = Some(t.max(1));
            }
        } else if a == "--quantize" && i + 1 < args.len() {
            let q_str = args[i + 1].to_lowercase();
            global_quantize = Some(q_str == "q8_0" || q_str == "q8");
        } else if a == "--q8" {
            global_quantize = Some(true);
        }
    }
    let theme = TerminalTheme::detect(color_arg.as_deref(), no_color_flag);

    // Default model weights path: check for discovered weights, with esm2_8m_weights.bin fallback
    let mut model_path = codon_infer::benchmark::discover_available_models()
        .into_iter()
        .next()
        .unwrap_or_else(|| "esm2_8m_weights.bin".to_string());

    // Subcommand dispatch
    let subcommand = if args.len() > 1 && !args[1].starts_with('-') {
        args[1].as_str()
    } else {
        // Fallback for backward compatibility (e.g., codon-infer --sequence "M" ...)
        "generate"
    };

    let sub_args_start = if args.len() > 1 && !args[1].starts_with('-') { 2 } else { 1 };

    match subcommand {
        "benchmark" => {
            let mut suite = "full".to_string();
            let mut model_arg: Option<String> = None;
            let mut export_json_path: Option<String> = None;
            let mut num_threads = global_threads.unwrap_or(default_threads);
            let mut quantize_q8 = global_quantize.unwrap_or(false);

            let mut i = sub_args_start;
            while i < args.len() {
                if (args[i] == "--suite" || args[i] == "-s") && i + 1 < args.len() {
                    suite = args[i + 1].clone();
                    i += 2;
                } else if args[i] == "--model" && i + 1 < args.len() {
                    model_arg = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--threads" || args[i] == "-t") && i + 1 < args.len() {
                    if let Ok(t) = args[i + 1].parse::<usize>() { num_threads = t.max(1); }
                    i += 2;
                } else if args[i] == "--quantize" && i + 1 < args.len() {
                    let q_str = args[i + 1].to_lowercase();
                    quantize_q8 = q_str == "q8_0" || q_str == "q8";
                    i += 2;
                } else if args[i] == "--q8" {
                    quantize_q8 = true;
                    i += 1;
                } else if args[i] == "--export-json" && i + 1 < args.len() {
                    export_json_path = Some(args[i + 1].clone());
                    i += 2;
                } else if args[i] == "--no-color" || args[i] == "--color" {
                    i += if args[i] == "--color" { 2 } else { 1 };
                } else {
                    eprintln!("Warning: unrecognized benchmark option '{}'", args[i]);
                    i += 1;
                }
            }

            let model_paths: Vec<String> = if let Some(ref m) = model_arg {
                if m == "all" {
                    codon_infer::benchmark::discover_available_models()
                } else {
                    vec![m.clone()]
                }
            } else {
                let discovered = codon_infer::benchmark::discover_available_models();
                if discovered.is_empty() {
                    vec![model_path]
                } else {
                    discovered
                }
            };

            codon_infer::benchmark::run_benchmark_cli(
                &model_paths,
                &suite,
                &theme,
                export_json_path.as_deref(),
                num_threads,
                quantize_q8,
            )?;
            Ok(())
        }

        "dms" => {
            let mut fasta_path: Option<String> = None;
            let mut sequence_arg: Option<String> = None;
            let mut window_size: usize = 128;
            let mut stride: usize = 64;
            let mut export_csv: Option<String> = None;
            let mut export_json: Option<String> = None;
            let mut num_threads = global_threads.unwrap_or(default_threads);
            let mut quantize_q8 = global_quantize.unwrap_or(false);

            let mut i = sub_args_start;
            while i < args.len() {
                if (args[i] == "--fasta" || args[i] == "-f") && i + 1 < args.len() {
                    fasta_path = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--sequence" || args[i] == "-s") && i + 1 < args.len() {
                    sequence_arg = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--threads" || args[i] == "-t") && i + 1 < args.len() {
                    if let Ok(t) = args[i + 1].parse::<usize>() { num_threads = t.max(1); }
                    i += 2;
                } else if args[i] == "--quantize" && i + 1 < args.len() {
                    let q_str = args[i + 1].to_lowercase();
                    quantize_q8 = q_str == "q8_0" || q_str == "q8";
                    i += 2;
                } else if args[i] == "--q8" {
                    quantize_q8 = true;
                    i += 1;
                } else if args[i] == "--window" && i + 1 < args.len() {
                    if let Ok(w) = args[i + 1].parse::<usize>() { window_size = w; }
                    i += 2;
                } else if args[i] == "--stride" && i + 1 < args.len() {
                    if let Ok(s) = args[i + 1].parse::<usize>() { stride = s; }
                    i += 2;
                } else if args[i] == "--export-csv" && i + 1 < args.len() {
                    export_csv = Some(args[i + 1].clone());
                    i += 2;
                } else if args[i] == "--export-json" && i + 1 < args.len() {
                    export_json = Some(args[i + 1].clone());
                    i += 2;
                } else if args[i] == "--model" && i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    i += 2;
                } else if args[i] == "--no-color" || args[i] == "--color" {
                    i += if args[i] == "--color" { 2 } else { 1 };
                } else {
                    eprintln!("Warning: unrecognized DMS option '{}'", args[i]);
                    i += 1;
                }
            }

            let record = load_input_record(fasta_path.as_deref(), sequence_arg.as_deref(), "target_protein")?;
            let (config, loaded_weights) = load_model_weights(&model_path, quantize_q8)?;

            let result = run_pipeline_dms_parallel(
                &record,
                window_size,
                stride,
                &config,
                loaded_weights.as_ref(),
                num_threads,
                &theme,
            );
            render_dms_heatmap(&result, 25, &theme);

            if let Some(ref csv_path) = export_csv {
                export_dms_csv(csv_path, &result.protein_id, &result.rows)?;
                println!("  [EXPORT] DMS rectangular matrix saved to '{}'.", csv_path);
            }
            if let Some(ref json_path) = export_json {
                export_dms_json(json_path, &result.protein_id, &result.sequence, &result.rows, result.elapsed_ms, result.muts_per_sec)?;
                println!("  [EXPORT] DMS analysis report saved to '{}'.", json_path);
            }
            Ok(())
        }

        "inpaint" => {
            let mut fasta_path: Option<String> = None;
            let mut sequence_arg: Option<String> = None;
            let mut num_threads = global_threads.unwrap_or(default_threads);
            let mut quantize_q8 = global_quantize.unwrap_or(false);

            let mut i = sub_args_start;
            while i < args.len() {
                if (args[i] == "--fasta" || args[i] == "-f") && i + 1 < args.len() {
                    fasta_path = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--sequence" || args[i] == "-s") && i + 1 < args.len() {
                    sequence_arg = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--threads" || args[i] == "-t") && i + 1 < args.len() {
                    if let Ok(t) = args[i + 1].parse::<usize>() { num_threads = t.max(1); }
                    i += 2;
                } else if args[i] == "--quantize" && i + 1 < args.len() {
                    let q_str = args[i + 1].to_lowercase();
                    quantize_q8 = q_str == "q8_0" || q_str == "q8";
                    i += 2;
                } else if args[i] == "--q8" {
                    quantize_q8 = true;
                    i += 1;
                } else if args[i] == "--model" && i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    i += 2;
                } else if args[i] == "--no-color" || args[i] == "--color" {
                    i += if args[i] == "--color" { 2 } else { 1 };
                } else {
                    eprintln!("Warning: unrecognized inpaint option '{}'", args[i]);
                    i += 1;
                }
            }

            let record = load_input_record(fasta_path.as_deref(), sequence_arg.as_deref(), "masked_protein")?;
            let (config, loaded_weights) = load_model_weights(&model_path, quantize_q8)?;

            run_pipeline_inpaint_parallel(&record, &config, loaded_weights.as_ref(), num_threads, &theme);
            Ok(())
        }

        "contacts" => {
            let mut fasta_path: Option<String> = None;
            let mut sequence_arg: Option<String> = None;
            let mut min_separation: usize = 6;
            let mut top_l_factor: f32 = 1.0;
            let mut export_csv: Option<String> = None;
            let mut export_json: Option<String> = None;
            let mut num_threads = global_threads.unwrap_or(default_threads);
            let mut quantize_q8 = global_quantize.unwrap_or(false);

            let mut i = sub_args_start;
            while i < args.len() {
                if (args[i] == "--fasta" || args[i] == "-f") && i + 1 < args.len() {
                    fasta_path = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--sequence" || args[i] == "-s") && i + 1 < args.len() {
                    sequence_arg = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--threads" || args[i] == "-t") && i + 1 < args.len() {
                    if let Ok(t) = args[i + 1].parse::<usize>() { num_threads = t.max(1); }
                    i += 2;
                } else if args[i] == "--quantize" && i + 1 < args.len() {
                    let q_str = args[i + 1].to_lowercase();
                    quantize_q8 = q_str == "q8_0" || q_str == "q8";
                    i += 2;
                } else if args[i] == "--q8" {
                    quantize_q8 = true;
                    i += 1;
                } else if args[i] == "--min-separation" && i + 1 < args.len() {
                    if let Ok(m) = args[i + 1].parse::<usize>() { min_separation = m; }
                    i += 2;
                } else if args[i] == "--top-l" && i + 1 < args.len() {
                    if let Ok(l) = args[i + 1].parse::<f32>() { top_l_factor = l; }
                    i += 2;
                } else if args[i] == "--export-csv" && i + 1 < args.len() {
                    export_csv = Some(args[i + 1].clone());
                    i += 2;
                } else if args[i] == "--export-json" && i + 1 < args.len() {
                    export_json = Some(args[i + 1].clone());
                    i += 2;
                } else if args[i] == "--model" && i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    i += 2;
                } else if args[i] == "--no-color" || args[i] == "--color" {
                    i += if args[i] == "--color" { 2 } else { 1 };
                } else {
                    eprintln!("Warning: unrecognized contacts option '{}'", args[i]);
                    i += 1;
                }
            }

            let record = load_input_record(fasta_path.as_deref(), sequence_arg.as_deref(), "target_structure")?;
            let (config, loaded_weights) = load_model_weights(&model_path, quantize_q8)?;

            let (top_contacts, enrichment, cys_indices) = run_pipeline_contacts_parallel(
                &record,
                min_separation,
                top_l_factor,
                &config,
                loaded_weights.as_ref(),
                num_threads,
                &theme,
            );

            if let Some(ref csv_path) = export_csv {
                export_contacts_csv(csv_path, &record.id, &top_contacts)?;
                println!("  [EXPORT] Predicted contact table saved to '{}'.", csv_path);
            }
            if let Some(ref json_path) = export_json {
                export_contacts_json(json_path, &record.id, &record.sequence, &cys_indices, enrichment, &top_contacts)?;
                println!("  [EXPORT] Contact analysis JSON saved to '{}'.", json_path);
            }
            Ok(())
        }

        "generate" => {
            let mut prompt = "M".to_string();
            let mut custom_seed: Option<u64> = None;
            let mut temperature: f32 = 0.7;
            let mut max_tokens: Option<usize> = None;
            let mut fasta_path: Option<String> = None;
            let mut num_threads = global_threads.unwrap_or(default_threads);
            let mut quantize_q8 = global_quantize.unwrap_or(false);

            let mut i = sub_args_start;
            while i < args.len() {
                if args[i] == "--model" && i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    i += 2;
                } else if (args[i] == "--sequence" || args[i] == "-s") && i + 1 < args.len() {
                    prompt = args[i + 1].clone();
                    i += 2;
                } else if (args[i] == "--fasta" || args[i] == "-f") && i + 1 < args.len() {
                    fasta_path = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--threads" || args[i] == "-t") && i + 1 < args.len() {
                    if let Ok(t) = args[i + 1].parse::<usize>() { num_threads = t.max(1); }
                    i += 2;
                } else if args[i] == "--quantize" && i + 1 < args.len() {
                    let q_str = args[i + 1].to_lowercase();
                    quantize_q8 = q_str == "q8_0" || q_str == "q8";
                    i += 2;
                } else if args[i] == "--q8" {
                    quantize_q8 = true;
                    i += 1;
                } else if args[i] == "--seed" && i + 1 < args.len() {
                    if let Ok(s) = args[i + 1].parse::<u64>() { custom_seed = Some(s); }
                    i += 2;
                } else if args[i] == "--temperature" && i + 1 < args.len() {
                    if let Ok(t) = args[i + 1].parse::<f32>() { temperature = t; }
                    i += 2;
                } else if args[i] == "--max-tokens" && i + 1 < args.len() {
                    if let Ok(m) = args[i + 1].parse::<usize>() { max_tokens = Some(m); }
                    i += 2;
                } else if args[i] == "--no-color" || args[i] == "--color" {
                    i += if args[i] == "--color" { 2 } else { 1 };
                } else {
                    eprintln!("Warning: unrecognized generate option '{}'", args[i]);
                    i += 1;
                }
            }

            if let Some(fpath) = fasta_path {
                let rec = load_input_record(Some(&fpath), None, "seed_record")?;
                prompt = rec.sequence;
            }

            println!("--- Codon-Infer Engine V1.5-dev (Zero-Deps Multi-Scale CLI) ---");
            println!("Active SIMD Backend:  {}", active_simd_backend());
            println!("Concurrency Threads:  {}", num_threads);
            println!("Execution Precision:  {}", if quantize_q8 { "Q8_0 (In-Memory Blocks)" } else { "FP32 (Canonical)" });

            let load_timer = Instant::now();
            println!("Loading model weights from '{}'...", model_path);
            let (config, loaded_weights) = load_model_weights(&model_path, quantize_q8)?;
            println!("Model weights loaded successfully for {} in {:?}!\n", config.model_name(), load_timer.elapsed());

            let mut run_state = RunState::new(&config);
            println!("Initial seed prompt: '{}'", prompt);
            println!("Starting autoregressive generation with stochastic sampling...");
            print!("Generation: {}", codon_infer::ui::format_sequence(&prompt, &theme));
            io::stdout().flush().unwrap();

            let seed = custom_seed.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0x123456789abcdef)
            });

            let mut print_cb = |c: char| {
                print!("{}", codon_infer::ui::format_aa(c, &theme));
                io::stdout().flush().unwrap();
            };

            let metrics = match loaded_weights {
                LoadedWeights::Fp32(ref weights) => {
                    generate_sequence_parallel(
                        &prompt,
                        temperature,
                        max_tokens,
                        seed,
                        &config,
                        weights,
                        &mut run_state,
                        num_threads,
                        Some(&mut print_cb),
                    )
                }
                LoadedWeights::Q8(ref weights_q8) => {
                    generate_sequence_q8(
                        &prompt,
                        temperature,
                        max_tokens,
                        seed,
                        &config,
                        weights_q8,
                        &mut run_state,
                        num_threads,
                        Some(&mut print_cb),
                    )
                }
            };

            println!("\n\n--- Performance & Validation Report ---");
            println!(" * Prompt Pre-fill Latency:            {:.2} ms", metrics.prefill_ms);
            println!(" * Token Generation Time:              {:.2} ms", metrics.generation_ms);
            println!(" * Tokens Generated:                   {} tokens", metrics.tokens_generated);
            println!(" * Throughput (Speed):                 {:.2} tokens/second", metrics.throughput_tps);
            println!("[Execution Completed Successfully]");
            Ok(())
        }

        unknown => {
            eprintln!("Error: Unknown subcommand '{}'.", unknown);
            eprintln!("Run 'codon-infer --help' for a list of available subcommands.");
            std::process::exit(1);
        }
    }
}
