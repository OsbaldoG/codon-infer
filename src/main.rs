use std::io::{self, Write};
use std::time::Instant;
use codon_infer::{
    active_simd_backend, export_contacts_csv, export_contacts_json,
    export_dms_csv, export_dms_json, generate_sequence, parse_fasta_file,
    parse_raw_sequence, render_dms_heatmap, run_pipeline_contacts,
    run_pipeline_dms, run_pipeline_inpaint, EsmConfig, EsmWeights,
    FastaRecord, RunState, TerminalTheme,
};

fn print_version() {
    println!("codon-infer v{}", env!("CARGO_PKG_VERSION"));
    println!("Engine: Pure Rust Zero-Dependencies (std)");
    println!("Vector Dispatch: {}", active_simd_backend());
    println!("Target Model: ESM-2 Multi-Scale [8M, 35M, 150M, 650M, 3B] (Meta AI / UniRef50)");
}

fn print_help(theme: &TerminalTheme) {
    let logo = if theme.enabled {
        "\x1b[38;2;139;92;246m╔═══════════════════════════════════════════════════════════════════════════════╗\n\
         ║   \x1b[38;2;59;130;246mCODON-INFER\x1b[38;2;139;92;246m: High-Performance Zero-Allocation ESM-2 Engine in Pure Rust     ║\n\
         ║   \x1b[38;2;16;185;129mVersion 1.4.0\x1b[38;2;139;92;246m | Multi-Scale CLI & Bioinformatic Analysis Pipeline (8M to 3B) ║\n\
         ╚═══════════════════════════════════════════════════════════════════════════════╝\x1b[0m"
    } else {
        "╔═══════════════════════════════════════════════════════════════════════════════╗\n\
         ║   CODON-INFER: High-Performance Zero-Allocation ESM-2 Engine in Pure Rust     ║\n\
         ║   Version 1.4.0 | Multi-Scale CLI & Bioinformatic Analysis Pipeline (8M to 3B) ║\n\
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
    println!("    --model <PATH>           Path to weights binary (or 'all' for multi-scale benchmark)");
    println!("    --export-json <PATH>     Export benchmark or analysis metrics to JSON");
    println!("    --color <auto|always|never> Control ANSI TrueColor terminal styling");
    println!("    --no-color               Disable all colored terminal escapes (NO_COLOR standard)");
    println!("\nSUBCOMMAND EXAMPLES:");
    println!("    codon-infer generate --sequence \"M\" --temperature 0.7 --max-tokens 50");
    println!("    codon-infer inpaint --sequence \"MCMPC*TTDHQMARKCDD\"");
    println!("    codon-infer dms --fasta protein.fasta --window 128 --export-csv dms.csv");
    println!("    codon-infer contacts --sequence \"MCMPCFTTDHQMARKCDDCCGGKGRGKCYGPQCLCR\" --export-csv contacts.csv");
    println!("    codon-infer benchmark --suite full");
    println!("    codon-infer benchmark --model esm2_35m_weights.bin --suite full");
    println!("    codon-infer benchmark --model esm2_650m_weights.bin --suite full");
    println!("    codon-infer benchmark --suite full --export-json benchmark_report.json");
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

fn load_model(model_path: &str) -> io::Result<(EsmConfig, EsmWeights)> {
    let config = match EsmConfig::detect_from_file(model_path) {
        Ok(cfg) => cfg,
        Err(_) => EsmConfig::esm2_8m(),
    };
    let weights = EsmWeights::from_file(model_path, &config)?;
    Ok((config, weights))
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

    // Detect color preferences early
    let mut color_arg: Option<String> = None;
    let mut no_color_flag = false;
    for (i, a) in args.iter().enumerate() {
        if a == "--no-color" {
            no_color_flag = true;
        } else if a == "--color" && i + 1 < args.len() {
            color_arg = Some(args[i + 1].clone());
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
            let mut i = sub_args_start;
            while i < args.len() {
                if (args[i] == "--suite" || args[i] == "-s") && i + 1 < args.len() {
                    suite = args[i + 1].clone();
                    i += 2;
                } else if args[i] == "--model" && i + 1 < args.len() {
                    model_arg = Some(args[i + 1].clone());
                    i += 2;
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

            let mut i = sub_args_start;
            while i < args.len() {
                if (args[i] == "--fasta" || args[i] == "-f") && i + 1 < args.len() {
                    fasta_path = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--sequence" || args[i] == "-s") && i + 1 < args.len() {
                    sequence_arg = Some(args[i + 1].clone());
                    i += 2;
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
            let (config, weights) = load_model(&model_path)?;

            let result = run_pipeline_dms(&record, window_size, stride, &config, &weights, &theme);
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

            let mut i = sub_args_start;
            while i < args.len() {
                if (args[i] == "--fasta" || args[i] == "-f") && i + 1 < args.len() {
                    fasta_path = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--sequence" || args[i] == "-s") && i + 1 < args.len() {
                    sequence_arg = Some(args[i + 1].clone());
                    i += 2;
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
            let (config, weights) = load_model(&model_path)?;

            run_pipeline_inpaint(&record, &config, &weights, &theme);
            Ok(())
        }

        "contacts" => {
            let mut fasta_path: Option<String> = None;
            let mut sequence_arg: Option<String> = None;
            let mut min_separation: usize = 6;
            let mut top_l_factor: f32 = 1.0;
            let mut export_csv: Option<String> = None;
            let mut export_json: Option<String> = None;

            let mut i = sub_args_start;
            while i < args.len() {
                if (args[i] == "--fasta" || args[i] == "-f") && i + 1 < args.len() {
                    fasta_path = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--sequence" || args[i] == "-s") && i + 1 < args.len() {
                    sequence_arg = Some(args[i + 1].clone());
                    i += 2;
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
            let (config, weights) = load_model(&model_path)?;

            let (top_contacts, enrichment, cys_indices) = run_pipeline_contacts(&record, min_separation, top_l_factor, &config, &weights, &theme);

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

            println!("--- Codon-Infer Engine V1.4 (Zero-Deps Multi-Scale CLI) ---");
            println!("Active SIMD Backend: {}", active_simd_backend());
            let load_timer = Instant::now();
            println!("Loading model weights from '{}'...", model_path);
            let (config, weights) = load_model(&model_path)?;
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

            let metrics = generate_sequence(
                &prompt,
                temperature,
                max_tokens,
                seed,
                &config,
                &weights,
                &mut run_state,
                Some(&mut print_cb),
            );

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
