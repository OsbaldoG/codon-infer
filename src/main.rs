use std::io::{self, Write};
use std::time::Instant;
use codon_infer::{
    active_simd_backend, export_contacts_csv, export_contacts_json,
    export_dms_csv, export_dms_json, export_ca_pdb, export_fold_tsv, export_fold_json,
    generate_sequence_parallel, generate_sequence_q8, generate_sequence_q4,
    parse_fasta_file, parse_raw_sequence, render_dms_heatmap,
    run_pipeline_contacts_parallel, run_pipeline_dms_with_mode,
    run_pipeline_inpaint_parallel, run_pipeline_fold_parallel, reconstruct_3d_ca,
    detect_optimal_threads, EsmConfig, EsmWeights, EsmWeightsQ8,
    EsmWeightsQ4, FastaRecord, ModelWeightsRef, RunState, TerminalTheme,
    CodqHeader, CODQ_MAGIC, DTYPE_Q4_0, DTYPE_Q8_0, DTYPE_FP32, QuantizationMode, DmsScoringMode,
};

fn print_version() {
    println!("codon-infer v{}", env!("CARGO_PKG_VERSION"));
    println!("Engine: Pure Rust Zero-Dependencies (std) + Tiled GEMM & Zero-Trunk 3D Reconstruction");
    println!("Vector Dispatch: {}", active_simd_backend());
    println!("Concurrency: Persistent Worker Pool with Atomic Barrier (zero OS thread spawning)");
    println!("Quantization: Multi-Format Block Engine (Q4_0 nibbles & Q8_0 blocks with CODQ header)");
    println!("Target Model: ESM-2 Multi-Scale [8M, 35M, 150M, 650M, 3B] (Meta AI / UniRef50)");
}

fn print_help(theme: &TerminalTheme) {
    let logo = if theme.enabled {
        "\x1b[38;2;139;92;246m╔═══════════════════════════════════════════════════════════════════════════════╗\n\
         ║   \x1b[38;2;59;130;246mCODON-INFER\x1b[38;2;139;92;246m: High-Performance Zero-Allocation ESM-2 Engine in Pure Rust     ║\n\
         ║   \x1b[38;2;16;185;129mVersion 1.8.0\x1b[38;2;139;92;246m     | Tiled GEMM, Canonical DMS, Contacts & 3D Fold Suite    ║\n\
         ╚═══════════════════════════════════════════════════════════════════════════════╝\x1b[0m"
    } else {
        "╔═══════════════════════════════════════════════════════════════════════════════╗\n\
         ║   CODON-INFER: High-Performance Zero-Allocation ESM-2 Engine in Pure Rust     ║\n\
         ║   Version 1.8.0     | Tiled GEMM, Canonical DMS, Contacts & 3D Fold Suite    ║\n\
         ╚═══════════════════════════════════════════════════════════════════════════════╝"
    };
    println!("{}", logo);
    println!("\nUSAGE:");
    println!("    codon-infer <SUBCOMMAND> [OPTIONS]");
    println!("\nSUBCOMMANDS:");
    println!("    generate     Autoregressive de novo sequence generation (causal mode)");
    println!("    inpaint      Masked sequence infilling & functional active-site reconstruction");
    println!("    dms          In silico Deep Mutational Scanning (Masked Marginal & 1-Pass Fast mode)");
    println!("    contacts     Zero-shot 3D contact map prediction (Official Multi-Layer APC + LR Head)");
    println!("    fold         Zero-trunk 3D protein structure folding & PDB atomic coordinate reconstruction");
    println!("    quantize     Export quantized weights binary in Q4_0 or Q8_0 block format");
    println!("    benchmark    Hardware validation suite (SIMD parity, cache thrashing, scaling)");
    println!("\nGLOBAL OPTIONS:");
    println!("    -h, --help               Print help information");
    println!("    -V, --version            Print engine version and active vector acceleration");
    println!("    -t, --threads <N>        Number of CPU threads (default: available physical cores)");
    println!("    --quantize <q4_0|q8_0|fp32> Quantization mode for execution (default: auto/fp32)");
    println!("    --q4                     Shortcut flag for '--quantize q4_0'");
    println!("    --q8                     Shortcut flag for '--quantize q8_0'");
    println!("    --model <PATH>           Path to weights binary (or 'all' for multi-scale benchmark)");
    println!("    --export-pdb <PATH>      Export reconstructed 3D C_alpha coordinates to standard PDB format");
    println!("    --export-json <PATH>     Export benchmark or analysis metrics to JSON");
    println!("    --color <auto|always|never> Control ANSI TrueColor terminal styling");
    println!("    --no-color               Disable all colored terminal escapes (NO_COLOR standard)");
    println!("\nSUBCOMMAND EXAMPLES:");
    println!("    codon-infer generate --sequence \"M\" --temperature 0.7 --max-tokens 50 --threads 4");
    println!("    codon-infer generate --sequence \"M\" --model esm2_650m_q4_0.bin --threads 4");
    println!("    codon-infer inpaint --sequence \"MCMPC*TTDHQMARKCDD\" --threads 4");
    println!("    codon-infer dms --fasta protein.fasta --window 128 --export-csv dms.csv --threads 4");
    println!("    codon-infer dms --fasta protein.fasta --fast --export-csv dms_fast.csv --threads 4");
    println!("    codon-infer contacts --sequence \"MCMPCFTTDHQMARKCDDCCGGKGRGKCYGPQCLCR\" --export-pdb contacts.pdb --threads 4");
    println!("    codon-infer fold --sequence \"MCMPCFTTDHQMARKCDDCCGGKGRGKCYGPQCLCR\" --export-pdb structure.pdb --threads 4");
    println!("    codon-infer quantize --model esm2_650m_weights.bin --output esm2_650m_q4_0.bin");
    println!("    codon-infer quantize q4_0 --model esm2_8m_weights.bin --output esm2_8m_q4_0.bin");
    println!("    codon-infer quantize --model esm2_8m_weights.bin --output esm2_8m_q8_0.bin");
    println!("    codon-infer benchmark --suite full --threads 4");
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
    Q4(EsmWeightsQ4),
}

impl LoadedWeights {
    pub fn as_ref(&self) -> ModelWeightsRef<'_> {
        match *self {
            LoadedWeights::Fp32(ref w) => ModelWeightsRef::Fp32(w),
            LoadedWeights::Q8(ref w) => ModelWeightsRef::Q8(w),
            LoadedWeights::Q4(ref w) => ModelWeightsRef::Q4(w),
        }
    }
}

pub fn load_model_weights(
    model_path: &str,
    quant_override: Option<QuantizationMode>,
) -> io::Result<(EsmConfig, LoadedWeights)> {
    // 1. Check for CODQ Header
    if let Ok(mut file) = std::fs::File::open(model_path) {
        use std::io::{Read, Seek};
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_ok() && magic == CODQ_MAGIC {
            let _ = file.rewind();
            #[allow(clippy::collapsible_if)]
            if let Ok(header) = CodqHeader::read_from(&mut file) {
                if let Ok(config) = header.to_config() {
                    match header.dtype {
                        DTYPE_Q4_0 => {
                            let weights = EsmWeightsQ4::from_file(model_path, &config)?;
                            return Ok((config, LoadedWeights::Q4(weights)));
                        }
                        DTYPE_Q8_0 => {
                            let weights = EsmWeightsQ8::from_file(model_path, &config)?;
                            return Ok((config, LoadedWeights::Q8(weights)));
                        }
                        DTYPE_FP32 => {
                            let weights = EsmWeights::from_file(model_path, &config)?;
                            match quant_override {
                                Some(QuantizationMode::Q4_0) => {
                                    return Ok((config, LoadedWeights::Q4(weights.quantize_q4())));
                                }
                                Some(QuantizationMode::Q8_0) => {
                                    return Ok((config, LoadedWeights::Q8(weights.quantize_q8())));
                                }
                                _ => return Ok((config, LoadedWeights::Fp32(weights))),
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let config = match EsmConfig::detect_from_file(model_path) {
        Ok(cfg) => cfg,
        Err(_) => EsmConfig::esm2_8m(),
    };

    let is_q4_file = model_path.ends_with(".q4")
        || model_path.ends_with(".q4.bin")
        || model_path.ends_with("_q4_0.bin")
        || model_path.ends_with("_q4.bin")
        || model_path.contains("q4_0")
        || model_path.contains("q4");

    if is_q4_file {
        let weights_q4 = EsmWeightsQ4::from_file(model_path, &config)?;
        return Ok((config, LoadedWeights::Q4(weights_q4)));
    }

    let file_len = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
    let is_q8_file = model_path.ends_with(".q8")
        || model_path.ends_with(".q8.bin")
        || model_path.ends_with("_q8_0.bin")
        || model_path.ends_with("_q8.bin")
        || model_path.contains("q8_0")
        || model_path.contains("q8")
        || file_len == 738_902_264;

    if is_q8_file {
        let weights_q8 = EsmWeightsQ8::from_file(model_path, &config)?;
        Ok((config, LoadedWeights::Q8(weights_q8)))
    } else {
        let weights = EsmWeights::from_file(model_path, &config)?;
        match quant_override {
            Some(QuantizationMode::Q4_0) => {
                let q4 = weights.quantize_q4();
                Ok((config, LoadedWeights::Q4(q4)))
            }
            Some(QuantizationMode::Q8_0) => {
                let q8 = weights.quantize_q8();
                Ok((config, LoadedWeights::Q8(q8)))
            }
            _ => Ok((config, LoadedWeights::Fp32(weights))),
        }
    }
}

pub fn load_model_weights_legacy(model_path: &str, quantize_q8: bool) -> io::Result<(EsmConfig, LoadedWeights)> {
    load_model_weights(model_path, if quantize_q8 { Some(QuantizationMode::Q8_0) } else { None })
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

    // Detect optimal hardware parallelism for vector workloads (physical cores on SMT)
    let default_threads = detect_optimal_threads();

    // Detect global preferences early
    let mut color_arg: Option<String> = None;
    let mut no_color_flag = false;
    let mut global_threads: Option<usize> = None;
    let mut global_quantize: Option<QuantizationMode> = None;

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
            if q_str == "q4_0" || q_str == "q4" {
                global_quantize = Some(QuantizationMode::Q4_0);
            } else if q_str == "q8_0" || q_str == "q8" {
                global_quantize = Some(QuantizationMode::Q8_0);
            } else if q_str == "fp32" {
                global_quantize = Some(QuantizationMode::Fp32);
            }
        } else if a == "--q8" {
            global_quantize = Some(QuantizationMode::Q8_0);
        } else if a == "--q4" {
            global_quantize = Some(QuantizationMode::Q4_0);
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
            let mut quant_mode = global_quantize;

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
                    if q_str == "q4_0" || q_str == "q4" {
                        quant_mode = Some(QuantizationMode::Q4_0);
                    } else if q_str == "q8_0" || q_str == "q8" {
                        quant_mode = Some(QuantizationMode::Q8_0);
                    } else if q_str == "fp32" {
                        quant_mode = Some(QuantizationMode::Fp32);
                    }
                    i += 2;
                } else if args[i] == "--q8" {
                    quant_mode = Some(QuantizationMode::Q8_0);
                    i += 1;
                } else if args[i] == "--q4" {
                    quant_mode = Some(QuantizationMode::Q4_0);
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

            let quantize_q8 = quant_mode == Some(QuantizationMode::Q8_0);

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
            let mut quant_mode = global_quantize;
            let mut dms_mode = DmsScoringMode::MaskedMarginal;

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
                    if q_str == "q4_0" || q_str == "q4" {
                        quant_mode = Some(QuantizationMode::Q4_0);
                    } else if q_str == "q8_0" || q_str == "q8" {
                        quant_mode = Some(QuantizationMode::Q8_0);
                    } else if q_str == "fp32" {
                        quant_mode = Some(QuantizationMode::Fp32);
                    }
                    i += 2;
                } else if args[i] == "--q8" {
                    quant_mode = Some(QuantizationMode::Q8_0);
                    i += 1;
                } else if args[i] == "--q4" {
                    quant_mode = Some(QuantizationMode::Q4_0);
                    i += 1;
                } else if args[i] == "--fast" || args[i] == "--wt-marginal" {
                    dms_mode = DmsScoringMode::WildtypeMarginal;
                    i += 1;
                } else if args[i] == "--mode" && i + 1 < args.len() {
                    let m_str = args[i + 1].to_lowercase();
                    if m_str == "wt" || m_str == "fast" || m_str == "wildtype" || m_str == "wt-marginal" {
                        dms_mode = DmsScoringMode::WildtypeMarginal;
                    } else {
                        dms_mode = DmsScoringMode::MaskedMarginal;
                    }
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
            let (config, loaded_weights) = load_model_weights(&model_path, quant_mode)?;

            let result = run_pipeline_dms_with_mode(
                &record,
                window_size,
                stride,
                &config,
                loaded_weights.as_ref(),
                num_threads,
                &theme,
                dms_mode,
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
            let mut quant_mode = global_quantize;

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
                    if q_str == "q4_0" || q_str == "q4" {
                        quant_mode = Some(QuantizationMode::Q4_0);
                    } else if q_str == "q8_0" || q_str == "q8" {
                        quant_mode = Some(QuantizationMode::Q8_0);
                    } else if q_str == "fp32" {
                        quant_mode = Some(QuantizationMode::Fp32);
                    }
                    i += 2;
                } else if args[i] == "--q8" {
                    quant_mode = Some(QuantizationMode::Q8_0);
                    i += 1;
                } else if args[i] == "--q4" {
                    quant_mode = Some(QuantizationMode::Q4_0);
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
            let (config, loaded_weights) = load_model_weights(&model_path, quant_mode)?;

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
            let mut export_pdb: Option<String> = None;
            let mut num_threads = global_threads.unwrap_or(default_threads);
            let mut quant_mode = global_quantize;

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
                    if q_str == "q4_0" || q_str == "q4" {
                        quant_mode = Some(QuantizationMode::Q4_0);
                    } else if q_str == "q8_0" || q_str == "q8" {
                        quant_mode = Some(QuantizationMode::Q8_0);
                    } else if q_str == "fp32" {
                        quant_mode = Some(QuantizationMode::Fp32);
                    }
                    i += 2;
                } else if args[i] == "--q8" {
                    quant_mode = Some(QuantizationMode::Q8_0);
                    i += 1;
                } else if args[i] == "--q4" {
                    quant_mode = Some(QuantizationMode::Q4_0);
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
                } else if args[i] == "--export-pdb" && i + 1 < args.len() {
                    export_pdb = Some(args[i + 1].clone());
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
            let (config, loaded_weights) = load_model_weights(&model_path, quant_mode)?;

            let contacts_res = run_pipeline_contacts_parallel(
                &record,
                min_separation,
                top_l_factor,
                &config,
                loaded_weights.as_ref(),
                num_threads,
                &theme,
            );

            if let Some(ref csv_path) = export_csv {
                export_contacts_csv(csv_path, &record.id, &contacts_res.top_contacts)?;
                println!("  [EXPORT] Predicted contact table saved to '{}'.", csv_path);
            }
            if let Some(ref json_path) = export_json {
                export_contacts_json(json_path, &record.id, &record.sequence, &contacts_res.cys_indices, contacts_res.cys_enrichment_ratio, &contacts_res.top_contacts)?;
                println!("  [EXPORT] Contact analysis JSON saved to '{}'.", json_path);
            }
            if let Some(ref pdb_path) = export_pdb {
                let (coords, num_restraints, _) = reconstruct_3d_ca(&contacts_res.contact_probs, record.sequence.len());
                export_ca_pdb(pdb_path, &record.sequence, &record.id, &coords)?;
                println!("  [EXPORT] Reconstructed 3D C_alpha coordinates saved to '{}' ({} tertiary restraints).", pdb_path, num_restraints);
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
            let mut quant_mode = global_quantize;

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
                    if q_str == "q4_0" || q_str == "q4" {
                        quant_mode = Some(QuantizationMode::Q4_0);
                    } else if q_str == "q8_0" || q_str == "q8" {
                        quant_mode = Some(QuantizationMode::Q8_0);
                    } else if q_str == "fp32" {
                        quant_mode = Some(QuantizationMode::Fp32);
                    }
                    i += 2;
                } else if args[i] == "--q8" {
                    quant_mode = Some(QuantizationMode::Q8_0);
                    i += 1;
                } else if args[i] == "--q4" {
                    quant_mode = Some(QuantizationMode::Q4_0);
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

            let load_timer = Instant::now();
            println!("Loading model weights from '{}'...", model_path);
            let (config, loaded_weights) = load_model_weights(&model_path, quant_mode)?;
            let prec_name = match loaded_weights {
                LoadedWeights::Fp32(_) => "FP32 (Canonical)",
                LoadedWeights::Q8(_) => "Q8_0 (In-Memory Blocks)",
                LoadedWeights::Q4(_) => "Q4_0 (4-bit Nibble Blocks)",
            };

            println!("--- Codon-Infer Engine V1.6 (Zero-Deps Multi-Scale CLI) ---");
            println!("Active SIMD Backend:  {}", active_simd_backend());
            println!("Concurrency Threads:  {}", num_threads);
            println!("Execution Precision:  {}", prec_name);
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
                LoadedWeights::Q4(ref weights_q4) => {
                    generate_sequence_q4(
                        &prompt,
                        temperature,
                        max_tokens,
                        seed,
                        &config,
                        weights_q4,
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

        "quantize" => {
            let mut format_arg: Option<String> = None;
            let mut input_model_arg: Option<String> = None;
            let mut output_path_arg: Option<String> = None;

            let mut i = sub_args_start;
            // Check for positional format: codon-infer quantize q4_0 ... or codon-infer quantize q8_0 ...
            if i < args.len() && !args[i].starts_with('-') {
                let lower = args[i].to_lowercase();
                if lower == "q4_0" || lower == "q4" || lower == "q8_0" || lower == "q8" {
                    format_arg = Some(lower);
                    i += 1;
                }
            }

            while i < args.len() {
                if (args[i] == "--model" || args[i] == "-m" || args[i] == "--input" || args[i] == "-i") && i + 1 < args.len() {
                    input_model_arg = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--output" || args[i] == "-o") && i + 1 < args.len() {
                    output_path_arg = Some(args[i + 1].clone());
                    i += 2;
                } else if (args[i] == "--format" || args[i] == "-f") && i + 1 < args.len() {
                    format_arg = Some(args[i + 1].to_lowercase());
                    i += 2;
                } else if !args[i].starts_with('-') && format_arg.is_none() {
                    let lower = args[i].to_lowercase();
                    if lower == "q4_0" || lower == "q4" || lower == "q8_0" || lower == "q8" {
                        format_arg = Some(lower);
                    }
                    i += 1;
                } else {
                    i += 1;
                }
            }

            // Determine whether to use Q4_0 or Q8_0:
            // Priority:
            // 1. Explicit format argument (--format or positional)
            // 2. Auto-detection from output filename:
            //    - if contains "q4_0" or "q4": Q4_0
            //    - if contains "q8_0" or "q8": Q8_0
            // 3. Default: Q8_0
            let is_q4 = match format_arg.as_deref() {
                Some("q4_0") | Some("q4") => true,
                Some("q8_0") | Some("q8") => false,
                _ => {
                    if let Some(ref out) = output_path_arg {
                        let out_lower = out.to_lowercase();
                        out_lower.contains("q4_0") || out_lower.contains("q4")
                    } else {
                        false
                    }
                }
            };

            let input_model = input_model_arg.unwrap_or_else(|| "esm2_650m_weights.bin".to_string());
            let output_path = output_path_arg.unwrap_or_else(|| {
                if is_q4 {
                    "esm2_650m_q4_0.bin".to_string()
                } else {
                    "esm2_650m_q8_0.bin".to_string()
                }
            });

            if is_q4 {
                println!("┌────────────────────────────────────────────────────────────────────────┐");
                println!("│ CODON-INFER: MOTOR DE CUANTIZACIÓN STREAMING Q4_0                      │");
                println!("└────────────────────────────────────────────────────────────────────────┘");
                println!("  -> Archivo fuente FP32: '{}'", input_model);
                println!("  -> Archivo destino Q4_0: '{}'", output_path);
                let t0 = Instant::now();
                let config = EsmConfig::detect_from_file(&input_model)?;
                println!("  -> Arquitectura detectada: {}", config.model_name());
                let file_size = match codon_infer::quantize_file_streaming_q4(&input_model, &output_path, &config) {
                    Ok(sz) => sz,
                    Err(_) => {
                        let fp32 = EsmWeights::from_file(&input_model, &config)?;
                        let q4 = fp32.quantize_q4();
                        q4.save_to_file(&output_path, &config)?;
                        std::fs::metadata(&output_path)?.len()
                    }
                };
                let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "  -> Streaming Q4_0 completado: {} bytes ({:.2} MB) en {:.1} ms.",
                    file_size,
                    file_size as f64 / 1_048_576.0,
                    total_ms
                );
                println!("==========================================================================");
                println!("  ESTADO: BINARIO Q4_0 GENERADO Y VALIDADO EN DISCO                      ");
                println!("==========================================================================\n");
            } else {
                println!("┌────────────────────────────────────────────────────────────────────────┐");
                println!("│ CODON-INFER: MOTOR DE CUANTIZACIÓN STREAMING Q8_0                      │");
                println!("└────────────────────────────────────────────────────────────────────────┘");
                println!("  -> Archivo fuente FP32: '{}'", input_model);
                println!("  -> Archivo destino Q8_0: '{}'", output_path);
                let t0 = Instant::now();
                let config = EsmConfig::detect_from_file(&input_model)?;
                println!("  -> Arquitectura detectada: {}", config.model_name());
                let file_size = match codon_infer::quantize_file_streaming_q8(&input_model, &output_path, &config) {
                    Ok(sz) => sz,
                    Err(_) => {
                        let fp32 = EsmWeights::from_file(&input_model, &config)?;
                        let q8 = fp32.quantize_q8();
                        q8.save_to_file_with_header(&output_path, &config)?;
                        std::fs::metadata(&output_path)?.len()
                    }
                };
                let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "  -> Streaming Q8_0 completado: {} bytes ({:.2} MB) en {:.1} ms.",
                    file_size,
                    file_size as f64 / 1_048_576.0,
                    total_ms
                );
                println!("==========================================================================");
                println!("  ESTADO: BINARIO Q8_0 GENERADO Y VALIDADO EN DISCO                      ");
                println!("==========================================================================\n");
            }
            Ok(())
        }

        "fold" => {
            let mut fasta_path: Option<String> = None;
            let mut sequence_arg: Option<String> = None;
            let mut export_pdb: Option<String> = None;
            let mut export_tsv: Option<String> = None;
            let mut export_json: Option<String> = None;
            let mut num_threads = global_threads.unwrap_or(default_threads);
            let mut quant_mode = global_quantize;

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
                    if q_str == "q4_0" || q_str == "q4" {
                        quant_mode = Some(QuantizationMode::Q4_0);
                    } else if q_str == "q8_0" || q_str == "q8" {
                        quant_mode = Some(QuantizationMode::Q8_0);
                    } else if q_str == "fp32" {
                        quant_mode = Some(QuantizationMode::Fp32);
                    }
                    i += 2;
                } else if args[i] == "--q8" {
                    quant_mode = Some(QuantizationMode::Q8_0);
                    i += 1;
                } else if args[i] == "--q4" {
                    quant_mode = Some(QuantizationMode::Q4_0);
                    i += 1;
                } else if args[i] == "--export-pdb" && i + 1 < args.len() {
                    export_pdb = Some(args[i + 1].clone());
                    i += 2;
                } else if args[i] == "--export-tsv" && i + 1 < args.len() {
                    export_tsv = Some(args[i + 1].clone());
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
                    eprintln!("Warning: unrecognized fold option '{}'", args[i]);
                    i += 1;
                }
            }

            let record = load_input_record(fasta_path.as_deref(), sequence_arg.as_deref(), "target_fold")?;
            let (config, loaded_weights) = load_model_weights(&model_path, quant_mode)?;

            let result = run_pipeline_fold_parallel(
                &record,
                &config,
                loaded_weights.as_ref(),
                num_threads,
                export_pdb.as_deref(),
                &theme,
            );

            if let Some(ref tsv_path) = export_tsv {
                export_fold_tsv(tsv_path, &result.protein_id, &result.sequence, &result.coords)?;
                println!("  [EXPORT] 3D coordinates TSV saved to '{}'.", tsv_path);
            }
            if let Some(ref json_path) = export_json {
                export_fold_json(json_path, &result)?;
                println!("  [EXPORT] 3D folding analysis JSON saved to '{}'.", json_path);
            }
            Ok(())
        }

        unknown => {
            eprintln!("Error: Unknown subcommand '{}'.", unknown);
            eprintln!("Run 'codon-infer --help' for a list of available subcommands.");
            std::process::exit(1);
        }
    }
}
