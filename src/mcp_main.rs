fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_arg: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        if (args[i] == "--model" || args[i] == "-m") && i + 1 < args.len() {
            model_arg = Some(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }

    let model_path = model_arg
        .or_else(|| std::env::var("CODON_INFER_MODEL").ok())
        .unwrap_or_else(|| {
            codon_infer::discover_available_models()
                .into_iter()
                .next()
                .unwrap_or_else(|| "esm2_8m_weights.bin".to_string())
        });

    codon_infer::mcp::run_stdio_server(&model_path)
}
