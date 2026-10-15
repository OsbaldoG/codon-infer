fn main() -> std::io::Result<()> {
    let model_path = if std::path::Path::new("esm2_8m_weights.bin").exists() {
        "esm2_8m_weights.bin"
    } else if std::path::Path::new("esm2_weights.bin").exists() {
        "esm2_weights.bin"
    } else {
        "/home/manuel/REA_GENOMICS/codon-infer/esm2_8m_weights.bin"
    };
    codon_infer::mcp::run_stdio_server(model_path)
}
