use std::time::Instant;
use codon_infer::{
    Tokenizer, EsmConfig, EsmWeights, EsmWeightsQ8, EsmWeightsQ4,
    BidirectionalRunState, forward_bidirectional_parallel_ext,
    forward_bidirectional_q8, forward_bidirectional_q4_ext,
};

fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

fn compute_rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum_sq: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum();
    (sum_sq / a.len() as f32).sqrt()
}

fn main() {
    let scales = [
        ("8M", "esm2_8m_weights.bin", "esm2_8m_q8_0.bin", "esm2_8m_q4_0.bin", EsmConfig::esm2_8m()),
        ("35M", "esm2_35m_weights.bin", "esm2_35m_q8_0.bin", "esm2_35m_q4_0.bin", EsmConfig::esm2_35m()),
        ("150M", "esm2_150m_weights.bin", "esm2_150m_q8_0.bin", "esm2_150m_q4_0.bin", EsmConfig::esm2_150m()),
        ("650M", "esm2_650m_weights.bin", "esm2_650m_q8_0_header.bin", "esm2_650m_q4_0.bin", EsmConfig::esm2_650m()),
    ];

    let seq = "AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS";
    let tokens = Tokenizer::encode(seq);
    let seq_len = tokens.len();
    let num_threads = 4;

    println!("=========================================================================================");
    println!("     CODON-INFER v1.6.0: SCRIPT DE DIAGNÓSTICO POR CAPA Y FIDELIDAD DE LOGITS           ");
    println!("=========================================================================================\n");
    println!("Secuencia de prueba: {} ({} tokens)", seq, seq_len);
    println!("Hilos de cómputo: {}\n", num_threads);

    for (scale_name, fp32_path, q8_path, q4_path, config) in &scales {
        println!("#########################################################################################");
        println!(" ESCALA: {} ({})", scale_name, config.model_name());
        println!("#########################################################################################");

        if !std::path::Path::new(fp32_path).exists() || !std::path::Path::new(q8_path).exists() || !std::path::Path::new(q4_path).exists() {
            println!("  [SKIP] Uno o más archivos de pesos no encontrados para {}\n", scale_name);
            continue;
        }

        let fp32_size = std::fs::metadata(fp32_path).unwrap().len();
        let q8_size = std::fs::metadata(q8_path).unwrap().len();
        let q4_size = std::fs::metadata(q4_path).unwrap().len();

        println!(" [1] Tamaños exactos en disco (stat -c %s):");
        println!("   * FP32: {:>12} bytes ({:>8.2} MB)", fp32_size, fp32_size as f64 / 1_048_576.0);
        println!("   * Q8_0: {:>12} bytes ({:>8.2} MB) | Compresión: {:.4}x", q8_size, q8_size as f64 / 1_048_576.0, fp32_size as f64 / q8_size as f64);
        println!("   * Q4_0: {:>12} bytes ({:>8.2} MB) | Compresión: {:.4}x", q4_size, q4_size as f64 / 1_048_576.0, fp32_size as f64 / q4_size as f64);

        let t0 = Instant::now();
        let weights_fp = EsmWeights::from_file(fp32_path, config).expect("FP32 weights load");
        let weights_q8 = EsmWeightsQ8::load_from_file(q8_path).expect("Q8 weights load").1;
        let weights_q4 = EsmWeightsQ4::load_from_file(q4_path).expect("Q4 weights load").1;
        println!("   * Carga de los 3 modelos completada en {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0);

        // Architecture & Classifier analysis
        let dim = config.dim;
        let req_cls_len = 3 * dim + dim * dim + config.vocab_size * dim + config.vocab_size;
        println!("\n [2] Inspección del cabezal LM y ligamiento:");
        println!("   * dim: {}, n_layers: {}, vocab_size: {}", dim, config.n_layers, config.vocab_size);
        println!("   * classifier_weights.len(): {} (requerido para LMHead dedicado: {})", weights_fp.classifier_weights.len(), req_cls_len);
        let has_dedicated_head = weights_fp.classifier_weights.len() >= req_cls_len;
        println!("   * Cabezal dedicado EsmLMHead presente: {}", has_dedicated_head);
        if has_dedicated_head {
            println!("   * Nota arquitectónica: Todos los modelos ESM-2 (8M, 35M, 150M, 650M) usan la MISMA ruta de código EsmLMHead.");
        }

        // Run forward passes with layer tracking
        let mut state_fp = BidirectionalRunState::new(config, seq_len + 8);
        let mut fp_layers: Vec<Vec<f32>> = Vec::new();
        forward_bidirectional_parallel_ext(&tokens, config, &weights_fp, &mut state_fp, num_threads, Some(&mut |_l, x_slice| {
            fp_layers.push(x_slice.to_vec());
        }));

        let mut state_q4 = BidirectionalRunState::new(config, seq_len + 8);
        let mut q4_layers: Vec<Vec<f32>> = Vec::new();
        forward_bidirectional_q4_ext(&tokens, config, &weights_q4, &mut state_q4, num_threads, Some(&mut |_l, x_slice| {
            q4_layers.push(x_slice.to_vec());
        }));

        let mut state_q8 = BidirectionalRunState::new(config, seq_len + 8);
        forward_bidirectional_q8(&tokens, config, &weights_q8, &mut state_q8, num_threads);

        // Embeddings and Logits similarity
        let emb_len = seq_len * dim;
        let cos_emb_q4 = compute_cosine_similarity(&state_fp.x[0..emb_len], &state_q4.x[0..emb_len]);
        let rmse_emb_q4 = compute_rmse(&state_fp.x[0..emb_len], &state_q4.x[0..emb_len]);
        let cos_emb_q8 = compute_cosine_similarity(&state_fp.x[0..emb_len], &state_q8.x[0..emb_len]);
        let rmse_emb_q8 = compute_rmse(&state_fp.x[0..emb_len], &state_q8.x[0..emb_len]);

        let log_len = seq_len * config.vocab_size;
        let cos_log_q4 = compute_cosine_similarity(&state_fp.logits[0..log_len], &state_q4.logits[0..log_len]);
        let rmse_log_q4 = compute_rmse(&state_fp.logits[0..log_len], &state_q4.logits[0..log_len]);
        let cos_log_q8 = compute_cosine_similarity(&state_fp.logits[0..log_len], &state_q8.logits[0..log_len]);
        let rmse_log_q8 = compute_rmse(&state_fp.logits[0..log_len], &state_q8.logits[0..log_len]);

        // Top-1 Argmax Token Agreement
        let mut top1_match_q8 = 0;
        let mut top1_match_q4 = 0;
        for p in 0..seq_len {
            let offset = p * config.vocab_size;
            let fp_slice = &state_fp.logits[offset..offset + config.vocab_size];
            let q8_slice = &state_q8.logits[offset..offset + config.vocab_size];
            let q4_slice = &state_q4.logits[offset..offset + config.vocab_size];

            let max_fp = fp_slice.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let max_q8 = q8_slice.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let max_q4 = q4_slice.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

            if max_fp == max_q8 { top1_match_q8 += 1; }
            if max_fp == max_q4 { top1_match_q4 += 1; }
        }

        println!("\n [3] Fidelidad numérica de representaciones:");
        println!("   * Embeddings Finales FP32 vs Q8_0: Coseno = {:.6}, RMSE = {:.6}", cos_emb_q8, rmse_emb_q8);
        println!("   * Embeddings Finales FP32 vs Q4_0: Coseno = {:.6}, RMSE = {:.6}", cos_emb_q4, rmse_emb_q4);
        println!("   * Logits Finales     FP32 vs Q8_0: Coseno = {:.6}, RMSE = {:.6} | Top-1 match: {}/{} ({:.1}%)", cos_log_q8, rmse_log_q8, top1_match_q8, seq_len, (top1_match_q8 as f64 / seq_len as f64) * 100.0);
        println!("   * Logits Finales     FP32 vs Q4_0: Coseno = {:.6}, RMSE = {:.6} | Top-1 match: {}/{} ({:.1}%)", cos_log_q4, rmse_log_q4, top1_match_q4, seq_len, (top1_match_q4 as f64 / seq_len as f64) * 100.0);

        println!("\n [4] Deriva capa a capa (Hidden States FP32 vs Q4_0):");
        for l in 0..config.n_layers {
            let cos_l = compute_cosine_similarity(&fp_layers[l], &q4_layers[l]);
            let rmse_l = compute_rmse(&fp_layers[l], &q4_layers[l]);
            println!("   - Capa {:>2}/{:>2}: Coseno = {:.6}, RMSE = {:.6}", l + 1, config.n_layers, cos_l, rmse_l);
        }
        println!();
    }
}
