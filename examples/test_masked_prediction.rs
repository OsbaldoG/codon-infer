use codon_infer::{
    Tokenizer, EsmConfig, EsmWeights, EsmWeightsQ4,
    BidirectionalRunState, forward_bidirectional_parallel,
    forward_bidirectional_q4,
};

fn main() {
    let unmasked_seq = "AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS";
    let orig_tokens = Tokenizer::encode(unmasked_seq);
    
    // Create sequences with each residue masked one by one (in-painting / masked residue prediction)
    println!("================================================================================");
    println!(" EVALUACIÓN EN TAREA REAL MLM (MASKED LANGUAGE MODELING):");
    println!(" Predicción de aminoácidos enmascarados (*) FP32 vs Q4_0");
    println!("================================================================================\n");

    let scales = [
        ("8M", "esm2_8m_weights.bin", "esm2_8m_q4_0.bin", EsmConfig::esm2_8m()),
        ("35M", "esm2_35m_weights.bin", "esm2_35m_q4_0.bin", EsmConfig::esm2_35m()),
        ("150M", "esm2_150m_weights.bin", "esm2_150m_q4_0.bin", EsmConfig::esm2_150m()),
    ];

    let masked_positions = vec![1, 4, 8, 12, 16, 20, 24, 28, 32, 36];

    for (scale, fp_p, q4_p, cfg) in scales {
        if !std::path::Path::new(fp_p).exists() || !std::path::Path::new(q4_p).exists() {
            continue;
        }
        let w_fp = EsmWeights::from_file(fp_p, &cfg).unwrap();
        let w_q4 = EsmWeightsQ4::load_from_file(q4_p).unwrap().1;
        let mut st_fp = BidirectionalRunState::new(&cfg, orig_tokens.len() + 8);
        let mut st_q4 = BidirectionalRunState::new(&cfg, orig_tokens.len() + 8);

        let mut fp_matches_orig = 0;
        let mut q4_matches_orig = 0;
        let mut fp_vs_q4_matches = 0;
        let total = masked_positions.len();

        for &pos in &masked_positions {
            let mut masked_tokens = orig_tokens.clone();
            masked_tokens[pos] = 32; // mask token '*'

            forward_bidirectional_parallel(&masked_tokens, &cfg, &w_fp, &mut st_fp, 4);
            forward_bidirectional_q4(&masked_tokens, &cfg, &w_q4, &mut st_q4, 4);

            let s_fp = &st_fp.logits[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];
            let s_q4 = &st_q4.logits[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];

            let top_fp = s_fp.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let top_q4 = s_q4.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let true_tok = orig_tokens[pos];

            if top_fp == true_tok { fp_matches_orig += 1; }
            if top_q4 == true_tok { q4_matches_orig += 1; }
            if top_fp == top_q4 { fp_vs_q4_matches += 1; }
        }

        println!("ESCALA {}:", scale);
        println!("  - Paridad Top-1 FP32 vs Q4_0 en posiciones enmascaradas: {}/{} ({:.1}%)",
            fp_vs_q4_matches, total, (fp_vs_q4_matches as f64 / total as f64) * 100.0);
        println!("  - Precisión recuperación residuo original FP32: {}/{} ({:.1}%)",
            fp_matches_orig, total, (fp_matches_orig as f64 / total as f64) * 100.0);
        println!("  - Precisión recuperación residuo original Q4_0: {}/{} ({:.1}%)\n",
            q4_matches_orig, total, (q4_matches_orig as f64 / total as f64) * 100.0);
    }
}
