use codon_infer::{
    Tokenizer, EsmConfig, EsmWeights,
    BidirectionalRunState, forward_bidirectional, forward_bidirectional_q8, forward_bidirectional_q4,
    compute_cosine_similarity,
};

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }
    fn next_range(&mut self, min: usize, max: usize) -> usize {
        min + (self.next_u32() as usize % (max - min))
    }
}

fn softmax_canonical(logits: &[f32]) -> [f32; 20] {
    let mut max_val = f32::NEG_INFINITY;
    for i in 4..=23 {
        if logits[i] > max_val { max_val = logits[i]; }
    }
    let mut sum_exp = 0.0f32;
    let mut exps = [0.0f32; 20];
    for i in 4..=23 {
        let e = (logits[i] - max_val).exp();
        exps[i - 4] = e;
        sum_exp += e;
    }
    for i in 0..20 {
        exps[i] /= sum_exp;
    }
    exps
}

fn kl_divergence_canonical(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    let p = softmax_canonical(p_logits);
    let q = softmax_canonical(q_logits);
    let mut kl = 0.0f32;
    for i in 0..20 {
        if p[i] > 1e-12 {
            let q_clamped = q[i].max(1e-12);
            kl += p[i] * (p[i] / q_clamped).ln();
        }
    }
    kl
}

fn centered_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        let db = y - mean_b;
        dot += da * db;
        norm_a += da * da;
        norm_b += db * db;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

fn argmax_canonical(logits: &[f32]) -> usize {
    let mut best_i = 4;
    let mut best_v = f32::NEG_INFINITY;
    for i in 4..=23 {
        if logits[i] > best_v {
            best_v = logits[i];
            best_i = i;
        }
    }
    best_i
}

fn main() {
    println!("=========================================================================================");
    println!("     VALIDACIÓN NUMÉRICA RIGUROSA: FP32 CORREGIDO vs Q8_0 vs Q4_0 (ESM-2 8M)            ");
    println!("=========================================================================================\n");

    let fp32_path = "esm2_8m_weights.bin";
    let q8_path = "esm2_8m_q8_0.bin";
    let q4_path = "esm2_8m_q4_0.bin";

    let config = EsmConfig::esm2_8m();
    println!("1. Cargando modelo FP32 corregido desde '{}'...", fp32_path);
    let weights_fp = EsmWeights::from_file(fp32_path, &config).expect("Carga FP32");

    println!("2. Cuantizando a Q8_0 y Q4_0 directamente desde FP32 corregido...");
    let weights_q8 = weights_fp.quantize_q8();
    let weights_q4 = weights_fp.quantize_q4();

    println!("   Guardando pesos cuantizados actualizados en disco...");
    weights_q8.save_to_file_with_header(q8_path, &config).expect("Guardar Q8");
    weights_q4.save_to_file(q4_path, &config).expect("Guardar Q4");

    let fasta_path = "test_peptidase.fasta";
    let raw_seq = if std::path::Path::new(fasta_path).exists() {
        let content = std::fs::read_to_string(fasta_path).expect("Read fasta");
        content.lines().filter(|l| !l.is_empty() && !l.starts_with('>')).collect::<Vec<_>>().join("")
    } else {
        "AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS".to_string()
    };
    println!("\nSecuencia de prueba ({} aa): {}", raw_seq.len(), raw_seq);

    let original_tokens = Tokenizer::encode(&raw_seq);
    let seq_len = original_tokens.len();
    let aa_len = raw_seq.len();

    // -------------------------------------------------------------------------
    // EXPERIMENTO 1: SECUENCIA SIN MÁSCARA (UNMASKED)
    // -------------------------------------------------------------------------
    println!("\n=========================================================================================");
    println!(" [EXPERIMENTO 1] SECUENCIA COMPLETA SIN MÁSCARA (UNMASKED)");
    println!("=========================================================================================");

    let mut state_fp = BidirectionalRunState::new(&config, seq_len);
    let mut state_q8 = BidirectionalRunState::new(&config, seq_len);
    let mut state_q4 = BidirectionalRunState::new(&config, seq_len);

    forward_bidirectional(&original_tokens, &config, &weights_fp, &mut state_fp);
    forward_bidirectional_q8(&original_tokens, &config, &weights_q8, &mut state_q8, 4);
    forward_bidirectional_q4(&original_tokens, &config, &weights_q4, &mut state_q4, 4);

    let mut fp_vs_true_matches = 0;
    let mut q8_vs_true_matches = 0;
    let mut q4_vs_true_matches = 0;

    let mut q8_vs_fp_matches = 0;
    let mut q4_vs_fp_matches = 0;

    let mut kl_q8_sum = 0.0f32;
    let mut kl_q4_sum = 0.0f32;

    let mut ccos_q8_sum = 0.0f32;
    let mut ccos_q4_sum = 0.0f32;

    for i in 0..aa_len {
        let pos = i + 1;
        let true_tok = original_tokens[pos];

        let fp_log = &state_fp.logits[pos * config.vocab_size..(pos + 1) * config.vocab_size];
        let q8_log = &state_q8.logits[pos * config.vocab_size..(pos + 1) * config.vocab_size];
        let q4_log = &state_q4.logits[pos * config.vocab_size..(pos + 1) * config.vocab_size];

        let fp_top1 = argmax_canonical(fp_log);
        let q8_top1 = argmax_canonical(q8_log);
        let q4_top1 = argmax_canonical(q4_log);

        if fp_top1 == true_tok { fp_vs_true_matches += 1; }
        if q8_top1 == true_tok { q8_vs_true_matches += 1; }
        if q4_top1 == true_tok { q4_vs_true_matches += 1; }

        if q8_top1 == fp_top1 { q8_vs_fp_matches += 1; }
        if q4_top1 == fp_top1 { q4_vs_fp_matches += 1; }

        kl_q8_sum += kl_divergence_canonical(fp_log, q8_log);
        kl_q4_sum += kl_divergence_canonical(fp_log, q4_log);

        ccos_q8_sum += centered_cosine_similarity(fp_log, q8_log);
        ccos_q4_sum += centered_cosine_similarity(fp_log, q4_log);
    }

    let global_cos_q8 = compute_cosine_similarity(&state_fp.logits[0..seq_len * config.vocab_size], &state_q8.logits[0..seq_len * config.vocab_size]);
    let global_cos_q4 = compute_cosine_similarity(&state_fp.logits[0..seq_len * config.vocab_size], &state_q4.logits[0..seq_len * config.vocab_size]);

    let global_ccos_q8 = centered_cosine_similarity(&state_fp.logits[0..seq_len * config.vocab_size], &state_q8.logits[0..seq_len * config.vocab_size]);
    let global_ccos_q4 = centered_cosine_similarity(&state_fp.logits[0..seq_len * config.vocab_size], &state_q4.logits[0..seq_len * config.vocab_size]);

    println!("* Precisión Top-1 frente al residuo real:");
    println!("  - FP32:  {:>2}/{} ({:.2}%)", fp_vs_true_matches, aa_len, fp_vs_true_matches as f32 / aa_len as f32 * 100.0);
    println!("  - Q8_0:  {:>2}/{} ({:.2}%)", q8_vs_true_matches, aa_len, q8_vs_true_matches as f32 / aa_len as f32 * 100.0);
    println!("  - Q4_0:  {:>2}/{} ({:.2}%)", q4_vs_true_matches, aa_len, q4_vs_true_matches as f32 / aa_len as f32 * 100.0);

    println!("\n* Fidelidad frente a FP32 corregido (Unmasked):");
    println!("  - Q8_0: Top-1 vs FP32 = {:>2}/{} ({:.2}%) | KL Promedio = {:.6} | Coseno Centrado = {:.6} (Global: {:.6}, Cos Estándar: {:.6})",
        q8_vs_fp_matches, aa_len, q8_vs_fp_matches as f32 / aa_len as f32 * 100.0,
        kl_q8_sum / aa_len as f32, ccos_q8_sum / aa_len as f32, global_ccos_q8, global_cos_q8);
    println!("  - Q4_0: Top-1 vs FP32 = {:>2}/{} ({:.2}%) | KL Promedio = {:.6} | Coseno Centrado = {:.6} (Global: {:.6}, Cos Estándar: {:.6})",
        q4_vs_fp_matches, aa_len, q4_vs_fp_matches as f32 / aa_len as f32 * 100.0,
        kl_q4_sum / aa_len as f32, ccos_q4_sum / aa_len as f32, global_ccos_q4, global_cos_q4);

    // -------------------------------------------------------------------------
    // EXPERIMENTO 2: 15% DE MÁSCARA ALEATORIA CON SEMILLA FIJA (SEED = 42)
    // -------------------------------------------------------------------------
    println!("\n=========================================================================================");
    println!(" [EXPERIMENTO 2] 15% DE MÁSCARA ALEATORIA CON SEMILLA FIJA (SEED = 42)");
    println!("=========================================================================================");

    let mut rng = SimpleRng::new(42);
    let num_masks = ((aa_len as f32) * 0.15).round() as usize; // 6 residuos en 39 aa
    let mut masked_positions = Vec::new();
    while masked_positions.len() < num_masks {
        let p = rng.next_range(1, aa_len + 1); // 1..=39
        if !masked_positions.contains(&p) {
            masked_positions.push(p);
        }
    }
    masked_positions.sort();

    let mut masked_tokens = original_tokens.clone();
    for &p in &masked_positions {
        masked_tokens[p] = 32; // <mask_1>
    }

    print!("Posiciones enmascaradas ({} posiciones): ", masked_positions.len());
    for &p in &masked_positions {
        let orig_c = raw_seq.chars().nth(p - 1).unwrap();
        print!("{}{} ", orig_c, p);
    }
    println!();

    forward_bidirectional(&masked_tokens, &config, &weights_fp, &mut state_fp);
    forward_bidirectional_q8(&masked_tokens, &config, &weights_q8, &mut state_q8, 4);
    forward_bidirectional_q4(&masked_tokens, &config, &weights_q4, &mut state_q4, 4);

    // Métricas en TODA la secuencia con 15% de máscara
    let mut m_fp_vs_true_all = 0;
    let mut m_q8_vs_true_all = 0;
    let mut m_q4_vs_true_all = 0;

    let mut m_q8_vs_fp_all = 0;
    let mut m_q4_vs_fp_all = 0;

    let mut m_kl_q8_all = 0.0f32;
    let mut m_kl_q4_all = 0.0f32;

    let mut m_ccos_q8_all = 0.0f32;
    let mut m_ccos_q4_all = 0.0f32;

    // Métricas EXCLUSIVAS en las posiciones enmascaradas
    let mut m_fp_vs_true_mask = 0;
    let mut m_q8_vs_true_mask = 0;
    let mut m_q4_vs_true_mask = 0;

    let mut m_q8_vs_fp_mask = 0;
    let mut m_q4_vs_fp_mask = 0;

    let mut m_kl_q8_mask = 0.0f32;
    let mut m_kl_q4_mask = 0.0f32;

    let mut m_ccos_q8_mask = 0.0f32;
    let mut m_ccos_q4_mask = 0.0f32;

    for i in 0..aa_len {
        let pos = i + 1;
        let true_tok = original_tokens[pos];
        let is_masked = masked_positions.contains(&pos);

        let fp_log = &state_fp.logits[pos * config.vocab_size..(pos + 1) * config.vocab_size];
        let q8_log = &state_q8.logits[pos * config.vocab_size..(pos + 1) * config.vocab_size];
        let q4_log = &state_q4.logits[pos * config.vocab_size..(pos + 1) * config.vocab_size];

        let fp_top1 = argmax_canonical(fp_log);
        let q8_top1 = argmax_canonical(q8_log);
        let q4_top1 = argmax_canonical(q4_log);

        let kl_8 = kl_divergence_canonical(fp_log, q8_log);
        let kl_4 = kl_divergence_canonical(fp_log, q4_log);

        let cc_8 = centered_cosine_similarity(fp_log, q8_log);
        let cc_4 = centered_cosine_similarity(fp_log, q4_log);

        if fp_top1 == true_tok { m_fp_vs_true_all += 1; }
        if q8_top1 == true_tok { m_q8_vs_true_all += 1; }
        if q4_top1 == true_tok { m_q4_vs_true_all += 1; }

        if q8_top1 == fp_top1 { m_q8_vs_fp_all += 1; }
        if q4_top1 == fp_top1 { m_q4_vs_fp_all += 1; }

        m_kl_q8_all += kl_8;
        m_kl_q4_all += kl_4;
        m_ccos_q8_all += cc_8;
        m_ccos_q4_all += cc_4;

        if is_masked {
            if fp_top1 == true_tok { m_fp_vs_true_mask += 1; }
            if q8_top1 == true_tok { m_q8_vs_true_mask += 1; }
            if q4_top1 == true_tok { m_q4_vs_true_mask += 1; }

            if q8_top1 == fp_top1 { m_q8_vs_fp_mask += 1; }
            if q4_top1 == fp_top1 { m_q4_vs_fp_mask += 1; }

            m_kl_q8_mask += kl_8;
            m_kl_q4_mask += kl_4;
            m_ccos_q8_mask += cc_8;
            m_ccos_q4_mask += cc_4;

            let orig_c = raw_seq.chars().nth(i).unwrap();
            println!("  [Mask pos {:>2}] Real={:?} | FP32 pred tok {} | Q8_0 pred tok {} | Q4_0 pred tok {} | KL(FP||Q4)={:.4}",
                pos, orig_c, fp_top1, q8_top1, q4_top1, kl_4);
        }
    }

    println!("\n* Resultados Globales con 15% de Máscara (39 residuos):");
    println!("  - Precisión Top-1 vs Residuo Real:");
    println!("    * FP32: {:>2}/{} ({:.2}%)", m_fp_vs_true_all, aa_len, m_fp_vs_true_all as f32 / aa_len as f32 * 100.0);
    println!("    * Q8_0: {:>2}/{} ({:.2}%)", m_q8_vs_true_all, aa_len, m_q8_vs_true_all as f32 / aa_len as f32 * 100.0);
    println!("    * Q4_0: {:>2}/{} ({:.2}%)", m_q4_vs_true_all, aa_len, m_q4_vs_true_all as f32 / aa_len as f32 * 100.0);

    println!("  - Paridad frente a FP32 corregido (Toda la secuencia):");
    println!("    * Q8_0: Top-1 vs FP32 = {:>2}/{} ({:.2}%) | KL = {:.6} | Coseno Centrado = {:.6}",
        m_q8_vs_fp_all, aa_len, m_q8_vs_fp_all as f32 / aa_len as f32 * 100.0,
        m_kl_q8_all / aa_len as f32, m_ccos_q8_all / aa_len as f32);
    println!("    * Q4_0: Top-1 vs FP32 = {:>2}/{} ({:.2}%) | KL = {:.6} | Coseno Centrado = {:.6}",
        m_q4_vs_fp_all, aa_len, m_q4_vs_fp_all as f32 / aa_len as f32 * 100.0,
        m_kl_q4_all / aa_len as f32, m_ccos_q4_all / aa_len as f32);

    println!("\n* Resultados EXCLUSIVOS en las {} Posiciones Enmascaradas (Mask Tokens):", masked_positions.len());
    let n_m = masked_positions.len() as f32;
    println!("  - Precisión de Reconstrucción vs Residuo Real Oculto:");
    println!("    * FP32: {:>2}/{} ({:.2}%)", m_fp_vs_true_mask, masked_positions.len(), m_fp_vs_true_mask as f32 / n_m * 100.0);
    println!("    * Q8_0: {:>2}/{} ({:.2}%)", m_q8_vs_true_mask, masked_positions.len(), m_q8_vs_true_mask as f32 / n_m * 100.0);
    println!("    * Q4_0: {:>2}/{} ({:.2}%)", m_q4_vs_true_mask, masked_positions.len(), m_q4_vs_true_mask as f32 / n_m * 100.0);

    println!("  - Paridad frente a FP32 corregido en Posiciones Enmascaradas:");
    println!("    * Q8_0: Top-1 vs FP32 = {:>2}/{} ({:.2}%) | KL = {:.6} | Coseno Centrado = {:.6}",
        m_q8_vs_fp_mask, masked_positions.len(), m_q8_vs_fp_mask as f32 / n_m * 100.0,
        m_kl_q8_mask / n_m, m_ccos_q8_mask / n_m);
    println!("    * Q4_0: Top-1 vs FP32 = {:>2}/{} ({:.2}%) | KL = {:.6} | Coseno Centrado = {:.6}",
        m_q4_vs_fp_mask, masked_positions.len(), m_q4_vs_fp_mask as f32 / n_m * 100.0,
        m_kl_q4_mask / n_m, m_ccos_q4_mask / n_m);

    println!("\n=========================================================================================");
    println!(" [RESUMEN FINAL] VALIDACIÓN COMPLETADA");
    println!("=========================================================================================\n");
}
