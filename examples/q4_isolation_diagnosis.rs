use codon_infer::{
    Tokenizer, EsmConfig, EsmWeights, EsmWeightsQ4,
    BidirectionalRunState, forward_bidirectional_parallel,
    forward_bidirectional_q4, dequantize_q4_to_vec,
};

fn softmax(out: &mut [f32], logits: &[f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max).exp();
        out[i] = e;
        sum += e;
    }
    for v in out.iter_mut() {
        *v /= sum;
    }
}

pub fn compute_top1(logits_a: &[f32], logits_b: &[f32], vocab_size: usize) -> (usize, usize, f64) {
    let n_tokens = logits_a.len() / vocab_size;
    let mut matches = 0;
    for t in 0..n_tokens {
        let slice_a = &logits_a[t * vocab_size..(t + 1) * vocab_size];
        let slice_b = &logits_b[t * vocab_size..(t + 1) * vocab_size];
        let arg_a = slice_a.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        let arg_b = slice_b.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        if arg_a == arg_b {
            matches += 1;
        }
    }
    (matches, n_tokens, (matches as f64 / n_tokens as f64) * 100.0)
}

pub fn compute_kl(logits_ref: &[f32], logits_target: &[f32], vocab_size: usize) -> f32 {
    let n_tokens = logits_ref.len() / vocab_size;
    let mut p = vec![0.0f32; vocab_size];
    let mut q = vec![0.0f32; vocab_size];
    let mut total_kl = 0.0f32;

    for t in 0..n_tokens {
        softmax(&mut p, &logits_ref[t * vocab_size..(t + 1) * vocab_size]);
        softmax(&mut q, &logits_target[t * vocab_size..(t + 1) * vocab_size]);
        let mut kl_t = 0.0f32;
        for i in 0..vocab_size {
            if p[i] > 1e-12 {
                let q_val = q[i].max(1e-12);
                kl_t += p[i] * (p[i] / q_val).ln();
            }
        }
        total_kl += kl_t;
    }
    total_kl / n_tokens as f32
}

pub fn compute_centered_cosine(a: &[f32], b: &[f32], vocab_size: usize) -> f32 {
    let n_tokens = a.len() / vocab_size;
    let mut sum_cos = 0.0f32;
    for t in 0..n_tokens {
        let sa = &a[t * vocab_size..(t + 1) * vocab_size];
        let sb = &b[t * vocab_size..(t + 1) * vocab_size];
        let mean_a = sa.iter().sum::<f32>() / vocab_size as f32;
        let mean_b = sb.iter().sum::<f32>() / vocab_size as f32;

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..vocab_size {
            let ca = sa[i] - mean_a;
            let cb = sb[i] - mean_b;
            dot += ca * cb;
            norm_a += ca * ca;
            norm_b += cb * cb;
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom > 1e-12 {
            sum_cos += dot / denom;
        } else {
            sum_cos += 1.0;
        }
    }
    sum_cos / n_tokens as f32
}

fn main() {
    let seq = "AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS";
    let tokens = Tokenizer::encode(seq);
    let seq_len = tokens.len();
    let num_threads = 4;

    println!("================================================================================");
    println!(" CODON-INFER v1.6.0: SUITE DE AISLAMIENTO Y DIAGNÓSTICO PROFUNDO DE BUG Q4");
    println!(" Secuencia: {} ({} tokens) | Threads: {}", seq, seq_len, num_threads);
    println!("================================================================================\n");

    let scales = [
        ("8M", "esm2_8m_weights.bin", "esm2_8m_q4_0.bin", EsmConfig::esm2_8m()),
        ("35M", "esm2_35m_weights.bin", "esm2_35m_q4_0.bin", EsmConfig::esm2_35m()),
        ("150M", "esm2_150m_weights.bin", "esm2_150m_q4_0.bin", EsmConfig::esm2_150m()),
        ("650M", "esm2_650m_weights.bin", "esm2_650m_q4_0.bin", EsmConfig::esm2_650m()),
    ];

    for (scale_name, fp_path, q4_path, config) in &scales {
        println!("################################################################################");
        println!(" ESCALA: {} ({})", scale_name, config.model_name());
        println!("################################################################################");

        if !std::path::Path::new(fp_path).exists() || !std::path::Path::new(q4_path).exists() {
            println!(" [SKIP] Archivo no encontrado\n");
            continue;
        }

        let weights_fp = EsmWeights::from_file(fp_path, config).expect("FP32 load");
        let weights_q4 = EsmWeightsQ4::load_from_file(q4_path).expect("Q4 load").1;

        // Baseline FP32
        let mut state_fp = BidirectionalRunState::new(config, seq_len + 8);
        forward_bidirectional_parallel(&tokens, config, &weights_fp, &mut state_fp, num_threads);
        let fp_logits = state_fp.logits[0..seq_len * config.vocab_size].to_vec();

        // Target Q4
        let mut state_q4 = BidirectionalRunState::new(config, seq_len + 8);
        forward_bidirectional_q4(&tokens, config, &weights_q4, &mut state_q4, num_threads);
        let q4_logits = state_q4.logits[0..seq_len * config.vocab_size].to_vec();

        let (t1_m, t1_tot, t1_pct) = compute_top1(&fp_logits, &q4_logits, config.vocab_size);
        let kl_q4 = compute_kl(&fp_logits, &q4_logits, config.vocab_size);
        let cent_cos_q4 = compute_centered_cosine(&fp_logits, &q4_logits, config.vocab_size);

        println!(" [BASELINE ACTUAL Q4_0]");
        println!("   * Top-1 Match: {}/{} ({:.1}%)", t1_m, t1_tot, t1_pct);
        println!("   * KL(FP32 || Q4): {:.6}", kl_q4);
        println!("   * Coseno centrado: {:.6}\n", cent_cos_q4);

        // PRUEBA B: Forward Q4 decuantizado a FP32 (usando matmul FP32 estándar)
        // Decuantizamos todas las matrices de Q4 a f32
        let dequant_wq = dequantize_q4_to_vec(&weights_q4.wq);
        let dequant_wk = dequantize_q4_to_vec(&weights_q4.wk);
        let dequant_wv = dequantize_q4_to_vec(&weights_q4.wv);
        let dequant_wo = dequantize_q4_to_vec(&weights_q4.wo);
        let dequant_fc1 = dequantize_q4_to_vec(&weights_q4.fc1_weight);
        let dequant_fc2 = dequantize_q4_to_vec(&weights_q4.fc2_weight);

        let weights_q4_as_fp32 = EsmWeights {
            token_embedding_table: weights_q4.token_embedding_table.clone(),
            ln_1_weight: weights_q4.ln_1_weight.clone(),
            ln_1_bias: weights_q4.ln_1_bias.clone(),
            wq: dequant_wq,
            bq: weights_q4.bq.clone(),
            wk: dequant_wk,
            bk: weights_q4.bk.clone(),
            wv: dequant_wv,
            bv: weights_q4.bv.clone(),
            wo: dequant_wo,
            bo: weights_q4.bo.clone(),
            ln_2_weight: weights_q4.ln_2_weight.clone(),
            ln_2_bias: weights_q4.ln_2_bias.clone(),
            fc1_weight: dequant_fc1,
            fc1_bias: weights_q4.fc1_bias.clone(),
            fc2_weight: dequant_fc2,
            fc2_bias: weights_q4.fc2_bias.clone(),
            ln_f_weight: weights_q4.ln_f_weight.clone(),
            ln_f_bias: weights_q4.ln_f_bias.clone(),
            classifier_weights: weights_q4.classifier_weights.clone(),
        };

        let mut state_dequant = BidirectionalRunState::new(config, seq_len + 8);
        forward_bidirectional_parallel(&tokens, config, &weights_q4_as_fp32, &mut state_dequant, num_threads);
        let dequant_logits = state_dequant.logits[0..seq_len * config.vocab_size].to_vec();

        let (t1_d_m, t1_d_tot, t1_d_pct) = compute_top1(&fp_logits, &dequant_logits, config.vocab_size);
        let kl_dequant = compute_kl(&fp_logits, &dequant_logits, config.vocab_size);
        let cent_cos_dequant = compute_centered_cosine(&fp_logits, &dequant_logits, config.vocab_size);

        println!(" [PRUEBA B: Q4_0 DECUANTIZADO A FP32 CON MATMUL FP32]");
        println!("   * Top-1 Match: {}/{} ({:.1}%)", t1_d_m, t1_d_tot, t1_d_pct);
        println!("   * KL(FP32 || Q4_dequant): {:.6}", kl_dequant);
        println!("   * Coseno centrado: {:.6}", cent_cos_dequant);

        // Compare direct kernel output vs dequantized matmul output
        let (_simd_vs_deq_m, _, simd_vs_deq_pct) = compute_top1(&dequant_logits, &q4_logits, config.vocab_size);
        let kl_kernel = compute_kl(&dequant_logits, &q4_logits, config.vocab_size);
        let cos_kernel = compute_centered_cosine(&dequant_logits, &q4_logits, config.vocab_size);
        println!("   * Paridad Kernel SIMD vs Matmul Decuantizado: Top-1={:.1}%, KL={:.6}, Coseno={:.6}\n", simd_vs_deq_pct, kl_kernel, cos_kernel);

        // If scale is 35M, run Prueba A (Aislamiento por matriz) and Prueba C (SIMD vs Escalar vs Hilos)
        if *scale_name == "35M" {
            println!(" [PRUEBA A: AISLAMIENTO MATRIZ POR MATRIZ EN 35M]");
            println!(" (Todas las matrices en FP32 excepto UNA sola decuantizada de Q4)");

            let matrices = [
                ("wq", &weights_q4_as_fp32.wq),
                ("wk", &weights_q4_as_fp32.wk),
                ("wv", &weights_q4_as_fp32.wv),
                ("wo", &weights_q4_as_fp32.wo),
                ("fc1", &weights_q4_as_fp32.fc1_weight),
                ("fc2", &weights_q4_as_fp32.fc2_weight),
            ];

            for (mat_name, q4_mat) in matrices {
                let mut test_w = EsmWeights::from_file(fp_path, config).expect("FP32 load");
                match mat_name {
                    "wq" => test_w.wq = q4_mat.clone(),
                    "wk" => test_w.wk = q4_mat.clone(),
                    "wv" => test_w.wv = q4_mat.clone(),
                    "wo" => test_w.wo = q4_mat.clone(),
                    "fc1" => test_w.fc1_weight = q4_mat.clone(),
                    "fc2" => test_w.fc2_weight = q4_mat.clone(),
                    _ => {}
                }
                let mut st_iso = BidirectionalRunState::new(config, seq_len + 8);
                forward_bidirectional_parallel(&tokens, config, &test_w, &mut st_iso, num_threads);
                let (iso_m, _, iso_pct) = compute_top1(&fp_logits, &st_iso.logits[0..seq_len * config.vocab_size], config.vocab_size);
                let iso_kl = compute_kl(&fp_logits, &st_iso.logits[0..seq_len * config.vocab_size], config.vocab_size);
                let iso_cos = compute_centered_cosine(&fp_logits, &st_iso.logits[0..seq_len * config.vocab_size], config.vocab_size);
                println!("   - Matriz {:<4} en Q4: Top-1 = {:>2}/{} ({:>5.1}%) | KL = {:.6} | Cos_cent = {:.6}", mat_name, iso_m, seq_len, iso_pct, iso_kl, iso_cos);
            }
            println!();

            println!(" [PRUEBA C: FORWARD 35M COMPLETO - KERNELS Y HILOS]");
            // 1 hilo vs 4 hilos
            let mut st_1t = BidirectionalRunState::new(config, seq_len + 8);
            forward_bidirectional_q4(&tokens, config, &weights_q4, &mut st_1t, 1);
            let (c_1t_m, _, c_1t_pct) = compute_top1(&fp_logits, &st_1t.logits[0..seq_len * config.vocab_size], config.vocab_size);
            println!("   * Q4 1 Thread (Secuencial): Top-1 = {:>2}/{} ({:.1}%)", c_1t_m, seq_len, c_1t_pct);

            let (p_m, _, p_pct) = compute_top1(&st_1t.logits[0..seq_len * config.vocab_size], &q4_logits, config.vocab_size);
            println!("   * Paridad 1 Thread vs 4 Threads: Top-1 = {:>2}/{} ({:.1}%)", p_m, seq_len, p_pct);
        }
        println!();
    }
}
