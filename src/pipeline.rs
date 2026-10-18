use std::time::Instant;
use crate::{
    forward_bidirectional, forward_bidirectional_parallel, forward_bidirectional_q8,
    forward_bidirectional_q4, forward_bidirectional_q3, log_softmax_canonical, BidirectionalRunState, EsmConfig,
    EsmWeights, ModelWeightsRef, Tokenizer,
};
use crate::benchmark::{ContactPair, CANONICAL_AAS};
use crate::export::DmsPositionRow;
use crate::fasta::FastaRecord;
use crate::ui::{aa_category, format_aa, format_dms_cell, format_sequence, ProgressBar, TerminalTheme};

/// Scoring mode for in silico Deep Mutational Scanning (DMS).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmsScoringMode {
    /// Canonical Masked Marginal scoring (O(L) forward passes, masking 1 residue at a time).
    MaskedMarginal,
    /// Fast 1-Pass Wildtype Marginal scoring (1 forward pass per window on unmasked sequence).
    WildtypeMarginal,
}

/// Result of a Deep Mutational Scanning (DMS) pipeline run.
pub struct DmsPipelineResult {
    pub protein_id: String,
    pub sequence: String,
    pub rows: Vec<DmsPositionRow>,
    pub elapsed_ms: f64,
    pub muts_per_sec: f64,
    pub catalytic_sites_detected: usize,
    pub scoring_mode: DmsScoringMode,
}

/// Executes Deep Mutational Scanning using an adaptive sliding window for arbitrarily long proteins.
pub fn run_pipeline_dms(
    record: &FastaRecord,
    window_size: usize,
    stride: usize,
    config: &EsmConfig,
    weights: &EsmWeights,
    theme: &TerminalTheme,
) -> DmsPipelineResult {
    run_pipeline_dms_parallel(
        record,
        window_size,
        stride,
        config,
        ModelWeightsRef::Fp32(weights),
        1,
        theme,
    )
}

/// Executes Deep Mutational Scanning with multithreading and optional Q8_0/Q4_0 weights (canonical Masked Marginal mode).
pub fn run_pipeline_dms_parallel(
    record: &FastaRecord,
    window_size: usize,
    stride: usize,
    config: &EsmConfig,
    weights: ModelWeightsRef,
    num_threads: usize,
    theme: &TerminalTheme,
) -> DmsPipelineResult {
    run_pipeline_dms_with_mode(
        record,
        window_size,
        stride,
        config,
        weights,
        num_threads,
        theme,
        DmsScoringMode::MaskedMarginal,
    )
}

/// Executes Deep Mutational Scanning with explicit scoring mode (`MaskedMarginal` or `WildtypeMarginal`).
#[allow(clippy::too_many_arguments)]
pub fn run_pipeline_dms_with_mode(
    record: &FastaRecord,
    window_size: usize,
    stride: usize,
    config: &EsmConfig,
    weights: ModelWeightsRef,
    num_threads: usize,
    theme: &TerminalTheme,
    mode: DmsScoringMode,
) -> DmsPipelineResult {
    let seq = &record.sequence;
    let n = seq.len();
    assert!(n > 0, "Sequence cannot be empty");

    let actual_window = if n <= window_size || window_size == 0 {
        n.min(config.max_seq_len)
    } else {
        window_size.min(config.max_seq_len)
    };

    let mut state = BidirectionalRunState::new(config, actual_window + 2);
    state.compute_contacts = false;
    let mut rows = Vec::with_capacity(n);

    let label = match mode {
        DmsScoringMode::MaskedMarginal => "Scanning in silico DMS (Masked Marginal)",
        DmsScoringMode::WildtypeMarginal => "Scanning in silico DMS (1-Pass WT Marginal)",
    };
    let mut progress = ProgressBar::new(n, label, *theme);
    let t0 = Instant::now();

    let seq_chars: Vec<char> = seq.chars().collect();

    match mode {
        DmsScoringMode::MaskedMarginal => {
            for p in 0..n {
                // Optimal window centered at position p to avoid RoPE boundary degradation
                let half_w = actual_window / 2;
                let w_start = if p < half_w {
                    0
                } else if p + half_w > n {
                    n.saturating_sub(actual_window)
                } else {
                    p - half_w
                };
                let w_end = (w_start + actual_window).min(n);
                let subseq: String = seq_chars[w_start..w_end].iter().collect();

                let rel_pos = p - w_start; // 0-based within window
                let mut tokens = Tokenizer::encode(&subseq);
                let tok_pos = rel_pos + 1; // 1-based token index (after <cls>)
                let wt_tok = tokens[tok_pos];
                let wt_char = seq_chars[p];

                tokens[tok_pos] = 32; // <mask_1>
                match weights {
                    ModelWeightsRef::Fp32(w) => {
                        if num_threads <= 1 {
                            forward_bidirectional(&tokens, config, w, &mut state);
                        } else {
                            forward_bidirectional_parallel(&tokens, config, w, &mut state, num_threads);
                        }
                    }
                    ModelWeightsRef::Q8(w_q8) => {
                        forward_bidirectional_q8(&tokens, config, w_q8, &mut state, num_threads);
                    }
                    ModelWeightsRef::Q4(w_q4) => {
                        forward_bidirectional_q4(&tokens, config, w_q4, &mut state, num_threads);
                    }
                    ModelWeightsRef::Q3(w_q3) => {
                        forward_bidirectional_q3(&tokens, config, w_q3, &mut state, num_threads);
                    }
                }

                let logits = &state.logits[tok_pos * config.vocab_size..(tok_pos + 1) * config.vocab_size];
                let log_probs = log_softmax_canonical(logits);

                // Shannon entropy calculation: H = - sum(p * ln(p))
                let mut entropy = 0.0f32;
                for i in 4..=23 {
                    let p_aa = log_probs[i].exp();
                    if p_aa > 1e-12 {
                        entropy -= p_aa * log_probs[i];
                    }
                }

                // Pure canonical Masked Marginal scoring: Δlog P = log P(x_i = m | x_{-i}) - log P(x_i = wt | x_{-i})
                let mut deltas = [0.0f32; 20];
                for (col_idx, &(_aa_char, aa_tok)) in CANONICAL_AAS.iter().enumerate() {
                    deltas[col_idx] = if aa_tok == wt_tok {
                        0.0f32
                    } else {
                        log_probs[aa_tok] - log_probs[wt_tok]
                    };
                }

                rows.push(DmsPositionRow::new(p + 1, wt_char, deltas, entropy));
                progress.update(p + 1);
            }
        }
        DmsScoringMode::WildtypeMarginal => {
            if n <= actual_window {
                let tokens = Tokenizer::encode(seq);
                match weights {
                    ModelWeightsRef::Fp32(w) => {
                        if num_threads <= 1 {
                            forward_bidirectional(&tokens, config, w, &mut state);
                        } else {
                            forward_bidirectional_parallel(&tokens, config, w, &mut state, num_threads);
                        }
                    }
                    ModelWeightsRef::Q8(w_q8) => {
                        forward_bidirectional_q8(&tokens, config, w_q8, &mut state, num_threads);
                    }
                    ModelWeightsRef::Q4(w_q4) => {
                        forward_bidirectional_q4(&tokens, config, w_q4, &mut state, num_threads);
                    }
                    ModelWeightsRef::Q3(w_q3) => {
                        forward_bidirectional_q3(&tokens, config, w_q3, &mut state, num_threads);
                    }
                }

                for p in 0..n {
                    let tok_pos = p + 1;
                    let wt_tok = tokens[tok_pos];
                    let wt_char = seq_chars[p];
                    let logits = &state.logits[tok_pos * config.vocab_size..(tok_pos + 1) * config.vocab_size];
                    let log_probs = log_softmax_canonical(logits);

                    let mut entropy = 0.0f32;
                    for i in 4..=23 {
                        let p_aa = log_probs[i].exp();
                        if p_aa > 1e-12 {
                            entropy -= p_aa * log_probs[i];
                        }
                    }

                    let mut deltas = [0.0f32; 20];
                    for (col_idx, &(_aa_char, aa_tok)) in CANONICAL_AAS.iter().enumerate() {
                        deltas[col_idx] = if aa_tok == wt_tok {
                            0.0f32
                        } else {
                            log_probs[aa_tok] - log_probs[wt_tok]
                        };
                    }

                    rows.push(DmsPositionRow::new(p + 1, wt_char, deltas, entropy));
                    progress.update(p + 1);
                }
            } else {
                let step = if stride > 0 && stride <= actual_window {
                    stride
                } else {
                    (actual_window / 2).max(1)
                };
                let mut covered = 0usize;
                let mut w_start = 0usize;
                while covered < n {
                    let w_end = (w_start + actual_window).min(n);
                    let subseq: String = seq_chars[w_start..w_end].iter().collect();
                    let tokens = Tokenizer::encode(&subseq);
                    match weights {
                        ModelWeightsRef::Fp32(w) => {
                            if num_threads <= 1 {
                                forward_bidirectional(&tokens, config, w, &mut state);
                            } else {
                                forward_bidirectional_parallel(&tokens, config, w, &mut state, num_threads);
                            }
                        }
                        ModelWeightsRef::Q8(w_q8) => {
                            forward_bidirectional_q8(&tokens, config, w_q8, &mut state, num_threads);
                        }
                        ModelWeightsRef::Q4(w_q4) => {
                            forward_bidirectional_q4(&tokens, config, w_q4, &mut state, num_threads);
                        }
                        ModelWeightsRef::Q3(w_q3) => {
                            forward_bidirectional_q3(&tokens, config, w_q3, &mut state, num_threads);
                        }
                    }

                    for p in covered..w_end {
                        let rel_pos = p - w_start;
                        let tok_pos = rel_pos + 1;
                        let wt_tok = tokens[tok_pos];
                        let wt_char = seq_chars[p];
                        let logits = &state.logits[tok_pos * config.vocab_size..(tok_pos + 1) * config.vocab_size];
                        let log_probs = log_softmax_canonical(logits);

                        let mut entropy = 0.0f32;
                        for i in 4..=23 {
                            let p_aa = log_probs[i].exp();
                            if p_aa > 1e-12 {
                                entropy -= p_aa * log_probs[i];
                            }
                        }

                        let mut deltas = [0.0f32; 20];
                        for (col_idx, &(_aa_char, aa_tok)) in CANONICAL_AAS.iter().enumerate() {
                            deltas[col_idx] = if aa_tok == wt_tok {
                                0.0f32
                            } else {
                                log_probs[aa_tok] - log_probs[wt_tok]
                            };
                        }

                        rows.push(DmsPositionRow::new(p + 1, wt_char, deltas, entropy));
                        progress.update(p + 1);
                    }
                    covered = w_end;
                    if w_end >= n {
                        break;
                    }
                    w_start = (w_start + step).min(n.saturating_sub(actual_window));
                }
            }
        }
    }
    progress.finish();

    let elapsed = t0.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let total_mutations = n * 20;
    let muts_per_sec = if elapsed.as_secs_f64() > 0.0 {
        total_mutations as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    let catalytic_sites_detected = rows.iter().filter(|r| r.mean_delta_logp <= -3.8).count();

    DmsPipelineResult {
        protein_id: record.id.clone(),
        sequence: seq.clone(),
        rows,
        elapsed_ms,
        muts_per_sec,
        catalytic_sites_detected,
        scoring_mode: mode,
    }
}

/// Renders a full TrueColor interactive heatmap table to stdout.
pub fn render_dms_heatmap(result: &DmsPipelineResult, max_display_rows: usize, theme: &TerminalTheme) {
    let mode_str = match result.scoring_mode {
        DmsScoringMode::MaskedMarginal => "Canonical Masked Marginal (L-Pass Exact)",
        DmsScoringMode::WildtypeMarginal => "Fast Wildtype Marginal (1-Pass Unmasked)",
    };
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║              CODON-INFER: IN SILICO DEEP MUTATIONAL SCANNING (DMS) HEATMAP                ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  * Target ID:           {}", result.protein_id);
    println!("  * Scoring Method:      {}", mode_str);
    println!("  * Sequence Length:     {} amino acids", result.sequence.len());
    println!("  * Total Mutations:     {} single-point variants evaluated", result.rows.len() * 20);
    println!("  * Scanning Latency:    {:.2} ms total ({:.1} muts/sec)", result.elapsed_ms, result.muts_per_sec);
    println!("  * Critical Sites:      {} residues identified with high mutational intolerance", result.catalytic_sites_detected);
    println!();

    println!("┌─────────┬────────────────────────────────────────────────────────────┬───────────────────┐");
    println!("│ Res Pos │ A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y │ Mean ΔLogP Status │");
    println!("├─────────┼────────────────────────────────────────────────────────────┼───────────────────┤");

    let display_rows = if result.rows.len() <= max_display_rows {
        &result.rows[..]
    } else {
        &result.rows[..max_display_rows]
    };

    for r in display_rows {
        let is_catalytic = r.mean_delta_logp <= -3.8;
        let is_core = r.mean_delta_logp <= -2.5 && !is_catalytic;
        let tag = if is_catalytic {
            "[CATALYTIC/CRIT]"
        } else if is_core {
            "[CORE CONSERVED]"
        } else {
            "                "
        };

        let wt_colored = format_aa(r.wt, theme);
        print!("│  {} {:>2}  │", wt_colored, r.pos);
        for col in 0..20 {
            let is_wt = CANONICAL_AAS[col].0 == r.wt;
            let d = r.deltas[col];
            print!("{}", format_dms_cell(d, is_wt, theme));
        }
        println!("│   {:>+6.2}   {:<13} │", r.mean_delta_logp, tag);
    }

    if result.rows.len() > max_display_rows {
        println!("│   ...   │ ... [Truncated {} additional positions for terminal display] ... │                   │",
            result.rows.len() - max_display_rows);
    }
    println!("├─────────┴────────────────────────────────────────────────────────────┴───────────────────┤");
    println!("│ Heatmap Density Legend:                                                                  │");
    println!("│   [█] Neutral/Beneficial (ΔLogP ≥ 0.0)      [▓] Mildly Deleterious (-1.0 ≤ ΔLogP < 0.0)  │");
    println!("│   [▒] Deleterious (-2.5 ≤ ΔLogP < -1.0)     [░] Strongly Deleterious (-4.0 ≤ ΔLogP < -2.5)│");
    println!("│   [·] Lethal / Strictly Conserved (ΔLogP < -4.0)                                         │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────┘\n");
}

/// Result of structural contact prediction and disulfide analysis.
#[derive(Debug, Clone)]
pub struct ContactsPipelineResult {
    pub protein_id: String,
    pub sequence_length: usize,
    pub cys_indices: Vec<usize>,
    pub cys_enrichment_ratio: f32,
    pub top_contacts: Vec<ContactPair>,
    pub contact_probs: Vec<f32>,
}

/// Computes zero-shot structural contact prediction & disulfide bridge analysis without stdout printing.
pub fn compute_pipeline_contacts(
    record: &FastaRecord,
    min_separation: usize,
    top_l_factor: f32,
    max_k: Option<usize>,
    config: &EsmConfig,
    weights: &EsmWeights,
) -> ContactsPipelineResult {
    compute_pipeline_contacts_parallel(
        record,
        min_separation,
        top_l_factor,
        max_k,
        config,
        ModelWeightsRef::Fp32(weights),
        1,
    )
}

/// Computes zero-shot structural contact prediction with multithreading and optional Q8_0/Q4_0 weights.
/// Uses the official ESM-2 multi-layer ContactPredictionHead (L * H symmetrized APC + logistic regression).
pub fn compute_pipeline_contacts_parallel(
    record: &FastaRecord,
    min_separation: usize,
    top_l_factor: f32,
    max_k: Option<usize>,
    config: &EsmConfig,
    weights: ModelWeightsRef,
    num_threads: usize,
) -> ContactsPipelineResult {
    let seq = &record.sequence;
    let n = seq.len();
    assert!(n >= 4, "Sequence too short for contact prediction");

    let tokens = Tokenizer::encode(seq);
    let seq_len = tokens.len();
    let mut state = BidirectionalRunState::new(config, seq_len);
    state.compute_contacts = true;
    match weights {
        ModelWeightsRef::Fp32(w) => {
            if num_threads <= 1 {
                forward_bidirectional(&tokens, config, w, &mut state);
            } else {
                forward_bidirectional_parallel(&tokens, config, w, &mut state, num_threads);
            }
        }
        ModelWeightsRef::Q8(w_q8) => {
            forward_bidirectional_q8(&tokens, config, w_q8, &mut state, num_threads);
        }
        ModelWeightsRef::Q4(w_q4) => {
            forward_bidirectional_q4(&tokens, config, w_q4, &mut state, num_threads);
        }
        ModelWeightsRef::Q3(w_q3) => {
            forward_bidirectional_q3(&tokens, config, w_q3, &mut state, num_threads);
        }
    }

    let seq_chars: Vec<char> = seq.chars().collect();
    let mut cys_indices = Vec::new();
    for (i, &c) in seq_chars.iter().enumerate() {
        if c == 'C' {
            cys_indices.push(i);
        }
    }

    let mut all_pairs = Vec::new();
    let mut cys_pairs_count = 0;
    let mut total_candidates_count = 0;

    for i in 0..n {
        for j in (i + min_separation)..n {
            let score = state.contact_probs[i * n + j];
            let is_cc = seq_chars[i] == 'C' && seq_chars[j] == 'C';
            total_candidates_count += 1;
            if is_cc {
                cys_pairs_count += 1;
            }

            all_pairs.push(ContactPair {
                pos1: i,
                res1: seq_chars[i],
                pos2: j,
                res2: seq_chars[j],
                apc_score: score,
                is_cys_cys: is_cc,
                is_known_disulfide: is_cc,
            });
        }
    }

    all_pairs.sort_by(|a, b| b.apc_score.partial_cmp(&a.apc_score).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = if let Some(k) = max_k {
        k.min(all_pairs.len())
    } else {
        ((n as f32 * top_l_factor).round() as usize).clamp(5, 50).min(all_pairs.len())
    };
    let top_contacts: Vec<ContactPair> = all_pairs.into_iter().take(top_k).collect();

    let top_cys_count = top_contacts.iter().filter(|c| c.is_cys_cys).count();
    let top_cys_density = if !top_contacts.is_empty() {
        top_cys_count as f32 / top_contacts.len() as f32
    } else {
        0.0
    };
    let bg_cys_density = if total_candidates_count > 0 {
        cys_pairs_count as f32 / total_candidates_count as f32
    } else {
        0.05
    };

    let enrichment = if bg_cys_density > 0.0 {
        top_cys_density / bg_cys_density
    } else {
        1.0
    };

    ContactsPipelineResult {
        protein_id: record.id.clone(),
        sequence_length: n,
        cys_indices,
        cys_enrichment_ratio: enrichment,
        top_contacts,
        contact_probs: state.contact_probs,
    }
}

/// Renders the contacts table to stdout with terminal theme formatting.
pub fn render_contacts_table(result: &ContactsPipelineResult, theme: &TerminalTheme) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║       CODON-INFER: ZERO-SHOT STRUCTURAL CONTACT & DISULFIDE DETECTION (APC)               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  * Target ID:                 {}", result.protein_id);
    println!("  * Sequence Length:           {} amino acids", result.sequence_length);
    let cys_str: Vec<String> = result.cys_indices.iter().map(|&idx| format!("C{}", idx + 1)).collect();
    println!("  * Cysteines Identified:      {}", if cys_str.is_empty() { "None".to_string() } else { cys_str.join(", ") });
    println!("  * Contact Density Ratio:     {:.2}x disulfide enrichment vs background", result.cys_enrichment_ratio);
    println!();

    println!("┌──────┬───────────┬───────────┬────────────┬───────────┬──────────────────────────────────┐");
    println!("│ Rank │ Residue 1 │ Residue 2 │ Separation │ APC Score │ Contact Classification           │");
    println!("├──────┼───────────┼───────────┼────────────┼───────────┼──────────────────────────────────┤");

    for (rank, cp) in result.top_contacts.iter().enumerate() {
        let sep = (cp.pos2 as isize - cp.pos1 as isize).abs();
        let ctype = if cp.is_cys_cys {
            "\x1b[38;2;245;158;11m★ Disulfide Bridge Candidate\x1b[0m"
        } else {
            "  Tertiary Spatial Contact"
        };
        let r1_colored = format_aa(cp.res1, theme);
        let r2_colored = format_aa(cp.res2, theme);

        println!("│  {:>2}  │   {:>2} ({})   │   {:>2} ({})   │    {:>3}     │  {:>+7.4}  │ {:<32} │",
            rank + 1, cp.pos1 + 1, r1_colored, cp.pos2 + 1, r2_colored, sep, cp.apc_score, ctype);
    }
    println!("└──────┴───────────┴───────────┴────────────┴───────────┴──────────────────────────────────┘\n");
}

/// Executes zero-shot structural contact prediction & disulfide bridge analysis.
pub fn run_pipeline_contacts(
    record: &FastaRecord,
    min_separation: usize,
    top_l_factor: f32,
    config: &EsmConfig,
    weights: &EsmWeights,
    theme: &TerminalTheme,
) -> ContactsPipelineResult {
    run_pipeline_contacts_parallel(
        record,
        min_separation,
        top_l_factor,
        config,
        ModelWeightsRef::Fp32(weights),
        1,
        theme,
    )
}

/// Executes zero-shot structural contact prediction & disulfide bridge analysis with multithreading and optional Q8_0 weights.
pub fn run_pipeline_contacts_parallel(
    record: &FastaRecord,
    min_separation: usize,
    top_l_factor: f32,
    config: &EsmConfig,
    weights: ModelWeightsRef,
    num_threads: usize,
    theme: &TerminalTheme,
) -> ContactsPipelineResult {
    let result = compute_pipeline_contacts_parallel(record, min_separation, top_l_factor, None, config, weights, num_threads);
    render_contacts_table(&result, theme);
    result
}

/// Candidate amino acid prediction at a masked position.
#[derive(Debug, Clone, PartialEq)]
pub struct InpaintCandidate {
    pub amino_acid: char,
    pub probability_pct: f32,
    pub category: &'static str,
}

/// Single reconstructed masked site.
#[derive(Debug, Clone, PartialEq)]
pub struct InpaintSite {
    pub position: usize,       // 1-based position in sequence
    pub token_idx: usize,      // Token index in model sequence (<cls> is 0)
    pub native_char: char,     // Usually '*'
    pub predicted_char: char,  // Highest probability amino acid
    pub candidates: Vec<InpaintCandidate>, // Top-5 or canonical candidates
}

/// Result of an inpainting pipeline run.
#[derive(Debug, Clone, PartialEq)]
pub struct InpaintPipelineResult {
    pub protein_id: String,
    pub original_sequence: String,
    pub reconstructed_sequence: String,
    pub sites: Vec<InpaintSite>,
}

/// Computes inpainting on a masked sequence without stdout printing.
pub fn compute_pipeline_inpaint(
    record: &FastaRecord,
    config: &EsmConfig,
    weights: &EsmWeights,
) -> InpaintPipelineResult {
    compute_pipeline_inpaint_parallel(record, config, ModelWeightsRef::Fp32(weights), 1)
}

/// Computes inpainting on a masked sequence with multithreading and optional Q8_0 weights.
pub fn compute_pipeline_inpaint_parallel(
    record: &FastaRecord,
    config: &EsmConfig,
    weights: ModelWeightsRef,
    num_threads: usize,
) -> InpaintPipelineResult {
    let seq = &record.sequence;
    let tokens = Tokenizer::encode(seq);
    let mut state = BidirectionalRunState::new(config, tokens.len());
    state.compute_contacts = false;
    match weights {
        ModelWeightsRef::Fp32(w) => {
            if num_threads <= 1 {
                forward_bidirectional(&tokens, config, w, &mut state);
            } else {
                forward_bidirectional_parallel(&tokens, config, w, &mut state, num_threads);
            }
        }
        ModelWeightsRef::Q8(w_q8) => {
            forward_bidirectional_q8(&tokens, config, w_q8, &mut state, num_threads);
        }
        ModelWeightsRef::Q4(w_q4) => {
            forward_bidirectional_q4(&tokens, config, w_q4, &mut state, num_threads);
        }
        ModelWeightsRef::Q3(w_q3) => {
            forward_bidirectional_q3(&tokens, config, w_q3, &mut state, num_threads);
        }
    }

    let mut mask_indices = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        if t == 32 {
            mask_indices.push(i);
        }
    }

    let mut reconstructed_tokens = tokens.clone();
    let mut sites = Vec::with_capacity(mask_indices.len());

    for &mask_idx in &mask_indices {
        let logits = &state.logits[mask_idx * config.vocab_size..(mask_idx + 1) * config.vocab_size];
        let log_probs = log_softmax_canonical(logits);

        let mut all_cands: Vec<(usize, f32)> = (4..=23)
            .map(|id| (id, log_probs[id]))
            .collect();
        all_cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_id = all_cands[0].0;
        reconstructed_tokens[mask_idx] = best_id;
        let pred_char = Tokenizer::decode(best_id).unwrap_or('?');

        let top5: Vec<InpaintCandidate> = all_cands
            .iter()
            .take(5)
            .map(|&(id, lp)| {
                let c = Tokenizer::decode(id).unwrap_or('?');
                InpaintCandidate {
                    amino_acid: c,
                    probability_pct: lp.exp() * 100.0,
                    category: aa_category(c),
                }
            })
            .collect();

        sites.push(InpaintSite {
            position: mask_idx, // 1-based index in sequence (since tokens[0] is <cls>)
            token_idx: mask_idx,
            native_char: '*',
            predicted_char: pred_char,
            candidates: top5,
        });
    }

    let mut reconstructed_seq = String::with_capacity(tokens.len());
    for &t in &reconstructed_tokens[1..reconstructed_tokens.len() - 1] {
        if let Some(c) = Tokenizer::decode(t) {
            reconstructed_seq.push(c);
        }
    }

    InpaintPipelineResult {
        protein_id: record.id.clone(),
        original_sequence: seq.clone(),
        reconstructed_sequence: reconstructed_seq,
        sites,
    }
}

/// Renders inpainting result table to stdout.
pub fn render_inpaint_result(result: &InpaintPipelineResult, theme: &TerminalTheme) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║              CODON-INFER: BIDIRECTIONAL INPAINTING & MOTIF RECONSTRUCTION                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  * Input Target ID:       {}", result.protein_id);
    println!("  * Input Sequence:        {}", format_sequence(&result.original_sequence, theme));
    println!("  * Masked Positions:      {} site(s) requiring reconstruction", result.sites.len());
    println!();

    if result.sites.is_empty() {
        println!("  [NOTE] No masked tokens ('*' or '<mask_1>') detected in input sequence.");
        println!("  To inpaint a protein, insert '*' at residues to reconstruct, e.g. 'MCMP*TTD'.\n");
        return;
    }

    println!("┌──────────┬────────┬────────┬─────────────────────────────────────────────────────────────┐");
    println!("│ Position │ Native │ Pred   │ Top-5 Candidate Probabilities & Physicochemical Class       │");
    println!("├──────────┼────────┼────────┼─────────────────────────────────────────────────────────────┤");

    for site in &result.sites {
        let mut top5_str = String::new();
        for cand in site.candidates.iter().take(5) {
            top5_str.push_str(&format!("{}({:.1}%) ", format_aa(cand.amino_acid, theme), cand.probability_pct));
        }

        let pred_colored = format_aa(site.predicted_char, theme);
        println!("│ Site {:>3} │   *    │   {}    │ {:<59} │",
            site.position, pred_colored, top5_str);
    }
    println!("└──────────┴────────┴────────┴─────────────────────────────────────────────────────────────┘");
    println!("  * Reconstructed Protein: {}\n", format_sequence(&result.reconstructed_sequence, theme));
}

/// Executes inpainting on a masked sequence and renders candidate distributions.
pub fn run_pipeline_inpaint(
    record: &FastaRecord,
    config: &EsmConfig,
    weights: &EsmWeights,
    theme: &TerminalTheme,
) -> InpaintPipelineResult {
    run_pipeline_inpaint_parallel(record, config, ModelWeightsRef::Fp32(weights), 1, theme)
}

/// Executes inpainting on a masked sequence with multithreading and optional Q8_0 weights.
pub fn run_pipeline_inpaint_parallel(
    record: &FastaRecord,
    config: &EsmConfig,
    weights: ModelWeightsRef,
    num_threads: usize,
    theme: &TerminalTheme,
) -> InpaintPipelineResult {
    let result = compute_pipeline_inpaint_parallel(record, config, weights, num_threads);
    render_inpaint_result(&result, theme);
    result
}

// ============================================================================
// ZERO-TRUNK 3D GEOMETRIC RECONSTRUCTION (STRUCT-2)
// Metric MDS + Biophysical Gradient Descent Folding Engine
// ============================================================================

/// Result of a 3D C_alpha folding reconstruction pipeline run (STRUCT-2).
#[derive(Debug, Clone)]
pub struct FoldPipelineResult {
    pub protein_id: String,
    pub sequence: String,
    pub coords: Vec<[f64; 3]>,
    pub num_restraints: usize,
    pub p_max: f64,
    pub forward_ms: f64,
    pub solve_ms: f64,
    pub radius_of_gyration: f64,
    pub allatom_residues: Option<Vec<crate::export::ReconstructedResidue>>,
    pub sidechain_metrics: Option<crate::export::SidechainPackingMetrics>,
}

struct RestraintPair {
    i: usize,
    j: usize,
    d_target: f64,
    weight: f64,
}

/// Stage A: Classical Metric MDS (Gram Matrix top-3 spectral projection) from estimated geodesic distances.
pub fn mds_spectral_init(target_dist: &[f64], n: usize) -> Vec<[f64; 3]> {
    if n <= 1 {
        return vec![[0.0, 0.0, 0.0]; n];
    }

    // 1. Floyd-Warshall geodesic shortest-path smoothing on upper-bound distance graph
    let mut d = target_dist.to_vec();
    for k in 0..n {
        for i in 0..n {
            let dik = d[i * n + k];
            for j in 0..n {
                let via = dik + d[k * n + j];
                if via < d[i * n + j] {
                    d[i * n + j] = via;
                }
            }
        }
    }

    // 2. Double-centered Gram matrix B = -0.5 * J * D^(2) * J
    let mut d2 = vec![0.0f64; n * n];
    let mut row_mean = vec![0.0f64; n];
    let mut grand_mean = 0.0f64;
    for i in 0..n {
        let mut r_sum = 0.0;
        for j in 0..n {
            let val = d[i * n + j] * d[i * n + j];
            d2[i * n + j] = val;
            r_sum += val;
        }
        row_mean[i] = r_sum / n as f64;
        grand_mean += row_mean[i];
    }
    grand_mean /= n as f64;

    let mut b_mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            b_mat[i * n + j] = -0.5 * (d2[i * n + j] - row_mean[i] - row_mean[j] + grand_mean);
        }
    }

    // 3. Power iteration with deflation for top 3 positive eigenvalues/eigenvectors
    let mut coords = vec![[0.0f64; 3]; n];
    for dim in 0..3 {
        let mut v: Vec<f64> = (0..n)
            .map(|i| (i as f64 * (dim as f64 + 1.37) + 0.5).sin())
            .collect();
        let mut lambda = 1.0f64;
        for _it in 0..60 {
            let mut next_v = vec![0.0f64; n];
            for i in 0..n {
                let mut sum = 0.0;
                let row = &b_mat[i * n..(i + 1) * n];
                for j in 0..n {
                    sum += row[j] * v[j];
                }
                next_v[i] = sum;
            }
            let norm = next_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                break;
            }
            lambda = norm;
            for i in 0..n {
                v[i] = next_v[i] / norm;
            }
        }

        // Enforce canonical sign convention: the largest component in magnitude must be positive
        let mut max_abs = 0.0;
        let mut max_val = 0.0;
        for &val in &v {
            if val.abs() > max_abs {
                max_abs = val.abs();
                max_val = val;
            }
        }
        if max_val < 0.0 {
            for x in &mut v {
                *x = -*x;
            }
        }

        let scale = lambda.max(0.1).sqrt();
        for i in 0..n {
            coords[i][dim] = v[i] * scale;
        }
        // Deflate B <- B - lambda * v * v^T
        for i in 0..n {
            for j in 0..n {
                b_mat[i * n + j] -= lambda * v[i] * v[j];
            }
        }
    }

    // Rescale MDS projection so mean consecutive C_alpha - C_alpha step is 3.80 Å
    if n > 1 {
        let mut mean_step = 0.0;
        for i in 0..(n - 1) {
            let dx = coords[i + 1][0] - coords[i][0];
            let dy = coords[i + 1][1] - coords[i][1];
            let dz = coords[i + 1][2] - coords[i][2];
            mean_step += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        mean_step /= (n - 1) as f64;
        if mean_step > 1e-6 {
            let factor = (3.80 / mean_step).clamp(0.5, 2.2);
            for pt in &mut coords {
                pt[0] *= factor;
                pt[1] *= factor;
                pt[2] *= factor;
            }
        }
    }
    coords
}

/// Generates an alternative initial conformation (helical/twisted ribbon) for multi-start optimization.
pub fn helical_init(n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut coords = Vec::with_capacity(n);
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next_rand = || -> f64 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / ((1u64 << 53) as f64)) - 0.5
    };
    let pitch = 1.5 + 0.4 * next_rand();
    let radius = 2.3 + 0.3 * next_rand();
    let turn = 1.7 + 0.3 * next_rand();
    for i in 0..n {
        let theta = i as f64 * turn;
        let x = radius * theta.cos() + 0.35 * next_rand();
        let y = radius * theta.sin() + 0.35 * next_rand();
        let z = pitch * (i as f64) * (1.0 - 0.35 * ((i as f64 * 0.18).sin()));
        coords.push([x, y, z]);
    }
    coords
}

/// Reconstructs 3D C_alpha coordinates from an N x N contact probability matrix `probs`.
/// Returns (coords, num_restraints, p_max).
pub fn reconstruct_3d_ca(probs: &[f32], n: usize) -> (Vec<[f64; 3]>, usize, f64) {
    if n == 0 {
        return (Vec::new(), 0, 0.0);
    }
    if n == 1 {
        return (vec![[0.0, 0.0, 0.0]], 0, 0.0);
    }
    if n == 2 {
        return (vec![[0.0, 0.0, 0.0], [3.80, 0.0, 0.0]], 0, 0.0);
    }

    // 1. Build target distance matrix and active tertiary restraints from contact map
    let mut cands: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in (i + 3)..n {
            let p = probs[i * n + j] as f64;
            cands.push((i, j, p));
        }
    }
    cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Keep top 1.5*N pairs with P >= 0.12 (or top 0.6*N for uncalibrated heuristic APC maps)
    let max_keep = ((n as f64 * 1.5).round() as usize).min(cands.len());
    let min_keep = ((n as f64 * 0.6).round() as usize).min(cands.len());

    let mut target_dist = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let sep = (j - i) as f64;
            let d_flory = if sep == 1.0 {
                3.80
            } else if sep == 2.0 {
                6.00
            } else {
                3.80 * sep.powf(0.56)
            };
            target_dist[i * n + j] = d_flory;
            target_dist[j * n + i] = d_flory;
        }
    }

    let mut restraints = Vec::new();
    let p_max = cands.first().map(|c| c.2).unwrap_or(0.5).max(0.10);

    for (rank, &(i, j, p)) in cands.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        let p_rel = (p / p_max).clamp(0.0, 1.0);
        if (rank < max_keep && p >= 0.12) || (rank < min_keep && p_rel >= 0.30) {
            let d_est = (7.8 - 2.2 * p_rel).clamp(5.4, 7.9);
            let sep_factor = (((j - i) as f64).min(24.0) / 10.0).sqrt();
            let w = (p_rel * p_rel) * sep_factor * 3.2;
            restraints.push(RestraintPair {
                i,
                j,
                d_target: d_est,
                weight: w,
            });
            if rank < n {
                target_dist[i * n + j] = d_est;
                target_dist[j * n + i] = d_est;
            }
        }
    }

    // Estimate local i, i+2 bend targets from local helix signal (P_{i,i+3} + P_{i,i+4})
    let mut local2_target = vec![6.10f64; n.saturating_sub(2)];
    for i in 0..n.saturating_sub(2) {
        let mut helix_sig = 0.0f64;
        if i + 3 < n {
            helix_sig += (probs[i * n + (i + 3)] as f64).max(0.0);
        }
        if i + 4 < n {
            helix_sig += (probs[i * n + (i + 4)] as f64).max(0.0);
        }
        let h_norm = (helix_sig / p_max).clamp(0.0, 1.0);
        local2_target[i] = 6.35 - 0.90 * h_norm;
    }

    // Target radius of gyration for globular single domain: R_g ≈ 2.25 * N^0.38
    let rg_target = 2.25 * (n as f64).powf(0.38);

    // Multi-start optimization: Start 0 = Spectral MDS, Starts 1..3 = Twisted Helical Ribbons
    let mut best_coords = vec![[0.0f64; 3]; n];
    let mut best_energy = f64::INFINITY;

    for start_idx in 0..4 {
        let mut x = if start_idx == 0 {
            mds_spectral_init(&target_dist, n)
        } else {
            helical_init(n, start_idx as u64 * 1337)
        };
        let mut vel = vec![[0.0f64; 3]; n];
        let mut grad = vec![[0.0f64; 3]; n];

        let total_steps = 450;
        for step in 0..total_steps {
            let progress = step as f64 / total_steps as f64;
            let lr = 0.025 * (1.0 - 0.85 * progress);
            let momentum = 0.82;

            for g in grad.iter_mut() {
                *g = [0.0, 0.0, 0.0];
            }

            // 1. Exact Peptide Bond Constraint |i - j| = 1: d = 3.80 Å
            let w_bond = 18.0 + 25.0 * progress;
            for i in 0..(n - 1) {
                let dx = x[i + 1][0] - x[i][0];
                let dy = x[i + 1][1] - x[i][1];
                let dz = x[i + 1][2] - x[i][2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                let diff = d - 3.80;
                let f = w_bond * diff / d;
                grad[i][0] -= f * dx;
                grad[i][1] -= f * dy;
                grad[i][2] -= f * dz;
                grad[i + 1][0] += f * dx;
                grad[i + 1][1] += f * dy;
                grad[i + 1][2] += f * dz;
            }

            // 2. Local Bend Angle Constraint |i - j| = 2: d in [5.35, 6.85] + soft pull to local2_target
            let w_bend = 4.5;
            for i in 0..n.saturating_sub(2) {
                let j = i + 2;
                let dx = x[j][0] - x[i][0];
                let dy = x[j][1] - x[i][1];
                let dz = x[j][2] - x[i][2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                let mut f = 0.5 * w_bend * (d - local2_target[i]) / d;
                if d < 5.30 {
                    f += 12.0 * (d - 5.30) / d;
                } else if d > 6.90 {
                    f += 12.0 * (d - 6.90) / d;
                }
                grad[i][0] -= f * dx;
                grad[i][1] -= f * dy;
                grad[i][2] -= f * dz;
                grad[j][0] += f * dx;
                grad[j][1] += f * dy;
                grad[j][2] += f * dz;
            }

            // 3. Tertiary Contact Restraints (robust Huber-like attractive potential)
            let w_contact_scale = 2.8;
            for r in &restraints {
                let dx = x[r.j][0] - x[r.i][0];
                let dy = x[r.j][1] - x[r.i][1];
                let dz = x[r.j][2] - x[r.i][2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                let diff = d - r.d_target;
                let capped_diff = diff.clamp(-2.5, 4.5);
                let f = w_contact_scale * r.weight * capped_diff / d;
                grad[r.i][0] -= f * dx;
                grad[r.i][1] -= f * dy;
                grad[r.i][2] -= f * dz;
                grad[r.j][0] += f * dx;
                grad[r.j][1] += f * dy;
                grad[r.j][2] += f * dz;
            }

            // 4. Steric Van der Waals Exclusion (|i - j| >= 3: d >= 4.0 Å) + Non-Contact Repulsion
            let w_vdw = 10.0 + 15.0 * progress;
            for i in 0..n {
                for j in (i + 3)..n {
                    let dx = x[j][0] - x[i][0];
                    let dy = x[j][1] - x[i][1];
                    let dz = x[j][2] - x[i][2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < 81.0 {
                        let d = d2.sqrt().max(1e-6);
                        if d < 4.10 {
                            let f = w_vdw * (d - 4.10) / d;
                            grad[i][0] -= f * dx;
                            grad[i][1] -= f * dy;
                            grad[i][2] -= f * dz;
                            grad[j][0] += f * dx;
                            grad[j][1] += f * dy;
                            grad[j][2] += f * dz;
                        } else if d < 9.0 {
                            let p = probs[i * n + j] as f64;
                            if p < 0.06 * p_max {
                                let f = 0.22 * (d - 9.0) / d;
                                grad[i][0] -= f * dx;
                                grad[i][1] -= f * dy;
                                grad[i][2] -= f * dz;
                                grad[j][0] += f * dx;
                                grad[j][1] += f * dy;
                                grad[j][2] += f * dz;
                            }
                        }
                    }
                }
            }

            // 5. Globular Radius of Gyration Regularizer
            let mut cx = 0.0;
            let mut cy = 0.0;
            let mut cz = 0.0;
            for pt in &x {
                cx += pt[0];
                cy += pt[1];
                cz += pt[2];
            }
            cx /= n as f64;
            cy /= n as f64;
            cz /= n as f64;
            let mut rg_sq = 0.0;
            for pt in &x {
                rg_sq += (pt[0] - cx).powi(2) + (pt[1] - cy).powi(2) + (pt[2] - cz).powi(2);
            }
            let rg = (rg_sq / n as f64).sqrt().max(1e-6);
            let rg_diff = rg - rg_target;
            let f_rg = 0.35 * rg_diff / (rg * n as f64);
            for i in 0..n {
                grad[i][0] += f_rg * (x[i][0] - cx);
                grad[i][1] += f_rg * (x[i][1] - cy);
                grad[i][2] += f_rg * (x[i][2] - cz);
            }

            // Update coordinates with clipped gradient + momentum
            for i in 0..n {
                let gn = (grad[i][0].powi(2) + grad[i][1].powi(2) + grad[i][2].powi(2)).sqrt();
                let clip = if gn > 15.0 { 15.0 / gn } else { 1.0 };
                for d in 0..3 {
                    vel[i][d] = momentum * vel[i][d] - lr * grad[i][d] * clip;
                    x[i][d] += vel[i][d];
                }
            }
        }

        // Compute final objective energy to select best multi-start basin
        let mut energy = 0.0f64;
        for i in 0..(n - 1) {
            let d = ((x[i + 1][0] - x[i][0]).powi(2)
                + (x[i + 1][1] - x[i][1]).powi(2)
                + (x[i + 1][2] - x[i][2]).powi(2))
            .sqrt();
            energy += 25.0 * (d - 3.80).powi(2);
        }
        for r in &restraints {
            let d = ((x[r.j][0] - x[r.i][0]).powi(2)
                + (x[r.j][1] - x[r.i][1]).powi(2)
                + (x[r.j][2] - x[r.i][2]).powi(2))
            .sqrt();
            let diff = (d - r.d_target).abs();
            let huber = if diff <= 2.5 {
                0.5 * diff * diff
            } else {
                2.5 * (diff - 1.25)
            };
            energy += 2.8 * r.weight * huber;
        }
        for i in 0..n {
            for j in (i + 3)..n {
                let d = ((x[j][0] - x[i][0]).powi(2)
                    + (x[j][1] - x[i][1]).powi(2)
                    + (x[j][2] - x[i][2]).powi(2))
                .sqrt();
                if d < 4.0 {
                    energy += 20.0 * (4.0 - d).powi(2);
                }
            }
        }

        if energy < best_energy {
            best_energy = energy;
            best_coords = x;
        }
    }

    let num_r = restraints.len();
    (best_coords, num_r, p_max)
}

/// Computes 3D protein folding reconstruction with explicit control over sidechain packing (STRUCT-4).
pub fn compute_pipeline_fold_parallel_opts(
    record: &FastaRecord,
    config: &EsmConfig,
    weights: ModelWeightsRef,
    num_threads: usize,
    pack_sidechains: bool,
) -> FoldPipelineResult {
    let seq = &record.sequence;
    let n = seq.len();
    assert!(n >= 4, "Sequence too short for 3D structural folding");

    let tokens = Tokenizer::encode(seq);
    let seq_len = tokens.len();
    let mut state = BidirectionalRunState::new(config, seq_len);
    state.compute_contacts = true;

    let t_fwd0 = Instant::now();
    match weights {
        ModelWeightsRef::Fp32(w) => {
            if num_threads <= 1 {
                forward_bidirectional(&tokens, config, w, &mut state);
            } else {
                forward_bidirectional_parallel(&tokens, config, w, &mut state, num_threads);
            }
        }
        ModelWeightsRef::Q8(w_q8) => {
            forward_bidirectional_q8(&tokens, config, w_q8, &mut state, num_threads);
        }
        ModelWeightsRef::Q4(w_q4) => {
            forward_bidirectional_q4(&tokens, config, w_q4, &mut state, num_threads);
        }
        ModelWeightsRef::Q3(w_q3) => {
            forward_bidirectional_q3(&tokens, config, w_q3, &mut state, num_threads);
        }
    }
    let forward_ms = t_fwd0.elapsed().as_secs_f64() * 1000.0;

    let t_solve0 = Instant::now();
    let (coords, num_restraints, p_max) = reconstruct_3d_ca(&state.contact_probs, n);
    let solve_ms = t_solve0.elapsed().as_secs_f64() * 1000.0;

    // Compute radius of gyration
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for pt in &coords {
        cx += pt[0];
        cy += pt[1];
        cz += pt[2];
    }
    cx /= n as f64;
    cy /= n as f64;
    cz /= n as f64;
    let mut rg_sq = 0.0;
    for pt in &coords {
        rg_sq += (pt[0] - cx).powi(2) + (pt[1] - cy).powi(2) + (pt[2] - cz).powi(2);
    }
    let radius_of_gyration = (rg_sq / n as f64).sqrt();

    // If sequence length >= 3, compute full all-atom coordinates via KinematicBackboneSolver:
    let (allatom_residues, sidechain_metrics) = if n >= 3 {
        let mut ws = crate::export::AllAtomWorkspace::new(n);
        let mut res_names = Vec::with_capacity(n);
        let mut res_seqs = Vec::with_capacity(n);
        for (i, ch) in seq.chars().enumerate() {
            let name3 = crate::export::aa1_to_aa3(ch);
            let mut b = [b' '; 3];
            let src = name3.as_bytes();
            for (idx, &byte) in src.iter().take(3).enumerate() {
                b[idx] = byte;
            }
            res_names.push(b);
            res_seqs.push((i + 1) as i32);
        }
        let solver = crate::export::KinematicBackboneSolver::default();
        let metrics = solver.reconstruct_into(&coords, &res_names, &res_seqs, &mut ws, pack_sidechains);
        (Some(ws.out_residues[0..n].to_vec()), metrics)
    } else {
        (None, None)
    };

    FoldPipelineResult {
        protein_id: record.id.clone(),
        sequence: seq.clone(),
        coords,
        num_restraints,
        p_max,
        forward_ms,
        solve_ms,
        radius_of_gyration,
        allatom_residues,
        sidechain_metrics,
    }
}

/// Computes 3D protein folding reconstruction from sequence without printing to stdout (defaults to sidechain packing).
pub fn compute_pipeline_fold_parallel(
    record: &FastaRecord,
    config: &EsmConfig,
    weights: ModelWeightsRef,
    num_threads: usize,
) -> FoldPipelineResult {
    compute_pipeline_fold_parallel_opts(record, config, weights, num_threads, true)
}

/// Renders a summary table of the 3D folding result to stdout.
pub fn render_fold_summary(
    result: &FoldPipelineResult,
    export_pdb_path: Option<&str>,
    all_atom: bool,
    theme: &TerminalTheme,
) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║         CODON-INFER: ZERO-TRUNK 3D PROTEIN FOLDING (STRUCT-2 / MDS + GRADIENT)            ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  * Target ID:                 {}", result.protein_id);
    println!("  * Sequence Length:           {} residues", result.sequence.len());
    println!("  * Sequence:                  {}", format_sequence(&result.sequence, theme));
    println!("  * Active Tertiary Restraints:{} contacts (peak contact prob: {:.4})", result.num_restraints, result.p_max);
    println!("  * Radius of Gyration (Rg):   {:.2} Å", result.radius_of_gyration);
    println!("  * Forward Attention Time:    {:.2} ms", result.forward_ms);
    println!("  * 3D Geometric Solver Time:  {:.2} ms (Multi-start MDS + 450 gradient steps)", result.solve_ms);
    println!("  * Total End-to-End Latency:  {:.2} ms", result.forward_ms + result.solve_ms);
    if let Some(path) = export_pdb_path {
        let mode_desc = if all_atom {
            if result.sidechain_metrics.is_some() {
                "All-Atom Backbone + Rotameric Sidechains (STRUCT-4)"
            } else {
                "All-Atom Backbone + CB (N, CA, C, O, CB)"
            }
        } else {
            "C-alpha Trace"
        };
        println!("  * Exported PDB Structure:    \x1b[38;2;16;185;129m{}\x1b[0m ({})", path, mode_desc);
    }
    if let Some(ref m) = result.sidechain_metrics {
        if all_atom {
            println!(
                "  * Sidechain Packing (DEE+ICM): Clashscore: {:.1} -> {:.1} (-{:.1}%), {} pruned rotamers in {:.2} ms ({} atoms)",
                m.initial_clashscore, m.final_clashscore, m.clash_reduction_pct, m.pruned_states, m.pack_time_ms, m.total_atoms
            );
        }
    }
    println!();

    println!("┌───────┬────────────┬───────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Index │ Amino Acid │ Three-Let │    X (Å)     │    Y (Å)     │    Z (Å)     │");
    println!("├───────┼────────────┼───────────┼──────────────┼──────────────┼──────────────┤");

    let display_n = result.coords.len().min(12);
    for (i, (ch, pt)) in result.sequence.chars().zip(result.coords.iter()).take(display_n).enumerate() {
        let res_colored = format_aa(ch, theme);
        println!(
            "│  {:>3}  │     {}      │    {}    │  {:>10.3}  │  {:>10.3}  │  {:>10.3}  │",
            i + 1,
            res_colored,
            crate::export::aa1_to_aa3(ch),
            pt[0],
            pt[1],
            pt[2]
        );
    }
    if result.coords.len() > display_n {
        println!(
            "│  ...  │    ...     │    ...    │  ... [{} residues omitted from preview] ...  │",
            result.coords.len() - display_n
        );
    }
    println!("└───────┴────────────┴───────────┴──────────────┴──────────────┴──────────────┘\n");
}

/// Executes 3D protein folding pipeline and optionally writes the PDB file to disk.
pub fn run_pipeline_fold_parallel(
    record: &FastaRecord,
    config: &EsmConfig,
    weights: ModelWeightsRef,
    num_threads: usize,
    export_pdb_path: Option<&str>,
    all_atom: bool,
    pack_sidechains: bool,
    theme: &TerminalTheme,
) -> FoldPipelineResult {
    let result = compute_pipeline_fold_parallel_opts(record, config, weights, num_threads, pack_sidechains);
    if let Some(path) = export_pdb_path {
        let write_res = if all_atom && result.allatom_residues.is_some() {
            crate::export::export_allatom_pdb(path, &result.protein_id, result.allatom_residues.as_ref().unwrap())
                .map(|_| ())
        } else {
            crate::export::export_ca_pdb(path, &result.sequence, &result.protein_id, &result.coords)
                .map(|_| ())
        };
        if let Err(e) = write_res {
            eprintln!("Warning: failed to write PDB to '{}': {}", path, e);
        }
    }
    render_fold_summary(&result, export_pdb_path, all_atom, theme);
    result
}

// ============================================================================
// STRUCTURE-SEQUENCE INVERSE FOLDING & CO-DESIGN (BIO-2 / MPNN-LITE)
// 3D Backbone k-NN Graph + Ramachandran Sterics + Desolvation Potentials
// ============================================================================

pub const STANDARD_AMINO_ACIDS: [char; 20] = [
    'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
    'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicoClass {
    Hydrophobic,
    Polar,
    Positive,
    Negative,
    Special,
}

impl PhysicoClass {
    pub fn from_char(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'L' | 'A' | 'V' | 'I' | 'F' | 'M' | 'W' => PhysicoClass::Hydrophobic,
            'S' | 'T' | 'N' | 'Q' | 'Y' => PhysicoClass::Polar,
            'K' | 'R' | 'H' => PhysicoClass::Positive,
            'D' | 'E' => PhysicoClass::Negative,
            'G' | 'P' | 'C' => PhysicoClass::Special,
            _ => PhysicoClass::Special,
        }
    }
}

pub fn get_token_id(c: char) -> usize {
    match c {
        'L' => 4, 'A' => 5, 'G' => 6, 'V' => 7, 'S' => 8, 'E' => 9, 'R' => 10, 'T' => 11, 'I' => 12, 'D' => 13,
        'P' => 14, 'K' => 15, 'Q' => 16, 'N' => 17, 'F' => 18, 'Y' => 19, 'M' => 20, 'H' => 21, 'W' => 22, 'C' => 23,
        _ => 3,
    }
}

pub fn calc_dihedral(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let b0 = crate::export::Vec3::from_array(p0).sub(crate::export::Vec3::from_array(p1));
    let b1 = crate::export::Vec3::from_array(p2).sub(crate::export::Vec3::from_array(p1));
    let b2 = crate::export::Vec3::from_array(p3).sub(crate::export::Vec3::from_array(p2));

    let b1_u = b1.normalize();
    let v = crate::export::Vec3([
        b0.0[0] - b1_u.0[0] * b0.dot(b1_u),
        b0.0[1] - b1_u.0[1] * b0.dot(b1_u),
        b0.0[2] - b1_u.0[2] * b0.dot(b1_u),
    ]);
    let w = crate::export::Vec3([
        b2.0[0] - b1_u.0[0] * b2.dot(b1_u),
        b2.0[1] - b1_u.0[1] * b2.dot(b1_u),
        b2.0[2] - b1_u.0[2] * b2.dot(b1_u),
    ]);

    let x = v.dot(w);
    let y = b1_u.cross(v).dot(w);
    y.atan2(x)
}

pub struct Bio2Workspace {
    pub burial: Vec<f64>,
    pub phi: Vec<f64>,
    pub psi: Vec<f64>,
    pub neighbors: Vec<(usize, f64)>,
    pub potentials: Vec<[f32; 20]>,
}

impl Bio2Workspace {
    pub fn new(max_len: usize) -> Self {
        Self {
            burial: vec![0.0; max_len],
            phi: vec![0.0; max_len],
            psi: vec![0.0; max_len],
            neighbors: vec![(0, 0.0); max_len],
            potentials: vec![[0.0f32; 20]; max_len],
        }
    }
}

pub struct Bio2CoDesigner {
    pub alpha_pack: f32,
    pub alpha_ram: f32,
    pub alpha_hb: f32,
    pub k_neighbors: usize,
}

impl Default for Bio2CoDesigner {
    fn default() -> Self {
        Self {
            alpha_pack: 0.35,
            alpha_ram: 0.65,
            alpha_hb: 0.45,
            k_neighbors: 30,
        }
    }
}

impl Bio2CoDesigner {
    /// Zero-allocation 3D potential evaluation writing directly into pre-allocated `Bio2Workspace`.
    pub fn compute_geometric_potentials_into(
        &self,
        residues: &[crate::export::ReconstructedResidue],
        use_neighbor_identities: bool,
        ws: &mut Bio2Workspace,
    ) {
        let n = residues.len();
        assert!(n <= ws.potentials.len(), "Residue count exceeds workspace capacity");

        // 1. Calculate Burial / Coordination Number B_i using virtual/real CB
        let mut b_sum = 0.0f64;
        for i in 0..n {
            let p_i = crate::export::Vec3::from_array(residues[i].cb.unwrap_or(residues[i].ca));
            let mut b_i = 0.0f64;
            for j in 0..n {
                if (i as isize - j as isize).abs() > 2 {
                    let p_j = crate::export::Vec3::from_array(residues[j].cb.unwrap_or(residues[j].ca));
                    let d = p_i.sub(p_j).norm();
                    b_i += 1.0 / (1.0 + ((d - 10.0) / 1.5).exp());
                }
            }
            ws.burial[i] = b_i;
            b_sum += b_i;
        }
        let b_mean = b_sum / n as f64;
        let mut b_var = 0.0f64;
        for i in 0..n {
            let diff = ws.burial[i] - b_mean;
            b_var += diff * diff;
        }
        let b_std = (b_var / n as f64).sqrt().max(1e-4);

        // 2. Calculate Dihedral Angles phi_i and psi_i
        for i in 0..n {
            ws.phi[i] = if i > 0 {
                calc_dihedral(residues[i - 1].c, residues[i].n, residues[i].ca, residues[i].c)
            } else {
                0.0
            };
            ws.psi[i] = if i + 1 < n {
                calc_dihedral(residues[i].n, residues[i].ca, residues[i].c, residues[i + 1].n)
            } else {
                0.0
            };
        }

        // 3. Compute potentials for each residue and amino acid candidate (0 heap allocations)
        let n_nb = n.saturating_sub(1);
        for i in 0..n {
            let norm_burial = (ws.burial[i] - b_mean) / b_std;
            let phi_deg = ws.phi[i].to_degrees();
            let psi_deg = ws.psi[i].to_degrees();

            let p_i = crate::export::Vec3::from_array(residues[i].ca);
            let mut nb_idx = 0;
            for j in 0..n {
                if j != i {
                    let d = p_i.sub(crate::export::Vec3::from_array(residues[j].ca)).norm();
                    ws.neighbors[nb_idx] = (j, d);
                    nb_idx += 1;
                }
            }
            let nb_slice = &mut ws.neighbors[0..n_nb];
            nb_slice.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let k_nn = &nb_slice[0..self.k_neighbors.min(n_nb)];

            let v_ab_i = crate::export::Vec3::from_array(residues[i].cb.unwrap_or(residues[i].ca)).sub(p_i).normalize();

            let is_in_helix = if i >= 2 && i + 2 < n {
                let d_prev = crate::export::Vec3::from_array(residues[i - 1].ca).sub(crate::export::Vec3::from_array(residues[i + 2].ca)).norm();
                let d_prev2 = crate::export::Vec3::from_array(residues[i - 2].ca).sub(crate::export::Vec3::from_array(residues[i + 1].ca)).norm();
                d_prev < 6.2 && d_prev2 < 6.2
            } else {
                false
            };

            let is_alpha = phi_deg >= -95.0 && phi_deg <= -40.0 && psi_deg >= -70.0 && psi_deg <= -15.0;
            let is_beta = phi_deg >= -165.0 && phi_deg <= -75.0 && (psi_deg >= 85.0 || psi_deg <= -150.0);

            for (a_idx, &aa) in STANDARD_AMINO_ACIDS.iter().enumerate() {
                let mut score = 0.0f32;

                let is_charged = matches!(aa, 'D' | 'E' | 'K' | 'R');
                let is_bulky_hydrophobic = matches!(aa, 'W' | 'F' | 'I' | 'L' | 'M' | 'V');
                let is_polar = matches!(aa, 'S' | 'T' | 'N' | 'Q' | 'H');
                let c_class = PhysicoClass::from_char(aa);

                // A. Desolvation and Surface Exposure Penalties (Threshold-based Born physics)
                if norm_burial > 1.0 {
                    if is_charged {
                        let mut has_salt_bridge = false;
                        if use_neighbor_identities {
                            for &(j, d) in k_nn {
                                if d < 5.5 && (i as isize - j as isize).abs() > 2 {
                                    let neighbor_res_char = crate::export::aa3_to_aa1(std::str::from_utf8(&residues[j].res_name).unwrap_or("UNK"));
                                    let neighbor_class = PhysicoClass::from_char(neighbor_res_char);
                                    if (c_class == PhysicoClass::Negative && neighbor_class == PhysicoClass::Positive)
                                        || (c_class == PhysicoClass::Positive && neighbor_class == PhysicoClass::Negative)
                                    {
                                        let p_j = crate::export::Vec3::from_array(residues[j].ca);
                                        let v_ab_j = crate::export::Vec3::from_array(residues[j].cb.unwrap_or(residues[j].ca)).sub(p_j).normalize();
                                        if v_ab_i.dot(v_ab_j) < -0.20 {
                                            has_salt_bridge = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        if !has_salt_bridge {
                            score -= self.alpha_pack * 1.8;
                        }
                    } else if is_bulky_hydrophobic {
                        score += self.alpha_pack * 0.6;
                    }
                } else if norm_burial < -1.0 {
                    if matches!(aa, 'W' | 'F' | 'I' | 'L') {
                        score -= self.alpha_pack * 1.2;
                    } else if is_charged || is_polar {
                        score += self.alpha_pack * 0.4;
                    }
                }

                // B. Dihedral / Stereochemical Filter (Ramachandran sterics & secondary structure)
                if aa == 'G' {
                    if phi_deg > 20.0 {
                        score += self.alpha_ram * 2.2;
                    } else if is_alpha {
                        score -= self.alpha_ram * 0.4;
                    }
                } else if phi_deg > 20.0 {
                    score -= self.alpha_ram * 2.5;
                }

                if is_alpha {
                    if matches!(aa, 'A' | 'L' | 'E' | 'M' | 'K' | 'R' | 'Q') {
                        score += self.alpha_ram * 0.25;
                    }
                } else if is_beta {
                    if matches!(aa, 'V' | 'I' | 'T' | 'Y' | 'F' | 'W') {
                        score += self.alpha_ram * 0.35;
                    }
                }

                if aa == 'P' {
                    if is_in_helix || is_alpha {
                        score -= self.alpha_ram * 3.0;
                    } else if phi_deg >= -90.0 && phi_deg <= -40.0 {
                        score += self.alpha_ram * 1.5;
                    } else if phi_deg > -30.0 || phi_deg < -110.0 {
                        score -= self.alpha_ram * 2.0;
                    }
                }

                // C. Contact / Electrostatic Complementarity
                if use_neighbor_identities {
                    let mut contact_bonus = 0.0f32;
                    for &(j, d) in k_nn {
                        if d < 6.5 && (i as isize - j as isize).abs() > 2 {
                            let p_j = crate::export::Vec3::from_array(residues[j].ca);
                            let v_ab_j = crate::export::Vec3::from_array(residues[j].cb.unwrap_or(residues[j].ca)).sub(p_j).normalize();
                            let pointing_face_to_face = v_ab_i.dot(v_ab_j) < -0.20;

                            if pointing_face_to_face {
                                let neighbor_res_char = crate::export::aa3_to_aa1(std::str::from_utf8(&residues[j].res_name).unwrap_or("UNK"));
                                let neighbor_class = PhysicoClass::from_char(neighbor_res_char);

                                if (c_class == PhysicoClass::Negative && neighbor_class == PhysicoClass::Positive)
                                    || (c_class == PhysicoClass::Positive && neighbor_class == PhysicoClass::Negative)
                                {
                                    contact_bonus += 1.5;
                                } else if (c_class == PhysicoClass::Negative && neighbor_class == PhysicoClass::Negative)
                                    || (c_class == PhysicoClass::Positive && neighbor_class == PhysicoClass::Positive)
                                {
                                    contact_bonus -= 1.5;
                                }

                                if matches!(aa, 'F' | 'Y' | 'W') && matches!(neighbor_res_char, 'F' | 'Y' | 'W') {
                                    contact_bonus += 1.2;
                                }
                            }
                        }
                    }
                    score += self.alpha_hb * contact_bonus;
                }

                ws.potentials[i][a_idx] = score;
            }
        }
    }
}

/// Candidate amino acid prediction from inverse folding co-design.
#[derive(Debug, Clone)]
pub struct DesignCandidate {
    pub amino_acid: char,
    pub score: f32,
    pub probability_pct: f32,
    pub category: &'static str,
}

/// Residue site output for sequence design.
#[derive(Debug, Clone)]
pub struct DesignPositionSite {
    pub position: usize,
    pub native_char: char,
    pub designed_char: char,
    pub candidates: Vec<DesignCandidate>,
}

/// Result of an inverse folding structure-sequence co-design run (BIO-2).
#[derive(Debug, Clone)]
pub struct DesignPipelineResult {
    pub protein_id: String,
    pub native_sequence: String,
    pub designed_sequence: String,
    pub sites: Vec<DesignPositionSite>,
    pub sequence_recovery_pct: f64,
    pub physicochemical_recovery_pct: f64,
    pub forward_ms: f64,
    pub potential_3d_us: f64,
    pub total_ms: f64,
}

/// Computes sequence design conditioned on 3D backbone atomic coordinates (BIO-2).
pub fn compute_pipeline_design_parallel(
    protein_id: &str,
    residues: &[crate::export::ReconstructedResidue],
    use_neighbor_identities: bool,
    config: &EsmConfig,
    weights: ModelWeightsRef,
    num_threads: usize,
) -> DesignPipelineResult {
    let n_res = residues.len();
    assert!(n_res >= 3, "Residue count too short for structure design");

    let wt_seq: String = residues
        .iter()
        .map(|r| crate::export::aa3_to_aa1(std::str::from_utf8(&r.res_name).unwrap_or("UNK")))
        .collect();

    let mut state = BidirectionalRunState::new(config, n_res + 2);
    let mut base_tokens = Tokenizer::encode(&wt_seq);
    let mut masked_logits = vec![0.0f32; n_res * config.vocab_size];

    let t_fwd0 = Instant::now();
    for i in 0..n_res {
        let token_pos = i + 1;
        let orig = base_tokens[token_pos];
        base_tokens[token_pos] = 32; // Mask token '*'
        match weights {
            ModelWeightsRef::Fp32(w) => {
                if num_threads <= 1 {
                    forward_bidirectional(&base_tokens, config, w, &mut state);
                } else {
                    forward_bidirectional_parallel(&base_tokens, config, w, &mut state, num_threads);
                }
            }
            ModelWeightsRef::Q8(w_q8) => {
                forward_bidirectional_q8(&base_tokens, config, w_q8, &mut state, num_threads);
            }
            ModelWeightsRef::Q4(w_q4) => {
                forward_bidirectional_q4(&base_tokens, config, w_q4, &mut state, num_threads);
            }
            ModelWeightsRef::Q3(w_q3) => {
                forward_bidirectional_q3(&base_tokens, config, w_q3, &mut state, num_threads);
            }
        }
        let slice = &state.logits[token_pos * config.vocab_size..(token_pos + 1) * config.vocab_size];
        masked_logits[i * config.vocab_size..(i + 1) * config.vocab_size].copy_from_slice(slice);
        base_tokens[token_pos] = orig;
    }
    let forward_ms = t_fwd0.elapsed().as_secs_f64() * 1000.0;

    let co_designer = Bio2CoDesigner::default();
    let mut ws = Bio2Workspace::new(n_res);

    let t_pot0 = Instant::now();
    co_designer.compute_geometric_potentials_into(residues, use_neighbor_identities, &mut ws);
    let potential_3d_us = t_pot0.elapsed().as_nanos() as f64 / 1000.0;

    let mut sites = Vec::with_capacity(n_res);
    let mut designed_chars = Vec::with_capacity(n_res);
    let mut correct_exact = 0;
    let mut correct_phys = 0;

    for (i, res) in residues.iter().enumerate() {
        let logits_slice = &masked_logits[i * config.vocab_size..(i + 1) * config.vocab_size];
        let wt_char = crate::export::aa3_to_aa1(std::str::from_utf8(&res.res_name).unwrap_or("UNK"));
        let wt_class = PhysicoClass::from_char(wt_char);

        let mut aa_scores: Vec<(char, f32)> = Vec::with_capacity(20);
        for (a_idx, &aa) in STANDARD_AMINO_ACIDS.iter().enumerate() {
            let tok_id = get_token_id(aa);
            let combined = logits_slice[tok_id] + ws.potentials[i][a_idx];
            aa_scores.push((aa, combined));
        }

        aa_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let max_s = aa_scores[0].1;
        let sum_exp: f32 = aa_scores.iter().map(|&(_, s)| (s - max_s).exp()).sum();

        let mut candidates = Vec::with_capacity(aa_scores.len());
        for &(aa, s) in &aa_scores {
            let prob_pct = ((s - max_s).exp() / sum_exp) * 100.0;
            candidates.push(DesignCandidate {
                amino_acid: aa,
                score: s,
                probability_pct: prob_pct,
                category: aa_category(aa),
            });
        }

        let best_aa = candidates[0].amino_acid;
        designed_chars.push(best_aa);

        if best_aa == wt_char {
            correct_exact += 1;
        }
        if PhysicoClass::from_char(best_aa) == wt_class {
            correct_phys += 1;
        }

        sites.push(DesignPositionSite {
            position: i + 1,
            native_char: wt_char,
            designed_char: best_aa,
            candidates,
        });
    }

    let recovery_pct = (correct_exact as f64 / n_res as f64) * 100.0;
    let phys_recovery_pct = (correct_phys as f64 / n_res as f64) * 100.0;
    let designed_sequence: String = designed_chars.into_iter().collect();

    DesignPipelineResult {
        protein_id: protein_id.to_string(),
        native_sequence: wt_seq,
        designed_sequence,
        sites,
        sequence_recovery_pct: recovery_pct,
        physicochemical_recovery_pct: phys_recovery_pct,
        forward_ms,
        potential_3d_us,
        total_ms: forward_ms + (potential_3d_us / 1000.0),
    }
}

/// Renders sequence design summary table to stdout.
pub fn render_design_summary(result: &DesignPipelineResult, max_display_rows: usize, theme: &TerminalTheme) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║       CODON-INFER: 3D STRUCTURE-GUIDED SEQUENCE CO-DESIGN & INVERSE FOLDING (BIO-2)       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  * Target ID:                 {}", result.protein_id);
    println!("  * Sequence Length:           {} residues", result.native_sequence.len());
    println!("  * Native Sequence:           {}", format_sequence(&result.native_sequence, theme));
    println!("  * Designed Sequence:         {}", format_sequence(&result.designed_sequence, theme));
    println!("  * Native Sequence Recovery:  {:.1}% exact matches", result.sequence_recovery_pct);
    println!("  * Physicochemical Recovery:  {:.1}% class conservation", result.physicochemical_recovery_pct);
    println!("  * 3D Potential Compute Time: {:.1} µs (0 dynamic heap allocations in hot loop)", result.potential_3d_us);
    println!("  * End-to-End Latency:        {:.2} ms total", result.total_ms);
    println!();

    println!("┌──────────┬────────┬────────┬─────────────────────────────────────────────────────────────┐");
    println!("│ Position │ Native │ Design │ Top-5 Candidates & Physicochemical Compatibility            │");
    println!("├──────────┼────────┼────────┼─────────────────────────────────────────────────────────────┤");

    let display_rows = if result.sites.len() <= max_display_rows {
        &result.sites[..]
    } else {
        &result.sites[..max_display_rows]
    };

    for site in display_rows {
        let is_match = site.native_char == site.designed_char;
        let native_colored = format_aa(site.native_char, theme);
        let designed_colored = format_aa(site.designed_char, theme);
        let match_marker = if is_match { "✓" } else { "·" };

        let mut top5_str = String::new();
        for cand in site.candidates.iter().take(5) {
            top5_str.push_str(&format!("{}({:.1}%) ", format_aa(cand.amino_acid, theme), cand.probability_pct));
        }

        println!(
            "│ Site {:>3} │   {}    │  {} {}  │ {:<59} │",
            site.position, native_colored, designed_colored, match_marker, top5_str
        );
    }

    if result.sites.len() > max_display_rows {
        println!(
            "│   ...    │  ...   │  ...   │ ... [Truncated {} additional positions for terminal display] │",
            result.sites.len() - max_display_rows
        );
    }
    println!("└──────────┴────────┴────────┴─────────────────────────────────────────────────────────────┘\n");
}

/// Executes sequence design from an input PDB file or residue slice.
pub fn run_pipeline_design_parallel(
    protein_id: &str,
    residues: &[crate::export::ReconstructedResidue],
    use_neighbor_identities: bool,
    config: &EsmConfig,
    weights: ModelWeightsRef,
    num_threads: usize,
    theme: &TerminalTheme,
) -> DesignPipelineResult {
    let result = compute_pipeline_design_parallel(
        protein_id,
        residues,
        use_neighbor_identities,
        config,
        weights,
        num_threads,
    );
    render_design_summary(&result, 25, theme);
    result
}



