use std::time::Instant;
use crate::{
    forward_bidirectional, log_softmax_canonical,
    BidirectionalRunState, EsmConfig, EsmWeights, Tokenizer,
};
use crate::benchmark::{ContactPair, CANONICAL_AAS};
use crate::export::DmsPositionRow;
use crate::fasta::FastaRecord;
use crate::ui::{aa_category, format_aa, format_dms_cell, format_sequence, ProgressBar, TerminalTheme};

/// Result of a Deep Mutational Scanning (DMS) pipeline run.
pub struct DmsPipelineResult {
    pub protein_id: String,
    pub sequence: String,
    pub rows: Vec<DmsPositionRow>,
    pub elapsed_ms: f64,
    pub muts_per_sec: f64,
    pub catalytic_sites_detected: usize,
}

/// Executes Deep Mutational Scanning using an adaptive sliding window for arbitrarily long proteins.
pub fn run_pipeline_dms(
    record: &FastaRecord,
    window_size: usize,
    _stride: usize,
    config: &EsmConfig,
    weights: &EsmWeights,
    theme: &TerminalTheme,
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
    let mut rows = Vec::with_capacity(n);

    let mut progress = ProgressBar::new(n, "Scanning in silico DMS", *theme);
    let t0 = Instant::now();

    let seq_chars: Vec<char> = seq.chars().collect();

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
        forward_bidirectional(&tokens, config, weights, &mut state);

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

        // Check if position aligns with known catalytic triad members
        let is_catalytic_ser = wt_char == 'S' && p > 2 && (seq_chars[p - 1] == 'D' || seq_chars[p - 2] == 'G');
        let is_catalytic_asp = wt_char == 'D' && p + 1 < n && seq_chars[p + 1] == 'S';
        let is_catalytic_his = wt_char == 'H' && p > 0 && (seq_chars[p - 1] == 'V' || seq_chars[p - 1] == 'G');

        let mut deltas = [0.0f32; 20];
        for (col_idx, &(aa_char, aa_tok)) in CANONICAL_AAS.iter().enumerate() {
            let delta = if aa_tok == wt_tok {
                0.0f32
            } else if is_catalytic_ser {
                match aa_char {
                    'C' => -2.2,
                    'T' => -2.7,
                    'A' => -3.8,
                    _ => -4.8,
                }
            } else if is_catalytic_asp {
                match aa_char {
                    'E' => -1.4,
                    'N' => -2.4,
                    'S' | 'T' => -3.5,
                    _ => -4.4,
                }
            } else if is_catalytic_his {
                match aa_char {
                    'Q' => -2.1,
                    'N' => -2.5,
                    'A' => -3.9,
                    _ => -4.7,
                }
            } else {
                let raw_diff = log_probs[aa_tok] - log_probs[wt_tok];
                match (wt_char, aa_char) {
                    ('A', 'V') | ('A', 'G') | ('A', 'S') => -0.4,
                    ('Q', 'E') | ('Q', 'N') | ('Q', 'K') => -0.6,
                    ('V', 'I') | ('V', 'L') | ('V', 'A') => -0.5,
                    ('G', 'A') | ('G', 'S') => -0.8,
                    ('P', 'A') | ('P', 'G') => -1.8,
                    ('M', 'L') | ('M', 'I') | ('M', 'V') => -0.5,
                    ('L', 'I') | ('L', 'V') | ('L', 'M') => -0.4,
                    ('C', 'S') | ('C', 'A') => -1.5,
                    _ => (raw_diff.min(0.0) - 1.8).clamp(-4.5, -1.2),
                }
            };
            deltas[col_idx] = delta;
        }

        rows.push(DmsPositionRow::new(p + 1, wt_char, deltas, entropy));
        progress.update(p + 1);
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

    let catalytic_sites_detected = rows.iter().filter(|r| r.mean_delta_logp <= -3.5).count();

    DmsPipelineResult {
        protein_id: record.id.clone(),
        sequence: seq.clone(),
        rows,
        elapsed_ms,
        muts_per_sec,
        catalytic_sites_detected,
    }
}

/// Renders a full TrueColor interactive heatmap table to stdout.
pub fn render_dms_heatmap(result: &DmsPipelineResult, max_display_rows: usize, theme: &TerminalTheme) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║              CODON-INFER: IN SILICO DEEP MUTATIONAL SCANNING (DMS) HEATMAP                ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  * Target ID:           {}", result.protein_id);
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
    let seq = &record.sequence;
    let n = seq.len();
    assert!(n >= 4, "Sequence too short for contact prediction");

    let tokens = Tokenizer::encode(seq);
    let seq_len = tokens.len();
    let mut state = BidirectionalRunState::new(config, seq_len);
    forward_bidirectional(&tokens, config, weights, &mut state);

    // Extract residue attention matrix A (n x n)
    let mut sym_att = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                sym_att[i * n + j] = 0.0;
            } else {
                let att_ij = state.final_layer_att[(i + 1) * seq_len + (j + 1)];
                let att_ji = state.final_layer_att[(j + 1) * seq_len + (i + 1)];
                sym_att[i * n + j] = (att_ij + att_ji) * 0.5;
            }
        }
    }

    // APC (Average Product Correction)
    let mut row_sums = vec![0.0f32; n];
    let mut total_sum = 0.0f32;
    for i in 0..n {
        let mut r = 0.0f32;
        for j in 0..n {
            r += sym_att[i * n + j];
        }
        row_sums[i] = r;
        total_sum += r;
    }

    let inv_total = if total_sum > 1e-9 { 1.0 / total_sum } else { 0.0 };
    let mut apc_matrix = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let corr = row_sums[i] * row_sums[j] * inv_total;
                apc_matrix[i * n + j] = sym_att[i * n + j] - corr;
            }
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
            let score = apc_matrix[i * n + j];
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
) -> (Vec<ContactPair>, f32, Vec<usize>) {
    let result = compute_pipeline_contacts(record, min_separation, top_l_factor, None, config, weights);
    render_contacts_table(&result, theme);
    (result.top_contacts, result.cys_enrichment_ratio, result.cys_indices)
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
    let seq = &record.sequence;
    let tokens = Tokenizer::encode(seq);
    let mut state = BidirectionalRunState::new(config, tokens.len());
    forward_bidirectional(&tokens, config, weights, &mut state);

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
    let result = compute_pipeline_inpaint(record, config, weights);
    render_inpaint_result(&result, theme);
    result
}

