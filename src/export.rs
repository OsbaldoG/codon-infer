use std::fs::File;
use std::io::{self, Write};
use crate::benchmark::{ContactPair, CANONICAL_AAS};

/// Rectangular data row for Deep Mutational Scanning (DMS) export.
#[derive(Debug, Clone)]
pub struct DmsPositionRow {
    pub pos: usize, // 1-based index
    pub wt: char,
    pub deltas: [f32; 20],
    pub mean_delta_logp: f32,
    pub entropy: f32,
    pub category: &'static str,
}

impl DmsPositionRow {
    pub fn new(
        pos: usize,
        wt: char,
        deltas: [f32; 20],
        entropy: f32,
    ) -> Self {
        // Compute mean of non-synonymous mutations (19 values)
        let sum: f32 = deltas.iter().filter(|&&d| d != 0.0).sum();
        let mean_delta_logp = sum / 19.0;

        let category = if mean_delta_logp <= -3.8 {
            "Catalytic / Strictly Intolerant"
        } else if mean_delta_logp <= -2.2 {
            "Conserved Structural Core"
        } else {
            "Tolerated Variable"
        };

        Self {
            pos,
            wt,
            deltas,
            mean_delta_logp,
            entropy,
            category,
        }
    }
}

/// Exports DMS matrix to publication-ready rectangular CSV.
pub fn export_dms_csv(path: &str, _protein_id: &str, rows: &[DmsPositionRow]) -> io::Result<()> {
    let mut file = File::create(path)?;
    // Header
    write!(file, "pos,wt")?;
    for &(aa, _) in &CANONICAL_AAS {
        write!(file, ",{}", aa)?;
    }
    writeln!(file, ",mean_delta_logp,entropy,category")?;

    for r in rows {
        write!(file, "{},{}", r.pos, r.wt)?;
        for &d in &r.deltas {
            write!(file, ",{:.3}", d)?;
        }
        writeln!(file, ",{:.3},{:.3},\"{}\"", r.mean_delta_logp, r.entropy, r.category)?;
    }
    file.flush()?;
    Ok(())
}

/// Serializes DMS results to structured JSON string.
pub fn dms_to_json_string(
    protein_id: &str,
    sequence: &str,
    rows: &[DmsPositionRow],
    elapsed_ms: f64,
    muts_per_sec: f64,
) -> String {
    let mut out = String::with_capacity(rows.len() * 250 + 256);
    out.push_str("{\n");
    out.push_str(&format!("  \"protein_id\": \"{}\",\n", protein_id));
    out.push_str(&format!("  \"sequence_length\": {},\n", sequence.len()));
    out.push_str(&format!("  \"sequence\": \"{}\",\n", sequence));
    out.push_str(&format!("  \"elapsed_ms\": {:.2},\n", elapsed_ms));
    out.push_str(&format!("  \"muts_per_sec\": {:.1},\n", muts_per_sec));

    let critical_sites: Vec<&DmsPositionRow> = rows.iter().filter(|r| r.mean_delta_logp <= -3.5).collect();
    out.push_str(&format!("  \"critical_sites_count\": {},\n", critical_sites.len()));
    out.push_str("  \"critical_sites\": [\n");
    for (i, r) in critical_sites.iter().enumerate() {
        let comma = if i + 1 < critical_sites.len() { "," } else { "" };
        out.push_str(&format!(
            "    {{\"pos\": {}, \"wt\": \"{}\", \"mean_delta_logp\": {:.3}, \"category\": \"{}\"}}{}\n",
            r.pos, r.wt, r.mean_delta_logp, r.category, comma
        ));
    }
    out.push_str("  ],\n");

    out.push_str("  \"positions\": [\n");
    for (i, r) in rows.iter().enumerate() {
        let comma = if i + 1 < rows.len() { "," } else { "" };
        out.push_str(&format!(
            "    {{\"pos\": {}, \"wt\": \"{}\", \"mean_delta_logp\": {:.3}, \"entropy\": {:.3}, \"category\": \"{}\", \"substitutions\": {{",
            r.pos, r.wt, r.mean_delta_logp, r.entropy, r.category
        ));
        for (j, &(aa, _)) in CANONICAL_AAS.iter().enumerate() {
            let sub_comma = if j + 1 < 20 { ", " } else { "" };
            out.push_str(&format!("\"{}\": {:.3}{}", aa, r.deltas[j], sub_comma));
        }
        out.push_str(&format!("}}}}{}\n", comma));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

/// Exports DMS results to structured JSON file.
pub fn export_dms_json(
    path: &str,
    protein_id: &str,
    sequence: &str,
    rows: &[DmsPositionRow],
    elapsed_ms: f64,
    muts_per_sec: f64,
) -> io::Result<()> {
    let mut file = File::create(path)?;
    let json = dms_to_json_string(protein_id, sequence, rows, elapsed_ms, muts_per_sec);
    file.write_all(json.as_bytes())?;
    file.flush()?;
    Ok(())
}

/// Exports structural contacts to CSV.
pub fn export_contacts_csv(path: &str, _protein_id: &str, pairs: &[ContactPair]) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "rank,pos1,res1,pos2,res2,separation,apc_score,is_cys_cys,is_disulfide_candidate"
    )?;
    for (rank, p) in pairs.iter().enumerate() {
        let sep = (p.pos2 as isize - p.pos1 as isize).abs();
        writeln!(
            file,
            "{},{},{},{},{},{},{:.5},{},{}",
            rank + 1,
            p.pos1 + 1,
            p.res1,
            p.pos2 + 1,
            p.res2,
            sep,
            p.apc_score,
            p.is_cys_cys,
            p.is_known_disulfide || p.is_cys_cys
        )?;
    }
    file.flush()?;
    Ok(())
}

/// Serializes structural contacts to structured JSON string.
pub fn contacts_to_json_string(
    protein_id: &str,
    sequence_len: usize,
    cys_indices: &[usize],
    enrichment: f32,
    pairs: &[ContactPair],
) -> String {
    let mut out = String::with_capacity(pairs.len() * 120 + 256);
    out.push_str("{\n");
    out.push_str(&format!("  \"protein_id\": \"{}\",\n", protein_id));
    out.push_str(&format!("  \"sequence_length\": {},\n", sequence_len));
    out.push_str(&format!("  \"cys_count\": {},\n", cys_indices.len()));
    let cys_str: Vec<String> = cys_indices.iter().map(|&idx| (idx + 1).to_string()).collect();
    out.push_str(&format!("  \"cys_positions\": [{}],\n", cys_str.join(", ")));
    out.push_str(&format!("  \"cys_enrichment_ratio\": {:.2},\n", enrichment));
    out.push_str(&format!("  \"top_contacts_count\": {},\n", pairs.len()));
    out.push_str("  \"top_contacts\": [\n");
    for (i, p) in pairs.iter().enumerate() {
        let comma = if i + 1 < pairs.len() { "," } else { "" };
        let sep = (p.pos2 as isize - p.pos1 as isize).abs();
        out.push_str(&format!(
            "    {{\"rank\": {}, \"pos1\": {}, \"res1\": \"{}\", \"pos2\": {}, \"res2\": \"{}\", \"separation\": {}, \"apc_score\": {:.5}, \"is_cys_cys\": {}, \"is_disulfide_candidate\": {}}}{}\n",
            i + 1, p.pos1 + 1, p.res1, p.pos2 + 1, p.res2, sep, p.apc_score, p.is_cys_cys, p.is_known_disulfide || p.is_cys_cys, comma
        ));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

/// Exports structural contacts to JSON file.
pub fn export_contacts_json(
    path: &str,
    protein_id: &str,
    sequence: &str,
    cys_indices: &[usize],
    enrichment: f32,
    pairs: &[ContactPair],
) -> io::Result<()> {
    let mut file = File::create(path)?;
    let json = contacts_to_json_string(protein_id, sequence.len(), cys_indices, enrichment, pairs);
    file.write_all(json.as_bytes())?;
    file.flush()?;
    Ok(())
}

/// Serializes inpainting motif reconstruction results into structured JSON string.
pub fn inpaint_to_json_string(
    protein_id: &str,
    original_sequence: &str,
    reconstructed_sequence: &str,
    sites: &[crate::pipeline::InpaintSite],
) -> String {
    let mut out = String::with_capacity(sites.len() * 300 + 256);
    out.push_str("{\n");
    out.push_str(&format!("  \"protein_id\": \"{}\",\n", protein_id));
    out.push_str(&format!("  \"sequence_length\": {},\n", original_sequence.len()));
    out.push_str(&format!("  \"original_sequence\": \"{}\",\n", original_sequence));
    out.push_str(&format!("  \"reconstructed_sequence\": \"{}\",\n", reconstructed_sequence));
    out.push_str(&format!("  \"masked_sites_count\": {},\n", sites.len()));
    out.push_str("  \"predictions\": [\n");
    for (i, site) in sites.iter().enumerate() {
        let comma = if i + 1 < sites.len() { "," } else { "" };
        out.push_str(&format!(
            "    {{\"position\": {}, \"native\": \"{}\", \"predicted\": \"{}\", \"candidates\": [",
            site.position, site.native_char, site.predicted_char
        ));
        for (j, cand) in site.candidates.iter().enumerate() {
            let cand_comma = if j + 1 < site.candidates.len() { ", " } else { "" };
            out.push_str(&format!(
                "{{\"amino_acid\": \"{}\", \"probability_pct\": {:.2}, \"category\": \"{}\"}}{}",
                cand.amino_acid, cand.probability_pct, cand.category, cand_comma
            ));
        }
        out.push_str(&format!("]}}{}\n", comma));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dms_row_and_csv_generation() {
        let deltas = [0.0; 20];
        let row = DmsPositionRow::new(1, 'M', deltas, 2.5);
        assert_eq!(row.pos, 1);
        assert_eq!(row.wt, 'M');

        let test_path = "test_export_scratch.csv";
        export_dms_csv(test_path, "test_protein", &[row]).expect("should export");
        assert!(std::path::Path::new(test_path).exists());
        let content = std::fs::read_to_string(test_path).unwrap();
        assert!(content.contains("pos,wt,A,C,D"));
        assert!(content.contains("1,M,0.000,0.000"));
        let _ = std::fs::remove_file(test_path);
    }

    #[test]
    fn test_json_string_serializers() {
        let deltas = [0.0; 20];
        let row = DmsPositionRow::new(1, 'M', deltas, 2.5);
        let dms_json = dms_to_json_string("test_prot", "M", &[row], 12.34, 162.0);
        assert!(dms_json.contains("\"protein_id\": \"test_prot\""));
        assert!(dms_json.contains("\"sequence_length\": 1"));
        assert!(dms_json.contains("\"positions\": ["));

        let pair = ContactPair {
            pos1: 0,
            res1: 'M',
            pos2: 5,
            res2: 'C',
            apc_score: 0.1234,
            is_cys_cys: false,
            is_known_disulfide: false,
        };
        let contacts_json = contacts_to_json_string("test_prot", 6, &[5], 1.25, &[pair]);
        assert!(contacts_json.contains("\"protein_id\": \"test_prot\""));
        assert!(contacts_json.contains("\"top_contacts\": ["));
        assert!(contacts_json.contains("\"apc_score\": 0.12340"));

        let inpaint_site = crate::pipeline::InpaintSite {
            position: 1,
            token_idx: 1,
            native_char: '*',
            predicted_char: 'C',
            candidates: vec![crate::pipeline::InpaintCandidate {
                amino_acid: 'C',
                probability_pct: 85.5,
                category: "Cysteine",
            }],
        };
        let inpaint_json = inpaint_to_json_string("test_prot", "*", "C", &[inpaint_site]);
        assert!(inpaint_json.contains("\"reconstructed_sequence\": \"C\""));
        assert!(inpaint_json.contains("\"predictions\": ["));
        assert!(inpaint_json.contains("85.50"));
    }
}
