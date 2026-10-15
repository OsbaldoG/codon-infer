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

/// Exports DMS results to structured JSON.
pub fn export_dms_json(
    path: &str,
    protein_id: &str,
    sequence: &str,
    rows: &[DmsPositionRow],
    elapsed_ms: f64,
    muts_per_sec: f64,
) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "{{")?;
    writeln!(file, "  \"protein_id\": \"{}\",", protein_id)?;
    writeln!(file, "  \"sequence_length\": {},", sequence.len())?;
    writeln!(file, "  \"sequence\": \"{}\",", sequence)?;
    writeln!(file, "  \"elapsed_ms\": {:.2},", elapsed_ms)?;
    writeln!(file, "  \"muts_per_sec\": {:.1},", muts_per_sec)?;

    // Critical sites summary
    let critical_sites: Vec<&DmsPositionRow> = rows.iter().filter(|r| r.mean_delta_logp <= -3.5).collect();
    writeln!(file, "  \"critical_sites\": [")?;
    for (i, r) in critical_sites.iter().enumerate() {
        let comma = if i + 1 < critical_sites.len() { "," } else { "" };
        writeln!(
            file,
            "    {{\"pos\": {}, \"wt\": \"{}\", \"mean_delta_logp\": {:.3}, \"category\": \"{}\"}}{}",
            r.pos, r.wt, r.mean_delta_logp, r.category, comma
        )?;
    }
    writeln!(file, "  ],")?;

    // Full positional matrix
    writeln!(file, "  \"positions\": [")?;
    for (i, r) in rows.iter().enumerate() {
        let comma = if i + 1 < rows.len() { "," } else { "" };
        write!(
            file,
            "    {{\"pos\": {}, \"wt\": \"{}\", \"mean_delta_logp\": {:.3}, \"entropy\": {:.3}, \"category\": \"{}\", \"substitutions\": {{",
            r.pos, r.wt, r.mean_delta_logp, r.entropy, r.category
        )?;
        for (j, &(aa, _)) in CANONICAL_AAS.iter().enumerate() {
            let sub_comma = if j + 1 < 20 { ", " } else { "" };
            write!(file, "\"{}\": {:.3}{}", aa, r.deltas[j], sub_comma)?;
        }
        writeln!(file, "}}}}{}", comma)?;
    }
    writeln!(file, "  ]")?;
    writeln!(file, "}}")?;
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

/// Exports structural contacts to JSON.
pub fn export_contacts_json(
    path: &str,
    protein_id: &str,
    sequence: &str,
    cys_indices: &[usize],
    enrichment: f32,
    pairs: &[ContactPair],
) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "{{")?;
    writeln!(file, "  \"protein_id\": \"{}\",", protein_id)?;
    writeln!(file, "  \"sequence_length\": {},", sequence.len())?;
    writeln!(file, "  \"cys_count\": {},", cys_indices.len())?;
    let cys_str: Vec<String> = cys_indices.iter().map(|&idx| (idx + 1).to_string()).collect();
    writeln!(file, "  \"cys_positions\": [{}],", cys_str.join(", "))?;
    writeln!(file, "  \"cys_enrichment_ratio\": {:.2},", enrichment)?;
    writeln!(file, "  \"top_contacts\": [")?;
    for (i, p) in pairs.iter().enumerate() {
        let comma = if i + 1 < pairs.len() { "," } else { "" };
        let sep = (p.pos2 as isize - p.pos1 as isize).abs();
        writeln!(
            file,
            "    {{\"rank\": {}, \"pos1\": {}, \"res1\": \"{}\", \"pos2\": {}, \"res2\": \"{}\", \"separation\": {}, \"apc_score\": {:.5}, \"is_cys_cys\": {}}}{}",
            i + 1, p.pos1 + 1, p.res1, p.pos2 + 1, p.res2, sep, p.apc_score, p.is_cys_cys, comma
        )?;
    }
    writeln!(file, "  ]")?;
    writeln!(file, "}}")?;
    file.flush()?;
    Ok(())
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
}
