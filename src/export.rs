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
    out.push_str(&format!("  \"protein_id\": \"{}\",\n", crate::mcp::escape_json_str(protein_id)));
    out.push_str(&format!("  \"sequence_length\": {},\n", sequence.len()));
    out.push_str(&format!("  \"sequence\": \"{}\",\n", crate::mcp::escape_json_str(sequence)));
    out.push_str(&format!("  \"elapsed_ms\": {:.2},\n", elapsed_ms));
    out.push_str(&format!("  \"muts_per_sec\": {:.1},\n", muts_per_sec));

    let critical_sites: Vec<&DmsPositionRow> = rows.iter().filter(|r| r.mean_delta_logp <= -3.8).collect();
    out.push_str(&format!("  \"critical_sites_count\": {},\n", critical_sites.len()));
    out.push_str("  \"critical_sites\": [\n");
    for (i, r) in critical_sites.iter().enumerate() {
        let comma = if i + 1 < critical_sites.len() { "," } else { "" };
        out.push_str(&format!(
            "    {{\"pos\": {}, \"wt\": \"{}\", \"mean_delta_logp\": {:.3}, \"category\": \"{}\"}}{}\n",
            r.pos, r.wt, r.mean_delta_logp, crate::mcp::escape_json_str(r.category), comma
        ));
    }
    out.push_str("  ],\n");

    out.push_str("  \"positions\": [\n");
    for (i, r) in rows.iter().enumerate() {
        let comma = if i + 1 < rows.len() { "," } else { "" };
        out.push_str(&format!(
            "    {{\"pos\": {}, \"wt\": \"{}\", \"mean_delta_logp\": {:.3}, \"entropy\": {:.3}, \"category\": \"{}\", \"substitutions\": {{",
            r.pos, r.wt, r.mean_delta_logp, r.entropy, crate::mcp::escape_json_str(r.category)
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
    out.push_str(&format!("  \"protein_id\": \"{}\",\n", crate::mcp::escape_json_str(protein_id)));
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
    out.push_str(&format!("  \"protein_id\": \"{}\",\n", crate::mcp::escape_json_str(protein_id)));
    out.push_str(&format!("  \"sequence_length\": {},\n", original_sequence.len()));
    out.push_str(&format!("  \"original_sequence\": \"{}\",\n", crate::mcp::escape_json_str(original_sequence)));
    out.push_str(&format!("  \"reconstructed_sequence\": \"{}\",\n", crate::mcp::escape_json_str(reconstructed_sequence)));
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
                cand.amino_acid, cand.probability_pct, crate::mcp::escape_json_str(cand.category), cand_comma
            ));
        }
        out.push_str(&format!("]}}{}\n", comma));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

/// Exports inpainting motif reconstruction results to a JSON file.
pub fn export_inpaint_json(
    path: &str,
    protein_id: &str,
    original_sequence: &str,
    reconstructed_sequence: &str,
    sites: &[crate::pipeline::InpaintSite],
) -> io::Result<()> {
    let mut file = File::create(path)?;
    let json = inpaint_to_json_string(protein_id, original_sequence, reconstructed_sequence, sites);
    file.write_all(json.as_bytes())?;
    file.flush()?;
    Ok(())
}

/// Converts standard single-letter amino acid code to three-letter IUPAC representation.
pub fn aa1_to_aa3(aa: char) -> &'static str {
    match aa.to_ascii_uppercase() {
        'A' => "ALA",
        'C' => "CYS",
        'D' => "ASP",
        'E' => "GLU",
        'F' => "PHE",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'K' => "LYS",
        'L' => "LEU",
        'M' => "MET",
        'N' => "ASN",
        'P' => "PRO",
        'Q' => "GLN",
        'R' => "ARG",
        'S' => "SER",
        'T' => "THR",
        'V' => "VAL",
        'W' => "TRP",
        'Y' => "TYR",
        _ => "UNK",
    }
}

/// Generates a standard Protein Data Bank (.pdb) string containing C_alpha backbone atoms with optional B-factors.
pub fn coords_to_pdb_string_with_bfactors(
    sequence: &str,
    protein_id: &str,
    coords: &[[f64; 3]],
    b_factors: Option<&[f64]>,
) -> String {
    let mut out = String::with_capacity(coords.len() * 82 + 256);
    out.push_str(&format!("HEADER    CODON-INFER STRUCT-2 3D RECONSTRUCTION {}\n", protein_id));
    out.push_str("TITLE     C-ALPHA BACKBONE SOLVED VIA ZERO-TRUNK MDS + GRADIENT DESCENT\n");
    for (i, (ch, pt)) in sequence.chars().zip(coords.iter()).enumerate() {
        let res_num = (i + 1) as i32;
        let res_name = aa1_to_aa3(ch);
        let bf = b_factors.and_then(|b| b.get(i).copied()).unwrap_or(50.0);
        out.push_str(&format!(
            "ATOM  {:5}  CA  {:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}           C\n",
            res_num,
            res_name,
            res_num,
            pt[0],
            pt[1],
            pt[2],
            bf
        ));
    }
    out.push_str("END\n");
    out
}

/// Generates a standard Protein Data Bank (.pdb) string containing C_alpha backbone atoms.
pub fn coords_to_pdb_string(
    sequence: &str,
    protein_id: &str,
    coords: &[[f64; 3]],
) -> String {
    coords_to_pdb_string_with_bfactors(sequence, protein_id, coords, None)
}

/// Exports reconstructed 3D C_alpha coordinates to standard Protein Data Bank (.pdb) format with optional B-factors.
pub fn export_ca_pdb_with_bfactors(
    path: &str,
    sequence: &str,
    protein_id: &str,
    coords: &[[f64; 3]],
    b_factors: Option<&[f64]>,
) -> io::Result<String> {
    let pdb_content = coords_to_pdb_string_with_bfactors(sequence, protein_id, coords, b_factors);
    let mut file = File::create(path)?;
    file.write_all(pdb_content.as_bytes())?;
    file.flush()?;
    Ok(pdb_content)
}

/// Exports reconstructed 3D C_alpha coordinates to standard Protein Data Bank (.pdb) format.
pub fn export_ca_pdb(
    path: &str,
    sequence: &str,
    protein_id: &str,
    coords: &[[f64; 3]],
) -> io::Result<String> {
    export_ca_pdb_with_bfactors(path, sequence, protein_id, coords, None)
}

/// Exports 3D coordinates to a tab-separated values (.tsv) file.
pub fn export_fold_tsv(
    path: &str,
    _protein_id: &str,
    sequence: &str,
    coords: &[[f64; 3]],
) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "residue_index\tamino_acid\tresname_3\tx\ty\tz")?;
    for (i, (ch, pt)) in sequence.chars().zip(coords.iter()).enumerate() {
        writeln!(
            file,
            "{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}",
            i + 1,
            ch,
            aa1_to_aa3(ch),
            pt[0],
            pt[1],
            pt[2]
        )?;
    }
    file.flush()?;
    Ok(())
}

/// Serializes 3D folding result into structured JSON string.
pub fn fold_to_json_string(
    result: &crate::pipeline::FoldPipelineResult,
    pdb_content: &str,
) -> String {
    let mut out = String::with_capacity(result.coords.len() * 120 + pdb_content.len() + 512);
    out.push_str("{\n");
    out.push_str(&format!("  \"protein_id\": \"{}\",\n", crate::mcp::escape_json_str(&result.protein_id)));
    out.push_str(&format!("  \"sequence_length\": {},\n", result.sequence.len()));
    out.push_str(&format!("  \"sequence\": \"{}\",\n", crate::mcp::escape_json_str(&result.sequence)));
    out.push_str(&format!("  \"num_restraints\": {},\n", result.num_restraints));
    out.push_str(&format!("  \"p_max\": {:.4},\n", result.p_max));
    out.push_str(&format!("  \"forward_ms\": {:.2},\n", result.forward_ms));
    out.push_str(&format!("  \"solve_ms\": {:.2},\n", result.solve_ms));
    out.push_str(&format!("  \"total_ms\": {:.2},\n", result.forward_ms + result.solve_ms));

    let mean_plddt = if !result.plddt.is_empty() {
        result.plddt.iter().sum::<f64>() / result.plddt.len() as f64
    } else {
        50.0
    };
    out.push_str(&format!("  \"mean_plddt\": {:.2},\n", mean_plddt));

    if let Some(ref m) = result.sidechain_metrics {
        out.push_str("  \"sidechain_packing\": {\n");
        out.push_str(&format!("    \"initial_clashscore\": {:.2},\n", m.initial_clashscore));
        out.push_str(&format!("    \"final_clashscore\": {:.2},\n", m.final_clashscore));
        out.push_str(&format!("    \"clash_reduction_pct\": {:.2},\n", m.clash_reduction_pct));
        out.push_str(&format!("    \"pruned_states\": {},\n", m.pruned_states));
        out.push_str(&format!("    \"total_atoms\": {},\n", m.total_atoms));
        out.push_str(&format!("    \"pack_time_ms\": {:.2}\n", m.pack_time_ms));
        out.push_str("  },\n");
    }

    // Coordinates array
    out.push_str("  \"coordinates\": [\n");
    for (i, (ch, pt)) in result.sequence.chars().zip(result.coords.iter()).enumerate() {
        let comma = if i + 1 < result.coords.len() { "," } else { "" };
        let plddt_val = result.plddt.get(i).copied().unwrap_or(50.0);
        out.push_str(&format!(
            "    {{\"index\": {}, \"res\": \"{}\", \"x\": {:.3}, \"y\": {:.3}, \"z\": {:.3}, \"plddt\": {:.2}}}{}\n",
            i + 1,
            ch,
            pt[0],
            pt[1],
            pt[2],
            plddt_val,
            comma
        ));
    }
    out.push_str("  ],\n");

    out.push_str("  \"pdb\": \"");
    out.push_str(&crate::mcp::escape_json_str(pdb_content));
    out.push_str("\"\n");
    out.push_str("}\n");
    out
}

/// Exports 3D folding result to structured JSON file.
pub fn export_fold_json(
    path: &str,
    result: &crate::pipeline::FoldPipelineResult,
) -> io::Result<()> {
    let pdb_content = if let Some(ref allatom) = result.allatom_residues {
        coords_to_allatom_pdb_string(&result.protein_id, allatom)
    } else {
        coords_to_pdb_string_with_bfactors(&result.sequence, &result.protein_id, &result.coords, Some(&result.plddt))
    };
    let json = fold_to_json_string(result, &pdb_content);
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    file.flush()?;
    Ok(())
}

/// Serializes inverse folding design results to structured JSON string.
pub fn design_to_json_string(result: &crate::pipeline::DesignPipelineResult) -> String {
    let mut out = String::with_capacity(result.sites.len() * 250 + 256);
    out.push_str("{\n");
    out.push_str(&format!("  \"protein_id\": \"{}\",\n", crate::mcp::escape_json_str(&result.protein_id)));
    out.push_str(&format!("  \"sequence_length\": {},\n", result.native_sequence.len()));
    out.push_str(&format!("  \"native_sequence\": \"{}\",\n", crate::mcp::escape_json_str(&result.native_sequence)));
    out.push_str(&format!("  \"designed_sequence\": \"{}\",\n", crate::mcp::escape_json_str(&result.designed_sequence)));
    out.push_str(&format!("  \"sequence_recovery_pct\": {:.2},\n", result.sequence_recovery_pct));
    out.push_str(&format!("  \"physicochemical_recovery_pct\": {:.2},\n", result.physicochemical_recovery_pct));
    out.push_str(&format!("  \"potential_3d_us\": {:.2},\n", result.potential_3d_us));
    out.push_str(&format!("  \"forward_ms\": {:.2},\n", result.forward_ms));
    out.push_str(&format!("  \"total_ms\": {:.2},\n", result.total_ms));

    out.push_str("  \"positions\": [\n");
    for (i, site) in result.sites.iter().enumerate() {
        let comma = if i + 1 < result.sites.len() { "," } else { "" };
        out.push_str(&format!(
            "    {{\"position\": {}, \"native\": \"{}\", \"designed\": \"{}\", \"is_match\": {}, \"candidates\": [",
            site.position, site.native_char, site.designed_char, site.native_char == site.designed_char
        ));
        for (j, cand) in site.candidates.iter().take(5).enumerate() {
            let cand_comma = if j + 1 < 5.min(site.candidates.len()) { ", " } else { "" };
            out.push_str(&format!(
                "{{\"amino_acid\": \"{}\", \"score\": {:.3}, \"probability_pct\": {:.2}, \"category\": \"{}\"}}{}",
                cand.amino_acid, cand.score, cand.probability_pct, crate::mcp::escape_json_str(cand.category), cand_comma
            ));
        }
        out.push_str(&format!("]}}{}\n", comma));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

/// Exports inverse folding design results to structured JSON file.
pub fn export_design_json(path: &str, result: &crate::pipeline::DesignPipelineResult) -> io::Result<()> {
    let mut file = File::create(path)?;
    let json = design_to_json_string(result);
    file.write_all(json.as_bytes())?;
    file.flush()?;
    Ok(())
}

// ============================================================================
// ALL-ATOM KINEMATIC BACKBONE & C-BETA RECONSTRUCTION (STRUCT-3)
// ============================================================================

/// 3D Vector for zero-allocation biophysical coordinate transformations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3(pub [f64; 3]);

impl Vec3 {
    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64) -> Self { Vec3([x, y, z]) }
    #[inline(always)]
    pub fn zero() -> Self { Vec3([0.0, 0.0, 0.0]) }
    #[inline(always)]
    pub fn from_array(a: [f64; 3]) -> Self { Vec3(a) }
    #[inline(always)]
    pub fn to_array(self) -> [f64; 3] { self.0 }

    #[inline(always)]
    pub fn dot(self, o: Self) -> f64 { self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2] }
    #[inline(always)]
    pub fn cross(self, o: Self) -> Self {
        Vec3([
            self.0[1] * o.0[2] - self.0[2] * o.0[1],
            self.0[2] * o.0[0] - self.0[0] * o.0[2],
            self.0[0] * o.0[1] - self.0[1] * o.0[0],
        ])
    }
    #[inline(always)]
    pub fn norm_sq(self) -> f64 { self.dot(self) }
    #[inline(always)]
    pub fn norm(self) -> f64 { self.norm_sq().sqrt() }
    #[inline(always)]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        if n < 1e-12 { Vec3::zero() } else { Vec3([self.0[0] / n, self.0[1] / n, self.0[2] / n]) }
    }
    #[inline(always)]
    pub fn add(self, o: Self) -> Self { Vec3([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]]) }
    #[inline(always)]
    pub fn sub(self, o: Self) -> Self { Vec3([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]]) }
    #[inline(always)]
    pub fn scale(self, s: f64) -> Self { Vec3([self.0[0] * s, self.0[1] * s, self.0[2] * s]) }
    #[inline(always)]
    pub fn dist(self, o: Self) -> f64 { self.sub(o).norm() }
}

/// Converts standard three-letter IUPAC representation to single-letter amino acid code.
pub fn aa3_to_aa1(res3: &str) -> char {
    match res3.trim().to_ascii_uppercase().as_str() {
        "ALA" => 'A', "CYS" => 'C', "ASP" => 'D', "GLU" => 'E',
        "PHE" => 'F', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
        "LYS" => 'K', "LEU" => 'L', "MET" => 'M', "ASN" => 'N',
        "PRO" => 'P', "GLN" => 'Q', "ARG" => 'R', "SER" => 'S',
        "THR" => 'T', "VAL" => 'V', "TRP" => 'W', "TYR" => 'Y',
        _ => 'X',
    }
}

/// Chemical interaction classification flags for heavy atoms (STRUCT-6).
pub const ATOM_POLAR_DONOR: u8    = 1 << 0; // H-bond donor (N-H, O-H)
pub const ATOM_POLAR_ACCEPTOR: u8 = 1 << 1; // H-bond acceptor (O, N lone pair)
pub const ATOM_CATION: u8         = 1 << 2; // Formal positive charge (Lys NZ, Arg NE/NH1/NH2, His ND1/NE2)
pub const ATOM_ANION: u8          = 1 << 3; // Formal negative charge (Asp OD1/OD2, Glu OE1/OE2)

/// Heavy atom within a sidechain residue beyond C_beta (STRUCT-4/6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SideChainAtom {
    pub name: [u8; 4],
    pub pos: [f64; 3],
    pub element: u8, // b'C', b'N', b'O', b'S'
    pub chem_flag: u8,
}

impl Default for SideChainAtom {
    fn default() -> Self {
        Self {
            name: [b' '; 4],
            pos: [0.0; 3],
            element: b'C',
            chem_flag: 0,
        }
    }
}

impl SideChainAtom {
    pub fn new(name: &str, pos: [f64; 3], element: u8) -> Self {
        Self::new_with_chem(name, pos, element, 0)
    }

    pub fn new_with_chem(name: &str, pos: [f64; 3], element: u8, chem_flag: u8) -> Self {
        let mut b = [b' '; 4];
        let bytes = name.as_bytes();
        let len = bytes.len().min(4);
        b[..len].copy_from_slice(&bytes[..len]);
        Self { name: b, pos, element, chem_flag }
    }

    pub fn name_str(&self) -> &str {
        std::str::from_utf8(&self.name).unwrap_or("XXXX").trim()
    }
}

/// Classifies sidechain and backbone heavy atoms into chemical interaction types (STRUCT-6).
pub fn classify_chem_flag(res_name: &str, atom_name: &str) -> u8 {
    let a_clean = atom_name.trim();
    let r_clean = res_name.trim();

    // Backbone atoms
    if a_clean == "N" {
        return if r_clean == "PRO" { 0 } else { ATOM_POLAR_DONOR };
    }
    if a_clean == "O" {
        return ATOM_POLAR_ACCEPTOR;
    }

    match r_clean {
        "SER" => match a_clean {
            "OG" => ATOM_POLAR_DONOR | ATOM_POLAR_ACCEPTOR,
            _ => 0,
        },
        "THR" => match a_clean {
            "OG1" => ATOM_POLAR_DONOR | ATOM_POLAR_ACCEPTOR,
            _ => 0,
        },
        "TYR" => match a_clean {
            "OH" => ATOM_POLAR_DONOR | ATOM_POLAR_ACCEPTOR,
            _ => 0,
        },
        "TRP" => match a_clean {
            "NE1" => ATOM_POLAR_DONOR,
            _ => 0,
        },
        "HIS" => match a_clean {
            "ND1" | "NE2" => ATOM_POLAR_DONOR | ATOM_POLAR_ACCEPTOR,
            _ => 0,
        },
        "ASP" => match a_clean {
            "OD1" | "OD2" => ATOM_POLAR_ACCEPTOR | ATOM_ANION,
            _ => 0,
        },
        "GLU" => match a_clean {
            "OE1" | "OE2" => ATOM_POLAR_ACCEPTOR | ATOM_ANION,
            _ => 0,
        },
        "ASN" => match a_clean {
            "OD1" => ATOM_POLAR_ACCEPTOR,
            "ND2" => ATOM_POLAR_DONOR,
            _ => 0,
        },
        "GLN" => match a_clean {
            "OE1" => ATOM_POLAR_ACCEPTOR,
            "NE2" => ATOM_POLAR_DONOR,
            _ => 0,
        },
        "LYS" => match a_clean {
            "NZ" => ATOM_POLAR_DONOR | ATOM_CATION,
            _ => 0,
        },
        "ARG" => match a_clean {
            "NE" | "NH1" | "NH2" => ATOM_POLAR_DONOR | ATOM_CATION,
            _ => 0,
        },
        _ => 0,
    }
}

/// Returns the normalized fraction of formal charge carried by an ionizable heavy atom (STRUCT-6).
#[inline]
pub fn charge_weight(atom_name: &str) -> f64 {
    match atom_name.trim() {
        "NZ" => 1.0,
        "OD1" | "OD2" | "OE1" | "OE2" => 0.5,
        "NH1" | "NH2" => 0.4,
        "NE" => 0.2,
        _ => 1.0,
    }
}

/// Represents a fully reconstructed amino acid residue with all backbone heavy atoms (N, CA, C, O),
/// C_beta, and full sidechain heavy atoms (STRUCT-3/4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconstructedResidue {
    pub res_name: [u8; 3],
    pub res_seq: i32,
    pub n: [f64; 3],
    pub ca: [f64; 3],
    pub c: [f64; 3],
    pub o: [f64; 3],
    pub cb: Option<[f64; 3]>,
    pub b_factor: f64,
    pub sidechain_len: u8,
    pub sidechain_atoms: [SideChainAtom; 10],
}

impl Default for ReconstructedResidue {
    fn default() -> Self {
        Self {
            res_name: *b"UNK",
            res_seq: 0,
            n: [0.0; 3],
            ca: [0.0; 3],
            c: [0.0; 3],
            o: [0.0; 3],
            cb: None,
            b_factor: 50.0,
            sidechain_len: 0,
            sidechain_atoms: [SideChainAtom::default(); 10],
        }
    }
}

/// Canonical Ramachandran macro-basins for backbone-dependent rotamer conditioning (STRUCT-5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RamachandranBasin {
    /// Right-handed alpha-helix basin (φ ≈ -60°, ψ ≈ -45°)
    Alpha,
    /// Extended beta-strand / beta-sheet basin (φ ≈ -120°, ψ ≈ +135°)
    Beta,
    /// Polyproline-II left-handed extended basin (φ ≈ -65°, ψ ≈ +145°)
    Ppii,
    /// Left-handed alpha-helix / turn basin (φ ≈ +60°, ψ ≈ +40°)
    AlphaL,
    /// General / coil / transition conformational basin
    General,
}

/// Classifies backbone dihedral angles (in degrees) into a canonical Ramachandran macro-basin.
#[inline]
pub fn classify_ramachandran(phi_deg: f64, psi_deg: f64) -> RamachandranBasin {
    if phi_deg >= -120.0 && phi_deg <= -30.0 && psi_deg >= -80.0 && psi_deg <= -10.0 {
        RamachandranBasin::Alpha
    } else if phi_deg >= -180.0 && phi_deg <= -40.0 && (psi_deg >= 80.0 || psi_deg <= -150.0) {
        if phi_deg >= -90.0 && phi_deg <= -40.0 && psi_deg >= 100.0 && psi_deg <= 180.0 {
            RamachandranBasin::Ppii
        } else {
            RamachandranBasin::Beta
        }
    } else if phi_deg >= 30.0 && phi_deg <= 100.0 && psi_deg >= -20.0 && psi_deg <= 90.0 {
        RamachandranBasin::AlphaL
    } else {
        RamachandranBasin::General
    }
}

/// Analytical 4-point dihedral angle calculation returning degrees in [-180.0, 180.0].
#[inline]
pub fn calc_dihedral_deg(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let b0 = Vec3::from_array(p0).sub(Vec3::from_array(p1));
    let b1 = Vec3::from_array(p2).sub(Vec3::from_array(p1));
    let b2 = Vec3::from_array(p3).sub(Vec3::from_array(p2));

    let b1_u = b1.normalize();
    let v = Vec3([
        b0.0[0] - b1_u.0[0] * b0.dot(b1_u),
        b0.0[1] - b1_u.0[1] * b0.dot(b1_u),
        b0.0[2] - b1_u.0[2] * b0.dot(b1_u),
    ]);
    let w = Vec3([
        b2.0[0] - b1_u.0[0] * b2.dot(b1_u),
        b2.0[1] - b1_u.0[1] * b2.dot(b1_u),
        b2.0[2] - b1_u.0[2] * b2.dot(b1_u),
    ]);
    let x = v.dot(w);
    let y = b1_u.cross(v).dot(w);
    y.atan2(x).to_degrees()
}

#[derive(Clone, Copy, Debug)]
pub struct RotamerAngles {
    pub chi1: f64, // radians
    pub chi2: f64, // radians
    pub chi3: f64, // radians
    pub chi4: f64, // radians
    pub prob: f64, // intrinsic probability [0.0, 1.0]
}

pub fn get_rotamer_candidates_bbdep(res_name: &str, basin: RamachandranBasin) -> &'static [RotamerAngles] {
    match res_name {
        "VAL" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.88 }, // trans (180°)
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.10 }, // gauche- (-60°)
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.02 }, // gauche+ (+60°)
            ],
            RamachandranBasin::Beta | RamachandranBasin::Ppii => &[
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.58 },
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.38 },
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.04 },
            ],
            RamachandranBasin::AlphaL => &[
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.52 },
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.38 },
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.10 },
            ],
            RamachandranBasin::General => &[
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.75 },
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.15 },
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.10 },
            ],
        },
        "LEU" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.0543, chi3: 0.0, chi4: 0.0, prob: 0.74 }, // g-, t
                RotamerAngles { chi1: 3.0543,  chi2: 1.1345, chi3: 0.0, chi4: 0.0, prob: 0.18 }, // t, g+
                RotamerAngles { chi1: -1.1345, chi2: 1.1345, chi3: 0.0, chi4: 0.0, prob: 0.08 }, // g-, g+
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.0543,  chi2: 1.1345, chi3: 0.0, chi4: 0.0, prob: 0.44 }, // t, g+
                RotamerAngles { chi1: -1.1345, chi2: 3.0543, chi3: 0.0, chi4: 0.0, prob: 0.38 }, // g-, t
                RotamerAngles { chi1: -1.1345, chi2: 1.1345, chi3: 0.0, chi4: 0.0, prob: 0.12 }, // g-, g+
                RotamerAngles { chi1: 3.0543,  chi2: 3.0543, chi3: 0.0, chi4: 0.0, prob: 0.06 }, // t, t
            ],
            RamachandranBasin::Ppii => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.0543, chi3: 0.0, chi4: 0.0, prob: 0.62 },
                RotamerAngles { chi1: 3.0543,  chi2: 1.1345, chi3: 0.0, chi4: 0.0, prob: 0.28 },
                RotamerAngles { chi1: -1.1345, chi2: 1.1345, chi3: 0.0, chi4: 0.0, prob: 0.10 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.0543, chi3: 0.0, chi4: 0.0, prob: 0.58 }, // g-, t
                RotamerAngles { chi1: -1.1345, chi2: 1.1345, chi3: 0.0, chi4: 0.0, prob: 0.15 }, // g-, g+
                RotamerAngles { chi1: 3.0543,  chi2: 1.1345, chi3: 0.0, chi4: 0.0, prob: 0.13 }, // t, g+
                RotamerAngles { chi1: 3.0543,  chi2: 3.0543, chi3: 0.0, chi4: 0.0, prob: 0.08 }, // t, t
                RotamerAngles { chi1: 1.1345,  chi2: 1.1345, chi3: 0.0, chi4: 0.0, prob: 0.04 }, // g+, g+
                RotamerAngles { chi1: 1.1345,  chi2: 3.0543, chi3: 0.0, chi4: 0.0, prob: 0.02 }, // g+, t
            ],
        },
        "ILE" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.0472, chi2: 2.9670, chi3: 0.0, chi4: 0.0, prob: 0.82 }, // g-, t
                RotamerAngles { chi1: 2.9670,  chi2: 2.9670, chi3: 0.0, chi4: 0.0, prob: 0.12 }, // t, t
                RotamerAngles { chi1: -1.0472, chi2: -1.0472,chi3: 0.0, chi4: 0.0, prob: 0.06 }, // g-, g-
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 2.9670,  chi2: 2.9670, chi3: 0.0, chi4: 0.0, prob: 0.56 }, // t, t
                RotamerAngles { chi1: -1.0472, chi2: 2.9670, chi3: 0.0, chi4: 0.0, prob: 0.32 }, // g-, t
                RotamerAngles { chi1: -1.0472, chi2: -1.0472,chi3: 0.0, chi4: 0.0, prob: 0.08 }, // g-, g-
                RotamerAngles { chi1: 2.9670,  chi2: -1.0472,chi3: 0.0, chi4: 0.0, prob: 0.04 }, // t, g-
            ],
            RamachandranBasin::Ppii => &[
                RotamerAngles { chi1: -1.0472, chi2: 2.9670, chi3: 0.0, chi4: 0.0, prob: 0.58 },
                RotamerAngles { chi1: 2.9670,  chi2: 2.9670, chi3: 0.0, chi4: 0.0, prob: 0.34 },
                RotamerAngles { chi1: -1.0472, chi2: -1.0472,chi3: 0.0, chi4: 0.0, prob: 0.08 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.0472, chi2: 2.9670, chi3: 0.0, chi4: 0.0, prob: 0.60 }, // g-, t
                RotamerAngles { chi1: 2.9670,  chi2: 2.9670, chi3: 0.0, chi4: 0.0, prob: 0.16 }, // t, t
                RotamerAngles { chi1: -1.0472, chi2: -1.0472,chi3: 0.0, chi4: 0.0, prob: 0.12 }, // g-, g-
                RotamerAngles { chi1: 2.9670,  chi2: -1.0472,chi3: 0.0, chi4: 0.0, prob: 0.06 }, // t, g-
                RotamerAngles { chi1: 1.0472,  chi2: 2.9670, chi3: 0.0, chi4: 0.0, prob: 0.04 }, // g+, t
                RotamerAngles { chi1: 1.0472,  chi2: -1.0472,chi3: 0.0, chi4: 0.0, prob: 0.02 }, // g+, g-
            ],
        },
        "SER" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.62 },
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.28 },
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.10 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.48 },
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.34 },
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.18 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.45 },
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.20 },
            ],
        },
        "THR" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.68 },
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.28 },
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.04 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.58 },
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.32 },
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.10 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.48 },
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.45 },
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.07 },
            ],
        },
        "CYS" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.65 },
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.10 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.50 },
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.15 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.0472, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.50 },
                RotamerAngles { chi1: 3.14159, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 1.0472,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.15 },
            ],
        },
        "MET" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 3.14159, chi4: 0.0, prob: 0.55 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 3.14159, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 3.14159, chi4: 0.0, prob: 0.12 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 3.14159, chi4: 0.0, prob: 0.08 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 3.14159, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 3.14159, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 3.14159, chi4: 0.0, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 3.14159, chi4: 0.0, prob: 0.18 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 3.14159, chi4: 0.0, prob: 0.40 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 3.14159, chi4: 0.0, prob: 0.20 },
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 3.14159, chi4: 0.0, prob: 0.15 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 3.14159, chi4: 0.0, prob: 0.10 },
                RotamerAngles { chi1: -1.1345, chi2: 1.1345,  chi3: 3.14159, chi4: 0.0, prob: 0.08 },
                RotamerAngles { chi1: 1.1345,  chi2: 3.14159, chi3: 3.14159, chi4: 0.0, prob: 0.07 },
            ],
        },
        "ASP" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: -0.5236, chi3: 0.0, chi4: 0.0, prob: 0.48 },
                RotamerAngles { chi1: -1.1345, chi2: 0.5236,  chi3: 0.0, chi4: 0.0, prob: 0.32 },
                RotamerAngles { chi1: 3.14159, chi2: -0.5236, chi3: 0.0, chi4: 0.0, prob: 0.12 },
                RotamerAngles { chi1: 3.14159, chi2: 0.5236,  chi3: 0.0, chi4: 0.0, prob: 0.08 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: -0.5236, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 3.14159, chi2: 0.5236,  chi3: 0.0, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: -1.1345, chi2: -0.5236, chi3: 0.0, chi4: 0.0, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: 0.5236,  chi3: 0.0, chi4: 0.0, prob: 0.18 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: -0.5236, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: -1.1345, chi2: 0.5236,  chi3: 0.0, chi4: 0.0, prob: 0.30 },
                RotamerAngles { chi1: 3.14159, chi2: -0.5236, chi3: 0.0, chi4: 0.0, prob: 0.15 },
                RotamerAngles { chi1: 3.14159, chi2: 0.5236,  chi3: 0.0, chi4: 0.0, prob: 0.10 },
                RotamerAngles { chi1: 1.1345,  chi2: -0.5236, chi3: 0.0, chi4: 0.0, prob: 0.05 },
                RotamerAngles { chi1: 1.1345,  chi2: 0.5236,  chi3: 0.0, chi4: 0.0, prob: 0.05 },
            ],
        },
        "ASN" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: -0.3491, chi3: 0.0, chi4: 0.0, prob: 0.48 },
                RotamerAngles { chi1: -1.1345, chi2: 0.3491,  chi3: 0.0, chi4: 0.0, prob: 0.32 },
                RotamerAngles { chi1: 3.14159, chi2: -0.3491, chi3: 0.0, chi4: 0.0, prob: 0.12 },
                RotamerAngles { chi1: 3.14159, chi2: 0.3491,  chi3: 0.0, chi4: 0.0, prob: 0.08 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: -0.3491, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 3.14159, chi2: 0.3491,  chi3: 0.0, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: -1.1345, chi2: -0.3491, chi3: 0.0, chi4: 0.0, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: 0.3491,  chi3: 0.0, chi4: 0.0, prob: 0.18 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: -0.3491, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: -1.1345, chi2: 0.3491,  chi3: 0.0, chi4: 0.0, prob: 0.30 },
                RotamerAngles { chi1: 3.14159, chi2: -0.3491, chi3: 0.0, chi4: 0.0, prob: 0.15 },
                RotamerAngles { chi1: 3.14159, chi2: 0.3491,  chi3: 0.0, chi4: 0.0, prob: 0.10 },
                RotamerAngles { chi1: 1.1345,  chi2: -0.3491, chi3: 0.0, chi4: 0.0, prob: 0.05 },
                RotamerAngles { chi1: 1.1345,  chi2: 0.3491,  chi3: 0.0, chi4: 0.0, prob: 0.05 },
            ],
        },
        "GLU" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.50 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.30 },
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.12 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.08 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.18 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.15 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.10 },
                RotamerAngles { chi1: -1.1345, chi2: 1.1345,  chi3: 0.0, chi4: 0.0, prob: 0.08 },
                RotamerAngles { chi1: 1.1345,  chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.07 },
            ],
        },
        "GLN" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.50 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.30 },
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.12 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.08 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.18 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.15 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 0.0, chi4: 0.0, prob: 0.10 },
                RotamerAngles { chi1: -1.1345, chi2: 1.1345,  chi3: 0.0, chi4: 0.0, prob: 0.08 },
                RotamerAngles { chi1: 1.1345,  chi2: 3.14159, chi3: 0.0, chi4: 0.0, prob: 0.07 },
            ],
        },
        "LYS" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.55 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.25 },
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.12 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.08 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.35 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.25 },
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.18 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.40 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.20 },
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.15 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.10 },
                RotamerAngles { chi1: -1.1345, chi2: 1.1345,  chi3: 3.14159, chi4: 3.14159, prob: 0.08 },
                RotamerAngles { chi1: 1.1345,  chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.07 },
            ],
        },
        "ARG" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.55 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.25 },
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.12 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.08 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.35 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.25 },
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.18 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.40 },
                RotamerAngles { chi1: -1.1345, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.20 },
                RotamerAngles { chi1: 3.14159, chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.15 },
                RotamerAngles { chi1: 3.14159, chi2: -1.1345, chi3: 3.14159, chi4: 3.14159, prob: 0.10 },
                RotamerAngles { chi1: -1.1345, chi2: 1.1345,  chi3: 3.14159, chi4: 3.14159, prob: 0.08 },
                RotamerAngles { chi1: 1.1345,  chi2: 3.14159, chi3: 3.14159, chi4: 3.14159, prob: 0.07 },
            ],
        },
        "PHE" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.55 },
                RotamerAngles { chi1: -1.1345, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 3.14159, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.07 },
                RotamerAngles { chi1: 3.14159, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.03 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.42 },
                RotamerAngles { chi1: 3.14159, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.14 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.45 },
                RotamerAngles { chi1: -1.1345, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: 3.14159, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.15 },
                RotamerAngles { chi1: 3.14159, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.10 },
                RotamerAngles { chi1: 1.1345,  chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.03 },
                RotamerAngles { chi1: 1.1345,  chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.02 },
            ],
        },
        "TYR" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.55 },
                RotamerAngles { chi1: -1.1345, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 3.14159, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.07 },
                RotamerAngles { chi1: 3.14159, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.03 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.42 },
                RotamerAngles { chi1: 3.14159, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.14 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.45 },
                RotamerAngles { chi1: -1.1345, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: 3.14159, chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.15 },
                RotamerAngles { chi1: 3.14159, chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.10 },
                RotamerAngles { chi1: 1.1345,  chi2: 1.3963, chi3: 0.0, chi4: 0.0, prob: 0.03 },
                RotamerAngles { chi1: 1.1345,  chi2: 1.7453, chi3: 0.0, chi4: 0.0, prob: 0.02 },
            ],
        },
        "TRP" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: -1.5708, chi3: 0.0, chi4: 0.0, prob: 0.58 },
                RotamerAngles { chi1: -1.1345, chi2: 1.5708,  chi3: 0.0, chi4: 0.0, prob: 0.32 },
                RotamerAngles { chi1: 3.14159, chi2: -1.5708, chi3: 0.0, chi4: 0.0, prob: 0.07 },
                RotamerAngles { chi1: 3.14159, chi2: 1.5708,  chi3: 0.0, chi4: 0.0, prob: 0.03 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: -1.5708, chi3: 0.0, chi4: 0.0, prob: 0.38 },
                RotamerAngles { chi1: 3.14159, chi2: 1.5708,  chi3: 0.0, chi4: 0.0, prob: 0.26 },
                RotamerAngles { chi1: -1.1345, chi2: -1.5708, chi3: 0.0, chi4: 0.0, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: 1.5708,  chi3: 0.0, chi4: 0.0, prob: 0.14 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: -1.5708, chi3: 0.0, chi4: 0.0, prob: 0.45 },
                RotamerAngles { chi1: -1.1345, chi2: 1.5708,  chi3: 0.0, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: 3.14159, chi2: -1.5708, chi3: 0.0, chi4: 0.0, prob: 0.15 },
                RotamerAngles { chi1: 3.14159, chi2: 1.5708,  chi3: 0.0, chi4: 0.0, prob: 0.10 },
                RotamerAngles { chi1: 1.1345,  chi2: -1.5708, chi3: 0.0, chi4: 0.0, prob: 0.03 },
                RotamerAngles { chi1: 1.1345,  chi2: 1.5708,  chi3: 0.0, chi4: 0.0, prob: 0.02 },
            ],
        },
        "HIS" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -1.1345, chi2: -1.3090, chi3: 0.0, chi4: 0.0, prob: 0.50 },
                RotamerAngles { chi1: -1.1345, chi2: 1.3090,  chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 3.14159, chi2: -1.3090, chi3: 0.0, chi4: 0.0, prob: 0.10 },
                RotamerAngles { chi1: 3.14159, chi2: 1.3090,  chi3: 0.0, chi4: 0.0, prob: 0.05 },
            ],
            RamachandranBasin::Beta => &[
                RotamerAngles { chi1: 3.14159, chi2: -1.3090, chi3: 0.0, chi4: 0.0, prob: 0.35 },
                RotamerAngles { chi1: 3.14159, chi2: 1.3090,  chi3: 0.0, chi4: 0.0, prob: 0.25 },
                RotamerAngles { chi1: -1.1345, chi2: -1.3090, chi3: 0.0, chi4: 0.0, prob: 0.22 },
                RotamerAngles { chi1: -1.1345, chi2: 1.3090,  chi3: 0.0, chi4: 0.0, prob: 0.18 },
            ],
            _ => &[
                RotamerAngles { chi1: -1.1345, chi2: -1.3090, chi3: 0.0, chi4: 0.0, prob: 0.40 },
                RotamerAngles { chi1: -1.1345, chi2: 1.3090,  chi3: 0.0, chi4: 0.0, prob: 0.30 },
                RotamerAngles { chi1: 3.14159, chi2: -1.3090, chi3: 0.0, chi4: 0.0, prob: 0.15 },
                RotamerAngles { chi1: 3.14159, chi2: 1.3090,  chi3: 0.0, chi4: 0.0, prob: 0.10 },
                RotamerAngles { chi1: 1.1345,  chi2: -1.3090, chi3: 0.0, chi4: 0.0, prob: 0.03 },
                RotamerAngles { chi1: 1.1345,  chi2: 1.3090,  chi3: 0.0, chi4: 0.0, prob: 0.02 },
            ],
        },
        "PRO" => match basin {
            RamachandranBasin::Alpha => &[
                RotamerAngles { chi1: -0.4363, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.75 }, // Cgamma-endo
                RotamerAngles { chi1: 0.4363,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.25 }, // Cgamma-exo
            ],
            RamachandranBasin::Beta | RamachandranBasin::Ppii => &[
                RotamerAngles { chi1: -0.4363, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.55 },
                RotamerAngles { chi1: 0.4363,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.45 },
            ],
            _ => &[
                RotamerAngles { chi1: -0.4363, chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.65 },
                RotamerAngles { chi1: 0.4363,  chi2: 0.0, chi3: 0.0, chi4: 0.0, prob: 0.35 },
            ],
        },
        _ => &[],
    }
}

pub fn get_rotamer_candidates(res_name: &str) -> &'static [RotamerAngles] {
    get_rotamer_candidates_bbdep(res_name, RamachandranBasin::General)
}

pub fn nerf_place_atom(a: Vec3, b: Vec3, c: Vec3, l: f64, theta: f64, chi: f64) -> Vec3 {
    let bc = c.sub(b).normalize();
    let ab = b.sub(a).normalize();
    let mut n = ab.cross(bc);
    if n.norm() < 1e-8 {
        n = Vec3::new(1.0, 0.0, 0.0).cross(bc);
        if n.norm() < 1e-8 {
            n = Vec3::new(0.0, 1.0, 0.0).cross(bc);
        }
    }
    let n = n.normalize();
    let n_perp = n.cross(bc);

    let d_vec = bc.scale(-theta.cos())
        .add(n_perp.scale(theta.sin() * chi.cos()))
        .add(n.scale(theta.sin() * chi.sin()))
        .scale(l);
    c.add(d_vec)
}

const L_CC: f64 = 1.53;
const L_CO: f64 = 1.42;
const L_CS: f64 = 1.81;
const L_CN: f64 = 1.48;
const L_RING: f64 = 1.39;

const T_TETRA: f64 = 1.91986; // 110.0° in radians
const T_PLANAR: f64 = 2.09440; // 120.0° in radians

pub fn build_sidechain_atoms_into(
    res_name: &str,
    n_pos: Vec3,
    ca_pos: Vec3,
    cb_pos: Vec3,
    rot: &RotamerAngles,
    out: &mut [SideChainAtom; 10],
) -> u8 {
    let chi1 = rot.chi1;
    let chi2 = rot.chi2;
    let chi3 = rot.chi3;
    let chi4 = rot.chi4;
    let mut len = 0usize;

    let mut push_atom = |name: &str, pos: Vec3, elem: u8| {
        if len < 10 {
            let chem = classify_chem_flag(res_name, name);
            out[len] = SideChainAtom::new_with_chem(name, pos.to_array(), elem, chem);
            len += 1;
        }
    };

    match res_name {
        "VAL" => {
            let cg1 = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let cg2 = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1 + 2.0944);
            push_atom("CG1", cg1, b'C');
            push_atom("CG2", cg2, b'C');
        }
        "LEU" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let cd1 = nerf_place_atom(ca_pos, cb_pos, cg, L_CC, T_TETRA, chi2);
            let cd2 = nerf_place_atom(ca_pos, cb_pos, cg, L_CC, T_TETRA, chi2 + 2.0944);
            push_atom("CG", cg, b'C');
            push_atom("CD1", cd1, b'C');
            push_atom("CD2", cd2, b'C');
        }
        "ILE" => {
            let cg1 = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let cg2 = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1 + 2.0944);
            let cd1 = nerf_place_atom(ca_pos, cb_pos, cg1, L_CC, T_TETRA, chi2);
            push_atom("CG1", cg1, b'C');
            push_atom("CG2", cg2, b'C');
            push_atom("CD1", cd1, b'C');
        }
        "SER" => {
            let og = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CO, T_TETRA, chi1);
            push_atom("OG", og, b'O');
        }
        "THR" => {
            let og1 = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CO, T_TETRA, chi1);
            let cg2 = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1 + 2.0944);
            push_atom("OG1", og1, b'O');
            push_atom("CG2", cg2, b'C');
        }
        "CYS" => {
            let sg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CS, T_TETRA, chi1);
            push_atom("SG", sg, b'S');
        }
        "MET" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let sd = nerf_place_atom(ca_pos, cb_pos, cg, L_CS, T_TETRA, chi2);
            let ce = nerf_place_atom(cb_pos, cg, sd, L_CS, 1.7628, chi3);
            push_atom("CG", cg, b'C');
            push_atom("SD", sd, b'S');
            push_atom("CE", ce, b'C');
        }
        "ASP" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let od1 = nerf_place_atom(ca_pos, cb_pos, cg, 1.25, T_PLANAR, chi2);
            let od2 = nerf_place_atom(ca_pos, cb_pos, cg, 1.25, T_PLANAR, chi2 + 3.14159);
            push_atom("CG", cg, b'C');
            push_atom("OD1", od1, b'O');
            push_atom("OD2", od2, b'O');
        }
        "ASN" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let od1 = nerf_place_atom(ca_pos, cb_pos, cg, 1.24, T_PLANAR, chi2);
            let nd2 = nerf_place_atom(ca_pos, cb_pos, cg, 1.33, T_PLANAR, chi2 + 3.14159);
            push_atom("CG", cg, b'C');
            push_atom("OD1", od1, b'O');
            push_atom("ND2", nd2, b'N');
        }
        "GLU" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let cd = nerf_place_atom(ca_pos, cb_pos, cg, L_CC, T_TETRA, chi2);
            let oe1 = nerf_place_atom(cb_pos, cg, cd, 1.25, T_PLANAR, chi3);
            let oe2 = nerf_place_atom(cb_pos, cg, cd, 1.25, T_PLANAR, chi3 + 3.14159);
            push_atom("CG", cg, b'C');
            push_atom("CD", cd, b'C');
            push_atom("OE1", oe1, b'O');
            push_atom("OE2", oe2, b'O');
        }
        "GLN" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let cd = nerf_place_atom(ca_pos, cb_pos, cg, L_CC, T_TETRA, chi2);
            let oe1 = nerf_place_atom(cb_pos, cg, cd, 1.24, T_PLANAR, chi3);
            let ne2 = nerf_place_atom(cb_pos, cg, cd, 1.33, T_PLANAR, chi3 + 3.14159);
            push_atom("CG", cg, b'C');
            push_atom("CD", cd, b'C');
            push_atom("OE1", oe1, b'O');
            push_atom("NE2", ne2, b'N');
        }
        "LYS" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let cd = nerf_place_atom(ca_pos, cb_pos, cg, L_CC, T_TETRA, chi2);
            let ce = nerf_place_atom(cb_pos, cg, cd, L_CC, T_TETRA, chi3);
            let nz = nerf_place_atom(cg, cd, ce, L_CN, T_TETRA, chi4);
            push_atom("CG", cg, b'C');
            push_atom("CD", cd, b'C');
            push_atom("CE", ce, b'C');
            push_atom("NZ", nz, b'N');
        }
        "ARG" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let cd = nerf_place_atom(ca_pos, cb_pos, cg, L_CC, T_TETRA, chi2);
            let ne = nerf_place_atom(cb_pos, cg, cd, L_CN, T_TETRA, chi3);
            let cz = nerf_place_atom(cg, cd, ne, 1.33, T_PLANAR, chi4);
            let nh1 = nerf_place_atom(cd, ne, cz, 1.33, T_PLANAR, 0.0);
            let nh2 = nerf_place_atom(cd, ne, cz, 1.33, T_PLANAR, 3.14159);
            push_atom("CG", cg, b'C');
            push_atom("CD", cd, b'C');
            push_atom("NE", ne, b'N');
            push_atom("CZ", cz, b'C');
            push_atom("NH1", nh1, b'N');
            push_atom("NH2", nh2, b'N');
        }
        "PHE" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let cd1 = nerf_place_atom(ca_pos, cb_pos, cg, L_RING, T_PLANAR, chi2);
            let cd2 = nerf_place_atom(ca_pos, cb_pos, cg, L_RING, T_PLANAR, chi2 + 3.14159);
            let ce1 = nerf_place_atom(cb_pos, cg, cd1, L_RING, T_PLANAR, 3.14159);
            let ce2 = nerf_place_atom(cb_pos, cg, cd2, L_RING, T_PLANAR, 3.14159);
            let cz = nerf_place_atom(cg, cd1, ce1, L_RING, T_PLANAR, 0.0);
            push_atom("CG", cg, b'C');
            push_atom("CD1", cd1, b'C');
            push_atom("CD2", cd2, b'C');
            push_atom("CE1", ce1, b'C');
            push_atom("CE2", ce2, b'C');
            push_atom("CZ", cz, b'C');
        }
        "TYR" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let cd1 = nerf_place_atom(ca_pos, cb_pos, cg, L_RING, T_PLANAR, chi2);
            let cd2 = nerf_place_atom(ca_pos, cb_pos, cg, L_RING, T_PLANAR, chi2 + 3.14159);
            let ce1 = nerf_place_atom(cb_pos, cg, cd1, L_RING, T_PLANAR, 3.14159);
            let ce2 = nerf_place_atom(cb_pos, cg, cd2, L_RING, T_PLANAR, 3.14159);
            let cz = nerf_place_atom(cg, cd1, ce1, L_RING, T_PLANAR, 0.0);
            let oh = nerf_place_atom(cd1, ce1, cz, 1.36, T_PLANAR, 3.14159);
            push_atom("CG", cg, b'C');
            push_atom("CD1", cd1, b'C');
            push_atom("CD2", cd2, b'C');
            push_atom("CE1", ce1, b'C');
            push_atom("CE2", ce2, b'C');
            push_atom("CZ", cz, b'C');
            push_atom("OH", oh, b'O');
        }
        "TRP" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let cd1 = nerf_place_atom(ca_pos, cb_pos, cg, 1.37, 2.21, chi2);
            let cd2 = nerf_place_atom(ca_pos, cb_pos, cg, 1.44, 2.21, chi2 + 3.14159);
            let ne1 = nerf_place_atom(cb_pos, cg, cd1, 1.38, 1.88, 0.0);
            let ce2 = nerf_place_atom(cg, cd1, ne1, 1.38, 1.92, 0.0);
            let ce3 = nerf_place_atom(cg, cd2, ce2, 1.40, T_PLANAR, 3.14159);
            let cz2 = nerf_place_atom(cd2, ce2, ne1, 1.40, T_PLANAR, 3.14159);
            let cz3 = nerf_place_atom(cd2, ce3, cz2, 1.40, T_PLANAR, 0.0);
            let ch2 = nerf_place_atom(ce3, cz3, cz2, 1.40, T_PLANAR, 0.0);
            push_atom("CG", cg, b'C');
            push_atom("CD1", cd1, b'C');
            push_atom("CD2", cd2, b'C');
            push_atom("NE1", ne1, b'N');
            push_atom("CE2", ce2, b'C');
            push_atom("CE3", ce3, b'C');
            push_atom("CZ2", cz2, b'C');
            push_atom("CZ3", cz3, b'C');
            push_atom("CH2", ch2, b'C');
        }
        "HIS" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, T_TETRA, chi1);
            let nd1 = nerf_place_atom(ca_pos, cb_pos, cg, 1.38, 2.14, chi2);
            let cd2 = nerf_place_atom(ca_pos, cb_pos, cg, 1.36, 2.14, chi2 + 3.14159);
            let ce1 = nerf_place_atom(cb_pos, cg, nd1, 1.32, 1.90, 0.0);
            let ne2 = nerf_place_atom(cg, nd1, ce1, 1.38, 1.95, 0.0);
            push_atom("CG", cg, b'C');
            push_atom("ND1", nd1, b'N');
            push_atom("CD2", cd2, b'C');
            push_atom("CE1", ce1, b'C');
            push_atom("NE2", ne2, b'N');
        }
        "PRO" => {
            let cg = nerf_place_atom(n_pos, ca_pos, cb_pos, L_CC, 1.80, chi1);
            let cd = nerf_place_atom(ca_pos, cb_pos, cg, L_CC, 1.80, -0.4);
            push_atom("CG", cg, b'C');
            push_atom("CD", cd, b'C');
        }
        _ => {}
    }

    len as u8
}

#[inline(always)]
pub fn vdw_radius(element: u8) -> f64 {
    match element {
        b'C' => 1.70,
        b'N' => 1.55,
        b'O' => 1.52,
        b'S' => 1.80,
        _ => 1.70,
    }
}

#[inline(always)]
pub fn soft_core_steric(dist: f64, elem_a: u8, elem_b: u8, alpha_soft: f64, w_clash: f64) -> f64 {
    let r_ab = vdw_radius(elem_a) + vdw_radius(elem_b);
    let r_cut = alpha_soft * r_ab;
    let mut energy = 0.0;
    if dist < r_cut {
        let pen = 1.0 - (dist / r_cut);
        energy += w_clash * pen * pen;
    }
    // Hard steric barrier at the MolProbity clash threshold (STRUCT-6):
    // Guarantees physical non-penetration and prevents attractive terms (H-bonds, salt bridges)
    // from overriding steric exclusion below 2.0 Å.
    if dist < CLASH_DIST_THRESHOLD {
        let pen_clash = 1.0 - (dist / CLASH_DIST_THRESHOLD);
        energy += 50.0 * pen_clash;
    }
    energy
}

/// Radial 10-12 Kortemme-Baker attractive hydrogen bond potential (STRUCT-6).
/// Active window: r in [2.30, 3.40] Å, optimal r0 = 2.80 Å.
/// Returns a negative stabilizing energy in kcal/mol.
#[inline]
pub fn hbond_energy(r: f64, ang_factor: f64, w_hb: f64) -> f64 {
    if r < 2.30 || r > 3.40 || ang_factor <= 0.0 || w_hb <= 0.0 {
        return 0.0;
    }
    let r0 = 2.80;
    let inv_r = r0 / r;
    let inv_r2 = inv_r * inv_r;
    let inv_r10 = inv_r2 * inv_r2 * inv_r2 * inv_r2 * inv_r2;
    let inv_r12 = inv_r10 * inv_r2;
    // Normalized 10-12 well: at r = r0, 5*(1) - 6*(1) = -1.0
    let f_rad = (5.0 * inv_r12 - 6.0 * inv_r10).clamp(-1.0, 0.0);

    // Smoothstep taper between 3.0 Å and 3.4 Å for C1 continuity
    let taper = if r > 3.00 {
        let t = (3.40 - r) / 0.40;
        t * t * (3.0 - 2.0 * t)
    } else {
        1.0
    };

    -w_hb * f_rad.abs() * ang_factor * taper
}

/// Screened electrostatic salt bridge interaction between charged residues (STRUCT-6).
/// Active window: r in [2.30, 4.20] Å, optimal r_opt = 2.90 Å, sigma = 0.60 Å.
/// Returns negative energy for attraction (cation-anion).
#[inline]
pub fn salt_bridge_energy(r: f64, w_salt: f64) -> f64 {
    if r < 2.30 || r > 4.20 || w_salt <= 0.0 {
        return 0.0;
    }
    let r_opt = 2.90;
    let sigma = 0.60;
    let delta = (r - r_opt) / sigma;
    let lorentz = 1.0 / (1.0 + delta * delta);

    // Smoothstep taper between 3.5 Å and 4.2 Å
    let taper = if r > 3.50 {
        let t = (4.20 - r) / 0.70;
        t * t * (3.0 - 2.0 * t)
    } else {
        1.0
    };

    -w_salt * lorentz * taper
}

/// Like-charge electrostatic repulsion between basic-basic or acidic-acidic groups (STRUCT-6).
#[inline]
pub fn like_charge_repulsion(r: f64, w_salt: f64) -> f64 {
    -0.5 * salt_bridge_energy(r, w_salt)
}

pub fn is_bonded_residue_atoms(res_i: i32, name_i: &str, res_j: i32, name_j: &str) -> bool {
    let diff = (res_i - res_j).abs();
    if diff == 0 {
        if (name_i == "N" && (name_j == "CA" || name_j == "C" || name_j == "CB" || name_j == "CD")) ||
           (name_j == "N" && (name_i == "CA" || name_i == "C" || name_i == "CB" || name_i == "CD")) {
            return true;
        }
        if (name_i == "CA" && (name_j == "C" || name_j == "O" || name_j == "CB" || name_j == "CD" || name_j.starts_with("CG") || name_j.starts_with("OG") || name_j.starts_with("SG"))) ||
           (name_j == "CA" && (name_i == "C" || name_i == "O" || name_i == "CB" || name_i == "CD" || name_i.starts_with("CG") || name_i.starts_with("OG") || name_i.starts_with("SG"))) {
            return true;
        }
        if (name_i == "C" && (name_j == "O" || name_j == "CB")) ||
           (name_j == "C" && (name_i == "O" || name_i == "CB")) {
            return true;
        }
        if (name_i == "CB" && (name_j.starts_with("CG") || name_j.starts_with("OG") || name_j.starts_with("SG") || name_j.starts_with("CD") || name_j.starts_with("OD") || name_j.starts_with("SD") || name_j.starts_with("ND"))) ||
           (name_j == "CB" && (name_i.starts_with("CG") || name_i.starts_with("OG") || name_i.starts_with("SG") || name_i.starts_with("CD") || name_i.starts_with("OD") || name_i.starts_with("SD") || name_i.starts_with("ND"))) {
            return true;
        }
        if (name_i.starts_with("CG") && (name_j.starts_with("CD") || name_j.starts_with("OD") || name_j.starts_with("SD") || name_j.starts_with("ND") || name_j.starts_with("CE") || name_j.starts_with("OE") || name_j.starts_with("NE"))) ||
           (name_j == "CG" && (name_i.starts_with("CD") || name_i.starts_with("OD") || name_i.starts_with("SD") || name_i.starts_with("ND") || name_i.starts_with("CE") || name_i.starts_with("OE") || name_i.starts_with("NE"))) {
            return true;
        }
        if (name_i.starts_with("CD") && (name_j.starts_with("CE") || name_j.starts_with("OE") || name_j.starts_with("NE") || name_j.starts_with("CZ") || name_j.starts_with("NZ") || name_j.starts_with("NH") || name_j.starts_with("OH"))) ||
           (name_j == "CD" && (name_i.starts_with("CE") || name_i.starts_with("OE") || name_i.starts_with("NE") || name_i.starts_with("CZ") || name_i.starts_with("NZ") || name_i.starts_with("NH") || name_i.starts_with("OH"))) {
            return true;
        }
        if (name_i.starts_with("CE") && (name_j.starts_with("CZ") || name_j.starts_with("NZ") || name_j.starts_with("NH") || name_j.starts_with("OH") || name_j.starts_with("CH") || name_j.starts_with("ND") || name_j.starts_with("NE") || name_j.starts_with("SD"))) ||
           (name_j.starts_with("CE") && (name_i.starts_with("CZ") || name_i.starts_with("NZ") || name_i.starts_with("NH") || name_i.starts_with("OH") || name_i.starts_with("CH") || name_i.starts_with("ND") || name_i.starts_with("NE") || name_i.starts_with("SD"))) {
            return true;
        }
        if (name_i.starts_with("CZ") && (name_j.starts_with("NE") || name_j.starts_with("NH") || name_j.starts_with("OH") || name_j.starts_with("CH"))) ||
           (name_j.starts_with("CZ") && (name_i.starts_with("NE") || name_i.starts_with("NH") || name_i.starts_with("OH") || name_i.starts_with("CH"))) {
            return true;
        }
        if (name_i.starts_with("CH") && (name_j.starts_with("CZ") || name_j.starts_with("CE"))) ||
           (name_j.starts_with("CH") && (name_i.starts_with("CZ") || name_i.starts_with("CE"))) {
            return true;
        }
        if (name_i.starts_with("NE") && name_j.starts_with("NH")) ||
           (name_j.starts_with("NE") && name_i.starts_with("NH")) {
            return true;
        }
        if (name_i.starts_with("CG") && name_j.starts_with("CG")) ||
           (name_i.starts_with("CD") && name_j.starts_with("CD")) ||
           (name_i.starts_with("CE") && name_j.starts_with("CE")) ||
           (name_i.starts_with("OD") && name_j.starts_with("OD")) ||
           (name_i.starts_with("OE") && name_j.starts_with("OE")) ||
           (name_i.starts_with("NH") && name_j.starts_with("NH")) ||
           (name_i.starts_with("ND") && name_j.starts_with("ND")) {
            return true;
        }
    } else if diff == 1 {
        let (first_name, second_name) = if res_i < res_j { (name_i, name_j) } else { (name_j, name_i) };
        if (first_name == "C" && (second_name == "N" || second_name == "CA")) ||
           (first_name == "O" && (second_name == "N" || second_name == "CA")) ||
           (first_name == "CA" && second_name == "N") {
            return true;
        }
    }
    false
}

pub fn compute_protein_clashscore(residues: &[ReconstructedResidue]) -> (usize, f64) {
    let b = compute_clash_breakdown(residues);
    (b.total_clashes, b.clashscore)
}

/// Clash distance threshold (Å) shared by `compute_protein_clashscore` and `compute_clash_breakdown`.
pub const CLASH_DIST_THRESHOLD: f64 = 2.0;
/// Pairs closer than this (Å) that survive the bonded-exclusion filter are almost certainly
/// covalent bonds missing from `is_bonded_residue_atoms` (or coincident atoms), not steric clashes.
pub const COVALENT_LIKE_THRESHOLD: f64 = 1.65;

/// Categorized decomposition of the internal clashscore (STRUCT-5a diagnostics).
///
/// "Backbone" (`bb`) = N, CA, C, O, CB (the 5-atom frame fixed during rotamer packing).
/// "Sidechain" (`sc`) = heavy atoms beyond CB placed by the rotamer packer.
/// The category counters always sum to `total_clashes`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClashBreakdown {
    pub total_atoms: usize,
    pub total_clashes: usize,
    pub clashscore: f64,
    /// bb–bb with |i−j| ≤ 1 (intra-residue and adjacent peptide geometry).
    pub bb_bb_local: usize,
    /// bb–bb with 2 ≤ |i−j| ≤ 4 (secondary-structure / turn geometry).
    pub bb_bb_short: usize,
    /// bb–bb with |i−j| ≥ 5 (tertiary fold collisions).
    pub bb_bb_long: usize,
    pub sc_bb_intra: usize,
    pub sc_bb_inter: usize,
    pub sc_sc_intra: usize,
    pub sc_sc_inter: usize,
    /// Subset of `total_clashes` with d < `COVALENT_LIKE_THRESHOLD` (likely missing bond exclusions).
    pub covalent_like: usize,
}

impl ClashBreakdown {
    #[inline]
    pub fn bb_bb(&self) -> usize { self.bb_bb_local + self.bb_bb_short + self.bb_bb_long }
    #[inline]
    pub fn sc_bb(&self) -> usize { self.sc_bb_intra + self.sc_bb_inter }
    #[inline]
    pub fn sc_sc(&self) -> usize { self.sc_sc_intra + self.sc_sc_inter }
    /// Converts a raw clash count into the same per-1000-atom scale as `clashscore`.
    #[inline]
    pub fn per_1000(&self, count: usize) -> f64 {
        if self.total_atoms == 0 { 0.0 } else { count as f64 * 1000.0 / self.total_atoms as f64 }
    }
}

/// Collects (position, name, is_sidechain) for every heavy atom of a residue into a fixed stack buffer.
#[inline]
fn collect_residue_atoms<'a>(r: &'a ReconstructedResidue, out: &mut [([f64; 3], &'a str, bool); 15]) -> usize {
    let mut c = 0;
    out[c] = (r.n, "N", false); c += 1;
    out[c] = (r.ca, "CA", false); c += 1;
    out[c] = (r.c, "C", false); c += 1;
    out[c] = (r.o, "O", false); c += 1;
    if let Some(cb) = r.cb {
        out[c] = (cb, "CB", false); c += 1;
    }
    for k in 0..r.sidechain_len as usize {
        out[c] = (r.sidechain_atoms[k].pos, r.sidechain_atoms[k].name_str(), true); c += 1;
    }
    c
}

/// Computes the internal clashscore (clashes with d < 2.0 Å per 1000 heavy atoms, after bonded
/// exclusions) together with its decomposition by atom class and sequence separation.
pub fn compute_clash_breakdown(residues: &[ReconstructedResidue]) -> ClashBreakdown {
    let mut out = ClashBreakdown::default();
    for r in residues {
        out.total_atoms += 4;
        if r.cb.is_some() { out.total_atoms += 1; }
        out.total_atoms += r.sidechain_len as usize;
    }
    if out.total_atoms == 0 { return out; }

    let thr_sq = CLASH_DIST_THRESHOLD * CLASH_DIST_THRESHOLD;
    let cov_sq = COVALENT_LIKE_THRESHOLD * COVALENT_LIKE_THRESHOLD;
    let empty: ([f64; 3], &str, bool) = ([0.0; 3], "", false);

    let record = |out: &mut ClashBreakdown, d_sq: f64, sc_a: bool, sc_b: bool, seq_sep: i32| {
        out.total_clashes += 1;
        if d_sq < cov_sq { out.covalent_like += 1; }
        match (sc_a, sc_b) {
            (false, false) => {
                if seq_sep <= 1 { out.bb_bb_local += 1; }
                else if seq_sep <= 4 { out.bb_bb_short += 1; }
                else { out.bb_bb_long += 1; }
            }
            (true, true) => {
                if seq_sep == 0 { out.sc_sc_intra += 1; } else { out.sc_sc_inter += 1; }
            }
            _ => {
                if seq_sep == 0 { out.sc_bb_intra += 1; } else { out.sc_bb_inter += 1; }
            }
        }
    };

    let n = residues.len();
    for i in 0..n {
        let r_i = &residues[i];
        let mut atoms_i = [empty; 15];
        let c_i = collect_residue_atoms(r_i, &mut atoms_i);

        for a in 0..c_i {
            let (pos_a, name_a, sc_a) = atoms_i[a];
            for b in (a + 1)..c_i {
                let (pos_b, name_b, sc_b) = atoms_i[b];
                if is_bonded_residue_atoms(r_i.res_seq, name_a, r_i.res_seq, name_b) {
                    continue;
                }
                let d_sq = (pos_a[0] - pos_b[0]).powi(2) + (pos_a[1] - pos_b[1]).powi(2) + (pos_a[2] - pos_b[2]).powi(2);
                if d_sq < thr_sq {
                    record(&mut out, d_sq, sc_a, sc_b, 0);
                }
            }
        }

        for j in (i + 1)..n {
            let r_j = &residues[j];
            let d_ca_sq = (r_i.ca[0] - r_j.ca[0]).powi(2) + (r_i.ca[1] - r_j.ca[1]).powi(2) + (r_i.ca[2] - r_j.ca[2]).powi(2);
            if d_ca_sq > 256.0 {
                continue;
            }
            let seq_sep = (r_i.res_seq - r_j.res_seq).abs();

            let mut atoms_j = [empty; 15];
            let c_j = collect_residue_atoms(r_j, &mut atoms_j);

            for a in 0..c_i {
                let (pos_a, name_a, sc_a) = atoms_i[a];
                for b in 0..c_j {
                    let (pos_b, name_b, sc_b) = atoms_j[b];
                    if is_bonded_residue_atoms(r_i.res_seq, name_a, r_j.res_seq, name_b) {
                        continue;
                    }
                    let d_sq = (pos_a[0] - pos_b[0]).powi(2) + (pos_a[1] - pos_b[1]).powi(2) + (pos_a[2] - pos_b[2]).powi(2);
                    if d_sq < thr_sq {
                        record(&mut out, d_sq, sc_a, sc_b, seq_sep);
                    }
                }
            }
        }
    }

    out.clashscore = (out.total_clashes as f64 / out.total_atoms as f64) * 1000.0;
    out
}

pub const MAX_ROTAMERS: usize = 9;
pub const MAX_SC_ATOMS: usize = 10;

#[derive(Clone, Copy, Debug, Default)]
pub struct SidechainPackingMetrics {
    pub initial_clashscore: f64,
    pub final_clashscore: f64,
    pub clash_reduction_pct: f64,
    pub pruned_states: usize,
    pub total_atoms: usize,
    pub pack_time_ms: f64,
}

pub struct RotamerWorkspace {
    pub max_len: usize,
    pub cand_atoms: Vec<[[SideChainAtom; MAX_SC_ATOMS]; MAX_ROTAMERS]>,
    pub cand_lens: Vec<[u8; MAX_ROTAMERS]>,
    pub self_energies: Vec<[f64; MAX_ROTAMERS]>,
    pub active_mask: Vec<[bool; MAX_ROTAMERS]>,
    pub n_rotamers: Vec<u8>,
    pub current_rotamer: Vec<u8>,
    pub neighbor_pairs: Vec<(usize, usize)>,
    pub pair_offsets: Vec<u32>,
    pub pair_energies: Vec<f64>,
}

impl RotamerWorkspace {
    pub fn new(max_len: usize) -> Self {
        Self {
            max_len,
            cand_atoms: vec![[[SideChainAtom::default(); MAX_SC_ATOMS]; MAX_ROTAMERS]; max_len],
            cand_lens: vec![[0u8; MAX_ROTAMERS]; max_len],
            self_energies: vec![[0.0f64; MAX_ROTAMERS]; max_len],
            active_mask: vec![[true; MAX_ROTAMERS]; max_len],
            n_rotamers: vec![0u8; max_len],
            current_rotamer: vec![0u8; max_len],
            neighbor_pairs: Vec::with_capacity(max_len * 32),
            pair_offsets: vec![u32::MAX; max_len * max_len],
            pair_energies: Vec::with_capacity(max_len * 32 * 81),
        }
    }

    pub fn ensure_capacity(&mut self, len: usize) {
        if len > self.max_len {
            self.max_len = len;
            self.cand_atoms.resize(len, [[SideChainAtom::default(); MAX_SC_ATOMS]; MAX_ROTAMERS]);
            self.cand_lens.resize(len, [0u8; MAX_ROTAMERS]);
            self.self_energies.resize(len, [0.0f64; MAX_ROTAMERS]);
            self.active_mask.resize(len, [true; MAX_ROTAMERS]);
            self.n_rotamers.resize(len, 0u8);
            self.current_rotamer.resize(len, 0u8);
            self.pair_offsets.resize(len * len, u32::MAX);
        }
    }
}

pub fn pack_sidechains(
    residues: &mut [ReconstructedResidue],
    ws: &mut RotamerWorkspace,
    alpha_soft: f64,
    w_clash: f64,
    w_rot: f64,
) -> SidechainPackingMetrics {
    pack_sidechains_with_potentials(residues, ws, alpha_soft, w_clash, w_rot, 2.50, 3.00)
}

/// Rotamer optimization engine with directional H-bonds and screened salt bridges (STRUCT-6).
pub fn pack_sidechains_with_potentials(
    residues: &mut [ReconstructedResidue],
    ws: &mut RotamerWorkspace,
    alpha_soft: f64,
    w_clash: f64,
    w_rot: f64,
    w_hb: f64,
    w_salt: f64,
) -> SidechainPackingMetrics {
    let timer = std::time::Instant::now();
    let n = residues.len();
    if n == 0 {
        return SidechainPackingMetrics::default();
    }
    ws.ensure_capacity(n);

    for &(i, j) in &ws.neighbor_pairs {
        ws.pair_offsets[i * ws.max_len + j] = u32::MAX;
        ws.pair_offsets[j * ws.max_len + i] = u32::MAX;
    }
    ws.neighbor_pairs.clear();
    ws.pair_energies.clear();

    for i in 0..n {
        let res_name = std::str::from_utf8(&residues[i].res_name).unwrap_or("UNK").trim();
        let basin = if i > 0 && i + 1 < n {
            let phi_deg = calc_dihedral_deg(residues[i - 1].c, residues[i].n, residues[i].ca, residues[i].c);
            let psi_deg = calc_dihedral_deg(residues[i].n, residues[i].ca, residues[i].c, residues[i + 1].n);
            classify_ramachandran(phi_deg, psi_deg)
        } else {
            RamachandranBasin::General
        };
        let rots = get_rotamer_candidates_bbdep(res_name, basin);
        let n_rots = rots.len().min(MAX_ROTAMERS);
        ws.n_rotamers[i] = n_rots as u8;

        let n_pos = Vec3::from_array(residues[i].n);
        let ca_pos = Vec3::from_array(residues[i].ca);
        let cb_pos = Vec3::from_array(residues[i].cb.unwrap_or(residues[i].ca));

        for r in 0..n_rots {
            let mut cand_buf = [SideChainAtom::default(); MAX_SC_ATOMS];
            let sc_len = build_sidechain_atoms_into(res_name, n_pos, ca_pos, cb_pos, &rots[r], &mut cand_buf);
            ws.cand_atoms[i][r] = cand_buf;
            ws.cand_lens[i][r] = sc_len;
            ws.active_mask[i][r] = true;

            let e_prob = -w_rot * (rots[r].prob + 1e-6).ln();

            let mut e_bb_clash = 0.0;
            let mut e_bb_hbond = 0.0;
            for j in 0..n {
                let d_ca = ca_pos.dist(Vec3::from_array(residues[j].ca));
                if d_ca > 15.0 {
                    continue;
                }
                let num_bb = if residues[j].cb.is_none() { 4 } else { 5 };
                let bb_atoms: [([f64; 3], u8, &str); 5] = [
                    (residues[j].n, b'N', "N"),
                    (residues[j].ca, b'C', "CA"),
                    (residues[j].c, b'C', "C"),
                    (residues[j].o, b'O', "O"),
                    (residues[j].cb.unwrap_or(residues[j].ca), b'C', "CB"),
                ];

                let (u_co, u_nh, has_amide) = if w_hb > 0.0 {
                    let c_pos = Vec3::from_array(residues[j].c);
                    let o_pos = Vec3::from_array(residues[j].o);
                    let co_diff = o_pos.sub(c_pos);
                    let u_co = if co_diff.norm() > 1e-6 { co_diff.normalize() } else { Vec3::zero() };

                    let res_j_name = residues[j].res_name;
                    let is_pro = res_j_name == *b"PRO";
                    let (u_nh, has_amide) = if !is_pro {
                        let n_pos = Vec3::from_array(residues[j].n);
                        let ca_j = Vec3::from_array(residues[j].ca);
                        let v_nh = if j > 0 {
                            let c_prev = Vec3::from_array(residues[j - 1].c);
                            let v1 = n_pos.sub(c_prev).normalize();
                            let v2 = n_pos.sub(ca_j).normalize();
                            v1.add(v2)
                        } else {
                            n_pos.sub(ca_j)
                        };
                        if v_nh.norm() > 1e-6 {
                            (v_nh.normalize(), true)
                        } else {
                            (Vec3::zero(), false)
                        }
                    } else {
                        (Vec3::zero(), false)
                    };
                    (u_co, u_nh, has_amide)
                } else {
                    (Vec3::zero(), Vec3::zero(), false)
                };

                for a_idx in 0..sc_len as usize {
                    let a = &cand_buf[a_idx];
                    let a_pos = Vec3::from_array(a.pos);
                    let a_name = a.name_str();

                    for &(b_pos, b_elem, b_name) in &bb_atoms[0..num_bb] {
                        if is_bonded_residue_atoms(residues[i].res_seq, a_name, residues[j].res_seq, b_name) {
                            continue;
                        }
                        let d = a_pos.dist(Vec3::from_array(b_pos));
                        e_bb_clash += soft_core_steric(d, a.element, b_elem, alpha_soft, w_clash);

                        // Directional H-bond potential (STRUCT-6)
                        if w_hb > 0.0 && d >= 2.30 && d <= 3.40 {
                            if (a.chem_flag & ATOM_POLAR_DONOR) != 0 && b_name == "O" {
                                let o_pos = Vec3::from_array(b_pos);
                                let u_od = a_pos.sub(o_pos).normalize();
                                let cos_th = u_co.dot(u_od);
                                if cos_th > 0.0 {
                                    e_bb_hbond += hbond_energy(d, cos_th * cos_th, w_hb);
                                }
                            } else if (a.chem_flag & ATOM_POLAR_ACCEPTOR) != 0 && b_name == "N" && has_amide {
                                let n_pos = Vec3::from_array(b_pos);
                                let u_na = a_pos.sub(n_pos).normalize();
                                let cos_th = u_nh.dot(u_na);
                                if cos_th > 0.0 {
                                    e_bb_hbond += hbond_energy(d, cos_th * cos_th, w_hb);
                                }
                            }
                        }
                    }
                }
            }
            ws.self_energies[i][r] = e_prob + e_bb_clash + e_bb_hbond;
        }
    }

    for i in 0..n {
        let n_rots = ws.n_rotamers[i] as usize;
        if n_rots > 0 {
            residues[i].sidechain_len = ws.cand_lens[i][0];
            residues[i].sidechain_atoms = ws.cand_atoms[i][0];
        } else {
            residues[i].sidechain_len = 0;
        }
    }
    let (_init_raw, initial_clashscore) = compute_protein_clashscore(residues);

    for i in 0..n {
        let cb_i = Vec3::from_array(residues[i].cb.unwrap_or(residues[i].ca));
        let n_rots_i = ws.n_rotamers[i] as usize;
        if n_rots_i == 0 { continue; }

        for j in (i + 1)..n {
            let n_rots_j = ws.n_rotamers[j] as usize;
            if n_rots_j == 0 { continue; }

            let cb_j = Vec3::from_array(residues[j].cb.unwrap_or(residues[j].ca));
            if cb_i.dist(cb_j) < 9.0 {
                let offset = ws.pair_energies.len() as u32;
                ws.neighbor_pairs.push((i, j));
                ws.pair_offsets[i * ws.max_len + j] = offset;
                ws.pair_offsets[j * ws.max_len + i] = offset | (1 << 31);

                for r_i in 0..MAX_ROTAMERS {
                    for r_j in 0..MAX_ROTAMERS {
                        if r_i < n_rots_i && r_j < n_rots_j {
                            let len_i = ws.cand_lens[i][r_i] as usize;
                            let len_j = ws.cand_lens[j][r_j] as usize;
                            let mut e_pair = 0.0;

                            for a_k in 0..len_i {
                                let a = &ws.cand_atoms[i][r_i][a_k];
                                let pos_a = Vec3::from_array(a.pos);
                                let name_a = a.name_str();

                                for b_k in 0..len_j {
                                    let b = &ws.cand_atoms[j][r_j][b_k];
                                    let name_b = b.name_str();
                                    if is_bonded_residue_atoms(residues[i].res_seq, name_a, residues[j].res_seq, name_b) {
                                        continue;
                                    }
                                    let d = pos_a.dist(Vec3::from_array(b.pos));
                                    e_pair += soft_core_steric(d, a.element, b.element, alpha_soft, w_clash);

                                    // Directional/distance H-bond (STRUCT-6)
                                    if w_hb > 0.0 && d >= 2.30 && d <= 3.40 {
                                        let donor_a = (a.chem_flag & ATOM_POLAR_DONOR) != 0;
                                        let acc_b = (b.chem_flag & ATOM_POLAR_ACCEPTOR) != 0;
                                        let donor_b = (b.chem_flag & ATOM_POLAR_DONOR) != 0;
                                        let acc_a = (a.chem_flag & ATOM_POLAR_ACCEPTOR) != 0;
                                        if (donor_a && acc_b) || (donor_b && acc_a) {
                                            e_pair += hbond_energy(d, 1.0, w_hb);
                                        }
                                    }

                                    // Screened Salt Bridge & Electrostatics (STRUCT-6)
                                    if w_salt > 0.0 && d >= 2.30 && d <= 4.20 {
                                        let cat_a = (a.chem_flag & ATOM_CATION) != 0;
                                        let ani_a = (a.chem_flag & ATOM_ANION) != 0;
                                        let cat_b = (b.chem_flag & ATOM_CATION) != 0;
                                        let ani_b = (b.chem_flag & ATOM_ANION) != 0;

                                        if (cat_a && ani_b) || (ani_a && cat_b) {
                                            let q_weight = charge_weight(name_a) * charge_weight(name_b);
                                            e_pair += q_weight * salt_bridge_energy(d, w_salt);
                                        } else if (cat_a && cat_b) || (ani_a && ani_b) {
                                            let q_weight = charge_weight(name_a) * charge_weight(name_b);
                                            e_pair += q_weight * like_charge_repulsion(d, w_salt);
                                        }
                                    }
                                }
                            }
                            ws.pair_energies.push(e_pair);
                        } else {
                            ws.pair_energies.push(0.0);
                        }
                    }
                }
            }
        }
    }

    let max_len = ws.max_len;
    let get_pair_e = |offsets: &[u32], energies: &[f64], i: usize, r_i: usize, j: usize, r_j: usize| -> f64 {
        let entry = offsets[i * max_len + j];
        if entry != u32::MAX {
            let is_transposed = (entry & (1 << 31)) != 0;
            let base = (entry & !(1 << 31)) as usize;
            if is_transposed {
                energies[base + r_j * MAX_ROTAMERS + r_i]
            } else {
                energies[base + r_i * MAX_ROTAMERS + r_j]
            }
        } else {
            0.0
        }
    };

    let mut pruned_states = 0;
    for _pass in 0..2 {
        for i in 0..n {
            let n_rots = ws.n_rotamers[i] as usize;
            if n_rots <= 1 { continue; }

            for r in 0..n_rots {
                if !ws.active_mask[i][r] { continue; }

                for t in 0..n_rots {
                    if r == t || !ws.active_mask[i][t] { continue; }

                    let mut sum_min_diff = 0.0;
                    for j in 0..n {
                        if i == j || ws.n_rotamers[j] == 0 { continue; }
                        if ws.pair_offsets[i * max_len + j] == u32::MAX { continue; }

                        let mut min_diff = f64::INFINITY;
                        let n_rots_j = ws.n_rotamers[j] as usize;
                        for s in 0..n_rots_j {
                            if !ws.active_mask[j][s] { continue; }
                            let diff = get_pair_e(&ws.pair_offsets, &ws.pair_energies, i, r, j, s)
                                     - get_pair_e(&ws.pair_offsets, &ws.pair_energies, i, t, j, s);
                            if diff < min_diff {
                                min_diff = diff;
                            }
                        }
                        if min_diff < f64::INFINITY {
                            sum_min_diff += min_diff;
                        }
                    }

                    let total_diff = ws.self_energies[i][r] - ws.self_energies[i][t] + sum_min_diff;
                    if total_diff > 1e-4 {
                        ws.active_mask[i][r] = false;
                        pruned_states += 1;
                        break;
                    }
                }
            }
        }
    }

    for i in 0..n {
        let n_rots = ws.n_rotamers[i] as usize;
        if n_rots == 0 {
            ws.current_rotamer[i] = 0;
            continue;
        }
        let mut best_r = 0;
        let mut min_e = f64::INFINITY;
        for r in 0..n_rots {
            if ws.active_mask[i][r] && ws.self_energies[i][r] < min_e {
                min_e = ws.self_energies[i][r];
                best_r = r;
            }
        }
        ws.current_rotamer[i] = best_r as u8;
    }

    for _sweep in 0..5 {
        let mut changed = false;
        for i in 0..n {
            let n_rots = ws.n_rotamers[i] as usize;
            if n_rots <= 1 { continue; }

            let mut best_r = ws.current_rotamer[i] as usize;
            let mut min_total_e = f64::INFINITY;

            for r in 0..n_rots {
                if !ws.active_mask[i][r] { continue; }
                let mut e_total = ws.self_energies[i][r];
                for j in 0..n {
                    if i == j || ws.n_rotamers[j] == 0 { continue; }
                    let cur_j = ws.current_rotamer[j] as usize;
                    e_total += get_pair_e(&ws.pair_offsets, &ws.pair_energies, i, r, j, cur_j);
                }
                if e_total < min_total_e {
                    min_total_e = e_total;
                    best_r = r;
                }
            }

            if best_r != ws.current_rotamer[i] as usize {
                ws.current_rotamer[i] = best_r as u8;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut total_atoms = 0;
    for i in 0..n {
        total_atoms += 4;
        if residues[i].cb.is_some() { total_atoms += 1; }

        let n_rots = ws.n_rotamers[i] as usize;
        if n_rots > 0 {
            let chosen = ws.current_rotamer[i] as usize;
            residues[i].sidechain_len = ws.cand_lens[i][chosen];
            residues[i].sidechain_atoms = ws.cand_atoms[i][chosen];
            total_atoms += residues[i].sidechain_len as usize;
        } else {
            residues[i].sidechain_len = 0;
        }
    }

    let (_final_raw, final_clashscore) = compute_protein_clashscore(residues);
    let clash_reduction_pct = if initial_clashscore > 0.0 {
        ((initial_clashscore - final_clashscore) / initial_clashscore) * 100.0
    } else {
        0.0
    };

    SidechainPackingMetrics {
        initial_clashscore,
        final_clashscore,
        clash_reduction_pct,
        pruned_states,
        total_atoms,
        pack_time_ms: timer.elapsed().as_secs_f64() * 1000.0,
    }
}

/// Pre-allocated workspace guaranteeing zero dynamic heap allocations during all-atom kinematic reconstruction.
pub struct AllAtomWorkspace {
    pub cas: Vec<Vec3>,
    pub e_x: Vec<Vec3>,
    pub e_y0: Vec<Vec3>,
    pub e_z0: Vec<Vec3>,
    pub thetas: Vec<f64>,
    pub v_th: Vec<f64>,
    pub c_preds: Vec<Vec3>,
    pub o_preds: Vec<Vec3>,
    pub n_preds: Vec<Vec3>,
    pub grad: Vec<f64>,
    pub out_residues: Vec<ReconstructedResidue>,
    pub rotamer_ws: RotamerWorkspace,
}

impl AllAtomWorkspace {
    pub fn new(max_len: usize) -> Self {
        Self {
            cas: vec![Vec3::zero(); max_len],
            e_x: vec![Vec3::zero(); max_len],
            e_y0: vec![Vec3::zero(); max_len],
            e_z0: vec![Vec3::zero(); max_len],
            thetas: vec![0.0; max_len],
            v_th: vec![0.0; max_len],
            c_preds: vec![Vec3::zero(); max_len],
            o_preds: vec![Vec3::zero(); max_len],
            n_preds: vec![Vec3::zero(); max_len],
            grad: vec![0.0; max_len],
            out_residues: vec![ReconstructedResidue::default(); max_len],
            rotamer_ws: RotamerWorkspace::new(max_len),
        }
    }
}

/// Analytical kinematic backbone & C_beta solver (STRUCT-3).
pub struct KinematicBackboneSolver {
    pub num_coarse_angles: usize,
    pub num_fine_steps: usize,
    pub learning_rate: f64,
    pub momentum: f64,
}

impl Default for KinematicBackboneSolver {
    fn default() -> Self {
        Self {
            num_coarse_angles: 24,
            num_fine_steps: 20,
            learning_rate: 0.04,
            momentum: 0.80,
        }
    }
}

impl KinematicBackboneSolver {
    /// Zero-allocation kinematic reconstruction writing directly into pre-allocated `AllAtomWorkspace`.
    /// When `pack_sidechains_flag` is true, performs rotamer optimization via NeRF + DEE + ICM (STRUCT-4).
    pub fn reconstruct_into(
        &self,
        ca_coords: &[[f64; 3]],
        res_names: &[[u8; 3]],
        res_seqs: &[i32],
        ws: &mut AllAtomWorkspace,
        pack_sidechains_flag: bool,
    ) -> Option<SidechainPackingMetrics> {
        self.reconstruct_into_with_bfactors(ca_coords, res_names, res_seqs, None, ws, pack_sidechains_flag)
    }

    /// Zero-allocation kinematic reconstruction writing directly into pre-allocated `AllAtomWorkspace`.
    /// When `b_factors` is provided, attaches dynamic residue confidence scores to each residue.
    /// When `pack_sidechains_flag` is true, performs rotamer optimization via NeRF + DEE + ICM (STRUCT-4/5).
    pub fn reconstruct_into_with_bfactors(
        &self,
        ca_coords: &[[f64; 3]],
        res_names: &[[u8; 3]],
        res_seqs: &[i32],
        b_factors: Option<&[f64]>,
        ws: &mut AllAtomWorkspace,
        pack_sidechains_flag: bool,
    ) -> Option<SidechainPackingMetrics> {
        let n = ca_coords.len();
        assert!(n >= 3 && n <= ws.cas.len(), "Invalid sequence length for workspace capacity");
        let num_bonds = n - 1;

        for i in 0..n {
            ws.cas[i] = Vec3::from_array(ca_coords[i]);
        }

        // 1. Build local orthonormal reference bases for each peptide bond i in 0..n-1
        for i in 0..num_bonds {
            let v = ws.cas[i + 1].sub(ws.cas[i]);
            let ex = v.normalize();
            ws.e_x[i] = ex;

            let ey = if i > 0 && i < num_bonds - 1 {
                let v_prev = ws.cas[i].sub(ws.cas[i - 1]);
                let v_next = ws.cas[i + 2].sub(ws.cas[i + 1]);
                let n_prev = v_prev.cross(ex).normalize();
                let n_next = ex.cross(v_next).normalize();
                let n_bis_raw = n_prev.add(n_next);
                let n_bis = if n_bis_raw.norm() < 1e-4 {
                    n_prev
                } else {
                    n_bis_raw.normalize()
                };
                let b_bis = n_bis.cross(ex).normalize();

                let d_span = ws.cas[i + 2].sub(ws.cas[i - 1]).norm();
                if d_span < 6.20 {
                    n_bis // Alpha-helix orientation
                } else if i % 2 == 0 {
                    b_bis
                } else {
                    b_bis.scale(-1.0)
                }
            } else if i == 0 {
                let v_next = ws.cas[2].sub(ws.cas[1]);
                ex.cross(v_next).normalize()
            } else {
                let v_prev = ws.cas[i].sub(ws.cas[i - 1]);
                v_prev.cross(ex).normalize()
            };

            let ez = ex.cross(ey).normalize();
            ws.e_y0[i] = ey;
            ws.e_z0[i] = ez;
            ws.thetas[i] = 0.0;
            ws.v_th[i] = 0.0;
        }

        // 2. Coarse 1D Angular Sweep (zero heap allocations)
        let n_coarse = self.num_coarse_angles;
        for i in 0..num_bonds {
            let mut best_th = 0.0;
            let mut best_score = 1e9;

            for k in 0..n_coarse {
                let th = -std::f64::consts::PI
                    + (2.0 * std::f64::consts::PI * k as f64) / n_coarse as f64;
                let ey = ws.e_y0[i].scale(th.cos()).add(ws.e_z0[i].scale(th.sin()));
                let c_i = ws.cas[i]
                    .add(ws.e_x[i].scale(1.4256))
                    .add(ey.scale(0.5382));
                let o_i = ws.cas[i]
                    .add(ws.e_x[i].scale(1.6399))
                    .add(ey.scale(1.7409));

                let mut score = 0.0;
                if i > 0 {
                    let th_prev = ws.thetas[i - 1];
                    let ey_prev = ws.e_y0[i - 1]
                        .scale(th_prev.cos())
                        .add(ws.e_z0[i - 1].scale(th_prev.sin()));
                    let n_i = ws.cas[i - 1]
                        .add(ws.e_x[i - 1].scale(2.3751))
                        .sub(ey_prev.scale(0.3688));

                    let v_n = n_i.sub(ws.cas[i]).normalize();
                    let v_c = c_i.sub(ws.cas[i]).normalize();
                    let cos_a = v_n.dot(v_c);
                    score += 15.0 * (cos_a - (-0.3584)).powi(2);
                }

                if i + 4 < n {
                    let d_ca = ws.cas[i + 4].sub(ws.cas[i]).norm();
                    if d_ca < 6.50 {
                        let target_n = ws.cas[i + 4]
                            .sub(ws.e_x[std::cmp::min(i + 3, num_bonds - 1)].scale(0.80));
                        let d_on = o_i.sub(target_n).norm();
                        score += 3.0 * (d_on - 2.95).powi(2);
                    }
                }

                if score < best_score {
                    best_score = score;
                    best_th = th;
                }
            }
            ws.thetas[i] = best_th;
        }

        // 3. Fine Gradient Descent Refinement with Heavy-Ball Momentum (Polyak)
        for _ in 0..self.num_fine_steps {
            ws.n_preds[0] = ws.cas[0];
            for i in 0..num_bonds {
                let th = ws.thetas[i];
                let ey = ws.e_y0[i].scale(th.cos()).add(ws.e_z0[i].scale(th.sin()));
                ws.c_preds[i] = ws.cas[i].add(ws.e_x[i].scale(1.4256)).add(ey.scale(0.5382));
                ws.n_preds[i + 1] = ws.cas[i].add(ws.e_x[i].scale(2.3751)).sub(ey.scale(0.3688));
                ws.grad[i] = 0.0;
            }

            for i in 1..n - 1 {
                let n_i = ws.n_preds[i];
                let c_i = ws.c_preds[i];
                let ca_i = ws.cas[i];

                let v_n = n_i.sub(ca_i);
                let v_c = c_i.sub(ca_i);
                let l_n = v_n.norm();
                let l_c = v_c.norm();
                let cos_a = v_n.dot(v_c) / (l_n * l_c);
                let diff = cos_a - (-0.3584);

                let th_i = ws.thetas[i];
                let d_ey_i = ws.e_y0[i].scale(-th_i.sin()).add(ws.e_z0[i].scale(th_i.cos()));
                let d_c = d_ey_i.scale(0.5382);
                let d_cos_i = v_n.dot(d_c) / (l_n * l_c) - (cos_a * v_c.dot(d_c)) / (l_c * l_c);
                ws.grad[i] += 2.0 * diff * d_cos_i;

                let th_im1 = ws.thetas[i - 1];
                let d_ey_im1 = ws.e_y0[i - 1]
                    .scale(-th_im1.sin())
                    .add(ws.e_z0[i - 1].scale(th_im1.cos()));
                let d_n = d_ey_im1.scale(-0.3688);
                let d_cos_im1 = d_n.dot(v_c) / (l_n * l_c) - (cos_a * v_n.dot(d_n)) / (l_n * l_n);
                ws.grad[i - 1] += 2.0 * diff * d_cos_im1;
            }

            for i in 0..num_bonds {
                let g = ws.grad[i].clamp(-1.5, 1.5);
                ws.v_th[i] = self.momentum * ws.v_th[i] + self.learning_rate * g;
                ws.thetas[i] -= ws.v_th[i];
            }
        }

        // 4. Final Forward Position Assignment
        let th_0 = ws.thetas[0];
        let ey_0 = ws.e_y0[0].scale(th_0.cos()).add(ws.e_z0[0].scale(th_0.sin()));
        ws.n_preds[0] = ws.cas[0].sub(ws.e_x[0].scale(0.94)).sub(ey_0.scale(1.11));

        for i in 0..num_bonds {
            let th = ws.thetas[i];
            let ey = ws.e_y0[i].scale(th.cos()).add(ws.e_z0[i].scale(th.sin()));
            ws.c_preds[i] = ws.cas[i].add(ws.e_x[i].scale(1.4256)).add(ey.scale(0.5382));
            ws.o_preds[i] = ws.cas[i].add(ws.e_x[i].scale(1.6399)).add(ey.scale(1.7409));
            ws.n_preds[i + 1] = ws.cas[i].add(ws.e_x[i].scale(2.3751)).sub(ey.scale(0.3688));
        }

        // Terminal C_{n-1} and O_{n-1} rotated by final peptide plane theta_{n-2}:
        let th_last = ws.thetas[num_bonds - 1];
        let ey_last = ws.e_y0[num_bonds - 1]
            .scale(th_last.cos())
            .add(ws.e_z0[num_bonds - 1].scale(th_last.sin()));
        ws.c_preds[n - 1] = ws.cas[n - 1]
            .add(ws.e_x[num_bonds - 1].scale(1.4256))
            .add(ey_last.scale(0.5382));
        ws.o_preds[n - 1] = ws.cas[n - 1]
            .add(ws.e_x[num_bonds - 1].scale(1.6399))
            .add(ey_last.scale(1.7409));

        // 5. Build full residue atoms with tetrahedral CB (L-chirality)
        for i in 0..n {
            let ca = ws.cas[i];
            let n_atom = ws.n_preds[i];
            let c_atom = ws.c_preds[i];
            let o_atom = ws.o_preds[i];

            let cb = if &res_names[i] == b"GLY" {
                None
            } else {
                let v_n = n_atom.sub(ca).normalize();
                let v_c = c_atom.sub(ca).normalize();
                let u1 = v_n.add(v_c).normalize();
                let u2 = v_n.cross(v_c).normalize();
                let cb_pos = ca.sub(u1.scale(0.9367)).add(u2.scale(1.2122));
                Some(cb_pos.to_array())
            };

            let b_factor = b_factors.and_then(|bf| bf.get(i).copied()).unwrap_or(50.0);

            ws.out_residues[i] = ReconstructedResidue {
                res_name: res_names[i],
                res_seq: res_seqs[i],
                n: n_atom.to_array(),
                ca: ca.to_array(),
                c: c_atom.to_array(),
                o: o_atom.to_array(),
                cb,
                b_factor,
                sidechain_len: 0,
                sidechain_atoms: [SideChainAtom::default(); 10],
            };
        }

        if pack_sidechains_flag {
            Some(pack_sidechains(
                &mut ws.out_residues[0..n],
                &mut ws.rotamer_ws,
                0.82,
                10.0,
                1.0,
            ))
        } else {
            None
        }
    }
}

/// Helper function to format PDB atom names into the 6-character field (Cols 12-17: space + 4-char name + space).
fn format_pdb_atom_name_field(name: &str) -> String {
    let inner = match name.len() {
        1 => format!(" {}  ", name),
        2 => format!(" {} ", name),
        3 => format!(" {}", name),
        _ => format!("{:4}", name),
    };
    format!(" {} ", inner)
}

/// Generates a Protein Data Bank (.pdb) string containing full all-atom backbone (N, CA, C, O) and C_beta atoms,
/// plus rotameric sidechain atoms when present (STRUCT-3/4/5).
/// Writes dynamic residue B-factors in columns 61-66.
pub fn coords_to_allatom_pdb_string(
    protein_id: &str,
    residues: &[ReconstructedResidue],
) -> String {
    let mut out = String::with_capacity(residues.len() * 600 + 256);
    out.push_str(&format!("HEADER    CODON-INFER STRUCT-3 ALL-ATOM RECONSTRUCTION {}\n", protein_id));
    let has_sidechains = residues.iter().any(|r| r.sidechain_len > 0);
    let title = if has_sidechains {
        "ALL-ATOM BACKBONE + ROTAMERIC SIDECHAINS (STRUCT-4/5 / NERF + DEE + ICM)"
    } else {
        "ALL-ATOM BACKBONE + CB SOLVED VIA PAULING-COREY KINEMATIC SOLVER"
    };
    out.push_str(&format!("TITLE     {}\n", title));

    let mut atom_idx = 1;
    for res in residues {
        let name_str = std::str::from_utf8(&res.res_name).unwrap_or("UNK");
        // N
        out.push_str(&format!(
            "ATOM  {:5}{}{:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}           N\n",
            atom_idx, format_pdb_atom_name_field("N"), name_str, res.res_seq, res.n[0], res.n[1], res.n[2], res.b_factor
        ));
        atom_idx += 1;

        // CA
        out.push_str(&format!(
            "ATOM  {:5}{}{:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}           C\n",
            atom_idx, format_pdb_atom_name_field("CA"), name_str, res.res_seq, res.ca[0], res.ca[1], res.ca[2], res.b_factor
        ));
        atom_idx += 1;

        // C
        out.push_str(&format!(
            "ATOM  {:5}{}{:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}           C\n",
            atom_idx, format_pdb_atom_name_field("C"), name_str, res.res_seq, res.c[0], res.c[1], res.c[2], res.b_factor
        ));
        atom_idx += 1;

        // O
        out.push_str(&format!(
            "ATOM  {:5}{}{:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}           O\n",
            atom_idx, format_pdb_atom_name_field("O"), name_str, res.res_seq, res.o[0], res.o[1], res.o[2], res.b_factor
        ));
        atom_idx += 1;

        // CB
        if let Some(cb) = res.cb {
            out.push_str(&format!(
                "ATOM  {:5}{}{:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}           C\n",
                atom_idx, format_pdb_atom_name_field("CB"), name_str, res.res_seq, cb[0], cb[1], cb[2], res.b_factor
            ));
            atom_idx += 1;
        }

        // Sidechain heavy atoms beyond CB (STRUCT-4)
        for k in 0..res.sidechain_len as usize {
            let sc_atom = &res.sidechain_atoms[k];
            let sc_name = sc_atom.name_str();
            let elem_char = (sc_atom.element as char).to_ascii_uppercase();
            out.push_str(&format!(
                "ATOM  {:5}{}{:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}          {:>2}\n",
                atom_idx, format_pdb_atom_name_field(sc_name), name_str, res.res_seq,
                sc_atom.pos[0], sc_atom.pos[1], sc_atom.pos[2], res.b_factor, elem_char
            ));
            atom_idx += 1;
        }
    }
    out.push_str("END\n");
    out
}

/// Exports all-atom reconstructed structure to a .pdb file.
pub fn export_allatom_pdb(
    path: &str,
    protein_id: &str,
    residues: &[ReconstructedResidue],
) -> io::Result<String> {
    let pdb_content = coords_to_allatom_pdb_string(protein_id, residues);
    let mut file = File::create(path)?;
    file.write_all(pdb_content.as_bytes())?;
    file.flush()?;
    Ok(pdb_content)
}

/// Parses an existing PDB file from disk into a vector of all-atom residues.
/// Reconstructs virtual tetrahedral C_beta when absent (e.g. GLY) to eliminate data leakage.
pub fn parse_pdb_atoms(path: &str) -> io::Result<Vec<ReconstructedResidue>> {
    let content = std::fs::read_to_string(path)?;
    let mut map: std::collections::BTreeMap<
        i32,
        (String, Option<[f64; 3]>, Option<[f64; 3]>, Option<[f64; 3]>, Option<[f64; 3]>, Option<[f64; 3]>, f64),
    > = std::collections::BTreeMap::new();

    for line in content.lines() {
        if !line.starts_with("ATOM  ") {
            continue;
        }
        let atom_name = line.get(12..16).map(|s| s.trim()).unwrap_or("");
        let res_name = line.get(17..20).map(|s| s.trim().to_string()).unwrap_or_else(|| "UNK".to_string());
        let res_seq: i32 = line.get(22..26).and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        let x: f64 = line.get(30..38).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        let y: f64 = line.get(38..46).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        let z: f64 = line.get(46..54).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        let b_factor: f64 = line.get(60..66).and_then(|s| s.trim().parse().ok()).unwrap_or(50.0);
        let coord = [x, y, z];

        let entry = map.entry(res_seq).or_insert((res_name, None, None, None, None, None, 50.0));
        match atom_name {
            "N" => entry.1 = Some(coord),
            "CA" => {
                entry.2 = Some(coord);
                entry.6 = b_factor;
            }
            "C" => entry.3 = Some(coord),
            "O" => entry.4 = Some(coord),
            "CB" => entry.5 = Some(coord),
            _ => {}
        }
    }

    let residues: Vec<ReconstructedResidue> = map.into_iter()
        .filter_map(|(res_seq, (res_name, n_opt, ca_opt, c_opt, o_opt, cb_opt, b_factor))| {
            let ca = ca_opt?;
            let n = n_opt?;
            let c = c_opt?;
            let o = o_opt?;
            let ca_v = Vec3::from_array(ca);
            let v_n = Vec3::from_array(n).sub(ca_v).normalize();
            let v_c = Vec3::from_array(c).sub(ca_v).normalize();
            let u1 = v_n.add(v_c).normalize();
            let u2 = v_n.cross(v_c).normalize();
            let virtual_cb = [
                ca_v.0[0] - 0.9367 * u1.0[0] + 1.2122 * u2.0[0],
                ca_v.0[1] - 0.9367 * u1.0[1] + 1.2122 * u2.0[1],
                ca_v.0[2] - 0.9367 * u1.0[2] + 1.2122 * u2.0[2],
            ];
            let is_gly = res_name == "GLY";
            let cb = if is_gly { Some(virtual_cb) } else { cb_opt.or(Some(virtual_cb)) };

            let mut name_bytes = [b' '; 3];
            let src_bytes = res_name.as_bytes();
            for (idx, &b) in src_bytes.iter().take(3).enumerate() {
                name_bytes[idx] = b;
            }

            Some(ReconstructedResidue {
                res_name: name_bytes,
                res_seq,
                n,
                ca,
                c,
                o,
                cb,
                b_factor,
                sidechain_len: 0,
                sidechain_atoms: [SideChainAtom::default(); 10],
            })
        })
        .collect();

    if residues.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("No valid ATOM backbone records found in PDB '{}'", path),
        ));
    }

    Ok(residues)
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
        let dms_json = dms_to_json_string("test\"prot\\1\n2", "M", &[row], 12.34, 162.0);
        assert!(dms_json.contains("\"protein_id\": \"test\\\"prot\\\\1\\n2\""));
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
        let contacts_json = contacts_to_json_string("test\"prot", 6, &[5], 1.25, &[pair]);
        assert!(contacts_json.contains("\"protein_id\": \"test\\\"prot\""));
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
        let inpaint_json = inpaint_to_json_string("test\"prot", "*", "C", &[inpaint_site]);
        assert!(inpaint_json.contains("\"protein_id\": \"test\\\"prot\""));
        assert!(inpaint_json.contains("\"reconstructed_sequence\": \"C\""));
        assert!(inpaint_json.contains("\"predictions\": ["));
        assert!(inpaint_json.contains("85.50"));
    }

    #[test]
    fn test_pdb_and_tsv_export() {
        let seq = "MCA";
        let coords = vec![[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]];
        let pdb = coords_to_pdb_string(seq, "test_pdb", &coords);
        assert!(pdb.contains("HEADER    CODON-INFER STRUCT-2 3D RECONSTRUCTION test_pdb"));
        assert!(pdb.contains("ATOM      1  CA  MET A   1       0.000   0.000   0.000"));
        assert!(pdb.contains("ATOM      2  CA  CYS A   2       3.800   0.000   0.000"));
        assert!(pdb.contains("ATOM      3  CA  ALA A   3       7.600   0.000   0.000"));
        assert!(pdb.ends_with("END\n"));

        let test_pdb_path = "test_ca_scratch.pdb";
        export_ca_pdb(test_pdb_path, seq, "test_pdb", &coords).expect("should export pdb");
        assert!(std::path::Path::new(test_pdb_path).exists());
        let _ = std::fs::remove_file(test_pdb_path);

        let test_tsv_path = "test_ca_scratch.tsv";
        export_fold_tsv(test_tsv_path, "test_pdb", seq, &coords).expect("should export tsv");
        assert!(std::path::Path::new(test_tsv_path).exists());
        let _ = std::fs::remove_file(test_tsv_path);
    }

    #[test]
    fn test_allatom_kinematic_reconstruction() {
        let ca_coords = [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [6.0, 3.1, 0.0],
            [9.0, 4.0, 2.0],
        ];
        let res_names = [*b"MET", *b"ALA", *b"GLY", *b"CYS"];
        let res_seqs = [1, 2, 3, 4];
        let mut ws = AllAtomWorkspace::new(10);
        let solver = KinematicBackboneSolver::default();
        solver.reconstruct_into(&ca_coords, &res_names, &res_seqs, &mut ws, false);

        assert_eq!(ws.out_residues.len(), 10);
        let r0 = &ws.out_residues[0];
        assert_eq!(&r0.res_name, b"MET");
        assert!(r0.cb.is_some());

        // Residue 2 is GLY -> cb must be None in output residues
        let r2 = &ws.out_residues[2];
        assert_eq!(&r2.res_name, b"GLY");
        assert!(r2.cb.is_none());

        let pdb_str = coords_to_allatom_pdb_string("test_allatom", &ws.out_residues[0..4]);
        assert!(pdb_str.contains("STRUCT-3 ALL-ATOM RECONSTRUCTION"));
        assert!(pdb_str.contains("ATOM      1  N   MET A   1"));
        assert!(pdb_str.contains("ATOM      2  CA  MET A   1"));
        assert!(pdb_str.contains("ATOM      3  C   MET A   1"));
        assert!(pdb_str.contains("ATOM      4  O   MET A   1"));
        assert!(pdb_str.contains("ATOM      5  CB  MET A   1"));
        assert!(pdb_str.ends_with("END\n"));
    }

    #[test]
    fn test_allatom_with_sidechain_packing() {
        let ca_coords = [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [6.0, 3.1, 0.0],
            [9.0, 4.0, 2.0],
        ];
        let res_names = [*b"MET", *b"LEU", *b"GLY", *b"CYS"];
        let res_seqs = [1, 2, 3, 4];
        let mut ws = AllAtomWorkspace::new(10);
        let solver = KinematicBackboneSolver::default();
        let metrics = solver.reconstruct_into(&ca_coords, &res_names, &res_seqs, &mut ws, true);

        assert!(metrics.is_some());
        let m = metrics.unwrap();
        assert!(m.total_atoms > 15);

        // MET (res 0) has CB + 3 sidechain atoms (CG, SD, CE)
        assert_eq!(ws.out_residues[0].sidechain_len, 3);
        assert_eq!(ws.out_residues[0].sidechain_atoms[0].name_str(), "CG");
        assert_eq!(ws.out_residues[0].sidechain_atoms[1].name_str(), "SD");
        assert_eq!(ws.out_residues[0].sidechain_atoms[2].name_str(), "CE");

        // LEU (res 1) has CB + 3 sidechain atoms (CG, CD1, CD2)
        assert_eq!(ws.out_residues[1].sidechain_len, 3);
        assert_eq!(ws.out_residues[1].sidechain_atoms[0].name_str(), "CG");
        assert_eq!(ws.out_residues[1].sidechain_atoms[1].name_str(), "CD1");
        assert_eq!(ws.out_residues[1].sidechain_atoms[2].name_str(), "CD2");

        // GLY (res 2) has 0 sidechain atoms
        assert_eq!(ws.out_residues[2].sidechain_len, 0);

        // CYS (res 3) has CB + 1 sidechain atom (SG)
        assert_eq!(ws.out_residues[3].sidechain_len, 1);
        assert_eq!(ws.out_residues[3].sidechain_atoms[0].name_str(), "SG");

        let pdb_str = coords_to_allatom_pdb_string("test_packed", &ws.out_residues[0..4]);
        assert!(pdb_str.contains("ALL-ATOM BACKBONE + ROTAMERIC SIDECHAINS"));
        assert!(pdb_str.contains("ATOM      6  CG  MET A   1"));
        assert!(pdb_str.contains("ATOM      7  SD  MET A   1"));
        assert!(pdb_str.contains("ATOM      8  CE  MET A   1"));
        assert!(pdb_str.ends_with("END\n"));
    }

    #[test]
    fn test_struct5_ramachandran_basins_and_bbdep_rotamers() {
        // 1. Verify Ramachandran classification
        assert_eq!(classify_ramachandran(-60.0, -45.0), RamachandranBasin::Alpha);
        assert_eq!(classify_ramachandran(-120.0, 130.0), RamachandranBasin::Beta);
        assert_eq!(classify_ramachandran(-75.0, 145.0), RamachandranBasin::Ppii);
        assert_eq!(classify_ramachandran(55.0, 45.0), RamachandranBasin::AlphaL);
        assert_eq!(classify_ramachandran(0.0, 0.0), RamachandranBasin::General);

        // 2. Dihedral angle calculation: cis (0 deg), trans (180 deg)
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [0.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];
        let p4_trans = [-1.0, 1.0, 0.0];
        let p4_cis = [1.0, 1.0, 0.0];

        let d_trans = calc_dihedral_deg(p1, p2, p3, p4_trans);
        let d_cis = calc_dihedral_deg(p1, p2, p3, p4_cis);
        assert!((d_trans.abs() - 180.0).abs() < 1e-4, "Trans dihedral should be ~180°");
        assert!(d_cis.abs() < 1e-4, "Cis dihedral should be ~0°");

        // 3. Backbone-dependent rotamer conditioning
        let rots_alpha = get_rotamer_candidates_bbdep("LEU", RamachandranBasin::Alpha);
        let rots_beta = get_rotamer_candidates_bbdep("LEU", RamachandranBasin::Beta);
        let rots_gen = get_rotamer_candidates_bbdep("LEU", RamachandranBasin::General);

        assert!(!rots_alpha.is_empty());
        assert!(!rots_beta.is_empty());
        assert!(!rots_gen.is_empty());
        // Alpha basin for Leu typically favors chi1 = -65° (-1.1345 rad), Beta favors chi1 = 175° (3.0543 rad)
        assert_eq!(rots_alpha[0].chi1, -1.1345);
        assert_eq!(rots_beta[0].chi1, 3.0543);

        // 4. Verify retrocompatible get_rotamer_candidates matches General basin
        let rots_legacy = get_rotamer_candidates("LEU");
        assert_eq!(rots_legacy.len(), rots_gen.len());
        for i in 0..rots_legacy.len() {
            assert_eq!(rots_legacy[i].chi1, rots_gen[i].chi1);
            assert_eq!(rots_legacy[i].chi2, rots_gen[i].chi2);
        }
    }

    #[test]
    fn test_dynamic_bfactors_and_pdb_columns() {
        let ca_coords = [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [6.0, 3.1, 0.0],
        ];
        let res_names = [*b"ALA", *b"CYS", *b"GLY"];
        let res_seqs = [1, 2, 3];
        let b_factors = [88.50, 92.15, 65.40];

        let mut ws = AllAtomWorkspace::new(10);
        let solver = KinematicBackboneSolver::default();
        solver.reconstruct_into_with_bfactors(
            &ca_coords,
            &res_names,
            &res_seqs,
            Some(&b_factors),
            &mut ws,
            true,
        );

        assert_eq!(ws.out_residues[0].b_factor, 88.50);
        assert_eq!(ws.out_residues[1].b_factor, 92.15);
        assert_eq!(ws.out_residues[2].b_factor, 65.40);

        let pdb_str = coords_to_allatom_pdb_string("TEST_BFAC", &ws.out_residues[0..3]);
        // Validate PDB format columns: occupancy 1.00 followed by {:6.2} B-factor
        assert!(pdb_str.contains("  1.00 88.50"));
        assert!(pdb_str.contains("  1.00 92.15"));
        assert!(pdb_str.contains("  1.00 65.40"));

        // CA-only PDB string with B-factors
        let ca_pdb = coords_to_pdb_string_with_bfactors("ACG", "TEST_CA", &ca_coords, Some(&b_factors));
        assert!(ca_pdb.contains("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 88.50           C"));
        assert!(ca_pdb.contains("ATOM      2  CA  CYS A   2       3.800   0.000   0.000  1.00 92.15           C"));
        assert!(ca_pdb.contains("ATOM      3  CA  GLY A   3       6.000   3.100   0.000  1.00 65.40           C"));
    }

    #[test]
    fn test_struct6_chemical_flags_and_weights() {
        // Serine hydroxyl: donor and acceptor
        let ser_og = classify_chem_flag("SER", "OG");
        assert_eq!(ser_og, ATOM_POLAR_DONOR | ATOM_POLAR_ACCEPTOR);

        // Aspartate carboxylate: acceptor and anion
        let asp_od1 = classify_chem_flag("ASP", "OD1");
        assert_eq!(asp_od1, ATOM_POLAR_ACCEPTOR | ATOM_ANION);
        assert_eq!(charge_weight("OD1"), 0.5);

        // Lysine amino: donor and cation
        let lys_nz = classify_chem_flag("LYS", "NZ");
        assert_eq!(lys_nz, ATOM_POLAR_DONOR | ATOM_CATION);
        assert_eq!(charge_weight("NZ"), 1.0);

        // Arginine guanidinium
        let arg_nh1 = classify_chem_flag("ARG", "NH1");
        assert_eq!(arg_nh1, ATOM_POLAR_DONOR | ATOM_CATION);
        assert_eq!(charge_weight("NH1"), 0.4);

        // Backbone atoms
        assert_eq!(classify_chem_flag("ALA", "N"), ATOM_POLAR_DONOR);
        assert_eq!(classify_chem_flag("PRO", "N"), 0); // Proline has no amide H
        assert_eq!(classify_chem_flag("GLY", "O"), ATOM_POLAR_ACCEPTOR);

        // Hydrophobic non-polar sidechain atoms
        assert_eq!(classify_chem_flag("VAL", "CG1"), 0);
        assert_eq!(classify_chem_flag("LEU", "CD1"), 0);
        assert_eq!(classify_chem_flag("PHE", "CZ"), 0);
    }

    #[test]
    fn test_struct6_hbond_potential() {
        let w_hb = 2.50;
        // Optimal distance r0 = 2.80 Å with ideal angle gives exact -w_hb
        let e_opt = hbond_energy(2.80, 1.0, w_hb);
        assert!((e_opt - (-2.50)).abs() < 1e-4, "Expected -2.50 at r0, got {}", e_opt);

        // Out of active window (< 2.30 or > 3.40) must be 0.0
        assert_eq!(hbond_energy(2.20, 1.0, w_hb), 0.0);
        assert_eq!(hbond_energy(3.50, 1.0, w_hb), 0.0);

        // Suboptimal angle scales as ang_factor (cos^2(theta))
        let e_half = hbond_energy(2.80, 0.5, w_hb);
        assert!((e_half - (-1.25)).abs() < 1e-4);

        // Orthogonal or backward angle gives 0.0
        assert_eq!(hbond_energy(2.80, 0.0, w_hb), 0.0);
        assert_eq!(hbond_energy(2.80, -0.5, w_hb), 0.0);
    }

    #[test]
    fn test_struct6_salt_bridge_potential() {
        let w_salt = 3.00;
        // Optimal distance r_opt = 2.90 Å gives exact -w_salt
        let e_opt = salt_bridge_energy(2.90, w_salt);
        assert!((e_opt - (-3.00)).abs() < 1e-4, "Expected -3.00 at r_opt, got {}", e_opt);

        // Like charge repulsion is +50% of attractive magnitude
        let e_rep = like_charge_repulsion(2.90, w_salt);
        assert!((e_rep - 1.50).abs() < 1e-4, "Expected +1.50 for like charges, got {}", e_rep);

        // Out of active window (< 2.30 or > 4.20) must be 0.0
        assert_eq!(salt_bridge_energy(2.10, w_salt), 0.0);
        assert_eq!(salt_bridge_energy(4.30, w_salt), 0.0);
        assert_eq!(like_charge_repulsion(2.10, w_salt), 0.0);
        assert_eq!(like_charge_repulsion(4.30, w_salt), 0.0);
    }
}
