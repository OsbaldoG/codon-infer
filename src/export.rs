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

/// Generates a standard Protein Data Bank (.pdb) string containing C_alpha backbone atoms.
pub fn coords_to_pdb_string(
    sequence: &str,
    protein_id: &str,
    coords: &[[f64; 3]],
) -> String {
    let mut out = String::with_capacity(coords.len() * 82 + 256);
    out.push_str(&format!("HEADER    CODON-INFER STRUCT-2 3D RECONSTRUCTION {}\n", protein_id));
    out.push_str("TITLE     C-ALPHA BACKBONE SOLVED VIA ZERO-TRUNK MDS + GRADIENT DESCENT\n");
    for (i, (ch, pt)) in sequence.chars().zip(coords.iter()).enumerate() {
        let res_num = (i + 1) as i32;
        let res_name = aa1_to_aa3(ch);
        out.push_str(&format!(
            "ATOM  {:5}  CA  {:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 50.00           C\n",
            res_num,
            res_name,
            res_num,
            pt[0],
            pt[1],
            pt[2]
        ));
    }
    out.push_str("END\n");
    out
}

/// Exports reconstructed 3D C_alpha coordinates to standard Protein Data Bank (.pdb) format.
pub fn export_ca_pdb(
    path: &str,
    sequence: &str,
    protein_id: &str,
    coords: &[[f64; 3]],
) -> io::Result<String> {
    let pdb_content = coords_to_pdb_string(sequence, protein_id, coords);
    let mut file = File::create(path)?;
    file.write_all(pdb_content.as_bytes())?;
    file.flush()?;
    Ok(pdb_content)
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

    // Coordinates array
    out.push_str("  \"coordinates\": [\n");
    for (i, (ch, pt)) in result.sequence.chars().zip(result.coords.iter()).enumerate() {
        let comma = if i + 1 < result.coords.len() { "," } else { "" };
        out.push_str(&format!(
            "    {{\"index\": {}, \"res\": \"{}\", \"x\": {:.3}, \"y\": {:.3}, \"z\": {:.3}}}{}\n",
            i + 1,
            ch,
            pt[0],
            pt[1],
            pt[2],
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
    let pdb_content = coords_to_pdb_string(&result.sequence, &result.protein_id, &result.coords);
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

/// Represents a fully reconstructed amino acid residue with all backbone heavy atoms (N, CA, C, O) and C_beta.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconstructedResidue {
    pub res_name: [u8; 3],
    pub res_seq: i32,
    pub n: [f64; 3],
    pub ca: [f64; 3],
    pub c: [f64; 3],
    pub o: [f64; 3],
    pub cb: Option<[f64; 3]>,
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
        }
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
    pub fn reconstruct_into(
        &self,
        ca_coords: &[[f64; 3]],
        res_names: &[[u8; 3]],
        res_seqs: &[i32],
        ws: &mut AllAtomWorkspace,
    ) {
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

            ws.out_residues[i] = ReconstructedResidue {
                res_name: res_names[i],
                res_seq: res_seqs[i],
                n: n_atom.to_array(),
                ca: ca.to_array(),
                c: c_atom.to_array(),
                o: o_atom.to_array(),
                cb,
            };
        }
    }
}

/// Generates a Protein Data Bank (.pdb) string containing full all-atom backbone (N, CA, C, O) and C_beta atoms.
pub fn coords_to_allatom_pdb_string(
    protein_id: &str,
    residues: &[ReconstructedResidue],
) -> String {
    let mut out = String::with_capacity(residues.len() * 400 + 256);
    out.push_str(&format!("HEADER    CODON-INFER STRUCT-3 ALL-ATOM RECONSTRUCTION {}\n", protein_id));
    out.push_str("TITLE     ALL-ATOM BACKBONE + CB SOLVED VIA PAULING-COREY KINEMATIC SOLVER\n");

    let mut atom_idx = 1;
    for res in residues {
        let name_str = std::str::from_utf8(&res.res_name).unwrap_or("UNK");
        out.push_str(&format!(
            "ATOM  {:5}  N   {:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 50.00           N\n",
            atom_idx, name_str, res.res_seq, res.n[0], res.n[1], res.n[2]
        ));
        atom_idx += 1;

        out.push_str(&format!(
            "ATOM  {:5}  CA  {:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 50.00           C\n",
            atom_idx, name_str, res.res_seq, res.ca[0], res.ca[1], res.ca[2]
        ));
        atom_idx += 1;

        out.push_str(&format!(
            "ATOM  {:5}  C   {:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 50.00           C\n",
            atom_idx, name_str, res.res_seq, res.c[0], res.c[1], res.c[2]
        ));
        atom_idx += 1;

        out.push_str(&format!(
            "ATOM  {:5}  O   {:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 50.00           O\n",
            atom_idx, name_str, res.res_seq, res.o[0], res.o[1], res.o[2]
        ));
        atom_idx += 1;

        if let Some(cb) = res.cb {
            out.push_str(&format!(
                "ATOM  {:5}  CB  {:3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 50.00           C\n",
                atom_idx, name_str, res.res_seq, cb[0], cb[1], cb[2]
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
        (String, Option<[f64; 3]>, Option<[f64; 3]>, Option<[f64; 3]>, Option<[f64; 3]>, Option<[f64; 3]>),
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
        let coord = [x, y, z];

        let entry = map.entry(res_seq).or_insert((res_name, None, None, None, None, None));
        match atom_name {
            "N" => entry.1 = Some(coord),
            "CA" => entry.2 = Some(coord),
            "C" => entry.3 = Some(coord),
            "O" => entry.4 = Some(coord),
            "CB" => entry.5 = Some(coord),
            _ => {}
        }
    }

    let residues: Vec<ReconstructedResidue> = map.into_iter()
        .filter_map(|(res_seq, (res_name, n_opt, ca_opt, c_opt, o_opt, cb_opt))| {
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
        solver.reconstruct_into(&ca_coords, &res_names, &res_seqs, &mut ws);

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
}
