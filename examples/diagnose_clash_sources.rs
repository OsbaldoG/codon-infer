use std::fs::File;
use std::io::{BufRead, BufReader};
use codon_infer::{
    SideChainAtom, ReconstructedResidue, RotamerWorkspace,
    pack_sidechains, compute_clash_breakdown, ClashBreakdown,
};

const TARGETS: [(&str, &str); 6] = [
    ("2RRU", "SQSTM_MOUSE_Tsuboyama_2023_2RRU"),
    ("2GP8", "VG08_BPP22_Tsuboyama_2023_2GP8"),
    ("2L2D", "OTU7A_HUMAN_Tsuboyama_2023_2L2D"),
    ("1AOY", "ARGR_ECOLI_Tsuboyama_2023_1AOY"),
    ("1O6X", "CBPA2_HUMAN_Tsuboyama_2023_1O6X"),
    ("3DKM", "HECD1_HUMAN_Tsuboyama_2023_3DKM"),
];

/// Parses complete PDB file extracting backbone and all deposited heavy atoms into ReconstructedResidues.
fn parse_crystal_pdb(path: &str) -> Vec<ReconstructedResidue> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open PDB {}: {}", path, e));
    let reader = BufReader::new(file);

    struct ResRaw {
        name: [u8; 3],
        seq: i32,
        n: Option<[f64; 3]>,
        ca: Option<[f64; 3]>,
        c: Option<[f64; 3]>,
        o: Option<[f64; 3]>,
        cb: Option<[f64; 3]>,
        sc_atoms: Vec<(String, [f64; 3], u8)>,
    }

    let mut map: std::collections::BTreeMap<i32, ResRaw> = std::collections::BTreeMap::new();

    for line in reader.lines().map_while(Result::ok) {
        if !line.starts_with("ATOM  ") {
            continue;
        }
        let atom_name = line[12..16].trim();
        let res_name = line[17..20].trim();
        let res_seq: i32 = line[22..26].trim().parse().unwrap_or(0);
        let x: f64 = line[30..38].trim().parse().unwrap_or(0.0);
        let y: f64 = line[38..46].trim().parse().unwrap_or(0.0);
        let z: f64 = line[46..54].trim().parse().unwrap_or(0.0);
        let coord = [x, y, z];

        let mut name_bytes = [b' '; 3];
        for (idx, &b) in res_name.as_bytes().iter().take(3).enumerate() {
            name_bytes[idx] = b;
        }

        let entry = map.entry(res_seq).or_insert_with(|| ResRaw {
            name: name_bytes,
            seq: res_seq,
            n: None,
            ca: None,
            c: None,
            o: None,
            cb: None,
            sc_atoms: Vec::new(),
        });

        match atom_name {
            "N" => entry.n = Some(coord),
            "CA" => entry.ca = Some(coord),
            "C" => entry.c = Some(coord),
            "O" => entry.o = Some(coord),
            "CB" => entry.cb = Some(coord),
            other => {
                let elem = match other.chars().next().unwrap_or('C') {
                    'N' => b'N',
                    'O' => b'O',
                    'S' => b'S',
                    _ => b'C',
                };
                entry.sc_atoms.push((other.to_string(), coord, elem));
            }
        }
    }

    let mut out = Vec::new();
    for (_, r) in map {
        if let (Some(n), Some(ca), Some(c), Some(o)) = (r.n, r.ca, r.c, r.o) {
            let mut res = ReconstructedResidue {
                res_name: r.name,
                res_seq: r.seq,
                n, ca, c, o,
                cb: r.cb,
                b_factor: 50.0,
                sidechain_len: 0,
                sidechain_atoms: [SideChainAtom::default(); 10],
            };
            let sc_count = r.sc_atoms.len().min(10);
            res.sidechain_len = sc_count as u8;
            for (i, (name, pos, elem)) in r.sc_atoms.into_iter().take(10).enumerate() {
                res.sidechain_atoms[i] = SideChainAtom::new(&name, pos, elem);
            }
            out.push(res);
        }
    }
    out
}

/// Parses predicted PDB (only backbone + CB).
fn parse_predicted_pdb(path: &str) -> Vec<ReconstructedResidue> {
    codon_infer::parse_pdb_atoms(path).unwrap_or_else(|e| panic!("Cannot open pred PDB {}: {}", path, e))
}

fn print_breakdown_row(label: &str, b: &ClashBreakdown) {
    println!(
        "{:<20} | {:>5} | {:>7.1} | {:>5} ({:>4.1}) | {:>5} ({:>4.1}) | {:>5} ({:>4.1}) | {:>5} ({:>4.1}) | {:>5} ({:>4.1}) | {:>4}",
        label,
        b.total_clashes,
        b.clashscore,
        b.bb_bb(),
        b.per_1000(b.bb_bb()),
        b.sc_bb(),
        b.per_1000(b.sc_bb()),
        b.sc_sc(),
        b.per_1000(b.sc_sc()),
        b.bb_bb_long,
        b.per_1000(b.bb_bb_long),
        b.bb_bb_local,
        b.per_1000(b.bb_bb_local),
        b.covalent_like,
    );
}

fn main() {
    println!("================================================================================================================");
    println!("  STRUCT-5a: DIAGNÓSTICO PROFUNDO DE FUENTES DE CHOQUE ESTÉRICO (CLASHSCORE BREAKDOWN)");
    println!("================================================================================================================");
    println!(
        "{:<20} | {:^5} | {:^7} | {:^13} | {:^13} | {:^13} | {:^13} | {:^13} | {:^4}",
        "Estado / Target", "Clash", "Score", "bb-bb (sc)", "sc-bb (sc)", "sc-sc (sc)", "bb-bb long", "bb-bb loc", "Cov<"
    );
    println!("----------------------------------------------------------------------------------------------------------------");

    let p_gt_dir = "/home/manuel/REA_GENOMICS/research/03_benchmarks_biologicos/pdb_ground_truth";
    let p_650_dir = "/home/manuel/REA_GENOMICS/research/03_benchmarks_biologicos/results_q3_benchmark";

    // 1. Control A: Cristal Nativo
    println!("\n[ CONTROL A: CRISTALES EXPERIMENTALES GROUND TRUTH ]");
    let mut total_gt_nat = ClashBreakdown::default();
    let mut total_gt_init = ClashBreakdown::default();
    let mut total_gt_pack = ClashBreakdown::default();

    for (tag, assay) in TARGETS {
        let path = format!("{}/{}.pdb", p_gt_dir, assay);
        let residues_crystal = parse_crystal_pdb(&path);
        let b_crystal = compute_clash_breakdown(&residues_crystal);
        print_breakdown_row(&format!("{} (Exp Dep)", tag), &b_crystal);

        // Strip sidechains and run initial rotamer assignment (r=0)
        let n = residues_crystal.len();
        let mut residues_pack = residues_crystal.clone();
        for r in &mut residues_pack {
            r.sidechain_len = 0;
        }

        // Pack sidechains with STRUCT-4
        let mut ws = RotamerWorkspace::new(n);
        let metrics = pack_sidechains(&mut residues_pack, &mut ws, 0.82, 10.0, 1.0);
        let b_packed = compute_clash_breakdown(&residues_pack);

        // Measure initial rotamer state (candidate 0)
        let mut residues_init = residues_crystal.clone();
        for i in 0..n {
            residues_init[i].sidechain_len = ws.cand_lens[i][0];
            residues_init[i].sidechain_atoms = ws.cand_atoms[i][0];
        }
        let b_init = compute_clash_breakdown(&residues_init);

        print_breakdown_row(&format!("{} (Init r=0)", tag), &b_init);
        print_breakdown_row(&format!("{} (STRUCT-4)", tag), &b_packed);
        println!("  └─> Metric report: init={:.1} final={:.1} drop={:.1}% pruned={}",
            metrics.initial_clashscore, metrics.final_clashscore, metrics.clash_reduction_pct, metrics.pruned_states);

        total_gt_nat.total_clashes += b_crystal.total_clashes;
        total_gt_nat.total_atoms += b_crystal.total_atoms;
        total_gt_nat.bb_bb_local += b_crystal.bb_bb_local;
        total_gt_nat.bb_bb_short += b_crystal.bb_bb_short;
        total_gt_nat.bb_bb_long += b_crystal.bb_bb_long;
        total_gt_nat.sc_bb_intra += b_crystal.sc_bb_intra;
        total_gt_nat.sc_bb_inter += b_crystal.sc_bb_inter;
        total_gt_nat.sc_sc_intra += b_crystal.sc_sc_intra;
        total_gt_nat.sc_sc_inter += b_crystal.sc_sc_inter;
        total_gt_nat.covalent_like += b_crystal.covalent_like;

        total_gt_init.total_clashes += b_init.total_clashes;
        total_gt_init.total_atoms += b_init.total_atoms;
        total_gt_init.bb_bb_local += b_init.bb_bb_local;
        total_gt_init.bb_bb_short += b_init.bb_bb_short;
        total_gt_init.bb_bb_long += b_init.bb_bb_long;
        total_gt_init.sc_bb_intra += b_init.sc_bb_intra;
        total_gt_init.sc_bb_inter += b_init.sc_bb_inter;
        total_gt_init.sc_sc_intra += b_init.sc_sc_intra;
        total_gt_init.sc_sc_inter += b_init.sc_sc_inter;
        total_gt_init.covalent_like += b_init.covalent_like;

        total_gt_pack.total_clashes += b_packed.total_clashes;
        total_gt_pack.total_atoms += b_packed.total_atoms;
        total_gt_pack.bb_bb_local += b_packed.bb_bb_local;
        total_gt_pack.bb_bb_short += b_packed.bb_bb_short;
        total_gt_pack.bb_bb_long += b_packed.bb_bb_long;
        total_gt_pack.sc_bb_intra += b_packed.sc_bb_intra;
        total_gt_pack.sc_bb_inter += b_packed.sc_bb_inter;
        total_gt_pack.sc_sc_intra += b_packed.sc_sc_intra;
        total_gt_pack.sc_sc_inter += b_packed.sc_sc_inter;
        total_gt_pack.covalent_like += b_packed.covalent_like;
    }
    total_gt_nat.clashscore = total_gt_nat.per_1000(total_gt_nat.total_clashes);
    total_gt_init.clashscore = total_gt_init.per_1000(total_gt_init.total_clashes);
    total_gt_pack.clashscore = total_gt_pack.per_1000(total_gt_pack.total_clashes);
    println!("----------------------------------------------------------------------------------------------------------------");
    print_breakdown_row("AVG CRISTAL EXP", &total_gt_nat);
    print_breakdown_row("AVG CRISTAL INIT", &total_gt_init);
    print_breakdown_row("AVG CRISTAL STRUCT4", &total_gt_pack);

    // 2. Control B: Traza Predicha ESM-2 650M
    println!("\n[ CONTROL B: TRAZAS PREDICHAS ESM-2 650M Q3_LUT ]");
    let mut total_pred_bb = ClashBreakdown::default();
    let mut total_pred_init = ClashBreakdown::default();
    let mut total_pred_pack = ClashBreakdown::default();

    for (tag, _) in TARGETS {
        let path = format!("{}/650M_{}_fold.pdb", p_650_dir, tag);
        let mut residues_pred = parse_predicted_pdb(&path);
        let n = residues_pred.len();

        // 2a. Backbone solo (sin cadenas laterales)
        let b_bb = compute_clash_breakdown(&residues_pred);
        print_breakdown_row(&format!("{} (BB 5-atom)", tag), &b_bb);

        // 2b. STRUCT-4 Sidechain Packing
        let mut ws = RotamerWorkspace::new(n);
        let metrics = pack_sidechains(&mut residues_pred, &mut ws, 0.82, 10.0, 1.0);
        let b_packed = compute_clash_breakdown(&residues_pred);

        // 2c. Initial candidate
        let mut residues_init = parse_predicted_pdb(&path);
        for i in 0..n {
            residues_init[i].sidechain_len = ws.cand_lens[i][0];
            residues_init[i].sidechain_atoms = ws.cand_atoms[i][0];
        }
        let b_init = compute_clash_breakdown(&residues_init);

        print_breakdown_row(&format!("{} (Init r=0)", tag), &b_init);
        print_breakdown_row(&format!("{} (STRUCT-4)", tag), &b_packed);
        println!("  └─> Metric report: init={:.1} final={:.1} drop={:.1}% pruned={}",
            metrics.initial_clashscore, metrics.final_clashscore, metrics.clash_reduction_pct, metrics.pruned_states);

        total_pred_bb.total_clashes += b_bb.total_clashes;
        total_pred_bb.total_atoms += b_bb.total_atoms;
        total_pred_bb.bb_bb_local += b_bb.bb_bb_local;
        total_pred_bb.bb_bb_short += b_bb.bb_bb_short;
        total_pred_bb.bb_bb_long += b_bb.bb_bb_long;
        total_pred_bb.covalent_like += b_bb.covalent_like;

        total_pred_init.total_clashes += b_init.total_clashes;
        total_pred_init.total_atoms += b_init.total_atoms;
        total_pred_init.bb_bb_local += b_init.bb_bb_local;
        total_pred_init.bb_bb_short += b_init.bb_bb_short;
        total_pred_init.bb_bb_long += b_init.bb_bb_long;
        total_pred_init.sc_bb_intra += b_init.sc_bb_intra;
        total_pred_init.sc_bb_inter += b_init.sc_bb_inter;
        total_pred_init.sc_sc_intra += b_init.sc_sc_intra;
        total_pred_init.sc_sc_inter += b_init.sc_sc_inter;
        total_pred_init.covalent_like += b_init.covalent_like;

        total_pred_pack.total_clashes += b_packed.total_clashes;
        total_pred_pack.total_atoms += b_packed.total_atoms;
        total_pred_pack.bb_bb_local += b_packed.bb_bb_local;
        total_pred_pack.bb_bb_short += b_packed.bb_bb_short;
        total_pred_pack.bb_bb_long += b_packed.bb_bb_long;
        total_pred_pack.sc_bb_intra += b_packed.sc_bb_intra;
        total_pred_pack.sc_bb_inter += b_packed.sc_bb_inter;
        total_pred_pack.sc_sc_intra += b_packed.sc_sc_intra;
        total_pred_pack.sc_sc_inter += b_packed.sc_sc_inter;
        total_pred_pack.covalent_like += b_packed.covalent_like;
    }
    total_pred_bb.clashscore = total_pred_bb.per_1000(total_pred_bb.total_clashes);
    total_pred_init.clashscore = total_pred_init.per_1000(total_pred_init.total_clashes);
    total_pred_pack.clashscore = total_pred_pack.per_1000(total_pred_pack.total_clashes);
    println!("----------------------------------------------------------------------------------------------------------------");
    print_breakdown_row("AVG PRED BACKBONE", &total_pred_bb);
    print_breakdown_row("AVG PRED INIT", &total_pred_init);
    print_breakdown_row("AVG PRED STRUCT4", &total_pred_pack);
    println!("================================================================================================================");
}
