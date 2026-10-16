use std::time::Instant;
use codon_infer::{EsmConfig, EsmWeightsQ4, EsmWeightsQ8};

fn main() {
    let q4_path = "esm2_650m_q4_0.bin";
    let q8_path = "esm2_650m_q8_0.bin";
    let cfg = EsmConfig::esm2_650m();

    println!("================================================================================");
    println!(" BENCHMARK COMPARATIVO: CARGA FRÍA (DISK I/O) VS CALIENTE (PAGE CACHE)");
    println!("================================================================================\n");

    if std::path::Path::new(q4_path).exists() {
        // Run 1: Warm Load of Q4_0 (already in page cache from recent usage)
        let t0 = Instant::now();
        let _w_q4_1 = EsmWeightsQ4::load_from_file(q4_path).unwrap();
        let d1 = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let _w_q4_2 = EsmWeightsQ4::load_from_file(q4_path).unwrap();
        let d2 = t1.elapsed().as_secs_f64() * 1000.0;

        println!("650M Q4_0 (414.6 MB):");
        println!("  - Carga 1 (Cache caliente): {:.2} ms ({:.2} MB/s)", d1, 414.62 / (d1 / 1000.0));
        println!("  - Carga 2 (Cache caliente): {:.2} ms ({:.2} MB/s)\n", d2, 414.62 / (d2 / 1000.0));
    }

    if std::path::Path::new(q8_path).exists() {
        let t0 = Instant::now();
        let _w_q8_1 = EsmWeightsQ8::from_file(q8_path, &cfg).unwrap();
        let d1 = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let _w_q8_2 = EsmWeightsQ8::from_file(q8_path, &cfg).unwrap();
        let d2 = t1.elapsed().as_secs_f64() * 1000.0;

        println!("650M Q8_0 (738.9 MB):");
        println!("  - Carga 1 (Cache caliente): {:.2} ms ({:.2} MB/s)", d1, 738.90 / (d1 / 1000.0));
        println!("  - Carga 2 (Cache caliente): {:.2} ms ({:.2} MB/s)\n", d2, 738.90 / (d2 / 1000.0));
    }
}
