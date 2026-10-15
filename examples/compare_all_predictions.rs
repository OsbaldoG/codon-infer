use codon_infer::{
    Tokenizer, EsmConfig, EsmWeights, EsmWeightsQ4,
    BidirectionalRunState, forward_bidirectional_parallel,
    forward_bidirectional_q4,
};

fn id_to_char(id: usize) -> char {
    match id {
        0 => '<', 1 => 'p', 2 => '>', 3 => '?',
        4=>'L',5=>'A',6=>'G',7=>'V',8=>'S',9=>'E',10=>'R',11=>'T',12=>'I',13=>'D',
        14=>'P',15=>'K',16=>'Q',17=>'N',18=>'F',19=>'Y',20=>'M',21=>'H',22=>'W',23=>'C',
        24=>'X',25=>'B',26=>'U',27=>'Z',28=>'O',29=>'.',30=>'-',32=>'*', _=>'?',
    }
}

fn main() {
    let seq = "AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS";
    let tokens = Tokenizer::encode(seq);
    let seq_len = tokens.len();

    let scales = [
        ("8M", "esm2_8m_weights.bin", "esm2_8m_q4_0.bin", EsmConfig::esm2_8m()),
        ("35M", "esm2_35m_weights.bin", "esm2_35m_q4_0.bin", EsmConfig::esm2_35m()),
        ("150M", "esm2_150m_weights.bin", "esm2_150m_q4_0.bin", EsmConfig::esm2_150m()),
        ("650M", "esm2_650m_weights.bin", "esm2_650m_q4_0.bin", EsmConfig::esm2_650m()),
    ];

    for (scale, fp_p, q4_p, cfg) in scales {
        let w_fp = EsmWeights::from_file(fp_p, &cfg).unwrap();
        let w_q4 = EsmWeightsQ4::load_from_file(q4_p).unwrap().1;
        let mut st_fp = BidirectionalRunState::new(&cfg, seq_len + 8);
        let mut st_q4 = BidirectionalRunState::new(&cfg, seq_len + 8);
        forward_bidirectional_parallel(&tokens, &cfg, &w_fp, &mut st_fp, 4);
        forward_bidirectional_q4(&tokens, &cfg, &w_q4, &mut st_q4, 4);

        let mut matches = 0;
        let mut fp_top = String::new();
        let mut q4_top = String::new();
        let mut in_str = String::new();
        for t in 0..seq_len {
            let tok = tokens[t];
            let s_fp = &st_fp.logits[t * 33..(t + 1) * 33];
            let s_q4 = &st_q4.logits[t * 33..(t + 1) * 33];
            let top_fp = s_fp.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let top_q4 = s_q4.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if top_fp == top_q4 { matches += 1; }
            in_str.push(id_to_char(tok));
            fp_top.push(id_to_char(top_fp));
            q4_top.push(id_to_char(top_q4));
        }
        println!("============================================================");
        println!("ESCALA {}", scale);
        println!("Input:   {}", in_str);
        println!("FP32:    {}", fp_top);
        println!("Q4_0:    {}", q4_top);
        println!("Matches: {}/{} ({:.1}%)\n", matches, seq_len, (matches as f64 / seq_len as f64) * 100.0);
    }
}
