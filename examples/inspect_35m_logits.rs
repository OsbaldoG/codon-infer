use codon_infer::{
    Tokenizer, EsmConfig, EsmWeights, EsmWeightsQ4,
    BidirectionalRunState, forward_bidirectional_parallel,
    forward_bidirectional_q4,
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
    let config = EsmConfig::esm2_35m();

    let weights_fp = EsmWeights::from_file("esm2_35m_weights.bin", &config).unwrap();
    let weights_q4 = EsmWeightsQ4::load_from_file("esm2_35m_q4_0.bin").unwrap().1;

    let mut state_fp = BidirectionalRunState::new(&config, seq_len + 8);
    forward_bidirectional_parallel(&tokens, &config, &weights_fp, &mut state_fp, 4);

    let mut state_q4 = BidirectionalRunState::new(&config, seq_len + 8);
    forward_bidirectional_q4(&tokens, &config, &weights_q4, &mut state_q4, 4);

    println!("POS | TOKEN | FP32 TOP1 (prob) | Q4 TOP1 (prob) | Q4 FP32_TOK_RANK | MATCH");
    println!("----+-------+------------------+----------------+------------------+------");

    let mut p_fp = vec![0.0f32; 33];
    let mut p_q4 = vec![0.0f32; 33];

    for t in 0..seq_len {
        let tok = tokens[t];
        let tok_ch = id_to_char(tok);

        let slice_fp = &state_fp.logits[t * 33..(t + 1) * 33];
        let slice_q4 = &state_q4.logits[t * 33..(t + 1) * 33];

        softmax(&mut p_fp, slice_fp);
        softmax(&mut p_q4, slice_q4);

        let mut sorted_fp: Vec<(usize, f32)> = p_fp.iter().cloned().enumerate().collect();
        sorted_fp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut sorted_q4: Vec<(usize, f32)> = p_q4.iter().cloned().enumerate().collect();
        sorted_q4.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_fp = sorted_fp[0];
        let top_q4 = sorted_q4[0];

        // Rank of FP32 top1 in Q4
        let rank_in_q4 = sorted_q4.iter().position(|x| x.0 == top_fp.0).unwrap() + 1;

        let is_match = if top_fp.0 == top_q4.0 { "YES" } else { "NO" };

        println!(
            "{:>3} | {:>5} | {} ({:.3})       | {} ({:.3})      | rank {:>2}          | {}",
            t, tok_ch,
            id_to_char(top_fp.0), top_fp.1,
            id_to_char(top_q4.0), top_q4.1,
            rank_in_q4, is_match
        );
    }
}
