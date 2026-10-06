use std::fs::File;
use codon_infer::{EsmConfig, EsmWeights, RunState, forward, Tokenizer};
use ndarray::Array;
use npyz::npz::NpzArchive;

/// Load an array from an NPZ file by name and return it as a dynamic array.
fn load_array_by_name(path: &str, name: &str) -> ndarray::ArrayD<f32> {
    let file = File::open(path).expect("Failed to open npz file");
    let mut archive = NpzArchive::new(file).expect("Failed to create npz archive");
    let npy_file = archive.by_name(name).expect("Failed to get array by name").unwrap_or_else(|| panic!("Array '{}' not found", name));
    let shape = npy_file.shape().iter().map(|&x| x as usize).collect::<Vec<_>>();
    let dtype = npy_file.dtype();
    let type_str = match dtype {
        npyz::DType::Plain(ref s) => s.to_string(),
        _ => panic!("Expected plain dtype"),
    };
    let vec = match type_str.as_str() {
        "<f4" | ">f4" => {
            let mut v = Vec::new();
            for item in npy_file.data::<f32>().expect("Failed to create data iterator for f32").map(|item| item.expect("Failed to read item")) {
                v.push(item);
            }
            v
        }
        "<f8" | ">f8" => {
            let mut v = Vec::new();
            for item in npy_file.data::<f64>().expect("Failed to create data iterator for f64").map(|item| item.expect("Failed to read item")) {
                v.push(item as f32);
            }
            v
        }
        _ => panic!("Unsupported dtype for f32 conversion: {} for array '{}'", type_str, name),
    };
    Array::from_shape_vec(shape, vec).expect("Failed to create array")
}

#[test]
fn test_numerical_parity_per_layer() {
    // List of sequences to test
    let sequences = [
        ("seq1", "MCMPCFTTDHQMARKCDD"),
        ("seq2", "ARK"),
        ("seq3", "ACDEFGHIKLMNPQRSTVWY"),
        ("seq4", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"),
        ("seq5", &"ACDEFGHIKLMNPQRSTVWY".repeat(20)),
        ("seq6", &"ACDEFGHIKLMNPQRSTVWY".repeat(50)),
    ];

    for (name, sequence) in &sequences {
        println!("Testing sequence {}: {}", name, sequence);
        let config = EsmConfig {
            dim: 320,
            hidden_dim: 1280,
            n_layers: 6,
            n_heads: 20,
            vocab_size: 33,
            max_seq_len: 1024,
        };

        let mut file = File::open("esm2_8m_weights.bin").expect("Failed to open model weights");
        let weights = EsmWeights::load(&mut file, &config).expect("Failed to load weights");

        let mut run_state = RunState::new(&config);
        let tokens = Tokenizer::encode(sequence);
        // Exclude <eos> for prompt
        let prompt_tokens = &tokens[..tokens.len()-1];

        // We will collect the logits for each position
        let mut all_logits = Vec::new();

        for (pos, &token) in prompt_tokens.iter().enumerate() {
            if pos == 0 {
                // We don't need to store embedding_out anymore
            }
            forward(token, pos, &config, &weights, &mut run_state);
            // After each forward call, state.logits contains the logits for the next token.
            // We want to store the logits for the current position? Actually, the logits after processing
            // token at position pos are for the token at position pos+1.
            // But the reference array is for the prompt tokens, where the i-th row corresponds to the
            // logits for the i-th token in the prompt (to predict the next token).
            // So after processing the token at position pos, we have the logits for the next token (pos+1).
            // We want to store these logits for each pos from 0 to len(prompt_tokens)-1.
            all_logits.extend_from_slice(&run_state.logits);
        }

        // Load reference logits
        let npz_path = format!("/workspaces/codon-infer/parity_reference_{}.npz", name);
        let logits_ref: ndarray::ArrayD<f32> = load_array_by_name(&npz_path, "logits");

        // Compare logits
        let ref_vec: Vec<f32> = logits_ref.into_iter().collect();
        assert_eq!(all_logits.len(), ref_vec.len(), "Logits length mismatch for {}", name);
        let mut mae_sum = 0.0f32;
        let mut max_abs = 0.0f32;
        for (&our, &ref_) in all_logits.iter().zip(ref_vec.iter()) {
            let diff = (our - ref_).abs();
            mae_sum += diff;
            if diff > max_abs {
                max_abs = diff;
            }
        }
        let mae = mae_sum / all_logits.len() as f32;
        println!("  MAE: {:.6e}", mae);
        assert!(mae < 1e-3, "MAE too high for {}: {}", name, mae);
    }
}