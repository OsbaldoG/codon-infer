#!/usr/bin/env python3
"""
Export ESM-2 8M weights from fair-esm to binary format compatible with codon-infer.
Output order matches Rust EsmWeights struct reading order.
PyTorch Linear weights are [out_features, in_features], forward: x @ W.T
Our matmul expects [out_dim, in_dim] and computes x @ W.T
So export weights in PyTorch native format (NO transpose).
"""
import torch
import esm
import struct

def export_weights():
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    
    state_dict = model.state_dict()
    
    # Print all keys and shapes
    for k, v in state_dict.items():
        print(f"{k}: {v.shape} ({v.numel()} elements)")
    
    with open("esm2_8m_weights.bin", "wb") as f:
        # 1. Token embeddings: [33, 320] (vocab_size, embed_dim)
        token_emb = state_dict["embed_tokens.weight"].numpy().astype('float32')
        f.write(struct.pack(f"{token_emb.size}f", *token_emb.flatten()))
        
        # 2. emb_layer_norm_after
        emb_ln_w = state_dict["emb_layer_norm_after.weight"].numpy().astype('float32')
        emb_ln_b = state_dict["emb_layer_norm_after.bias"].numpy().astype('float32')
        f.write(struct.pack(f"{emb_ln_w.size}f", *emb_ln_w.flatten()))
        f.write(struct.pack(f"{emb_ln_b.size}f", *emb_ln_b.flatten()))
        
        # 3. All ln_1_weight across layers: [6, 320]
        for l in range(6):
            ln1_w = state_dict[f"layers.{l}.self_attn_layer_norm.weight"].numpy().astype('float32')
            f.write(struct.pack(f"{ln1_w.size}f", *ln1_w.flatten()))
        
        # 4. All ln_1_bias across layers: [6, 320]
        for l in range(6):
            ln1_b = state_dict[f"layers.{l}.self_attn_layer_norm.bias"].numpy().astype('float32')
            f.write(struct.pack(f"{ln1_b.size}f", *ln1_b.flatten()))
        
        # 5. All q_proj weights: [6, 320, 320] (out_features=320, in_features=320)
        for l in range(6):
            q_w = state_dict[f"layers.{l}.self_attn.q_proj.weight"].numpy().astype('float32')  # NO transpose
            f.write(struct.pack(f"{q_w.size}f", *q_w.flatten()))
        
        # 6. All q_proj biases: [6, 320]
        for l in range(6):
            q_b = state_dict[f"layers.{l}.self_attn.q_proj.bias"].numpy().astype('float32')
            f.write(struct.pack(f"{q_b.size}f", *q_b.flatten()))
        
        # 7. All k_proj weights: [6, 320, 320]
        for l in range(6):
            k_w = state_dict[f"layers.{l}.self_attn.k_proj.weight"].numpy().astype('float32')
            f.write(struct.pack(f"{k_w.size}f", *k_w.flatten()))
        
        # 8. All k_proj biases: [6, 320]
        for l in range(6):
            k_b = state_dict[f"layers.{l}.self_attn.k_proj.bias"].numpy().astype('float32')
            f.write(struct.pack(f"{k_b.size}f", *k_b.flatten()))
        
        # 9. All v_proj weights: [6, 320, 320]
        for l in range(6):
            v_w = state_dict[f"layers.{l}.self_attn.v_proj.weight"].numpy().astype('float32')
            f.write(struct.pack(f"{v_w.size}f", *v_w.flatten()))
        
        # 10. All v_proj biases: [6, 320]
        for l in range(6):
            v_b = state_dict[f"layers.{l}.self_attn.v_proj.bias"].numpy().astype('float32')
            f.write(struct.pack(f"{v_b.size}f", *v_b.flatten()))
        
        # 11. All out_proj weights: [6, 320, 320]
        for l in range(6):
            o_w = state_dict[f"layers.{l}.self_attn.out_proj.weight"].numpy().astype('float32')
            f.write(struct.pack(f"{o_w.size}f", *o_w.flatten()))
        
        # 12. All out_proj biases: [6, 320]
        for l in range(6):
            o_b = state_dict[f"layers.{l}.self_attn.out_proj.bias"].numpy().astype('float32')
            f.write(struct.pack(f"{o_b.size}f", *o_b.flatten()))
        
        # 13. All ln_2_weight across layers: [6, 320]
        for l in range(6):
            ln2_w = state_dict[f"layers.{l}.final_layer_norm.weight"].numpy().astype('float32')
            f.write(struct.pack(f"{ln2_w.size}f", *ln2_w.flatten()))
        
        # 14. All ln_2_bias across layers: [6, 320]
        for l in range(6):
            ln2_b = state_dict[f"layers.{l}.final_layer_norm.bias"].numpy().astype('float32')
            f.write(struct.pack(f"{ln2_b.size}f", *ln2_b.flatten()))
        
        # 15. All fc1 weights: [6, 1280, 320] (out_features=1280, in_features=320)
        for l in range(6):
            fc1_w = state_dict[f"layers.{l}.fc1.weight"].numpy().astype('float32')  # NO transpose
            f.write(struct.pack(f"{fc1_w.size}f", *fc1_w.flatten()))
        
        # 16. All fc1 biases: [6, 1280]
        for l in range(6):
            fc1_b = state_dict[f"layers.{l}.fc1.bias"].numpy().astype('float32')
            f.write(struct.pack(f"{fc1_b.size}f", *fc1_b.flatten()))
        
        # 17. All fc2 weights: [6, 320, 1280] (out_features=320, in_features=1280)
        for l in range(6):
            fc2_w = state_dict[f"layers.{l}.fc2.weight"].numpy().astype('float32')  # NO transpose
            f.write(struct.pack(f"{fc2_w.size}f", *fc2_w.flatten()))
        
        # 18. All fc2 biases: [6, 320]
        for l in range(6):
            fc2_b = state_dict[f"layers.{l}.fc2.bias"].numpy().astype('float32')
            f.write(struct.pack(f"{fc2_b.size}f", *fc2_b.flatten()))
        
        # 19. lm_head.layer_norm weight: [320]
        lm_ln_w = state_dict["lm_head.layer_norm.weight"].numpy().astype('float32')
        f.write(struct.pack(f"{lm_ln_w.size}f", *lm_ln_w.flatten()))
        
        # 20. lm_head.layer_norm bias: [320]
        lm_ln_b = state_dict["lm_head.layer_norm.bias"].numpy().astype('float32')
        f.write(struct.pack(f"{lm_ln_b.size}f", *lm_ln_b.flatten()))
        
        # 21. lm_head.dense weight: [320, 320] (out=320, in=320)
        lm_dense_w = state_dict["lm_head.dense.weight"].numpy().astype('float32')  # NO transpose
        f.write(struct.pack(f"{lm_dense_w.size}f", *lm_dense_w.flatten()))
        
        # 22. lm_head.dense bias: [320]
        lm_dense_b = state_dict["lm_head.dense.bias"].numpy().astype('float32')
        f.write(struct.pack(f"{lm_dense_b.size}f", *lm_dense_b.flatten()))
        
        # 23. lm_head.weight (classifier): [33, 320] (out=33, in=320)
        lm_head_w = state_dict["lm_head.weight"].numpy().astype('float32')  # NO transpose
        f.write(struct.pack(f"{lm_head_w.size}f", *lm_head_w.flatten()))
        
        # 24. lm_head.bias (classifier): [33]
        lm_head_b = state_dict["lm_head.bias"].numpy().astype('float32')
        f.write(struct.pack(f"{lm_head_b.size}f", *lm_head_b.flatten()))
    
    print("Weights exported to esm2_8m_weights.bin")

if __name__ == "__main__":
    export_weights()