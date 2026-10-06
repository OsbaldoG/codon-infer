#!/usr/bin/env python3
"""
Export PyTorch/ESM reference logits for numerical parity testing.
Uses facebookresearch/esm pretrained model esm2_t6_8M_UR50D.
Runs with causal attention mask to match codon-infer's autoregressive forward pass.
"""
import torch
import esm
import json
import struct
import sys

def export_parity_reference():
    sequence = "MCMPCFTTDHQMARKCDD"
    
    print(f"Loading ESM-2 8M model (esm2_t6_8M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    
    batch_converter = alphabet.get_batch_converter()
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    print(f"Input tokens shape: {batch_tokens.shape}")
    print(f"Input tokens: {batch_tokens[0].tolist()}")
    
    # Full sequence with <cls> and <eos>
    full_tokens = batch_tokens[0]  # [20] = [<cls>, 18 AAs, <eos>]
    # Prompt tokens: exclude <eos> -> [<cls>, 18 AAs] = 19 tokens
    prompt_tokens = full_tokens[:-1]  # [19]
    seq_len = prompt_tokens.shape[0]  # 19
    
    print(f"Prompt tokens (no EOS): {prompt_tokens.tolist()}")
    print(f"Prompt length: {seq_len}")
    
    # Create causal mask: upper triangular (no future attention)
    # Shape: [seq_len, seq_len]
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    # Run model layer by layer with causal mask
    with torch.no_grad():
        # Embedding
        x = model.embed_scale * model.embed_tokens(prompt_tokens.unsqueeze(0))  # [1, 19, 320]
        
        # Token dropout not applied in eval mode
        
        padding_mask = prompt_tokens.unsqueeze(0).eq(model.padding_idx)  # [1, 19]
        if not padding_mask.any():
            padding_mask = None
        
        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)  # [19, 1, 320]
        
        # Apply transformer layers with causal mask
        for layer_idx, layer in enumerate(model.layers):
            x, _ = layer(
                x,
                self_attn_mask=causal_mask,
                self_attn_padding_mask=padding_mask,
                need_head_weights=False,
            )
        
        # emb_layer_norm_after
        x = model.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E) = [1, 19, 320]
        
        # lm_head
        x = model.lm_head(x)  # [1, 19, 33]
        
        logits = x[0]  # [19, 33]
        print(f"Logits shape: {logits.shape}")
        
        # Logits at last position (after last amino acid, predicting <eos>)
        final_logits = logits[-1].float().numpy()  # Position 18
        print(f"Final logits shape: {final_logits.shape}")
        print(f"Final logits (first 10): {final_logits[:10]}")
        print(f"Final logits stats: min={final_logits.min():.6f}, max={final_logits.max():.6f}, mean={final_logits.mean():.6f}")
    
    output_bin = "parity_reference.bin"
    with open(output_bin, "wb") as f:
        f.write(struct.pack("I", len(final_logits)))
        f.write(struct.pack(f"{len(final_logits)}f", *final_logits))
    print(f"Exported binary reference to {output_bin}")
    
    output_json = "parity_reference.json"
    with open(output_json, "w") as f:
        json.dump({
            "sequence": sequence,
            "vocab_size": int(final_logits.shape[0]),
            "logits": final_logits.tolist()
        }, f)
    print(f"Exported JSON reference to {output_json}")
    
    return final_logits

if __name__ == "__main__":
    export_parity_reference()