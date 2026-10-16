#!/usr/bin/env python3
"""
Generate ground-truth reference outputs from HuggingFace ESM-2 (8M)
for comparison against codon-infer FP32 engine.
Model: facebook/esm2_t6_8M_UR50D
Input: test_peptidase.fasta (39 aa + BOS/EOS = 41 tokens)
Output: scripts/ref_8m.npz
"""

import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def read_fasta(path: str) -> str:
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                lines.append(line)
    return "".join(lines)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fasta_path = os.path.join(repo_root, "test_peptidase.fasta")
    out_npz = os.path.join(repo_root, "scripts", "ref_8m.npz")

    if not os.path.exists(fasta_path):
        print(f"Error: {fasta_path} not found.", file=sys.stderr)
        sys.exit(1)

    sequence = read_fasta(fasta_path)
    print(f"Read FASTA sequence ({len(sequence)} aa): {sequence}")

    model_id = "facebook/esm2_t6_8M_UR50D"
    print(f"Loading HuggingFace model & tokenizer: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.eval()

    inputs = tokenizer(sequence, return_tensors="pt")
    input_ids = inputs["input_ids"] # shape [1, 41]
    tokens = input_ids[0].cpu().numpy()
    print(f"Token IDs ({len(tokens)} tokens): {tokens.tolist()}")

    # Capture actual raw output of layer 5 before emb_layer_norm_after
    layer_outputs = {}
    def make_hook(idx):
        def hook(mod, inp, outp):
            layer_outputs[idx] = outp[0].detach().cpu().numpy().astype(np.float32)
        return hook

    for i in range(model.config.num_hidden_layers):
        model.esm.encoder.layer[i].register_forward_hook(make_hook(i))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states:
    # 0: initial word embeddings [1, 41, 320] (with token_dropout compensation if enabled)
    # 1..5: post layer 0..4
    # 6: last_hidden_state (already normalized by emb_layer_norm_after)
    initial_embed = outputs.hidden_states[0][0].detach().cpu().numpy().astype(np.float32)

    layer_states = [layer_outputs[i] for i in range(model.config.num_hidden_layers)]
    stacked_layers = np.stack(layer_states, axis=0) # [6, 41, 320]

    # Output after emb_layer_norm_after: exactly outputs.hidden_states[6]
    emb_ln_after = outputs.hidden_states[6][0].detach().cpu().numpy().astype(np.float32)

    # Logits
    logits = outputs.logits[0].detach().cpu().numpy().astype(np.float32) # [41, 33]

    # Top-1 prediction check without mask
    pred_ids = logits.argmax(axis=-1)
    pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids.tolist())
    pred_seq = "".join(pred_tokens[1:-1])
    matches = sum(a == b for a, b in zip(sequence, pred_seq))
    acc = matches / len(sequence) * 100.0
    print(f"Top-1 predicted sequence: {pred_seq}")
    print(f"Unmasked Top-1 accuracy:  {matches}/{len(sequence)} ({acc:.2f}%)")

    np.savez(
        out_npz,
        tokens=tokens,
        initial_embed=initial_embed,
        layer_0=layer_states[0],
        layer_1=layer_states[1],
        layer_2=layer_states[2],
        layer_3=layer_states[3],
        layer_4=layer_states[4],
        layer_5=layer_states[5],
        layers=stacked_layers,
        emb_layer_norm_after=emb_ln_after,
        logits=logits,
    )

    size_kb = os.path.getsize(out_npz) / 1024
    print(f"Saved reference outputs to {out_npz} ({size_kb:.2f} KB)")


if __name__ == "__main__":
    main()
