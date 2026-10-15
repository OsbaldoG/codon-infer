#!/usr/bin/env python3
"""
codon-infer: ESM-2 Binary Weights Exporter (8M, 35M, 150M)
===========================================================
Exports ESM-2 model weights from HuggingFace Transformers or Meta fair-esm
into the standardized binary format expected by codon-infer.

Standardized output filenames:
  - esm2_8m_weights.bin   (~30 MB FP32)
  - esm2_35m_weights.bin  (~134 MB FP32)
  - esm2_150m_weights.bin (~592 MB FP32)

Usage:
  python scripts/export_weights.py --model 8M
  python scripts/export_weights.py --model 35M
  python scripts/export_weights.py --model 150M
  python scripts/export_weights.py --model all
"""

import argparse
import os
import struct
import sys

MODELS = {
    "8M": {
        "hf_name": "facebook/esm2_t6_8M_UR50D",
        "fair_func": "esm2_t6_8M_UR50D",
        "output": "esm2_8m_weights.bin",
        "n_layers": 6,
        "dim": 320,
        "hidden_dim": 1280,
    },
    "35M": {
        "hf_name": "facebook/esm2_t12_35M_UR50D",
        "fair_func": "esm2_t12_35M_UR50D",
        "output": "esm2_35m_weights.bin",
        "n_layers": 12,
        "dim": 480,
        "hidden_dim": 1920,
    },
    "150M": {
        "hf_name": "facebook/esm2_t30_150M_UR50D",
        "fair_func": "esm2_t30_150M_UR50D",
        "output": "esm2_150m_weights.bin",
        "n_layers": 30,
        "dim": 640,
        "hidden_dim": 2560,
    },
    "650M": {
        "hf_name": "facebook/esm2_t33_650M_UR50D",
        "fair_func": "esm2_t33_650M_UR50D",
        "output": "esm2_650m_weights.bin",
        "n_layers": 33,
        "dim": 1280,
        "hidden_dim": 5120,
    },
    "3B": {
        "hf_name": "facebook/esm2_t36_3B_UR50D",
        "fair_func": "esm2_t36_3B_UR50D",
        "output": "esm2_3b_weights.bin",
        "n_layers": 36,
        "dim": 2560,
        "hidden_dim": 10240,
    },
}

def export_fair_esm(meta, out_path):
    import esm
    import torch

    func_name = meta["fair_func"]
    print(f"Loading fair-esm model: {func_name} ...")
    model_fn = getattr(esm.pretrained, func_name)
    model, _ = model_fn()
    model.eval()
    state_dict = model.state_dict()
    n_layers = meta["n_layers"]

    with open(out_path, "wb") as f:
        # 1. Token embeddings: [33, dim]
        token_emb = state_dict["embed_tokens.weight"].numpy().astype("float32")
        f.write(struct.pack(f"{token_emb.size}f", *token_emb.flatten()))

        # 2. ln_1_weight across layers: [n_layers, dim]
        for l in range(n_layers):
            w = state_dict[f"layers.{l}.self_attn_layer_norm.weight"].numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 3. ln_1_bias across layers: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"layers.{l}.self_attn_layer_norm.bias"].numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 4. wq across layers: [n_layers, dim, dim]
        for l in range(n_layers):
            w = state_dict[f"layers.{l}.self_attn.q_proj.weight"].numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 5. bq across layers: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"layers.{l}.self_attn.q_proj.bias"].numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 6. wk across layers: [n_layers, dim, dim]
        for l in range(n_layers):
            w = state_dict[f"layers.{l}.self_attn.k_proj.weight"].numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 7. bk across layers: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"layers.{l}.self_attn.k_proj.bias"].numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 8. wv across layers: [n_layers, dim, dim]
        for l in range(n_layers):
            w = state_dict[f"layers.{l}.self_attn.v_proj.weight"].numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 9. bv across layers: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"layers.{l}.self_attn.v_proj.bias"].numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 10. wo across layers: [n_layers, dim, dim]
        for l in range(n_layers):
            w = state_dict[f"layers.{l}.self_attn.out_proj.weight"].numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 11. bo across layers: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"layers.{l}.self_attn.out_proj.bias"].numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 12. ln_2_weight across layers: [n_layers, dim]
        for l in range(n_layers):
            w = state_dict[f"layers.{l}.final_layer_norm.weight"].numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 13. ln_2_bias across layers: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"layers.{l}.final_layer_norm.bias"].numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 14. fc1_weight across layers: [n_layers, dim, hidden_dim]
        for l in range(n_layers):
            w = state_dict[f"layers.{l}.fc1.weight"].numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 15. fc1_bias across layers: [n_layers, hidden_dim]
        for l in range(n_layers):
            b = state_dict[f"layers.{l}.fc1.bias"].numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 16. fc2_weight across layers: [n_layers, hidden_dim, dim]
        for l in range(n_layers):
            w = state_dict[f"layers.{l}.fc2.weight"].numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 17. fc2_bias across layers: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"layers.{l}.fc2.bias"].numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 18. ln_f_weight and ln_f_bias: [dim]
        ln_f_w = state_dict["emb_layer_norm_after.weight"].numpy().astype("float32")
        ln_f_b = state_dict["emb_layer_norm_after.bias"].numpy().astype("float32")
        f.write(struct.pack(f"{ln_f_w.size}f", *ln_f_w.flatten()))
        f.write(struct.pack(f"{ln_f_b.size}f", *ln_f_b.flatten()))

        # 19. Classifier weights (EsmLMHead) if present
        if "lm_head.layer_norm.weight" in state_dict:
            for k in [
                "lm_head.layer_norm.weight",
                "lm_head.layer_norm.bias",
                "lm_head.dense.weight",
                "lm_head.dense.bias",
                "lm_head.weight",
                "lm_head.bias",
            ]:
                if k in state_dict:
                    t = state_dict[k].numpy().astype("float32")
                    f.write(struct.pack(f"{t.size}f", *t.flatten()))

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Exported {out_path} ({size_mb:.2f} MB)")

def export_huggingface(meta, out_path):
    from transformers import AutoModelForMaskedLM
    import torch

    hf_name = meta["hf_name"]
    print(f"Loading HuggingFace model: {hf_name} ...")
    model = AutoModelForMaskedLM.from_pretrained(hf_name)
    model.eval()
    state_dict = model.state_dict()
    n_layers = meta["n_layers"]

    with open(out_path, "wb") as f:
        # 1. Token embeddings: [33, dim]
        token_emb = state_dict["esm.embeddings.word_embeddings.weight"].cpu().numpy().astype("float32")
        f.write(struct.pack(f"{token_emb.size}f", *token_emb.flatten()))

        # 2. ln_1_weight: [n_layers, dim]
        for l in range(n_layers):
            w = state_dict[f"esm.encoder.layer.{l}.attention.LayerNorm.weight"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 3. ln_1_bias: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"esm.encoder.layer.{l}.attention.LayerNorm.bias"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 4. wq: [n_layers, dim, dim]
        for l in range(n_layers):
            w = state_dict[f"esm.encoder.layer.{l}.attention.self.query.weight"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 5. bq: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"esm.encoder.layer.{l}.attention.self.query.bias"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 6. wk: [n_layers, dim, dim]
        for l in range(n_layers):
            w = state_dict[f"esm.encoder.layer.{l}.attention.self.key.weight"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 7. bk: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"esm.encoder.layer.{l}.attention.self.key.bias"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 8. wv: [n_layers, dim, dim]
        for l in range(n_layers):
            w = state_dict[f"esm.encoder.layer.{l}.attention.self.value.weight"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 9. bv: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"esm.encoder.layer.{l}.attention.self.value.bias"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 10. wo: [n_layers, dim, dim]
        for l in range(n_layers):
            w = state_dict[f"esm.encoder.layer.{l}.attention.output.dense.weight"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 11. bo: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"esm.encoder.layer.{l}.attention.output.dense.bias"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 12. ln_2_weight: [n_layers, dim]
        for l in range(n_layers):
            w = state_dict[f"esm.encoder.layer.{l}.LayerNorm.weight"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 13. ln_2_bias: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"esm.encoder.layer.{l}.LayerNorm.bias"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 14. fc1_weight: [n_layers, dim, hidden_dim]
        for l in range(n_layers):
            w = state_dict[f"esm.encoder.layer.{l}.intermediate.dense.weight"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 15. fc1_bias: [n_layers, hidden_dim]
        for l in range(n_layers):
            b = state_dict[f"esm.encoder.layer.{l}.intermediate.dense.bias"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 16. fc2_weight: [n_layers, hidden_dim, dim]
        for l in range(n_layers):
            w = state_dict[f"esm.encoder.layer.{l}.output.dense.weight"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{w.size}f", *w.flatten()))

        # 17. fc2_bias: [n_layers, dim]
        for l in range(n_layers):
            b = state_dict[f"esm.encoder.layer.{l}.output.dense.bias"].cpu().numpy().astype("float32")
            f.write(struct.pack(f"{b.size}f", *b.flatten()))

        # 18. ln_f_weight and ln_f_bias: [dim]
        ln_f_w = state_dict["esm.encoder.emb_layer_norm_after.weight"].cpu().numpy().astype("float32")
        ln_f_b = state_dict["esm.encoder.emb_layer_norm_after.bias"].cpu().numpy().astype("float32")
        f.write(struct.pack(f"{ln_f_w.size}f", *ln_f_w.flatten()))
        f.write(struct.pack(f"{ln_f_b.size}f", *ln_f_b.flatten()))

        # 19. Classifier weights (EsmLMHead) if present
        for k in [
            "lm_head.layer_norm.weight",
            "lm_head.layer_norm.bias",
            "lm_head.dense.weight",
            "lm_head.dense.bias",
            "lm_head.decoder.weight",
            "lm_head.bias",
        ]:
            if k in state_dict:
                t = state_dict[k].cpu().numpy().astype("float32")
                f.write(struct.pack(f"{t.size}f", *t.flatten()))

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Exported {out_path} ({size_mb:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Export ESM-2 weights to codon-infer binary format.")
    parser.add_argument(
        "--model",
        choices=["8M", "35M", "150M", "650M", "3B", "all"],
        default="8M",
        help="Model scale to export (default: 8M)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "transformers", "fair-esm"],
        default="auto",
        help="Python backend to use (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save exported .bin files (default: current directory)",
    )
    args = parser.parse_args()

    # Determine backend
    backend = args.backend
    if backend == "auto":
        try:
            import transformers
            backend = "transformers"
        except ImportError:
            try:
                import esm
                backend = "fair-esm"
            except ImportError:
                print("Error: Neither 'transformers' nor 'esm' (fair-esm) is installed.", file=sys.stderr)
                print("Install with: pip install transformers torch  OR  pip install fair-esm torch", file=sys.stderr)
                sys.exit(1)

    targets = ["8M", "35M", "150M", "650M", "3B"] if args.model == "all" else [args.model]
    for m in targets:
        meta = MODELS[m]
        out_path = os.path.join(args.output_dir, meta["output"])
        print(f"\n--- Exporting ESM-2 {m} via {backend} to {out_path} ---")
        if backend == "transformers":
            export_huggingface(meta, out_path)
        else:
            export_fair_esm(meta, out_path)

    print("\nAll requested weights exported successfully.")

if __name__ == "__main__":
    main()
