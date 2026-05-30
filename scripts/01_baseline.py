"""Phase 1: Establish full-precision perplexity and memory baselines."""

import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.eval import load_wikitext2_slice, evaluate_perplexity

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"


def main() -> None:
    print("=== Phase 1: Baseline Evaluation ===\n")

    print(f"Loading tokenizer and model ({MODEL_NAME}) in fp16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Memory measurements
    total_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    expert_memory = 0
    for layer in model.model.layers:
        experts = layer.mlp.experts
        expert_memory += experts.gate_up_proj.numel() * experts.gate_up_proj.element_size()
        expert_memory += experts.down_proj.numel() * experts.down_proj.element_size()

    print(f"Total model memory:  {total_memory / 1e9:.2f} GB")
    print(f"Expert-only memory:  {expert_memory / 1e9:.2f} GB")
    print(f"Expert share:        {expert_memory / total_memory * 100:.1f}%\n")

    # Perplexity evaluation
    print("Loading WikiText-2 slice (500 samples, seq_len=512)...")
    input_ids = load_wikitext2_slice(tokenizer, n_samples=500, seq_len=512)
    print(f"Loaded {input_ids.size(0)} sequences\n")

    print("Evaluating perplexity...")
    ppl = evaluate_perplexity(model, input_ids, batch_size=1)
    print(f"\nBaseline perplexity: {ppl:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "perplexity": round(ppl, 4),
        "total_memory_bytes": total_memory,
        "expert_memory_bytes": expert_memory,
        "n_samples": input_ids.size(0),
        "seq_len": 512,
        "model": MODEL_NAME,
    }
    with open("results/baseline.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPhase 1 complete. Results saved to results/baseline.json")


if __name__ == "__main__":
    main()
