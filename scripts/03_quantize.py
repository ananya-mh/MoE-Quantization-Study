"""Phase 3: Apply INT8, uniform INT4, and activation-aware INT4 quantization."""

import gc
import json
import os
import sys
import torch
from functools import partial
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.quantize import (
    apply_quantization_to_expert,
    compute_expert_memory,
    simulate_int8,
    simulate_int4,
)
from utils.eval import load_wikitext2_slice, evaluate_perplexity

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"


def _load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )


def _unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _quantize_all_experts(model, simulate_fn, desc: str) -> int:
    """Apply simulate_fn to every expert. Returns theoretical memory in bytes."""
    total_memory = 0
    num_layers = len(model.model.layers)

    for layer_idx in tqdm(range(num_layers), desc=desc):
        experts = model.model.layers[layer_idx].mlp.experts
        num_experts = experts.gate_up_proj.shape[0]
        for expert_idx in range(num_experts):
            apply_quantization_to_expert(experts, expert_idx, simulate_fn)

    return total_memory


def _compute_total_expert_memory(model, bit_width: int) -> int:
    total = 0
    for layer in model.model.layers:
        experts = layer.mlp.experts
        num_experts = experts.gate_up_proj.shape[0]
        for expert_idx in range(num_experts):
            total += compute_expert_memory(experts, expert_idx, bit_width)
    return total


def run_int8_all(tokenizer, input_ids) -> dict:
    print("\n--- Variant 1: INT8 All Experts ---")
    model = _load_model()
    _quantize_all_experts(model, simulate_int8, "INT8 quantization")
    expert_memory = _compute_total_expert_memory(model, bit_width=8)
    ppl = evaluate_perplexity(model, input_ids, batch_size=1)
    print(f"INT8 perplexity: {ppl:.4f}")
    print(f"INT8 expert memory: {expert_memory / 1e9:.2f} GB")
    _unload_model(model)
    return {"perplexity": round(ppl, 4), "expert_memory_bytes": expert_memory}


def run_int4_uniform(tokenizer, input_ids) -> dict:
    print("\n--- Variant 2: Uniform INT4 All Experts ---")
    model = _load_model()
    _quantize_all_experts(model, simulate_int4, "INT4 quantization")
    expert_memory = _compute_total_expert_memory(model, bit_width=4)
    ppl = evaluate_perplexity(model, input_ids, batch_size=1)
    print(f"INT4 uniform perplexity: {ppl:.4f}")
    print(f"INT4 uniform expert memory: {expert_memory / 1e9:.2f} GB")
    _unload_model(model)
    return {"perplexity": round(ppl, 4), "expert_memory_bytes": expert_memory}


def run_activation_aware(tokenizer, input_ids) -> dict:
    print("\n--- Variant 3: Activation-Aware Mixed Precision ---")

    freq_path = "results/expert_frequencies.json"
    if not os.path.exists(freq_path):
        print(f"ERROR: {freq_path} not found. Run Phase 2 first.")
        sys.exit(1)

    with open(freq_path) as f:
        freq_data = json.load(f)
    hot_set = {tuple(x) for x in freq_data["hot_experts"]}
    cold_set = {tuple(x) for x in freq_data["cold_experts"]}
    print(f"Hot experts: {len(hot_set)}, Cold experts: {len(cold_set)}")

    model = _load_model()
    num_layers = len(model.model.layers)
    int8_count = 0
    int4_count = 0

    for layer_idx in tqdm(range(num_layers), desc="Activation-aware quantization"):
        experts = model.model.layers[layer_idx].mlp.experts
        num_experts = experts.gate_up_proj.shape[0]
        for expert_idx in range(num_experts):
            if (layer_idx, expert_idx) in cold_set:
                apply_quantization_to_expert(experts, expert_idx, simulate_int4)
                int4_count += 1
            else:
                apply_quantization_to_expert(experts, expert_idx, simulate_int8)
                int8_count += 1

    print(f"Applied INT8 to {int8_count} experts, INT4 to {int4_count} experts")

    # Compute mixed memory
    expert_memory = 0
    for layer_idx in range(num_layers):
        experts = model.model.layers[layer_idx].mlp.experts
        num_experts = experts.gate_up_proj.shape[0]
        for expert_idx in range(num_experts):
            if (layer_idx, expert_idx) in cold_set:
                expert_memory += compute_expert_memory(experts, expert_idx, bit_width=4)
            else:
                expert_memory += compute_expert_memory(experts, expert_idx, bit_width=8)

    ppl = evaluate_perplexity(model, input_ids, batch_size=1)
    print(f"Activation-aware perplexity: {ppl:.4f}")
    print(f"Activation-aware expert memory: {expert_memory / 1e9:.2f} GB")
    _unload_model(model)
    return {
        "perplexity": round(ppl, 4),
        "expert_memory_bytes": expert_memory,
        "int8_experts": int8_count,
        "int4_experts": int4_count,
    }


def main() -> None:
    print("=== Phase 3: Quantization Evaluation ===\n")

    # Load baseline for sanity check
    baseline_path = "results/baseline.json"
    if not os.path.exists(baseline_path):
        print(f"ERROR: {baseline_path} not found. Run Phase 1 first.")
        sys.exit(1)
    with open(baseline_path) as f:
        baseline = json.load(f)
    baseline_ppl = baseline["perplexity"]
    print(f"Baseline perplexity: {baseline_ppl:.4f}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Loading WikiText-2 slice...")
    input_ids = load_wikitext2_slice(tokenizer, n_samples=500, seq_len=512)
    print(f"Loaded {input_ids.size(0)} sequences\n")

    results = {}

    # Variant 1: INT8
    results["int8_all"] = run_int8_all(tokenizer, input_ids)
    if results["int8_all"]["perplexity"] > baseline_ppl + 2.0:
        print(f"\n!!! WARNING: INT8 perplexity ({results['int8_all']['perplexity']:.4f}) "
              f"is >{baseline_ppl + 2.0:.4f} (baseline + 2). Possible quantization bug! !!!")

    # Variant 2: Uniform INT4
    results["int4_uniform"] = run_int4_uniform(tokenizer, input_ids)

    # Variant 3: Activation-aware
    results["activation_aware"] = run_activation_aware(tokenizer, input_ids)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Phase 3 complete. Results saved to results/quantization_results.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
