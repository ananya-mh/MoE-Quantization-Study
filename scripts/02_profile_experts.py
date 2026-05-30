"""Phase 2: Profile expert activation frequencies and classify hot/cold experts."""

import json
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.eval import load_wikitext2_slice
from utils.profiling import profile_expert_activations, classify_experts

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"


def main() -> None:
    print("=== Phase 2: Expert Activation Profiling ===\n")

    print(f"Loading tokenizer and model ({MODEL_NAME}) in fp16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading WikiText-2 slice for profiling...")
    input_ids = load_wikitext2_slice(tokenizer, n_samples=10, seq_len=512)
    print(f"Profiling on {input_ids.size(0)} sequences ({input_ids.numel()} tokens)\n")

    print("Profiling expert activations...")
    activation_counts = profile_expert_activations(model, input_ids)

    print("Classifying experts into hot/cold buckets...")
    hot_set, cold_set = classify_experts(activation_counts, hot_fraction=0.33)
    print(f"  Hot experts:  {len(hot_set)}")
    print(f"  Cold experts: {len(cold_set)}")

    # Save results
    os.makedirs("results", exist_ok=True)

    serializable_counts = {str(k): v for k, v in activation_counts.items()}
    results = {
        "activation_counts": serializable_counts,
        "hot_experts": sorted([list(t) for t in hot_set]),
        "cold_experts": sorted([list(t) for t in cold_set]),
    }
    with open("results/expert_frequencies.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate histogram
    _plot_histogram(activation_counts)

    print(f"\nPhase 2 complete. Results saved to results/")


def _plot_histogram(activation_counts: dict[int, dict[int, int]]) -> None:
    """Plot aggregated expert activation frequency histogram."""
    aggregated: dict[int, int] = {}
    for layer_counts in activation_counts.values():
        for expert_id, count in layer_counts.items():
            aggregated[expert_id] = aggregated.get(expert_id, 0) + count

    expert_ids = sorted(aggregated.keys())
    counts = [aggregated[eid] for eid in expert_ids]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Bar chart: aggregated across layers
    axes[0].bar(expert_ids, counts, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Expert ID")
    axes[0].set_ylabel("Total Activation Count (all layers)")
    axes[0].set_title("Expert Activation Frequency (Aggregated)")

    # Heatmap: per-layer
    num_layers = max(activation_counts.keys()) + 1
    num_experts = max(max(ec.keys()) for ec in activation_counts.values()) + 1
    heatmap = np.zeros((num_layers, num_experts))
    for layer_idx, expert_counts in activation_counts.items():
        for expert_id, count in expert_counts.items():
            heatmap[layer_idx, expert_id] = count

    im = axes[1].imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    axes[1].set_xlabel("Expert ID")
    axes[1].set_ylabel("Layer")
    axes[1].set_title("Expert Activation Heatmap (per layer)")
    plt.colorbar(im, ax=axes[1], label="Activation Count")

    plt.tight_layout()
    plt.savefig("results/expert_frequency_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved histogram to results/expert_frequency_histogram.png")


if __name__ == "__main__":
    main()
