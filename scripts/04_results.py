"""Phase 4: Generate comparison table and quality-vs-memory plot."""

import csv
import json
import os
import sys
import matplotlib.pyplot as plt


def main() -> None:
    print("=== Phase 4: Results Summary ===\n")

    # Load all results
    for path in ["results/baseline.json", "results/quantization_results.json"]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run earlier phases first.")
            sys.exit(1)

    with open("results/baseline.json") as f:
        baseline = json.load(f)
    with open("results/quantization_results.json") as f:
        quant = json.load(f)

    baseline_ppl = baseline["perplexity"]
    baseline_expert_mem = baseline["expert_memory_bytes"]

    variants = [
        ("FP16 Baseline", baseline_ppl, baseline_expert_mem),
        ("INT8 All", quant["int8_all"]["perplexity"], quant["int8_all"]["expert_memory_bytes"]),
        ("INT4 Uniform", quant["int4_uniform"]["perplexity"], quant["int4_uniform"]["expert_memory_bytes"]),
        ("Activation-Aware", quant["activation_aware"]["perplexity"], quant["activation_aware"]["expert_memory_bytes"]),
    ]

    # Print comparison table
    header = f"{'Variant':<20} | {'Perplexity':>12} | {'Expert Mem (MB)':>16} | {'Delta PPL':>10} | {'Mem Savings':>12}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    rows = []
    for name, ppl, mem in variants:
        mem_mb = mem / 1e6
        if name == "FP16 Baseline":
            delta_ppl = "--"
            mem_savings = "--"
        else:
            delta_ppl = f"+{ppl - baseline_ppl:.4f}"
            mem_savings = f"{(1 - mem / baseline_expert_mem) * 100:.1f}%"

        print(f"{name:<20} | {ppl:>12.4f} | {mem_mb:>16.1f} | {delta_ppl:>10} | {mem_savings:>12}")
        rows.append([name, f"{ppl:.4f}", f"{mem_mb:.1f}", delta_ppl, mem_savings])

    print(sep)

    # Save CSV
    with open("results/comparison_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "Perplexity", "Expert Memory (MB)", "Delta PPL", "Mem Savings"])
        writer.writerows(rows)
    print("\nSaved table to results/comparison_table.csv")

    # Generate quality-vs-memory plot
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    markers = ["o", "s", "D", "^"]

    for i, (name, ppl, mem) in enumerate(variants):
        mem_mb = mem / 1e6
        ax.scatter(mem_mb, ppl, c=colors[i], marker=markers[i], s=120, zorder=5, label=name)
        ax.annotate(
            name,
            (mem_mb, ppl),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=9,
        )

    ax.set_xlabel("Expert Memory (MB)", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("Quality vs. Memory: Expert Quantization Variants", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/quality_vs_memory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot to results/quality_vs_memory.png")

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
