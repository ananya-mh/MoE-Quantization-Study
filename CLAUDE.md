# MoE Expert Quantization Project

## What This Project Is
A study of activation-aware weight quantization for Mixture-of-Experts language models. We quantize expert FFN weights at different precision levels (INT8, group-wise INT4) and test whether allocating precision based on expert activation frequency preserves more quality than uniform quantization.

Target model: OLMoE-1B-7B (fallback: Qwen1.5-MoE-A2.7B)
Environment: Google Colab (free tier T4 GPU, 16GB VRAM)

## Project Structure
```
moe-expert-quantization/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── scripts/
│   ├── 00_go_no_go.py        # Phase 0: verify router logit extraction works
│   ├── 01_baseline.py         # Phase 1: full-precision perplexity + memory
│   ├── 02_profile_experts.py  # Phase 2: log expert activation frequencies
│   ├── 03_quantize.py         # Phase 3: INT8, uniform INT4, activation-aware INT4
│   └── 04_results.py          # Phase 4: comparison table + plots
├── utils/
│   ├── quantize.py            # Quantization helpers (per-channel INT8, group-wise INT4)
│   ├── eval.py                # Perplexity evaluation loop
│   └── profiling.py           # Router hook and expert frequency tallying
├── results/
│   └── (generated tables and plots go here)
└── notebooks/
    └── colab_runner.ipynb     # Single notebook to run everything on Colab
```

## Key Technical Constraints
- **Memory**: T4 has 16GB VRAM. OLMoE-1B-7B is ~14GB in fp16. Load in fp16, not fp32.
- **No training**: This is inference-only. We never fine-tune or train anything.
- **Router logits**: OLMoE exposes expert selections via `output_router_logits=True` in the forward pass. Always pass this flag when profiling.
- **Eval dataset**: Use a 500-sample slice of WikiText-2 for perplexity. Do not use the full dataset — it's slow and unnecessary for a comparative study.
- **Quantization scope**: Only quantize expert FFN weights (gate_proj, up_proj, down_proj inside each expert). Leave attention layers, router weights, embeddings, and layer norms at full precision.

## Workflow Per Phase
Each phase has its own script in `scripts/`. Run them in order. Each script should:
1. Print a clear summary of what it's doing
2. Save outputs (metrics, plots, tensors) to `results/`
3. Exit cleanly with a pass/fail message

### Phase 0 — Go/No-Go Check
Load OLMoE-1B-7B, run one forward pass with `output_router_logits=True`, confirm we can extract per-token expert selections. If this fails, try Qwen1.5-MoE-A2.7B before giving up.

### Phase 1 — Baseline
Run perplexity eval on the unquantized model. Record: perplexity, total model memory, expert-only memory. Save to `results/baseline.json`.

### Phase 2 — Expert Activation Profiling
Run ~500 tokens through the model, hook the router, tally per-expert activation counts. Bucket experts into "hot" (top third by frequency) and "cold" (bottom third). Save counts to `results/expert_frequencies.json`. Generate a histogram plot to `results/expert_frequency_histogram.png`.

### Phase 3 — Quantization
Implement three variants, each evaluated on the same perplexity slice:
1. **INT8**: Per-channel INT8 on all expert FFN weights. If perplexity spikes (>2 points above baseline), treat as a bug — do not proceed to INT4 until fixed.
2. **Uniform INT4**: Group-wise INT4 (group_size=128) on all expert FFN weights.
3. **Activation-aware INT4**: Cold experts at INT4, hot experts at INT8.

Save per-variant metrics to `results/quantization_results.json`.

### Phase 4 — Results
Generate a comparison table and a quality-vs-memory plot from the saved JSON files. Save to `results/`.

## Code Style
- Python 3.10+, type hints on function signatures
- Use PyTorch only (no ONNX, no TensorRT, no custom CUDA)
- Keep dependencies minimal: torch, transformers, datasets, matplotlib, numpy
- No classes where a function will do. Keep it flat and readable.
- Every script runnable standalone: `python scripts/01_baseline.py`
- Print progress with tqdm for any loop over data

## Quantization Implementation Notes
- **Per-channel INT8**: For each output channel, compute scale = max(abs(weight)) / 127. Store quantized weights as int8 + per-channel float32 scales.
- **Group-wise INT4**: Reshape weight into groups of 128 along the input dimension. Per group: scale = max(abs(group)) / 7, zero_point = 0 (symmetric). Pack two int4 values per byte if measuring memory; for simplicity, storing as int8 with range [-7, 7] is acceptable.
- **Dequantization**: Always dequantize before matmul (simulate quantization, don't write custom kernels). The goal is measuring quality impact, not building a production quantizer.
- Do NOT use bitsandbytes, GPTQ, AutoGPTQ, or any external quantization library. The point is implementing it from scratch.

## Common Pitfalls to Avoid
- Do not load the model in fp32 — it won't fit on a T4.
- Do not quantize the router. It's tiny and critical for correctness.
- Do not run full WikiText-2 eval — use a 500-sample slice.
- Do not add attention quantization, mixed-precision attention, or other scope expansions.
- If INT8 perplexity is bad, the quantization math is wrong. Fix it before moving on.

## README Expectations
The final README.md should include:
1. One-paragraph project summary
2. How to run (one command per phase)
3. Results table: variant | perplexity | memory | delta from baseline
4. The expert frequency histogram
5. The quality-vs-memory plot
6. A short "what I found" section (3-5 sentences)