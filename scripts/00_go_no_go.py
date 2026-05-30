"""Phase 0: Verify router logit extraction works on the target MoE model."""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "allenai/OLMoE-1B-7B-0924",
    "Qwen/Qwen1.5-MoE-A2.7B",
]


def try_model(model_name: str) -> bool:
    print(f"\n{'='*60}")
    print(f"Trying: {model_name}")
    print(f"{'='*60}")

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Loading model in fp16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"GPU memory used: {mem_gb:.2f} GB")

        print("Running forward pass with output_router_logits=True...")
        inputs = tokenizer("The Mixture of Experts architecture", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)

        router_logits = outputs.router_logits
        if router_logits is None:
            print("FAIL: router_logits is None")
            return False

        print(f"Router logits: {len(router_logits)} layers")
        for i, logits in enumerate(router_logits):
            print(f"  Layer {i}: shape {logits.shape}")

        # Verify expert access path
        layer_0 = model.model.layers[0]
        experts = layer_0.mlp.experts
        print(f"\nExpert weights access path verified:")
        print(f"  gate_up_proj shape: {experts.gate_up_proj.shape}")
        print(f"  down_proj shape:    {experts.down_proj.shape}")

        num_layers = len(model.model.layers)
        num_experts = experts.gate_up_proj.shape[0]
        print(f"\nModel summary: {num_layers} layers, {num_experts} experts/layer")

        print(f"\n>>> PASS: {model_name} is ready <<<")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def main() -> None:
    print("=== Phase 0: Go/No-Go Check ===")
    print("Verifying router logit extraction on MoE model\n")

    for model_name in MODELS:
        if try_model(model_name):
            sys.exit(0)

    print("\nAll models failed. Cannot proceed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
