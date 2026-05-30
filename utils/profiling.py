import torch
from collections import defaultdict
from tqdm import tqdm
from transformers import PreTrainedModel


def profile_expert_activations(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
) -> dict[int, dict[int, int]]:
    """Hook routers and tally per-layer per-expert activation counts.

    Returns {layer_idx: {expert_idx: count}}.
    """
    device = next(model.parameters()).device
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    hooks = []

    for layer_idx, layer in enumerate(model.model.layers):
        router = layer.mlp.gate

        def make_hook(lidx: int):
            def hook_fn(module, input, output):
                # output = (router_logits, router_scores, router_indices)
                router_indices = output[2]  # [num_tokens, top_k]
                for expert_id in router_indices.reshape(-1).tolist():
                    counts[lidx][expert_id] += 1
            return hook_fn

        hooks.append(router.register_forward_hook(make_hook(layer_idx)))

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(input_ids.size(0)), desc="Profiling expert activations"):
            batch = input_ids[i : i + 1].to(device)
            model(batch)

    for h in hooks:
        h.remove()

    return {k: dict(v) for k, v in counts.items()}


def classify_experts(
    activation_counts: dict[int, dict[int, int]],
    hot_fraction: float = 0.33,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Classify experts into hot (top third) and cold (bottom third) per layer.

    Returns (hot_set, cold_set) of (layer_idx, expert_idx) tuples.
    """
    hot_set: set[tuple[int, int]] = set()
    cold_set: set[tuple[int, int]] = set()

    for layer_idx, expert_counts in activation_counts.items():
        sorted_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)
        n_experts = len(sorted_experts)
        n_hot = max(1, int(n_experts * hot_fraction))
        n_cold = max(1, int(n_experts * hot_fraction))

        for expert_id, _ in sorted_experts[:n_hot]:
            hot_set.add((layer_idx, expert_id))
        for expert_id, _ in sorted_experts[-n_cold:]:
            cold_set.add((layer_idx, expert_id))

    return hot_set, cold_set
