import math
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel


def load_wikitext2_slice(
    tokenizer: PreTrainedTokenizer,
    n_samples: int = 500,
    seq_len: int = 512,
) -> torch.Tensor:
    """Load WikiText-2 test split, tokenize, chunk into fixed-length sequences."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings["input_ids"].squeeze(0)

    total_len = input_ids.size(0)
    num_chunks = total_len // seq_len
    actual_samples = min(n_samples, num_chunks)
    if actual_samples < n_samples:
        print(f"Warning: only {actual_samples} sequences available (requested {n_samples})")

    input_ids = input_ids[: actual_samples * seq_len].reshape(actual_samples, seq_len)
    return input_ids


def evaluate_perplexity(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    batch_size: int = 1,
) -> float:
    """Compute perplexity over input_ids using cross-entropy loss."""
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    n_samples = input_ids.size(0)

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Evaluating perplexity"):
            batch = input_ids[i : i + batch_size].to(device)
            outputs = model(batch, labels=batch)
            num_tokens = batch.numel() - batch.size(0)  # exclude first token per sequence
            total_nll += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity
