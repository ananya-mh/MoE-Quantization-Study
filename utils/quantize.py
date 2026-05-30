import torch
import torch.nn as nn
from typing import Callable


def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel (per-row) symmetric INT8 quantization."""
    w = weight.float()
    abs_max = w.abs().amax(dim=1)
    scales = abs_max / 127.0
    scales = scales.clamp(min=1e-10)
    quantized = (w / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return quantized, scales


def dequantize_int8(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return (quantized.float() * scales.unsqueeze(1)).to(torch.float16)


def quantize_int4(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-wise symmetric INT4 quantization along the input (column) dimension."""
    w = weight.float()
    out_dim, in_dim = w.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by group_size {group_size}"
    num_groups = in_dim // group_size
    grouped = w.reshape(out_dim, num_groups, group_size)
    abs_max = grouped.abs().amax(dim=2)
    scales = abs_max / 7.0
    scales = scales.clamp(min=1e-10)
    quantized = (grouped / scales.unsqueeze(2)).round().clamp(-7, 7).to(torch.int8)
    return quantized, scales


def dequantize_int4(quantized: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    out_dim = quantized.shape[0]
    in_dim = quantized.shape[1] * quantized.shape[2]
    dequantized = (quantized.float() * scales.unsqueeze(2)).reshape(out_dim, in_dim)
    return dequantized.to(torch.float16)


def simulate_int8(weight: torch.Tensor) -> torch.Tensor:
    """Quantize to INT8 and immediately dequantize. Returns fp16."""
    q, s = quantize_int8(weight)
    return dequantize_int8(q, s)


def simulate_int4(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Quantize to INT4 and immediately dequantize. Returns fp16."""
    q, s = quantize_int4(weight, group_size)
    return dequantize_int4(q, s, group_size)


def apply_quantization_to_expert(
    experts_module: nn.Module,
    expert_idx: int,
    simulate_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Replace one expert's weights with simulated-quantized versions in-place."""
    with torch.no_grad():
        gate_up = experts_module.gate_up_proj.data[expert_idx].cpu()
        experts_module.gate_up_proj.data[expert_idx] = simulate_fn(gate_up).to(experts_module.gate_up_proj.device)

        down = experts_module.down_proj.data[expert_idx].cpu()
        experts_module.down_proj.data[expert_idx] = simulate_fn(down).to(experts_module.down_proj.device)


def compute_expert_memory(experts_module: nn.Module, expert_idx: int, bit_width: int = 16) -> int:
    """Theoretical memory in bytes for one expert at the given bit-width."""
    gate_up_shape = experts_module.gate_up_proj.data[expert_idx].shape
    down_shape = experts_module.down_proj.data[expert_idx].shape
    gate_up_elements = gate_up_shape[0] * gate_up_shape[1]
    down_elements = down_shape[0] * down_shape[1]
    total_elements = gate_up_elements + down_elements

    if bit_width == 16:
        return total_elements * 2
    elif bit_width == 8:
        weight_bytes = total_elements * 1
        scale_bytes = (gate_up_shape[0] + down_shape[0]) * 4
        return weight_bytes + scale_bytes
    elif bit_width == 4:
        weight_bytes = total_elements // 2
        group_size = 128
        num_groups_gate_up = gate_up_shape[0] * (gate_up_shape[1] // group_size)
        num_groups_down = down_shape[0] * (down_shape[1] // group_size)
        scale_bytes = (num_groups_gate_up + num_groups_down) * 4
        return weight_bytes + scale_bytes
    else:
        raise ValueError(f"Unsupported bit_width: {bit_width}")
