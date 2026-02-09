"""
Merge Tinker LoRA adapter weights with gpt-oss-120b model on Modal.

Requirements:
1. Print merge statistics for each tensor:
   - L2, L1, L0 norms for: lora_A, lora_B, original, merged, diff
2. Handle mixed precision correctly:
   - BF16: attention (q/k/v/o_proj) - 145 tensors
   - MXFP4: MoE experts (gate_up_proj fused, down_proj) - 108 tensors
3. Merge all 253 LoRA-targeted weights (no weights skipped)
4. CPU-only execution (no GPU required)
5. Process shard-by-shard to limit memory usage
6. Optimize compute with Numba JIT (parallel, fastmath)

Model precision (gpt-oss-120b):
- BF16: self_attn.{q,k,v,o}_proj, router, embed_tokens, lm_head
- MXFP4: mlp.experts.{gate_up_proj, down_proj}
  - Block size 32, E2M1 values (4-bit), E8M0 scales (8-bit)


Execution
uv run modal volume list
uv run modal volume delete merged-model

# take the sampler path
uv run modal run --detach merge_adapter.py --tinker-model "tinker://...:train:0/sampler_weights/final"

# test the merged model with Modal inference
uv run modal deploy inference_merged.py
./notebook.sh
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable

import modal
import numpy as np
import torch

# =============================================================================
# Constants
# =============================================================================

# FP4 E2M1 values: [0, 0.5, 1, 1.5, 2, 3, 4, 6] for positive, same negated for negative
FP4_VALUES = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)

# Quantization boundaries (midpoints between adjacent FP4 values)
FP4_BOUNDARIES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)

BASE_MODEL = "openai/gpt-oss-120b"
NUM_LAYERS = 36
NUM_EXPERTS = 128

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class NormStats:
    """Per-element normalized statistics."""

    l2_avg: float  # RMS value
    l1_avg: float  # Mean absolute value
    l0_avg: float  # Fraction of non-zero elements


@dataclass
class MergeStats:
    """Statistics for a merged weight tensor."""

    lora_A: NormStats
    lora_B: NormStats
    original: NormStats
    merged: NormStats
    diff: NormStats
    num_elements: int
    num_changed: int
    precision: str
    fp4_dist_before: list[int] | None = None
    fp4_dist_after: list[int] | None = None
    bf16_dists: dict[str, list[int]] | None = None


@dataclass
class LoraInfo:
    """LoRA adapter weights and scaling factor."""

    lora_A: torch.Tensor
    lora_B: torch.Tensor
    scaling: float


# =============================================================================
# Numba JIT Functions (lazy initialization)
# =============================================================================

_numba_dequantize: Callable | None = None
_numba_quantize: Callable | None = None


def _init_numba():
    """Initialize numba JIT functions. Called once in Modal container."""
    global _numba_dequantize, _numba_quantize
    if _numba_dequantize is not None:
        return

    from numba import njit, prange  # type: ignore[import-not-found]

    @njit(parallel=True, cache=True, fastmath=True)
    def dequantize(blocks, scales, fp4_values):
        """Dequantize MXFP4 blocks to float32."""
        n_exp, out_dim, n_scales, _ = blocks.shape
        result = np.empty((n_exp, out_dim, n_scales * 32), dtype=np.float32)

        for e in prange(n_exp):
            for o in range(out_dim):
                for s in range(n_scales):
                    scale = 2.0 ** (int(scales[e, o, s]) - 127)
                    base = s * 32
                    for p in range(16):
                        b = blocks[e, o, s, p]
                        result[e, o, base + p * 2] = fp4_values[b & 0x0F] * scale
                        result[e, o, base + p * 2 + 1] = fp4_values[b >> 4] * scale
        return result

    @njit(parallel=True, cache=True, fastmath=True)
    def quantize(tensor, n_scales, boundaries):
        """Quantize float32 tensor to MXFP4."""
        n_exp, out_dim, _ = tensor.shape
        blocks = np.empty((n_exp, out_dim, n_scales, 16), dtype=np.uint8)
        scales = np.empty((n_exp, out_dim, n_scales), dtype=np.uint8)

        for e in prange(n_exp):
            for o in range(out_dim):
                for s in range(n_scales):
                    base = s * 32
                    # Find max absolute value in block
                    abs_max = 0.0
                    for i in range(32):
                        v = tensor[e, o, base + i]
                        av = v if v >= 0 else -v
                        if av > abs_max:
                            abs_max = av

                    # Compute scale exponent
                    if abs_max < 1e-12:
                        exp = -127
                    else:
                        exp = int(np.ceil(np.log2(abs_max / 6.0)))
                        exp = max(-127, min(127, exp))

                    scale_val = 2.0**exp
                    scales[e, o, s] = np.uint8(exp + 127)

                    # Quantize and pack pairs into bytes
                    for p in range(16):
                        n0 = tensor[e, o, base + p * 2] / scale_val
                        n1 = tensor[e, o, base + p * 2 + 1] / scale_val
                        a0, a1 = (n0 if n0 >= 0 else -n0), (n1 if n1 >= 0 else -n1)

                        # Find bucket via linear search
                        q0 = q1 = 0
                        for b in range(7):
                            if a0 >= boundaries[b]:
                                q0 = b + 1
                            if a1 >= boundaries[b]:
                                q1 = b + 1
                        if n0 < 0:
                            q0 += 8
                        if n1 < 0:
                            q1 += 8

                        blocks[e, o, s, p] = np.uint8(q0 | (q1 << 4))

        return blocks, scales

    _numba_dequantize = dequantize
    _numba_quantize = quantize
    log("Numba JIT compiled")


# =============================================================================
# MXFP4 Operations
# =============================================================================


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 tensor to float32."""
    assert _numba_dequantize is not None, "Call _init_numba() first"
    result = _numba_dequantize(blocks.numpy(), scales.numpy(), FP4_VALUES)
    return torch.from_numpy(result)


def quantize_mxfp4(
    tensor: torch.Tensor, blocks_shape: tuple
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize float32 tensor to MXFP4."""
    assert _numba_quantize is not None, "Call _init_numba() first"
    blocks, scales = _numba_quantize(tensor.numpy(), blocks_shape[2], FP4_BOUNDARIES)
    return torch.from_numpy(blocks), torch.from_numpy(scales)


def count_fp4_distribution(blocks: torch.Tensor) -> list[int]:
    """Count occurrences of each FP4 value index (0-15) in packed blocks."""
    b = blocks.numpy().ravel()
    counts = np.bincount(b & 0x0F, minlength=16) + np.bincount(b >> 4, minlength=16)
    return counts.tolist()


# NOTE: each bucket do NOT have the same number of representable values
BF16_BOUNDARIES = torch.tensor([1 / 65556, 1 / 256, 1 / 64, 1 / 16, 1, 2])
BF16_LABELS: list[str] = [
    "[0,1/65556)", "[1/65556,1/256)",
    "[1/256,1/64)", "[1/64,1/16)",
    "[1/16,1)", "[1,16)", "[2,inf)",
]


def count_bf16_distribution(t: torch.Tensor) -> list[int]:
    """Count values in magnitude buckets, split by sign: [8 pos, 8 neg]."""
    f = t.float()
    a = f.abs()
    buckets = torch.bucketize(a, BF16_BOUNDARIES)
    pos_mask = f >= 0
    pos_counts = torch.bincount(buckets[pos_mask], minlength=8)[:8]
    neg_counts = torch.bincount(buckets[~pos_mask], minlength=8)[:8]
    return pos_counts.tolist() + neg_counts.tolist()


# =============================================================================
# Merge Functions
# =============================================================================


def apply_lora_delta(
    base: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
    is_batched: bool = False,
) -> torch.Tensor:
    """Apply LoRA delta to base tensor."""
    A, B = lora_A.float(), lora_B.float()
    if is_batched:
        n = base.shape[0]
        delta = torch.bmm(B.expand(n, -1, -1) * scaling, A.expand(n, -1, -1))
    else:
        delta = torch.nn.functional.linear(A.T, B * scaling).T
    return base.float() + delta


def merge_bf16(base: torch.Tensor, lora: LoraInfo) -> tuple[torch.Tensor, MergeStats]:
    """Merge LoRA into BF16 tensor."""
    base_f = base.float()
    merged = apply_lora_delta(base_f, lora.lora_A, lora.lora_B, lora.scaling)
    bf16_dists = {
        "lora_A": count_bf16_distribution(lora.lora_A),
        "lora_B": count_bf16_distribution(lora.lora_B),
        "original": count_bf16_distribution(base_f),
        "merged": count_bf16_distribution(merged),
    }
    stats = compute_stats(
        lora.lora_A, lora.lora_B, base_f, merged, "bf16",
        bf16_dists=bf16_dists,
    )
    return merged.to(base.dtype), stats


def merge_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    lora: LoraInfo,
) -> tuple[torch.Tensor, torch.Tensor, MergeStats]:
    """Merge LoRA into MXFP4 tensor."""
    dist_before = count_fp4_distribution(blocks)

    t0 = time.perf_counter()
    original = dequantize_mxfp4(blocks, scales)
    t_dq = time.perf_counter() - t0

    t0 = time.perf_counter()
    merged = apply_lora_delta(
        original, lora.lora_A, lora.lora_B, lora.scaling, is_batched=True
    )
    t_mm = time.perf_counter() - t0

    t0 = time.perf_counter()
    merged_blocks, merged_scales = quantize_mxfp4(merged.contiguous(), blocks.shape)
    t_q = time.perf_counter() - t0

    dist_after = count_fp4_distribution(merged_blocks)

    merged_dq = dequantize_mxfp4(merged_blocks, merged_scales)
    stats = compute_stats(
        lora.lora_A, lora.lora_B, original, merged_dq, "mxfp4",
        fp4_dist_before=dist_before, fp4_dist_after=dist_after,
    )

    log(f"    Timing: dequant={t_dq:.2f}s, matmul={t_mm:.2f}s, quant={t_q:.2f}s")
    return merged_blocks, merged_scales, stats


def merge_fused_gate_up(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    gate_lora: LoraInfo,
    up_lora: LoraInfo,
) -> tuple[torch.Tensor, torch.Tensor, MergeStats]:
    """Merge LoRA into fused gate_up_proj MXFP4 tensor."""
    dist_before = count_fp4_distribution(blocks)

    t0 = time.perf_counter()
    original = dequantize_mxfp4(blocks, scales)
    t_dq = time.perf_counter() - t0

    n_exp, fused_dim, in_dim = original.shape
    half = fused_dim // 2
    merged = original.clone()

    t0 = time.perf_counter()
    # Apply gate and up deltas to respective halves
    gate_delta = apply_lora_delta(
        torch.zeros(n_exp, half, in_dim),
        gate_lora.lora_A,
        gate_lora.lora_B,
        gate_lora.scaling,
        True,
    )
    merged[:, :half, :] += gate_delta
    del gate_delta

    up_delta = apply_lora_delta(
        torch.zeros(n_exp, half, in_dim),
        up_lora.lora_A,
        up_lora.lora_B,
        up_lora.scaling,
        True,
    )
    merged[:, half:, :] += up_delta
    del up_delta
    t_mm = time.perf_counter() - t0

    t0 = time.perf_counter()
    merged_blocks, merged_scales = quantize_mxfp4(merged.contiguous(), blocks.shape)
    t_q = time.perf_counter() - t0

    dist_after = count_fp4_distribution(merged_blocks)

    merged_dq = dequantize_mxfp4(merged_blocks, merged_scales)
    combined_A = torch.cat([gate_lora.lora_A, up_lora.lora_A], dim=0)
    combined_B = torch.cat([gate_lora.lora_B, up_lora.lora_B], dim=0)
    stats = compute_stats(
        combined_A, combined_B, original, merged_dq, "mxfp4",
        fp4_dist_before=dist_before, fp4_dist_after=dist_after,
    )

    log(f"    Timing: dequant={t_dq:.2f}s, matmul={t_mm:.2f}s, quant={t_q:.2f}s")
    return merged_blocks, merged_scales, stats


# =============================================================================
# Statistics
# =============================================================================


def compute_norm_stats(t: torch.Tensor) -> NormStats:
    """Compute normalized statistics for a tensor."""
    n = t.numel()
    return NormStats(
        l2_avg=torch.norm(t).item() / (n**0.5),
        l1_avg=t.abs().sum().item() / n,
        l0_avg=(t != 0).sum().item() / n,
    )


def compute_stats(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    original: torch.Tensor,
    merged: torch.Tensor,
    precision: str,
    fp4_dist_before: list[int] | None = None,
    fp4_dist_after: list[int] | None = None,
    bf16_dists: dict[str, list[int]] | None = None,
) -> MergeStats:
    """Compute merge statistics."""
    diff = merged - original
    n = original.numel()
    num_changed = int((diff != 0).sum().item())
    return MergeStats(
        lora_A=compute_norm_stats(lora_A),
        lora_B=compute_norm_stats(lora_B),
        original=compute_norm_stats(original),
        merged=compute_norm_stats(merged),
        diff=NormStats(
            l2_avg=torch.norm(diff).item() / (n**0.5),
            l1_avg=diff.abs().sum().item() / n,
            l0_avg=num_changed / n,
        ),
        num_elements=n,
        num_changed=num_changed,
        precision=precision,
        fp4_dist_before=fp4_dist_before,
        fp4_dist_after=fp4_dist_after,
        bf16_dists=bf16_dists,
    )


def fmt_fp4_dist(dist: list[int]) -> tuple[str, str]:
    """Format FP4 value distribution as positive and negative rows."""
    labels = ["+0", "+.5", "+1", "+1.5", "+2", "+3", "+4", "+6",
              "-0", "-.5", "-1", "-1.5", "-2", "-3", "-4", "-6"]
    total = sum(dist)
    pos = "   ".join(f"{labels[i]:>4}: {dist[i]/total:6.2%}" for i in range(8))
    neg = "   ".join(f"{labels[i]:>4}: {dist[i]/total:6.2%}" for i in range(8, 16))
    return pos, neg


def fmt_bf16_dist(dist: list[int]) -> tuple[str, str]:
    """Format BF16 magnitude distribution as positive and negative rows.

    dist has 16 elements: [8 pos buckets, 8 neg buckets].
    """
    total = sum(dist)
    pos = "   ".join(f"{BF16_LABELS[i]:>13}: {dist[i]/total:6.2%}" for i in range(len(BF16_LABELS)))
    neg = "   ".join(f"{BF16_LABELS[i]:>13}: {dist[i+8]/total:6.2%}" for i in range(len(BF16_LABELS)))
    return pos, neg


def log_stats(stats: MergeStats):
    """Log merge statistics."""
    log(f"    {stats.precision}, {stats.num_elements:,} elements")
    for name, ns in [
        ("lora_A", stats.lora_A),
        ("lora_B", stats.lora_B),
        ("original", stats.original),
        ("merged", stats.merged),
    ]:
        log(f"    {name}: L2={ns.l2_avg:.6f}, L1={ns.l1_avg:.6f}, L0={ns.l0_avg:.6f}")
    log(
        f"    diff: L2={stats.diff.l2_avg:.6f}, L1={stats.diff.l1_avg:.6f}, "
        f"L0={stats.diff.l0_avg:.6f} ({stats.num_changed:,} changed)"
    )
    if stats.fp4_dist_before is not None and stats.fp4_dist_after is not None:
        before_pos, before_neg = fmt_fp4_dist(stats.fp4_dist_before)
        after_pos, after_neg = fmt_fp4_dist(stats.fp4_dist_after)
        log(f"    FP4 before (pos): {before_pos}")
        log(f"    FP4 before (neg): {before_neg}")
        log(f"    FP4 after  (pos): {after_pos}")
        log(f"    FP4 after  (neg): {after_neg}")
    if stats.bf16_dists is not None:
        for name in ["lora_A", "lora_B", "original", "merged"]:
            pos, neg = fmt_bf16_dist(stats.bf16_dists[name])
            log(f"    BF16 {name:>8} (pos): {pos}")
            log(f"    BF16 {name:>8} (neg): {neg}")


def log_aggregate_stats(all_stats: dict[str, MergeStats]):
    """Log aggregate statistics."""
    log("\n=== Aggregate Merge Statistics ===")

    for precision in ["bf16", "mxfp4"]:
        stats_list = [s for s in all_stats.values() if s.precision == precision]
        if not stats_list:
            continue

        total_elem = sum(s.num_elements for s in stats_list)
        total_changed = sum(s.num_changed for s in stats_list)

        def avg(getter):
            return NormStats(
                l2_avg=sum(getter(s).l2_avg for s in stats_list) / len(stats_list),
                l1_avg=sum(getter(s).l1_avg for s in stats_list) / len(stats_list),
                l0_avg=sum(getter(s).l0_avg for s in stats_list) / len(stats_list),
            )

        log(
            f"\n{precision.upper()} ({len(stats_list)} tensors, {total_elem:,} elements):"
        )
        for name, ns in [
            ("LoRA A", avg(lambda s: s.lora_A)),
            ("LoRA B", avg(lambda s: s.lora_B)),
            ("Original", avg(lambda s: s.original)),
            ("Merged", avg(lambda s: s.merged)),
            ("Diff", avg(lambda s: s.diff)),
        ]:
            log(f"  {name}:\tL2={ns.l2_avg:.6f}, L1={ns.l1_avg:.6f}, L0={ns.l0_avg:.6f}")
        log(
            f"  Changed: {total_changed:,} / {total_elem:,} ({100 * total_changed / total_elem:.4f}%)"
        )

        # Aggregate FP4 value distributions
        fp4_before = [s.fp4_dist_before for s in stats_list if s.fp4_dist_before is not None]
        fp4_after = [s.fp4_dist_after for s in stats_list if s.fp4_dist_after is not None]
        if fp4_before and fp4_after:
            total_before = [sum(d[i] for d in fp4_before) for i in range(16)]
            total_after = [sum(d[i] for d in fp4_after) for i in range(16)]
            before_pos, before_neg = fmt_fp4_dist(total_before)
            after_pos, after_neg = fmt_fp4_dist(total_after)
            log(f"  FP4 before (pos): {before_pos}")
            log(f"  FP4 before (neg): {before_neg}")
            log(f"  FP4 after  (pos): {after_pos}")
            log(f"  FP4 after  (neg): {after_neg}")

        # Aggregate BF16 magnitude distributions
        bf16_stats = [s.bf16_dists for s in stats_list if s.bf16_dists is not None]
        if bf16_stats:
            for name in ["lora_A", "lora_B", "original", "merged"]:
                totals = [sum(d[name][i] for d in bf16_stats) for i in range(16)]
                pos, neg = fmt_bf16_dist(totals)
                log(f"  BF16 {name:>8} (pos): {pos}")
                log(f"  BF16 {name:>8} (neg): {neg}")


# =============================================================================
# LoRA Mapping
# =============================================================================


def build_lora_mappings(weights: dict, config: dict) -> tuple[dict, dict]:
    """Build mappings from base model weight names to LoRA info."""
    scaling = config["lora_alpha"] / config["r"]
    bf16_map, mxfp4_map = {}, {}

    for name in weights:
        if ".lora_A.weight" not in name:
            continue

        lora_A = weights[name]
        lora_B = weights[name.replace(".lora_A.weight", ".lora_B.weight")]

        # Transform to base model naming
        base = name.replace(".lora_A.weight", "").replace("base_model.model.", "")
        base = base.replace("model.unembed_tokens", "lm_head").replace(
            ".attn.", ".self_attn."
        )

        info = LoraInfo(lora_A=lora_A, lora_B=lora_B, scaling=scaling)

        if ".mlp.experts." in base:
            base = (
                base.replace(".w1", ".gate_proj")
                .replace(".w2", ".down_proj")
                .replace(".w3", ".up_proj")
            )
            mxfp4_map[base] = info
        else:
            bf16_map[base + ".weight"] = info

    log(f"LoRA mappings: {len(bf16_map)} BF16, {len(mxfp4_map)} MXFP4")
    return bf16_map, mxfp4_map


# =============================================================================
# Utilities
# =============================================================================


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# =============================================================================
# Modal App
# =============================================================================

merge_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "torch",
        "safetensors",
        "huggingface-hub",
        "tinker",
        "pydantic>=2.0,<3.0",
        "numba",
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
merged_model_vol = modal.Volume.from_name("merged-model", create_if_missing=True)

app = modal.App("merge-tinker-adapter")


@app.function(
    image=merge_image,
    cpu=4,
    memory=64 * 1024,
    timeout=120 * 60,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/merged": merged_model_vol},
    secrets=[modal.Secret.from_name("tinker-credentials")],
)
def merge_adapter(tinker_model: str, output_subdir: str):
    """Merge adapter weights into base model shard-by-shard."""
    import re
    import tarfile
    import urllib.request
    import tinker
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    _init_numba()

    # Download adapter if needed
    model_id = re.search(r"tinker://([a-f0-9-]+)", tinker_model)
    model_id = model_id.group(1) if model_id else "unknown"
    adapter_dir = f"/merged/adapters/{model_id}"

    config_path = weights_path = None
    if os.path.exists(f"{adapter_dir}/adapter"):
        for root, _, files in os.walk(f"{adapter_dir}/adapter"):
            for f in files:
                if f == "adapter_config.json":
                    config_path = os.path.join(root, f)
                elif f == "adapter_model.safetensors":
                    weights_path = os.path.join(root, f)

    if not (config_path and weights_path):
        log(f"Downloading adapter: {tinker_model}")
        os.makedirs(adapter_dir, exist_ok=True)
        sc = tinker.ServiceClient()
        url = (
            sc.create_rest_client()
            .get_checkpoint_archive_url_from_tinker_path(tinker_model)
            .result()
            .url
        )
        archive = f"{adapter_dir}/adapter.tar"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, "r") as tar:
            tar.extractall(f"{adapter_dir}/adapter")
        for root, _, files in os.walk(f"{adapter_dir}/adapter"):
            for f in files:
                if f == "adapter_config.json":
                    config_path = os.path.join(root, f)
                elif f == "adapter_model.safetensors":
                    weights_path = os.path.join(root, f)
    else:
        log("Using cached adapter")

    # Load adapter
    assert config_path is not None, "adapter_config.json not found"
    assert weights_path is not None, "adapter_model.safetensors not found"
    with open(config_path) as f:
        adapter_config = json.load(f)
    adapter_weights = load_file(weights_path, device="cpu")
    log(
        f"Adapter: r={adapter_config['r']}, alpha={adapter_config['lora_alpha']}, {len(adapter_weights)} tensors"
    )

    # Find base model
    cache_dir = f"/root/.cache/huggingface/hub/models--{BASE_MODEL.replace('/', '--')}/snapshots"
    model_path = None
    if os.path.exists(cache_dir):
        snapshots = os.listdir(cache_dir)
        if snapshots:
            model_path = f"{cache_dir}/{snapshots[0]}"
    if not model_path or not glob.glob(f"{model_path}/*.safetensors"):
        raise ValueError("Model not found in cache. Run inference.py first.")
    log(f"Base model: {model_path}")

    # Build mappings and process
    bf16_map, mxfp4_map = build_lora_mappings(adapter_weights, adapter_config)

    output_path = f"/merged/{output_subdir}"
    os.makedirs(output_path, exist_ok=True)

    all_stats = {}
    merged_keys = set()
    shard_files = sorted(glob.glob(f"{model_path}/*.safetensors"))

    for i, shard_file in enumerate(shard_files):
        shard_name = os.path.basename(shard_file)
        log(f"Shard {i + 1}/{len(shard_files)}: {shard_name}")

        with safe_open(shard_file, framework="pt", device="cpu") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        merged = {}

        # BF16 weights
        for key, tensor in tensors.items():
            if key in bf16_map:
                merged[key], stats = merge_bf16(tensor, bf16_map[key])
                all_stats[key] = stats
                merged_keys.add(key)
                log(f"  BF16: {key}")
                log_stats(stats)

        # MXFP4 weights
        for layer in range(NUM_LAYERS):
            prefix = f"model.layers.{layer}.mlp.experts"

            # down_proj
            blocks_key = f"{prefix}.down_proj_blocks"
            if blocks_key in tensors:
                lora_key = f"{prefix}.down_proj"
                mb, ms, stats = merge_mxfp4(
                    tensors[blocks_key],
                    tensors[f"{prefix}.down_proj_scales"],
                    mxfp4_map[lora_key],
                )
                merged[blocks_key], merged[f"{prefix}.down_proj_scales"] = mb, ms
                all_stats[lora_key] = stats
                merged_keys.add(lora_key)
                log(f"  MXFP4: {lora_key}")
                log_stats(stats)

            # gate_up_proj (fused)
            blocks_key = f"{prefix}.gate_up_proj_blocks"
            if blocks_key in tensors:
                mb, ms, stats = merge_fused_gate_up(
                    tensors[blocks_key],
                    tensors[f"{prefix}.gate_up_proj_scales"],
                    mxfp4_map[f"{prefix}.gate_proj"],
                    mxfp4_map[f"{prefix}.up_proj"],
                )
                merged[blocks_key], merged[f"{prefix}.gate_up_proj_scales"] = mb, ms
                all_stats[f"{prefix}.gate_up_proj"] = stats
                merged_keys.update([f"{prefix}.gate_proj", f"{prefix}.up_proj"])
                log(f"  MXFP4 fused: layer {layer} gate_up_proj")
                log_stats(stats)

        # Copy unmodified tensors
        for key in tensors:
            if key not in merged:
                merged[key] = tensors[key]

        save_file(merged, os.path.join(output_path, shard_name))
        log(f"  Saved: {shard_name}")

    # Verify completeness
    expected = set(bf16_map.keys()) | set(mxfp4_map.keys())
    missing = expected - merged_keys
    if missing:
        log(f"\nWARNING: {len(missing)} weights not merged")
    else:
        log(f"\n=== Successfully merged ALL {len(merged_keys)} adapter weights ===")

    # Copy config files
    for cfg in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    ]:
        src = os.path.join(model_path, cfg)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, cfg))

    log_aggregate_stats(all_stats)

    with open(os.path.join(output_path, "merge_stats.json"), "w") as f:
        json.dump({k: asdict(v) for k, v in all_stats.items()}, f, indent=2)

    merged_model_vol.commit()
    log(f"Saved to: {output_path}")
    return all_stats


@app.local_entrypoint()
def main(tinker_model: str):
    """Merge Tinker adapter with base model."""
    print(f"Merging: {tinker_model}")
    print("Output: /merged/model")

    merge_adapter.remote(tinker_model, "model")

    print("\n=== Merge Complete ===")
